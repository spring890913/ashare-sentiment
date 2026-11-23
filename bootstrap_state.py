"""Bootstrap or gap-fill per-stock limit-up state (data/daily_state.json) & recent daily bars.

Uses Tencent short-window Kline (with Sina Kline fallback) to reconstruct exact
limit-up stocks and consecutive-board streaks on the latest confirmed close date,
so daily_update.py only needs a 3-second full-market snapshot (qt.gtimg.cn) going forward.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import json
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STOCK_LIST_PATH = DATA_DIR / "stock_list_a.csv"
STATE_PATH = DATA_DIR / "daily_state.json"

_tl = threading.local()


def _get_session() -> requests.Session:
    if not hasattr(_tl, "sess"):
        s = requests.Session()
        ad = requests.adapters.HTTPAdapter(pool_connections=24, pool_maxsize=24)
        s.mount("https://", ad)
        s.mount("http://", ad)
        s.headers.update({"User-Agent": "Mozilla/5.0"})
        _tl.sess = s
    return _tl.sess


def _fetch_short_kline(tc_code: str, start_date: str, end_date: str) -> list[tuple] | None:
    """Return list of (date, open, close, high, low, is_exdiv) tuples."""
    sess = _get_session()
    # Primary: Tencent fqkline (includes explicit ex-div markers)
    try:
        r = sess.get(
            "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
            params={"param": f"{tc_code},day,{start_date},{end_date},30,"},
            timeout=8,
        )
        if r.status_code == 200:
            data = r.json().get("data", {})
            if isinstance(data, dict) and tc_code in data:
                node = data[tc_code]
                rows = node.get("day") or node.get("qfqday")
                if rows and len(rows) >= 2:
                    return [
                        (
                            str(row[0]),
                            float(row[1]),
                            float(row[2]),
                            float(row[3]),
                            float(row[4]),
                            len(row) > 6 and isinstance(row[6], dict),
                        )
                        for row in rows
                    ]
    except Exception:
        pass

    # Fallback: Sina Kline API
    try:
        r = sess.get(
            "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
            "CN_MarketData.getKLineData",
            params={"symbol": tc_code, "scale": "240", "ma": "no", "datalen": "25"},
            timeout=8,
        )
        if r.status_code == 200 and r.text.strip().startswith("["):
            arr = r.json()
            out = []
            for item in arr:
                dt = str(item["day"])
                if start_date <= dt <= end_date:
                    out.append((
                        dt,
                        float(item["open"]),
                        float(item["close"]),
                        float(item["high"]),
                        float(item["low"]),
                        False,
                    ))
            if len(out) >= 2:
                return out
    except Exception:
        pass
    return None


def compute_stock_zt_on_date(
    tc_code: str, name: str, lim: float, is_st: bool, target_date: str, start_date: str
) -> dict | None:
    if is_st:
        return None
    bars = _fetch_short_kline(tc_code, start_date, target_date)
    if not bars:
        return None
    # Keep bars up to target_date
    bars = [b for b in bars if b[0] <= target_date]
    if len(bars) < 2 or bars[-1][0] != target_date:
        return None

    n = len(bars)
    opens = np.array([b[1] for b in bars], dtype=np.float64)
    closes = np.array([b[2] for b in bars], dtype=np.float64)
    highs = np.array([b[3] for b in bars], dtype=np.float64)
    lows = np.array([b[4] for b in bars], dtype=np.float64)
    exdiv = np.array([b[5] for b in bars], dtype=bool)

    prev_c = np.roll(closes, 1)
    prev_c[0] = np.nan
    zt_px = np.floor(prev_c * (1.0 + lim) * 100.0 + 0.5) / 100.0
    valid = (~exdiv) & (prev_c > 0) & (np.arange(n) >= 1)
    is_zt = valid & (closes >= zt_px - 1e-6) & (closes <= zt_px + 0.015) & (closes >= highs - 1e-6)

    if not is_zt[-1]:
        return None

    # Count trailing streak ending on target_date
    lb = 0
    for i in range(n - 1, 0, -1):
        if is_zt[i]:
            lb += 1
        else:
            break
    is_yizi = bool(opens[-1] >= zt_px[-1] - 1e-6 and lows[-1] >= zt_px[-1] - 1e-6)
    return {
        "code": tc_code,
        "name": name,
        "close": round(float(closes[-1]), 3),
        "lb": int(lb),
        "yizi": is_yizi,
        "lim": lim,
    }


def bootstrap_state(target_date: str | None = None, max_workers: int = 16) -> dict:
    if target_date is None:
        st_df = pd.read_csv(DATA_DIR / "short_term_daily.csv")
        target_date = str(st_df["date"].iloc[-1])

    dt_obj = datetime.strptime(target_date, "%Y-%m-%d")
    start_date = (dt_obj - timedelta(days=35)).strftime("%Y-%m-%d")

    stk_df = pd.read_csv(STOCK_LIST_PATH, dtype=str)
    tasks = [
        (r["tc_code"], r["name"], float(r["lim"]), bool(int(r["is_st"])))
        for _, r in stk_df.iterrows()
        if not bool(int(r["is_st"]))
    ]
    print(f"Bootstrapping daily_state for {target_date} ({len(tasks)} non-ST stocks, window {start_date}..{target_date})...", flush=True)
    t0 = time.time()
    zt_stocks: dict[str, dict] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [
            ex.submit(compute_stock_zt_on_date, c, nm, lim, st, target_date, start_date)
            for c, nm, lim, st in tasks
        ]
        for fut in as_completed(futs):
            done += 1
            res = fut.result()
            if res is not None:
                code = res.pop("code")
                zt_stocks[code] = res
            if done % 1000 == 0 or done == len(tasks):
                print(f"  Progress: {done}/{len(tasks)} | zt found={len(zt_stocks)} ({time.time()-t0:.1f}s)", flush=True)

    state = {
        "last_confirmed_date": target_date,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "zt_count": len(zt_stocks),
        "max_lb": max((v["lb"] for v in zt_stocks.values()), default=0),
        "zt_stocks": dict(sorted(zt_stocks.items(), key=lambda kv: (-kv[1]["lb"], kv[0]))),
    }
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Saved {STATE_PATH}: date={target_date}, zt_count={state['zt_count']}, max_lb={state['max_lb']} in {time.time()-t0:.1f}s", flush=True)
    return state


if __name__ == "__main__":
    bootstrap_state()
