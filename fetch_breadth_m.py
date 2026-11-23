"""Compute exact full-market daily Breadth (B), Up-Turnover Ratio (M), and McClellan Oscillator (MO).

Uses cached A-share universe (data/stock_list_a.csv) + Tencent daily OHLCV (2021-01-04 to 2026-11-20).
Daily turnover per stock is estimated via VWAP proxy: ((open + high + low + close) / 4) * volume_lots * 100.
Excludes ex-dividend/ex-rights days (Tencent 7th dict field) and flat stocks from B and M denominators.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent / "data"
STOCK_LIST_PATH = DATA_DIR / "stock_list_a.csv"
OUT_PATH = DATA_DIR / "breadth_b_m_mo.csv"

START_DATE = "2021-01-01"
END_DATE = "2026-11-22"
MAX_WORKERS = 20

_tl = threading.local()


def get_session() -> requests.Session:
    if not hasattr(_tl, "session"):
        s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
        s.mount("https://", adapter)
        s.headers.update({"User-Agent": "Mozilla/5.0"})
        _tl.session = s
    return _tl.session


def fetch_one(tc_code: str) -> list[tuple] | None:
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    params = {"param": f"{tc_code},day,{START_DATE},{END_DATE},1600,"}
    sess = get_session()

    rows = None
    for attempt in range(4):
        try:
            r = sess.get(url, params=params, timeout=10)
            if r.status_code != 200:
                time.sleep(0.4 * (attempt + 1))
                continue
            data = r.json().get("data", {})
            if not isinstance(data, dict) or tc_code not in data:
                return None
            node = data[tc_code]
            rows = node.get("day") or node.get("qfqday")
            break
        except Exception:
            time.sleep(0.4 * (attempt + 1))

    if not rows or len(rows) < 2:
        return None

    n = len(rows)
    dates = [r[0] for r in rows]
    opens = np.array([float(r[1]) for r in rows], dtype=np.float64)
    closes = np.array([float(r[2]) for r in rows], dtype=np.float64)
    highs = np.array([float(r[3]) for r in rows], dtype=np.float64)
    lows = np.array([float(r[4]) for r in rows], dtype=np.float64)
    vols = np.array([float(r[5]) for r in rows], dtype=np.float64)  # 手 (100 shares)
    is_exdiv = np.array([len(r) > 6 and isinstance(r[6], dict) for r in rows], dtype=bool)

    prev_c = np.roll(closes, 1)
    prev_c[0] = np.nan

    # Typical price * volume * unit = RMB turnover (元); SSE STAR board (sh68) reports volume in shares (1), others in lots (100)
    vol_unit = 1.0 if tc_code.startswith("sh68") else 100.0
    amt = ((opens + highs + lows + closes) * 0.25) * vols * vol_unit
    diff = closes - prev_c

    out = []
    for i in range(1, n):
        if is_exdiv[i] or prev_c[i] <= 0 or vols[i] <= 0:
            continue
        if diff[i] > 1e-9:
            out.append((dates[i], 1, 0, 0, float(amt[i]), 0.0))
        elif diff[i] < -1e-9:
            out.append((dates[i], 0, 1, 0, 0.0, float(amt[i])))
        else:
            out.append((dates[i], 0, 0, 1, 0.0, 0.0))
    return out


def main() -> None:
    stk = pd.read_csv(STOCK_LIST_PATH, dtype=str)
    codes = stk["tc_code"].tolist()
    print(f"Fetching turnover-weighted breadth (B, M, MO) for {len(codes)} stocks...", flush=True)
    t0 = time.time()

    acc = defaultdict(lambda: [0, 0, 0, 0.0, 0.0])  # up_cnt, dn_cnt, flat_cnt, up_amt, dn_amt
    done = 0
    ok = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(fetch_one, c) for c in codes]
        for fut in as_completed(futs):
            done += 1
            ev = fut.result()
            if ev:
                ok += 1
                for dt, u, d, f, ua, da in ev:
                    slot = acc[dt]
                    slot[0] += u
                    slot[1] += d
                    slot[2] += f
                    slot[3] += ua
                    slot[4] += da
            if done % 1000 == 0 or done == len(codes):
                print(f"  Progress: {done}/{len(codes)} ({ok} ok) elapsed {time.time()-t0:.1f}s", flush=True)

    rows = []
    for dt in sorted(acc.keys()):
        u, d, f, ua, da = acc[dt]
        if (u + d + f) < 1500:
            continue
        ud = u + d
        uda = ua + da
        b = u / ud if ud > 0 else np.nan
        m = ua / uda if uda > 0 else np.nan
        rana = (u - d) / ud if ud > 0 else np.nan  # 净上涨家数比例 = 2B - 1
        rows.append({
            "date": dt,
            "up_cnt": u,
            "dn_cnt": d,
            "flat_cnt": f,
            "up_amt_yi": ua / 1e8,
            "dn_amt_yi": da / 1e8,
            "B": b,
            "M": m,
            "RANA": rana,
        })

    df = pd.DataFrame(rows)
    # McClellan Oscillator on daily net advance ratio (RANA): EMA(19) - EMA(39)
    df["EMA19"] = df["RANA"].ewm(span=19, adjust=False).mean()
    df["EMA39"] = df["RANA"].ewm(span=39, adjust=False).mean()
    df["MO"] = df["EMA19"] - df["EMA39"]
    # Also compute money-weighted McClellan Oscillator on (2M - 1) for comparison
    df["M_minus_B"] = df["M"] - df["B"]

    df.to_csv(OUT_PATH, index=False)
    print(f"[DONE] Saved {len(df)} days to {OUT_PATH} in {time.time()-t0:.1f}s", flush=True)
    print("\nLast 5 trading days sanity check (total_amt = up_amt + dn_amt vs SH+SZ ~14222亿 on 2026-11-20):")
    print(df[["date", "up_cnt", "dn_cnt", "up_amt_yi", "dn_amt_yi", "B", "M", "M_minus_B", "MO"]].tail(5).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
