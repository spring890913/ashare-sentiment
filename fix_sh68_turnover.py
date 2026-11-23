"""Correct STAR board (sh68) 100x volume-unit overcount in data/breadth_b_m_mo.csv via Sina Kline API."""

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
RAW_PATH = DATA_DIR / "breadth_b_m_mo_raw.csv"
OUT_PATH = DATA_DIR / "breadth_b_m_mo.csv"

_tl = threading.local()


def get_session() -> requests.Session:
    if not hasattr(_tl, "session"):
        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"})
        _tl.session = s
    return _tl.session


def fetch_sh68_one(tc_code: str) -> list[tuple] | None:
    url = "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
    params = {"symbol": tc_code, "scale": "240", "ma": "no", "datalen": "1500"}
    sess = get_session()
    rows = None
    for attempt in range(4):
        try:
            r = sess.get(url, params=params, timeout=10)
            if r.status_code == 200 and r.text.strip().startswith("["):
                rows = r.json()
                break
            time.sleep(0.5 * (attempt + 1))
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    if not rows or len(rows) < 2:
        return None

    dates = [x["day"] for x in rows]
    opens = np.array([float(x["open"]) for x in rows], dtype=np.float64)
    highs = np.array([float(x["high"]) for x in rows], dtype=np.float64)
    lows = np.array([float(x["low"]) for x in rows], dtype=np.float64)
    closes = np.array([float(x["close"]) for x in rows], dtype=np.float64)
    vols = np.array([float(x["volume"]) for x in rows], dtype=np.float64)  # shares (1 share)
    prev_c = np.roll(closes, 1)
    prev_c[0] = np.nan

    amt = ((opens + highs + lows + closes) * 0.25) * vols  # true RMB turnover (元)
    diff = closes - prev_c

    out = []
    for i in range(1, len(rows)):
        if dates[i] < "2021-01-01" or prev_c[i] <= 0 or vols[i] <= 0:
            continue
        if diff[i] > 1e-9:
            out.append((dates[i], float(amt[i]), 0.0))
        elif diff[i] < -1e-9:
            out.append((dates[i], 0.0, float(amt[i])))
    return out


def main() -> None:
    if not RAW_PATH.exists():
        # Preserve original raw file once so this script is 100% idempotent
        Path(RAW_PATH).write_bytes(OUT_PATH.read_bytes())

    base = pd.read_csv(RAW_PATH)
    stk = pd.read_csv(STOCK_LIST_PATH, dtype=str)
    sh68_codes = stk[stk["tc_code"].str.startswith("sh68")]["tc_code"].tolist()
    print(f"Fetching {len(sh68_codes)} STAR-board (sh68) stocks via Sina...", flush=True)
    t0 = time.time()

    acc = defaultdict(lambda: [0.0, 0.0])  # true up_amt, true dn_amt (元)
    done = 0
    ok = 0
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(fetch_sh68_one, c) for c in sh68_codes]
        for fut in as_completed(futs):
            done += 1
            ev = fut.result()
            if ev:
                ok += 1
                for dt, ua, da in ev:
                    acc[dt][0] += ua
                    acc[dt][1] += da
            if done % 150 == 0 or done == len(sh68_codes):
                print(f"  Progress: {done}/{len(sh68_codes)} ({ok} ok) elapsed {time.time()-t0:.1f}s", flush=True)

    # Subtract 99 * true_sh68_amt (since raw counted 100 * true_sh68_amt instead of 1 * true_sh68_amt)
    sh68_up_yi = base["date"].map(lambda d: acc[d][0] / 1e8)
    sh68_dn_yi = base["date"].map(lambda d: acc[d][1] / 1e8)

    fixed = base.copy()
    fixed["up_amt_yi"] = (fixed["up_amt_yi"] - 99.0 * sh68_up_yi).clip(lower=0.0)
    fixed["dn_amt_yi"] = (fixed["dn_amt_yi"] - 99.0 * sh68_dn_yi).clip(lower=0.0)
    fixed["total_amt_yi"] = fixed["up_amt_yi"] + fixed["dn_amt_yi"]
    fixed["M"] = fixed["up_amt_yi"] / fixed["total_amt_yi"].replace(0.0, np.nan)
    fixed["M_minus_B"] = fixed["M"] - fixed["B"]

    fixed.to_csv(OUT_PATH, index=False)
    print(f"\n[OK] Corrected {len(fixed)} days in {OUT_PATH} ({time.time()-t0:.1f}s)")
    print("\nSanity check vs actual SH+SZ turnover (~14222亿 on 2026-11-20, ~13051亿 on 11-19, ~13697亿 on 11-18):")
    cols = ["date", "up_cnt", "dn_cnt", "up_amt_yi", "dn_amt_yi", "total_amt_yi", "B", "M", "M_minus_B", "MO"]
    print(fixed[cols].tail(6).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
