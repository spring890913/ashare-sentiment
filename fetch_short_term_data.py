"""Build exact full-market A-share daily limit-up / limit-down / consecutive-board dataset (2021-2026).

Data sources:
  1. Baostock query_all_stock: official A-share universe (sh60/68, sz00/30) + stock names (ST filter)
  2. Tencent Kline API (unadjusted daily OHLCV + explicit ex-dividend markers): 2021-01-04 to 2026-11-20

Computes exact exchange-rounded limit-up / limit-down / broken-board / consecutive-board /
promotion-rate / yesterday-limit-up-premium / big-loss ("核按钮") series across ~5,000+ stocks,
plus realistic tradeable short-term relay returns (excluding untradeable one-line limit-ups).
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading
import time

import baostock as bs
import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

STOCK_LIST_PATH = DATA_DIR / "stock_list_a.csv"
OUT_DAILY_PATH = DATA_DIR / "short_term_daily.csv"

START_DATE = "2021-01-01"
END_DATE = "2026-11-22"
MAX_WORKERS = 24

_thread_local = threading.local()


def get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
        s.mount("https://", adapter)
        s.headers.update({"User-Agent": "Mozilla/5.0"})
        _thread_local.session = s
    return _thread_local.session


def get_stock_list() -> pd.DataFrame:
    if STOCK_LIST_PATH.exists() and STOCK_LIST_PATH.stat().st_size > 10000:
        df = pd.read_csv(STOCK_LIST_PATH, dtype=str)
        print(f"[CACHE] Loaded {len(df)} A-share stocks from {STOCK_LIST_PATH}", flush=True)
        return df

    print("Fetching full A-share stock list from Baostock...", flush=True)
    t0 = time.time()
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"Baostock login failed: {lg.error_msg}")
    try:
        rs = bs.query_all_stock(day="2026-11-20")
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
    finally:
        bs.logout()

    df = pd.DataFrame(rows, columns=["bs_code", "tradeStatus", "name"])
    # Filter to Shanghai/Shenzhen A-shares (exclude indices, ETFs, B-shares)
    mask = (
        df["bs_code"].str.startswith("sh.60")
        | df["bs_code"].str.startswith("sh.68")
        | df["bs_code"].str.startswith("sz.00")
        | df["bs_code"].str.startswith("sz.30")
    )
    df = df[mask].copy().reset_index(drop=True)
    df["tc_code"] = df["bs_code"].str.replace(".", "", regex=False)
    df["is_st"] = df["name"].str.contains("ST|退", case=False, na=False).astype(int)
    df["lim"] = np.where(
        df["tc_code"].str.startswith(("sz30", "sh68")), "0.20", "0.10"
    )
    df.to_csv(STOCK_LIST_PATH, index=False)
    print(f"[OK] Saved {len(df)} A-share stocks in {time.time()-t0:.1f}s", flush=True)
    return df


def fetch_and_process_one(tc_code: str, lim: float, is_st: bool) -> list[tuple] | None:
    """Fetch unadjusted daily OHLCV for one stock and extract per-day limit-up/down events."""
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    params = {"param": f"{tc_code},day,{START_DATE},{END_DATE},1600,"}
    sess = get_session()

    rows = None
    for attempt in range(3):
        try:
            r = sess.get(url, params=params, timeout=10)
            if r.status_code != 200:
                time.sleep(0.3 * (attempt + 1))
                continue
            data = r.json().get("data", {})
            if not isinstance(data, dict) or tc_code not in data:
                return None
            node = data[tc_code]
            rows = node.get("day") or node.get("qfqday")
            break
        except Exception:
            time.sleep(0.3 * (attempt + 1))

    if not rows or len(rows) < 6:
        return None

    n = len(rows)
    dates = [r[0] for r in rows]
    opens = np.array([float(r[1]) for r in rows], dtype=np.float64)
    closes = np.array([float(r[2]) for r in rows], dtype=np.float64)
    highs = np.array([float(r[3]) for r in rows], dtype=np.float64)
    lows = np.array([float(r[4]) for r in rows], dtype=np.float64)
    # Mark ex-dividend / ex-rights rows (Tencent attaches dict as 7th element)
    is_exdiv = np.array([len(r) > 6 and isinstance(r[6], dict) for r in rows], dtype=bool)

    # New IPOs listed after 2021-01-15 have no price limit in first 5 trading days
    new_ipo = dates[0] > "2021-01-15"
    start_idx = 5 if new_ipo else 1

    prev_c = np.roll(closes, 1)
    prev_c[0] = np.nan

    # Exact A-share tick rounding (round half up to 0.01 RMB)
    zt_px = np.floor(prev_c * (1.0 + lim) * 100.0 + 0.5) / 100.0
    dt_px = np.floor(prev_c * (1.0 - lim) * 100.0 + 0.5) / 100.0

    ret_c = closes / prev_c - 1.0
    ret_o = opens / prev_c - 1.0

    # Strictly exclude ST stocks & ex-dividend days from limit-up/down classification
    valid = (~is_exdiv) & (prev_c > 0.0) & (np.arange(n) >= start_idx)
    if is_st:
        # Still count up/down market breadth for non-exdiv valid days, but not limit-ups/downs
        is_zt = np.zeros(n, dtype=bool)
        is_zb = np.zeros(n, dtype=bool)
        is_dt = np.zeros(n, dtype=bool)
        is_yizi = np.zeros(n, dtype=bool)
    else:
        # Limit-up: closed at exact limit-up price (= high) and within +0.015 tick tolerance
        is_zt = valid & (closes >= zt_px - 1e-6) & (closes <= zt_px + 0.015) & (closes >= highs - 1e-6)
        is_yizi = is_zt & (opens >= zt_px - 1e-6) & (lows >= zt_px - 1e-6)
        # Broken limit-up (炸板): touched limit-up intraday, failed to close at limit-up
        is_zb = valid & (highs >= zt_px - 1e-6) & (highs <= zt_px + 0.015) & (closes < zt_px - 1e-6)
        # Limit-down: closed at exact limit-down price (= low)
        is_dt = valid & (closes <= dt_px + 1e-6) & (closes >= dt_px - 0.015) & (closes <= lows + 1e-6)

    # Consecutive limit-up streak
    lb = np.zeros(n, dtype=np.int16)
    for i in range(1, n):
        if is_zt[i]:
            lb[i] = lb[i - 1] + 1

    # Intraday pullback from high (for big-loss / "大面" detection)
    dd_from_high = closes / np.where(highs > 0, highs, np.nan) - 1.0

    # Pack lightweight per-day events for aggregation
    # Only emit days with valid prev_c
    events = []
    for i in range(start_idx, n):
        if is_exdiv[i] or prev_c[i] <= 0:
            continue
        prev_zt = bool(is_zt[i - 1])
        prev_lb = int(lb[i - 1])
        # Next-day tradeable relay return (if bought at open[i] when NOT yizi limit-up, sold at close[i+1])
        # Only valid if i+1 exists and is not ex-dividend!
        can_buy_open = (opens[i] < zt_px[i] - 1e-6) and (opens[i] > 0)
        next_o2c = np.nan
        if prev_zt and can_buy_open and (i + 1 < n) and (not is_exdiv[i + 1]):
            next_o2c = float(closes[i + 1] / opens[i] - 1.0)
        # Same-day open-to-close return for tradeable prev_zt
        same_o2c = float(closes[i] / opens[i] - 1.0) if (prev_zt and can_buy_open) else np.nan

        events.append((
            dates[i],
            1 if ret_c[i] > 0 else 0,
            1 if ret_c[i] < 0 else 0,
            1 if ret_c[i] >= 0.05 else 0,
            1 if ret_c[i] <= -0.05 else 0,
            1 if is_zt[i] else 0,
            1 if (is_zt[i] and lim > 0.15) else 0,
            1 if is_yizi[i] else 0,
            1 if is_zb[i] else 0,
            1 if is_dt[i] else 0,
            int(lb[i]),
            1 if (is_zt[i] and not is_yizi[i]) else 0,
            prev_zt,
            prev_lb,
            float(ret_c[i]) if prev_zt else 0.0,
            float(ret_o[i]) if prev_zt else 0.0,
            1 if (prev_zt and (ret_c[i] <= -0.05 or dd_from_high[i] <= -0.08)) else 0,
            same_o2c,
            next_o2c,
        ))
    return events


def main() -> None:
    stk_df = get_stock_list()
    tasks = [
        (row["tc_code"], float(row["lim"]), bool(int(row["is_st"])))
        for _, row in stk_df.iterrows()
    ]

    print(f"Downloading & processing {len(tasks)} stocks ({START_DATE} -> {END_DATE}) with {MAX_WORKERS} threads...", flush=True)
    t0 = time.time()

    # Daily accumulators
    acc = defaultdict(lambda: {
        "total": 0, "up": 0, "down": 0, "big_up": 0, "big_down": 0,
        "zt": 0, "zt_20cm": 0, "yizi": 0, "zb": 0, "dt": 0,
        "lb2_plus": 0, "lb3_plus": 0, "max_lb": 0, "max_lb_ex_yizi": 0,
        "prev_zt_cnt": 0, "prev_zt_promo": 0,
        "prev_1b_cnt": 0, "prev_1b_promo": 0,
        "prev_2b_plus_cnt": 0, "prev_2b_plus_promo": 0,
        "prev_zt_ret_sum": 0.0, "prev_zt_open_sum": 0.0,
        "prev_lb_cnt": 0, "prev_lb_ret_sum": 0.0,
        "big_loss_cnt": 0,
        "tradeable_zt_cnt": 0, "same_o2c_sum": 0.0,
        "relay_next_cnt": 0, "relay_next_sum": 0.0,
        "relay_lb_cnt": 0, "relay_lb_sum": 0.0,
    })

    done = 0
    ok = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(fetch_and_process_one, c, lim, st) for c, lim, st in tasks]
        for fut in as_completed(futs):
            done += 1
            ev = fut.result()
            if ev:
                ok += 1
                for (
                    dt, up, dn, bup, bdn, zt, zt20, yz, zb, dt_f, lb, non_yz_zt,
                    p_zt, p_lb, p_ret_c, p_ret_o, b_loss, s_o2c, n_o2c
                ) in ev:
                    d = acc[dt]
                    d["total"] += 1
                    d["up"] += up
                    d["down"] += dn
                    d["big_up"] += bup
                    d["big_down"] += bdn
                    if zt:
                        d["zt"] += 1
                        d["zt_20cm"] += zt20
                        d["yizi"] += yz
                        if lb >= 2:
                            d["lb2_plus"] += 1
                        if lb >= 3:
                            d["lb3_plus"] += 1
                        if lb > d["max_lb"]:
                            d["max_lb"] = lb
                        if non_yz_zt and lb > d["max_lb_ex_yizi"]:
                            d["max_lb_ex_yizi"] = lb
                    if zb:
                        d["zb"] += 1
                    if dt_f:
                        d["dt"] += 1
                    if p_zt:
                        d["prev_zt_cnt"] += 1
                        d["prev_zt_ret_sum"] += p_ret_c
                        d["prev_zt_open_sum"] += p_ret_o
                        d["big_loss_cnt"] += b_loss
                        if zt:
                            d["prev_zt_promo"] += 1
                        if p_lb == 1:
                            d["prev_1b_cnt"] += 1
                            if zt:
                                d["prev_1b_promo"] += 1
                        elif p_lb >= 2:
                            d["prev_2b_plus_cnt"] += 1
                            d["prev_lb_cnt"] += 1
                            d["prev_lb_ret_sum"] += p_ret_c
                            if zt:
                                d["prev_2b_plus_promo"] += 1
                        if not np.isnan(s_o2c):
                            d["tradeable_zt_cnt"] += 1
                            d["same_o2c_sum"] += s_o2c
                        if not np.isnan(n_o2c):
                            d["relay_next_cnt"] += 1
                            d["relay_next_sum"] += n_o2c
                            if p_lb >= 2:
                                d["relay_lb_cnt"] += 1
                                d["relay_lb_sum"] += n_o2c

            if done % 500 == 0 or done == len(tasks):
                print(f"  Progress: {done}/{len(tasks)} ({ok} ok) elapsed {time.time()-t0:.1f}s", flush=True)

    rows = []
    for dt in sorted(acc.keys()):
        d = acc[dt]
        if d["total"] < 1500:  # skip partial/holiday glitch dates
            continue
        zt, zb, dt_c = d["zt"], d["zb"], d["dt"]
        p_cnt = d["prev_zt_cnt"]
        rows.append({
            "date": dt,
            "total_stocks": d["total"],
            "up_count": d["up"],
            "down_count": d["down"],
            "up_ratio": d["up"] / d["total"],
            "big_up": d["big_up"],
            "big_down": d["big_down"],
            "zt_count": zt,
            "zt_20cm": d["zt_20cm"],
            "yizi_count": d["yizi"],
            "zb_count": zb,
            "zb_rate": zb / (zt + zb) if (zt + zb) > 0 else 0.0,
            "dt_count": dt_c,
            "lb2_count": d["lb2_plus"],
            "lb3_count": d["lb3_plus"],
            "max_lb": d["max_lb"],
            "max_lb_ex_yizi": d["max_lb_ex_yizi"],
            "prev_zt_cnt": p_cnt,
            "promo_rate": d["prev_zt_promo"] / p_cnt if p_cnt > 0 else 0.0,
            "promo_1to2": d["prev_1b_promo"] / d["prev_1b_cnt"] if d["prev_1b_cnt"] > 0 else 0.0,
            "promo_high": d["prev_2b_plus_promo"] / d["prev_2b_plus_cnt"] if d["prev_2b_plus_cnt"] > 0 else 0.0,
            "prev_zt_ret": d["prev_zt_ret_sum"] / p_cnt if p_cnt > 0 else 0.0,
            "prev_zt_open_ret": d["prev_zt_open_sum"] / p_cnt if p_cnt > 0 else 0.0,
            "prev_lb_ret": d["prev_lb_ret_sum"] / d["prev_lb_cnt"] if d["prev_lb_cnt"] > 0 else 0.0,
            "big_loss_rate": d["big_loss_cnt"] / p_cnt if p_cnt > 0 else 0.0,
            "tradeable_zt_cnt": d["tradeable_zt_cnt"],
            "prev_zt_same_o2c": d["same_o2c_sum"] / d["tradeable_zt_cnt"] if d["tradeable_zt_cnt"] > 0 else 0.0,
            "relay_next_ret": d["relay_next_sum"] / d["relay_next_cnt"] if d["relay_next_cnt"] > 0 else np.nan,
            "relay_lb_next_ret": d["relay_lb_sum"] / d["relay_lb_cnt"] if d["relay_lb_cnt"] > 0 else np.nan,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_DAILY_PATH, index=False)
    print(f"\n[DONE] Saved {len(out_df)} trading days to {OUT_DAILY_PATH} in {time.time()-t0:.1f}s", flush=True)
    print("\nLast 5 trading days (compare 2026-11-20 against Eastmoney: zt=28, max_lb=4, zb=7, dt=58):")
    cols = ["date", "zt_count", "dt_count", "zb_count", "zb_rate", "max_lb", "lb2_count", "promo_rate", "prev_zt_ret", "big_loss_rate", "up_ratio"]
    print(out_df[cols].tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
