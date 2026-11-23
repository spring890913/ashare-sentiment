"""Daily Incremental A-Share Sentiment Pipeline & Live Command Center Builder.

Fetches full-market A-share quotes (~5,200 stocks + 9 indices) in ~3 seconds via
Tencent batch quote API (qt.gtimg.cn), updates ACSI & STSI/MO/M-B indicators
strictly causally (zero look-ahead), writes structured daily outputs, rebuilds
self-contained HTML dashboards (results_live/index.html & docs/index.html),
and optionally pushes a daily briefing via Webhook (--notify / WEBHOOK_URL).

Modes:
  - Post-close (>= 15:00 BJT on a new trading day): safely commits the new bar
    into data/*.csv and data/daily_state.json (idempotent).
  - Intraday (09:25-14:59 BJT on a new trading day): preserves confirmed EOD
    history untouched AND computes a live intraday preview side-by-side.
  - Multi-day gap: automatically self-heals missing trading days via short Kline.

Usage:
  .venv/bin/python daily_update.py              # standard daily run
  .venv/bin/python daily_update.py --fast       # skip slow AkShare sub-feeds (~4s)
  .venv/bin/python daily_update.py --notify     # also send Webhook alert
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

warnings.filterwarnings("ignore")

from explore_factors import load_all
from explore_short_term import build_stsi, classify_cycle_phase
from run_full_backtest import build_acsi, strategy_hysteresis
from run_short_term_backtest import (
    rolling_z_cdf,
    strat_cycle_from_health,
    strat_style_weight,
    exec_dual_engine,
    START_DATE as ST_START,
    COST_ETF,
    CASH_MMF,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LIVE_DIR = BASE_DIR / "results_live"
DOCS_DIR = BASE_DIR / "docs"
LIVE_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

STOCK_LIST_PATH = DATA_DIR / "stock_list_a.csv"
STATE_PATH = DATA_DIR / "daily_state.json"
ST_DAILY_PATH = DATA_DIR / "short_term_daily.csv"
BM_DAILY_PATH = DATA_DIR / "breadth_b_m_mo.csv"
TEMPLATE_PATH = BASE_DIR / "dashboard_template.html"

BJT = timezone(timedelta(hours=8))

INDEX_MAP = {
    "hs300": "sh000300",
    "sh_comp": "sh000001",
    "sz_comp": "sz399106",
    "sse50": "sh000016",
    "zz500": "sh000905",
    "zz1000": "sh000852",
    "chinext": "sz399006",
    "zz_all": "sh000985",
    "zz_div": "sh000922",
}


def _sf(val: str, default: float = 0.0) -> float:
    try:
        return float(val) if val else default
    except Exception:
        return default


def fetch_tencent_quotes(codes: list[str], batch_size: int = 80, max_workers: int = 6) -> dict[str, dict]:
    """Fetch live/close batch quotes for A-shares and indices in ~3 seconds."""
    chunks = [codes[i : i + batch_size] for i in range(0, len(codes), batch_size)]
    sess = requests.Session()
    ad = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16)
    sess.mount("https://", ad)
    sess.headers.update({"User-Agent": "Mozilla/5.0"})

    def _fetch_chunk(ch: list[str]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        url = f"https://qt.gtimg.cn/q={','.join(ch)}"
        for attempt in range(3):
            try:
                r = sess.get(url, timeout=8)
                if r.status_code != 200:
                    time.sleep(0.3 * (attempt + 1))
                    continue
                for line in r.text.splitlines():
                    if not line.startswith("v_"):
                        continue
                    eq = line.find('="')
                    if eq < 0:
                        continue
                    code = line[2:eq]
                    parts = line[eq + 2 :].rstrip('";').split("~")
                    if len(parts) < 49:
                        continue
                    ts = parts[30]
                    dt_str = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}" if len(ts) >= 8 else ""
                    tm_str = f"{ts[8:10]}:{ts[10:12]}:{ts[12:14]}" if len(ts) >= 14 else ""
                    out[code] = {
                        "code": code,
                        "name": parts[1],
                        "close": _sf(parts[3]),
                        "prev_close": _sf(parts[4]),
                        "open": _sf(parts[5]),
                        "volume": _sf(parts[6]),
                        "high": _sf(parts[33]),
                        "low": _sf(parts[34]),
                        "amt_wan": _sf(parts[37]),  # 万元 across all boards incl. STAR
                        "zt_px": _sf(parts[47], -1.0),
                        "dt_px": _sf(parts[48], -1.0),
                        "date": dt_str,
                        "time": tm_str,
                    }
                return out
            except Exception:
                time.sleep(0.4 * (attempt + 1))
        return out

    result: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for chunk_res in ex.map(_fetch_chunk, chunks):
            result.update(chunk_res)
    return result


def compute_snapshot_bars(
    stock_quotes: dict[str, dict],
    stk_df: pd.DataFrame,
    prev_state: dict,
    snap_date: str,
) -> tuple[dict, dict, dict, dict]:
    """Compute exact daily short_term_daily row, breadth row, new state, and top board details."""
    prev_zt_map: dict[str, dict] = {}
    if prev_state and str(prev_state.get("last_confirmed_date", "")) < snap_date:
        prev_zt_map = prev_state.get("zt_stocks", {})

    lim_lookup = dict(zip(stk_df["tc_code"], stk_df["lim"].astype(float)))

    total = up = dn = big_up = big_dn = 0
    up_amt_yi = dn_amt_yi = 0.0
    zt_cnt = zt_20 = yizi_cnt = zb_cnt = dt_cnt = 0
    lb2 = lb3 = max_lb = max_lb_ex_yz = 0

    prev_zt_cnt = prev_zt_promo = 0
    prev_1b_cnt = prev_1b_promo = 0
    prev_2b_cnt = prev_2b_promo = 0
    prev_zt_ret_sum = prev_zt_open_sum = prev_lb_ret_sum = 0.0
    prev_lb_cnt = big_loss_cnt = 0
    tradeable_zt_cnt = same_o2c_sum = 0.0

    new_zt_stocks: dict[str, dict] = {}
    prev_zt_details: list[dict] = []

    for code, q in stock_quotes.items():
        if code not in lim_lookup:
            continue
        op, cl, hi, lo, pc = q["open"], q["close"], q["high"], q["low"], q["prev_close"]
        amt_yi = q["amt_wan"] / 1e4
        if op <= 0 or cl <= 0 or pc <= 0 or amt_yi <= 0:
            continue
        if q["date"] and q["date"] != snap_date:
            continue

        total += 1
        ret_c = cl / pc - 1.0
        ret_o = op / pc - 1.0
        dd_hi = cl / hi - 1.0 if hi > 0 else 0.0

        if cl > pc:
            up += 1
            up_amt_yi += amt_yi
        elif cl < pc:
            dn += 1
            dn_amt_yi += amt_yi
        if ret_c >= 0.05:
            big_up += 1
        elif ret_c <= -0.05:
            big_dn += 1

        name = q["name"]
        is_st = ("ST" in name.upper()) or ("退" in name)
        lim = lim_lookup[code]
        zt_px = q["zt_px"]
        dt_px = q["dt_px"]
        # Fallback tick-rounded limits if quote omits them; ignore no-limit IPOs
        if zt_px <= 0:
            zt_px = np.floor(pc * (1.0 + lim) * 100.0 + 0.5) / 100.0
        if dt_px <= 0:
            dt_px = np.floor(pc * (1.0 - lim) * 100.0 + 0.5) / 100.0
        no_limit_ipo = (zt_px / pc - 1.0) > 0.25

        is_zt = is_yizi = is_zb = is_dt = False
        if not is_st and not no_limit_ipo:
            is_zt = (cl >= zt_px - 1e-4) and (cl >= hi - 1e-4)
            is_yizi = is_zt and (op >= zt_px - 1e-4) and (lo >= zt_px - 1e-4)
            is_zb = (hi >= zt_px - 1e-4) and (cl < zt_px - 1e-4)
            is_dt = (cl <= dt_px + 1e-4) and (cl <= lo + 1e-4)

        p_info = prev_zt_map.get(code)
        p_zt = p_info is not None
        p_lb = int(p_info["lb"]) if p_zt else 0
        cur_lb = (p_lb + 1) if is_zt else 0

        if is_zt:
            zt_cnt += 1
            if lim > 0.15:
                zt_20 += 1
            if is_yizi:
                yizi_cnt += 1
            if cur_lb >= 2:
                lb2 += 1
            if cur_lb >= 3:
                lb3 += 1
            if cur_lb > max_lb:
                max_lb = cur_lb
            if (not is_yizi) and cur_lb > max_lb_ex_yz:
                max_lb_ex_yz = cur_lb
            new_zt_stocks[code] = {
                "name": name,
                "close": round(cl, 3),
                "lb": int(cur_lb),
                "yizi": bool(is_yizi),
                "lim": lim,
                "amt_yi": round(amt_yi, 2),
                "ret_pct": round(ret_c * 100.0, 2),
            }

        if is_zb:
            zb_cnt += 1
        if is_dt:
            dt_cnt += 1

        if p_zt:
            prev_zt_cnt += 1
            prev_zt_ret_sum += ret_c
            prev_zt_open_sum += ret_o
            is_big_loss = (ret_c <= -0.05) or (dd_hi <= -0.08)
            if is_big_loss:
                big_loss_cnt += 1
            if is_zt:
                prev_zt_promo += 1
            if p_lb == 1:
                prev_1b_cnt += 1
                if is_zt:
                    prev_1b_promo += 1
            elif p_lb >= 2:
                prev_2b_cnt += 1
                prev_lb_cnt += 1
                prev_lb_ret_sum += ret_c
                if is_zt:
                    prev_2b_promo += 1
            can_buy_open = op < zt_px - 1e-4
            if can_buy_open:
                tradeable_zt_cnt += 1
                same_o2c_sum += cl / op - 1.0
            prev_zt_details.append({
                "code": code,
                "name": name,
                "prev_lb": p_lb,
                "ret_pct": round(ret_c * 100.0, 2),
                "open_pct": round(ret_o * 100.0, 2),
                "promoted": bool(is_zt),
                "big_loss": bool(is_big_loss),
            })

    st_row = {
        "date": snap_date,
        "total_stocks": total,
        "up_count": up,
        "down_count": dn,
        "up_ratio": up / total if total > 0 else np.nan,
        "big_up": big_up,
        "big_down": big_dn,
        "zt_count": zt_cnt,
        "zt_20cm": zt_20,
        "yizi_count": yizi_cnt,
        "zb_count": zb_cnt,
        "zb_rate": zb_cnt / (zt_cnt + zb_cnt) if (zt_cnt + zb_cnt) > 0 else 0.0,
        "dt_count": dt_cnt,
        "lb2_count": lb2,
        "lb3_count": lb3,
        "max_lb": max_lb,
        "max_lb_ex_yizi": max_lb_ex_yz,
        "prev_zt_cnt": prev_zt_cnt,
        "promo_rate": prev_zt_promo / prev_zt_cnt if prev_zt_cnt > 0 else 0.0,
        "promo_1to2": prev_1b_promo / prev_1b_cnt if prev_1b_cnt > 0 else 0.0,
        "promo_high": prev_2b_promo / prev_2b_cnt if prev_2b_cnt > 0 else 0.0,
        "prev_zt_ret": prev_zt_ret_sum / prev_zt_cnt if prev_zt_cnt > 0 else 0.0,
        "prev_zt_open_ret": prev_zt_open_sum / prev_zt_cnt if prev_zt_cnt > 0 else 0.0,
        "prev_lb_ret": prev_lb_ret_sum / prev_lb_cnt if prev_lb_cnt > 0 else 0.0,
        "big_loss_rate": big_loss_cnt / prev_zt_cnt if prev_zt_cnt > 0 else 0.0,
        "tradeable_zt_cnt": int(tradeable_zt_cnt),
        "prev_zt_same_o2c": same_o2c_sum / tradeable_zt_cnt if tradeable_zt_cnt > 0 else 0.0,
        "relay_next_ret": np.nan,
        "relay_lb_next_ret": np.nan,
    }

    nz = up + dn
    tot_amt = up_amt_yi + dn_amt_yi
    B = up / nz if nz > 0 else np.nan
    M = up_amt_yi / tot_amt if tot_amt > 0 else np.nan
    RANA = (up - dn) / nz if nz > 0 else np.nan
    bm_row = {
        "date": snap_date,
        "up_cnt": up,
        "dn_cnt": dn,
        "flat_cnt": max(total - nz, 0),
        "up_amt_yi": round(up_amt_yi, 4),
        "dn_amt_yi": round(dn_amt_yi, 4),
        "total_amt_yi": round(tot_amt, 4),
        "B": B,
        "M": M,
        "RANA": RANA,
        "M_minus_B": (M - B) if (not np.isnan(M) and not np.isnan(B)) else np.nan,
    }

    sorted_zt = dict(sorted(new_zt_stocks.items(), key=lambda kv: (-kv[1]["lb"], -kv[1]["amt_yi"])))
    new_state = {
        "last_confirmed_date": snap_date,
        "updated_at": datetime.now(BJT).strftime("%Y-%m-%d %H:%M:%S BJT"),
        "zt_count": len(sorted_zt),
        "max_lb": max_lb,
        "zt_stocks": sorted_zt,
    }

    details = {
        "top_zt": [
            {"code": k, **v} for k, v in list(sorted_zt.items())[:12]
        ],
        "prev_zt_board": sorted(prev_zt_details, key=lambda x: (-x["prev_lb"], -x["ret_pct"]))[:12],
    }
    return st_row, bm_row, new_state, details


def refresh_akshare_feeds(timeout_per_feed: int = 18) -> dict[str, str]:
    """Non-blocking incremental refresh of ACSI slow sentiment sub-feeds."""
    import akshare as ak

    feeds = [
        ("margin_info.csv", "日期", lambda: ak.stock_margin_account_info()),
        ("qvix_50etf.csv", "date", lambda: ak.index_option_50etf_qvix()),
        ("below_net_asset.csv", "date", lambda: ak.stock_a_below_net_asset_statistics(symbol="全部A股")),
        ("ebs_hs300.csv", "日期", lambda: ak.stock_ebs_lg()),
    ]
    status: dict[str, str] = {}
    today_bjt = datetime.now(BJT).strftime("%Y-%m-%d")

    for fname, dcol, fn in feeds:
        path = DATA_DIR / fname
        last_dt = ""
        if path.exists():
            try:
                old = pd.read_csv(path)
                last_dt = str(old[dcol].iloc[-1])
            except Exception:
                pass
        if last_dt >= today_bjt:
            status[fname] = f"cached ({last_dt})"
            continue

        ex = ThreadPoolExecutor(max_workers=1)
        fut = ex.submit(fn)
        try:
            df = fut.result(timeout=timeout_per_feed)
            if df is not None and len(df) > 100:
                df[dcol] = pd.to_datetime(df[dcol]).dt.strftime("%Y-%m-%d")
                df = df.sort_values(dcol).drop_duplicates(dcol)
                df.to_csv(path, index=False)
                new_dt = str(df[dcol].iloc[-1])
                status[fname] = f"updated ({new_dt})"
            else:
                status[fname] = f"empty -> kept ({last_dt})"
        except FuturesTimeout:
            status[fname] = f"timeout -> kept ({last_dt})"
        except Exception as e:
            status[fname] = f"err({type(e).__name__}) -> kept ({last_dt})"
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
    return status


def upsert_csv_row(path: Path, row: dict, date_col: str = "date") -> None:
    df = pd.read_csv(path)
    dt = str(row[date_col])
    df = df[df[date_col].astype(str) != dt]
    new_row = {c: row.get(c, np.nan) for c in df.columns}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df = df.sort_values(date_col).reset_index(drop=True)
    df.to_csv(path, index=False)


def upsert_index_bars(idx_quotes: dict[str, dict], snap_date: str) -> None:
    for alias, tc in INDEX_MAP.items():
        q = idx_quotes.get(tc)
        p = DATA_DIR / f"idx_{alias}.csv"
        if not q or not p.exists() or q["close"] <= 0:
            continue
        if q["date"] and q["date"] != snap_date:
            continue
        pct = (q["close"] / q["prev_close"] - 1.0) * 100.0 if q["prev_close"] > 0 else 0.0
        upsert_csv_row(
            p,
            {
                "date": snap_date,
                "open": q["open"],
                "high": q["high"],
                "low": q["low"],
                "close": q["close"],
                "volume": q["volume"] * 100.0,
                "amount": q["amt_wan"] * 1e4,  # 元
                "pctChg": round(pct, 6),
            },
        )


def compute_unified_panel(
    overlay_st_row: dict | None = None,
    overlay_bm_row: dict | None = None,
    overlay_idx_quotes: dict[str, dict] | None = None,
    overlay_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build full causal ACSI (2016+) and STSI+MO+M-B (2021+) panels, optionally with intraday overlay."""
    # 1. ACSI panel
    ac_raw = load_all()
    if overlay_idx_quotes and overlay_date:
        ts = pd.Timestamp(overlay_date)
        if ts not in ac_raw.index:
            ac_raw.loc[ts] = np.nan
            ac_raw = ac_raw.sort_index()
        for alias, tc in INDEX_MAP.items():
            q = overlay_idx_quotes.get(tc)
            if not q or q["close"] <= 0:
                continue
            if alias == "hs300":
                ac_raw.loc[ts, "hs300_open"] = q["open"]
                ac_raw.loc[ts, "hs300_close"] = q["close"]
                ac_raw.loc[ts, "hs300_amount"] = q["amt_wan"] * 1e4
            elif alias == "sh_comp":
                ac_raw.loc[ts, "sh_close"] = q["close"]
                ac_raw.loc[ts, "sh_amount"] = q["amt_wan"] * 1e4
            elif alias == "sz_comp":
                ac_raw.loc[ts, "sz_amount"] = q["amt_wan"] * 1e4
            else:
                if f"{alias}_close" in ac_raw.columns:
                    ac_raw.loc[ts, f"{alias}_close"] = q["close"]
                    ac_raw.loc[ts, f"{alias}_open"] = q["open"]
                    ac_raw.loc[ts, f"{alias}_amount"] = q["amt_wan"] * 1e4
        # Avoid biasing slow 756d ACSI volume activity with partial-day intraday turnover
        ref_tm = (overlay_idx_quotes.get("sh000300") or {}).get("time", "")
        if ref_tm >= "15:00:00" and pd.notna(ac_raw.loc[ts, "sh_amount"]) and pd.notna(ac_raw.loc[ts, "sz_amount"]):
            ac_raw.loc[ts, "total_amount"] = ac_raw.loc[ts, "sh_amount"] + ac_raw.loc[ts, "sz_amount"]
        ac_raw["hs300_ret"] = ac_raw["hs300_close"].pct_change()

    acsi_df = build_acsi(ac_raw, window=756, min_periods=240, smooth_days=5)
    acsi_df["acsi_pos"] = strategy_hysteresis(acsi_df["acsi"], 30.0, 70.0)

    # 2. Short-term STSI + MO + M-B panel
    st = pd.read_csv(ST_DAILY_PATH)
    st["date"] = pd.to_datetime(st["date"])
    st = st.sort_values("date").drop_duplicates("date").set_index("date")
    if overlay_st_row and overlay_date:
        ts = pd.Timestamp(overlay_date)
        st.loc[ts] = {c: overlay_st_row.get(c, np.nan) for c in st.columns}
        st = st.sort_index()

    # Join major indices
    for alias in ["hs300", "zz500", "zz1000", "chinext", "sh_comp"]:
        p = DATA_DIR / f"idx_{alias}.csv"
        if p.exists():
            idx = pd.read_csv(p)
            idx["date"] = pd.to_datetime(idx["date"])
            idx = idx.sort_values("date").drop_duplicates("date").set_index("date")
            if overlay_idx_quotes and overlay_date:
                q = overlay_idx_quotes.get(INDEX_MAP[alias])
                if q and q["close"] > 0:
                    ts = pd.Timestamp(overlay_date)
                    idx.loc[ts, "open"] = q["open"]
                    idx.loc[ts, "close"] = q["close"]
                    idx = idx.sort_index()
            st[f"{alias}_open"] = idx["open"]
            st[f"{alias}_close"] = idx["close"]
            st[f"{alias}_ret"] = idx["close"].pct_change()

    stsi_df = build_stsi(st, window=240, min_periods=60, smooth_days=3)
    df = st.join(stsi_df)

    bm = pd.read_csv(BM_DAILY_PATH)
    bm["date"] = pd.to_datetime(bm["date"])
    bm = bm.sort_values("date").drop_duplicates("date").set_index("date")
    if overlay_bm_row and overlay_date:
        ts = pd.Timestamp(overlay_date)
        for col in ["up_cnt", "dn_cnt", "flat_cnt", "up_amt_yi", "dn_amt_yi", "total_amt_yi", "B", "M", "RANA", "M_minus_B"]:
            if col in overlay_bm_row:
                bm.loc[ts, col] = overlay_bm_row[col]
        bm = bm.sort_index()
    # Always recompute causal EMA(19)-EMA(39) MO dynamically so new days are 100% exact!
    bm["ema19"] = bm["RANA"].ewm(span=19, adjust=False).mean()
    bm["ema39"] = bm["RANA"].ewm(span=39, adjust=False).mean()
    bm["MO"] = bm["ema19"] - bm["ema39"]

    df = df.join(bm[["up_cnt", "dn_cnt", "up_amt_yi", "dn_amt_yi", "B", "M", "RANA", "MO", "M_minus_B"]])
    df["s_B"] = rolling_z_cdf(df["B"])
    df["s_M"] = rolling_z_cdf(df["M"])
    df["s_MO"] = rolling_z_cdf(df["MO"])
    df["MB10"] = df["M_minus_B"].rolling(10).mean()
    df["s_MB10"] = rolling_z_cdf(df["MB10"])

    df["health_base"] = ((df["s_seal_rate"] + df["s_anti_nuke"]) / 2.0).rolling(5, min_periods=1).mean()
    df["health_mo"] = ((df["s_seal_rate"] + df["s_anti_nuke"] + df["s_MO"]) / 3.0).rolling(5, min_periods=1).mean()
    df["phase"] = classify_cycle_phase(df["stsi"], df["stsi_mom"])

    # Signals
    df["pos_mo"] = strat_cycle_from_health(df["health_mo"], df["stsi"], 60.0, 40.0, 20.0)
    df["w_core"] = strat_style_weight(df["s_MB10"], 65.0, 35.0)
    df["relay_ok"] = ((df["stsi"] <= 25.0) & (df["big_loss_rate"] < 0.50) & (df["MO"] < 0.0)).astype(int)

    # Join ACSI onto short-term panel
    df["acsi"] = acsi_df["acsi"].reindex(df.index).ffill()
    df["acsi_pos"] = acsi_df["acsi_pos"].reindex(df.index).ffill()
    return acsi_df, df


def summarize_day_signal(acsi_df: pd.DataFrame, df: pd.DataFrame, row_ts: pd.Timestamp, mode_label: str) -> dict:
    r = df.loc[row_ts]
    a_val = float(r["acsi"])
    a_pos = float(r["acsi_pos"])
    h_mo = float(r["health_mo"])
    stsi_v = float(r["stsi"])
    mo_v = float(r["MO"])
    s_mb = float(r["s_MB10"])
    mb_raw = float(r["M_minus_B"])
    b_v = float(r["B"])
    m_v = float(r["M"])
    bl_v = float(r["big_loss_rate"])
    pos = float(r["pos_mo"])
    w_core = float(r["w_core"])
    relay_ok = bool(int(r["relay_ok"]))

    # ACSI verdict
    if a_val <= 20:
        acsi_zone, acsi_action, acsi_tone = "极度恐慌 (0-20)", "分批重仓逆向布局", "green"
    elif a_val <= 30:
        acsi_zone, acsi_action, acsi_tone = "偏恐慌 (20-30)", "触发逆向买点 · 满仓持有", "green"
    elif a_val >= 70:
        acsi_zone, acsi_action, acsi_tone = "贪婪/过热 (≥70)", "逆向减仓防守 · 兑现收益", "red"
    else:
        acsi_zone = "中性偏冷" if a_val < 50 else "中性偏暖"
        acsi_action = "维持满仓持有" if a_pos >= 0.9 else ("维持空仓观望" if a_pos <= 0.1 else "中性半仓配置")
        acsi_tone = "blue" if a_pos >= 0.5 else "amber"

    # Swing timing verdict + distance to flip
    if pos >= 0.99:
        if stsi_v <= 20.0:
            timing_reason = f"触发冰点抄底开关 (STSI {stsi_v:.1f} ≤ 20)"
            timing_dist = f"冰点保护生效中；常规退潮线 Health_MO ≤ 40 (当前 {h_mo:.1f})"
        else:
            timing_reason = f"Health_MO 强势持仓 ({h_mo:.1f})"
            timing_dist = f"距翻空退潮线 (40) 还有 +{h_mo - 40.0:.1f} 分缓冲"
        timing_action = "100% 满仓进攻"
        timing_tone = "green"
    else:
        timing_reason = f"Health_MO 处于弱势区 ({h_mo:.1f})"
        timing_dist = f"距右侧进场 (60) 差 {60.0 - h_mo:.1f} 分；距冰点抄底 (STSI≤20) 差 {stsi_v - 20.0:.1f} 分"
        timing_action = "0% 空仓防守 (货币基金)"
        timing_tone = "amber"

    # Style verdict + distance to flip
    if w_core >= 0.99:
        style_target = "创业板指 (核心龙头)"
        style_etf = "创业板ETF (159915)"
        style_dist = f"s_MB10={s_mb:.1f}，距切回中证1000 (≤35) 还有 {s_mb - 35.0:.1f} 分"
        style_tone = "purple"
    elif w_core <= 0.01:
        style_target = "中证1000 (小盘扩散)"
        style_etf = "中证1000ETF (512100)"
        style_dist = f"s_MB10={s_mb:.1f}，距切往创业板 (≥65) 还差 {65.0 - s_mb:.1f} 分"
        style_tone = "cyan"
    else:
        style_target = "50/50 均衡配置"
        style_etf = "创业板ETF + 中证1000ETF"
        style_dist = f"s_MB10={s_mb:.1f} 处于缓冲带 (35-65)"
        style_tone = "blue"

    # Combined dual-engine ETF directive
    if pos >= 0.99:
        dual_directive = f"100% 满仓 · {style_target}"
    else:
        dual_directive = f"0% 空仓防守（风格偏向：{style_target}）"

    # Relay verdict
    if stsi_v >= 75.0:
        relay_action = "高潮避坑 · 禁止追高"
        relay_reason = f"STSI={stsi_v:.1f} ≥ 75（历史高潮追涨停5年亏-42.8%）"
        relay_tone = "red"
    elif relay_ok:
        relay_action = "允许轻仓左侧试错"
        relay_reason = f"STSI={stsi_v:.1f}≤25 且 大面率={bl_v*100:.1f}%<50% 且 MO={mo_v*100:+.2f}%<0"
        relay_tone = "green"
    else:
        blockers = []
        if stsi_v > 25.0:
            blockers.append(f"STSI={stsi_v:.1f}>25")
        if bl_v >= 0.50:
            blockers.append(f"大面率={bl_v*100:.1f}%≥50%(防核按钮)")
        if mo_v >= 0.0:
            blockers.append(f"MO={mo_v*100:+.2f}%≥0")
        relay_action = "空仓观望 · 等待出清"
        relay_reason = "未满足：" + "，".join(blockers)
        relay_tone = "amber"

    return {
        "date": row_ts.strftime("%Y-%m-%d"),
        "mode_label": mode_label,
        "phase": str(r["phase"]),
        "acsi": round(a_val, 1),
        "acsi_pos": round(a_pos * 100.0, 0),
        "acsi_zone": acsi_zone,
        "acsi_action": acsi_action,
        "acsi_tone": acsi_tone,
        "health_mo": round(h_mo, 1),
        "stsi": round(stsi_v, 1),
        "mo_pct": round(mo_v * 100.0, 2),
        "s_mo": round(float(r["s_MO"]), 1),
        "s_seal": round(float(r["s_seal_rate"]), 1),
        "s_anti_nuke": round(float(r["s_anti_nuke"]), 1),
        "s_mb10": round(s_mb, 1),
        "mb_raw_pct": round(mb_raw * 100.0, 2),
        "b_pct": round(b_v * 100.0, 1),
        "m_pct": round(m_v * 100.0, 1),
        "up_cnt": int(r["up_cnt"]) if pd.notna(r["up_cnt"]) else 0,
        "dn_cnt": int(r["dn_cnt"]) if pd.notna(r["dn_cnt"]) else 0,
        "tot_amt_yi": round(float(r["up_amt_yi"] + r["dn_amt_yi"]), 0) if pd.notna(r["up_amt_yi"]) else 0,
        "zt_count": int(r["zt_count"]),
        "dt_count": int(r["dt_count"]),
        "zb_count": int(r["zb_count"]),
        "zb_rate_pct": round(float(r["zb_rate"]) * 100.0, 1),
        "max_lb": int(r["max_lb"]),
        "lb2_count": int(r["lb2_count"]),
        "prev_zt_cnt": int(r["prev_zt_cnt"]),
        "promo_rate_pct": round(float(r["promo_rate"]) * 100.0, 1),
        "big_loss_pct": round(bl_v * 100.0, 1),
        "prev_zt_ret_pct": round(float(r["prev_zt_ret"]) * 100.0, 2),
        "pos_pct": int(round(pos * 100)),
        "w_core_pct": int(round(w_core * 100)),
        "timing_action": timing_action,
        "timing_reason": timing_reason,
        "timing_dist": timing_dist,
        "timing_tone": timing_tone,
        "style_target": style_target,
        "style_etf": style_etf,
        "style_dist": style_dist,
        "style_tone": style_tone,
        "dual_directive": dual_directive,
        "relay_action": relay_action,
        "relay_reason": relay_reason,
        "relay_tone": relay_tone,
        "hs300_close": round(float(r["hs300_close"]), 2) if pd.notna(r["hs300_close"]) else None,
        "hs300_ret_pct": round(float(r["hs300_ret"]) * 100.0, 2) if pd.notna(r["hs300_ret"]) else 0.0,
        "zz1000_close": round(float(r["zz1000_close"]), 2) if pd.notna(r["zz1000_close"]) else None,
        "zz1000_ret_pct": round(float(r["zz1000_ret"]) * 100.0, 2) if pd.notna(r["zz1000_ret"]) else 0.0,
        "chinext_close": round(float(r["chinext_close"]), 2) if pd.notna(r["chinext_close"]) else None,
        "chinext_ret_pct": round(float(r["chinext_ret"]) * 100.0, 2) if pd.notna(r["chinext_ret"]) else 0.0,
    }


def _clean(vals, ndigits: int = 2) -> list:
    out = []
    for v in vals:
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            out.append(None)
        else:
            out.append(round(float(v), ndigits))
    return out


def build_payload(
    acsi_eod: pd.DataFrame,
    df_eod: pd.DataFrame,
    acsi_live: pd.DataFrame | None,
    df_live: pd.DataFrame | None,
    meta: dict,
    board_details: dict,
) -> dict:
    """Build compact, zero-NaN JSON payload powering the Command Center dashboard."""
    eod_ts = df_eod.index[-1]
    eod_card = summarize_day_signal(acsi_eod, df_eod, eod_ts, f"收盘确认 · {eod_ts.strftime('%Y-%m-%d')}")
    live_card = None
    if df_live is not None and acsi_live is not None:
        live_ts = df_live.index[-1]
        live_card = summarize_day_signal(
            acsi_live,
            df_live,
            live_ts,
            f"盘中实时 · {live_ts.strftime('%Y-%m-%d')} {meta['quote_time']} BJT",
        )

    # Full daily backtest equity curves (confirmed EOD history from ST_START)
    r_main, r_b50, hp_main, hw_core, _ = exec_dual_engine(
        df_eod["chinext_open"], df_eod["chinext_close"],
        df_eod["zz1000_open"], df_eod["zz1000_close"],
        df_eod["w_core"], df_eod["pos_mo"],
        cost_one_way=COST_ETF, cash_rf_annual=CASH_MMF, use_open=True, start_date=ST_START,
    )
    r_def, _, _, _, _ = exec_dual_engine(
        df_eod["hs300_open"], df_eod["hs300_close"],
        df_eod["zz1000_open"], df_eod["zz1000_close"],
        df_eod["w_core"], df_eod["pos_mo"],
        cost_one_way=COST_ETF, cash_rf_annual=CASH_MMF, use_open=True, start_date=ST_START,
    )
    eq_main = (1.0 + r_main).cumprod()
    eq_def = (1.0 + r_def).cumprod()
    eq_b50 = (1.0 + r_b50).cumprod()

    # Keep FULL daily resolution for recent 300 days + weekly downsample for older days?
    # Better: ALL ~1,305 daily points is only ~110KB JSON, or 3-day stride + last 120 full daily days (~35KB)!
    sub = df_eod.loc[eq_main.index].copy()
    sub["eq_main"] = eq_main
    sub["eq_def"] = eq_def
    sub["eq_b50"] = eq_b50
    sub["hp_pos"] = hp_main
    sub["hw_core"] = hw_core
    sub["cy_zz_ratio"] = (sub["chinext_close"] / sub["chinext_close"].iloc[0]) / (
        sub["zz1000_close"] / sub["zz1000_close"].iloc[0]
    )

    n = len(sub)
    # Keep every 6th day for older history + EVERY single trading day in the last 60 days!
    keep_idx = sorted(set(list(range(0, max(0, n - 60), 6)) + list(range(max(0, n - 60), n))))
    s = sub.iloc[keep_idx]

    st_series = {
        "dates": [d.strftime("%Y-%m-%d") for d in s.index],
        "eq_main": _clean(s["eq_main"], 3),
        "eq_def": _clean(s["eq_def"], 3),
        "eq_b50": _clean(s["eq_b50"], 3),
        "health_mo": _clean(s["health_mo"], 1),
        "stsi": _clean(s["stsi"], 1),
        "mo_pct": _clean(s["MO"] * 100.0, 1),
        "s_mb10": _clean(s["s_MB10"], 1),
        "mb_raw_pct": _clean(s["M_minus_B"] * 100.0, 1),
        "big_loss_pct": _clean(s["big_loss_rate"] * 100.0, 1),
        "cy_zz_ratio": _clean(s["cy_zz_ratio"], 3),
        "pos": _clean(s["pos_mo"], 0),
        "w_core": _clean(s["w_core"], 0),
        "zz1000": _clean(s["zz1000_close"], 0),
        "chinext": _clean(s["chinext_close"], 0),
    }

    # ACSI 10-year series (2016-03-01+, stride 12 + last 30 daily)
    ac_sub = acsi_eod.loc["2016-03-01":].copy()
    ac_ret = ac_sub["hs300_close"].pct_change().fillna(0.0)
    ac_hp = ac_sub["acsi_pos"].shift(2).fillna(0.5)
    ac_to = ac_hp.diff().abs().fillna(0.0)
    rf_d = (1.0 + CASH_MMF) ** (1.0 / 243.0) - 1.0
    ac_sr = ac_hp * ac_ret + (1.0 - ac_hp) * rf_d - ac_to * COST_ETF
    ac_sub["eq_acsi"] = (1.0 + ac_sr).cumprod()
    ac_sub["eq_hs300"] = (1.0 + ac_ret).cumprod()

    na = len(ac_sub)
    keep_a = sorted(set(list(range(0, max(0, na - 30), 12)) + list(range(max(0, na - 30), na))))
    sa = ac_sub.iloc[keep_a]
    acsi_series = {
        "dates": [d.strftime("%Y-%m-%d") for d in sa.index],
        "acsi": _clean(sa["acsi"], 1),
        "hs300": _clean(sa["hs300_close"], 0),
        "eq_acsi": _clean(sa["eq_acsi"], 3),
        "eq_hs300": _clean(sa["eq_hs300"], 3),
        "pos": _clean(sa["acsi_pos"], 0),
    }

    # Recent 25 trading days matrix (include live intraday row on top if active!)
    active_df = df_live if df_live is not None else df_eod
    recent_rows = []
    for ts in reversed(active_df.index[-25:]):
        r = active_df.loc[ts]
        is_live_row = (df_live is not None) and (ts == df_live.index[-1])
        p_val = int(round(float(r["pos_mo"]) * 100))
        w_val = int(round(float(r["w_core"]) * 100))
        style_lbl = "创业板" if w_val >= 99 else ("中证1000" if w_val <= 1 else "均衡")
        dir_lbl = f"满仓·{style_lbl}" if p_val >= 99 else "空仓防守"
        relay_lbl = "高潮避坑" if float(r["stsi"]) >= 75 else ("可试错" if int(r["relay_ok"]) else "观望")
        recent_rows.append({
            "date": ts.strftime("%Y-%m-%d"),
            "is_live": is_live_row,
            "zz1000_ret": round(float(r["zz1000_ret"]) * 100.0, 2) if pd.notna(r["zz1000_ret"]) else 0.0,
            "chinext_ret": round(float(r["chinext_ret"]) * 100.0, 2) if pd.notna(r["chinext_ret"]) else 0.0,
            "acsi": round(float(r["acsi"]), 1),
            "health_mo": round(float(r["health_mo"]), 1),
            "mo_pct": round(float(r["MO"]) * 100.0, 2),
            "s_mb10": round(float(r["s_MB10"]), 1),
            "mb_raw": round(float(r["M_minus_B"]) * 100.0, 1),
            "stsi": round(float(r["stsi"]), 1),
            "zt": int(r["zt_count"]),
            "dt": int(r["dt_count"]),
            "big_loss": round(float(r["big_loss_rate"]) * 100.0, 1),
            "directive": dir_lbl,
            "pos": p_val,
            "w_core": w_val,
            "relay": relay_lbl,
        })

    # Performance headline metrics (aligned strictly with metrics.py rf=2%)
    yrs = len(r_main) / 243.0
    tot_m = float(eq_main.iloc[-1] - 1.0)
    cagr_m = float((eq_main.iloc[-1]) ** (1.0 / yrs) - 1.0)
    mdd_m = float((eq_main / eq_main.cummax() - 1.0).min())
    ex_m = r_main - rf_d
    sh_m = float(ex_m.mean() / r_main.std() * np.sqrt(243)) if r_main.std() > 0 else 0.0

    tot_b = float(eq_b50.iloc[-1] - 1.0)
    cagr_b = float((eq_b50.iloc[-1]) ** (1.0 / yrs) - 1.0)
    mdd_b = float((eq_b50 / eq_b50.cummax() - 1.0).min())

    yrs_a = len(ac_sr) / 243.0
    cagr_a = float((ac_sub["eq_acsi"].iloc[-1]) ** (1.0 / yrs_a) - 1.0)
    mdd_a = float((ac_sub["eq_acsi"] / ac_sub["eq_acsi"].cummax() - 1.0).min())
    cagr_h = float((ac_sub["eq_hs300"].iloc[-1]) ** (1.0 / yrs_a) - 1.0)
    mdd_h = float((ac_sub["eq_hs300"] / ac_sub["eq_hs300"].cummax() - 1.0).min())

    return {
        "meta": meta,
        "eod": eod_card,
        "live": live_card,
        "perf": {
            "years": round(yrs, 2),
            "main_total_pct": round(tot_m * 100.0, 1),
            "main_cagr_pct": round(cagr_m * 100.0, 2),
            "main_mdd_pct": round(mdd_m * 100.0, 1),
            "main_sharpe": round(sh_m, 2),
            "bench_total_pct": round(tot_b * 100.0, 1),
            "bench_cagr_pct": round(cagr_b * 100.0, 2),
            "bench_mdd_pct": round(mdd_b * 100.0, 1),
            "acsi_years": round(yrs_a, 1),
            "acsi_cagr_pct": round(cagr_a * 100.0, 2),
            "acsi_mdd_pct": round(mdd_a * 100.0, 1),
            "hs300_cagr_pct": round(cagr_h * 100.0, 2),
            "hs300_mdd_pct": round(mdd_h * 100.0, 1),
        },
        "st_series": st_series,
        "acsi_series": acsi_series,
        "recent": recent_rows,
        "boards": board_details,
    }


def write_daily_briefing(payload: dict) -> str:
    card = payload["live"] or payload["eod"]
    eod = payload["eod"]
    lines = [
        f"# A股情绪作战室 · 每日信号播报 ({card['mode_label']})",
        "",
        f"- **波段双引擎指令**：**{card['dual_directive']}**",
        f"  - 仓位引擎：`Health_MO = {card['health_mo']}` | `MO = {card['mo_pct']:+.2f}%` | `STSI = {card['stsi']}`（{card['phase']}）→ {card['timing_reason']}",
        f"  - 风格引擎：`s_MB10 = {card['s_mb10']}`（当日 `M-B = {card['mb_raw_pct']:+.1f}%`，`B = {card['b_pct']:.1f}%`，`M = {card['m_pct']:.1f}%`）→ {card['style_etf']}",
        f"- **大周期定海神针**：`ACSI = {card['acsi']}`（{card['acsi_zone']}）→ {card['acsi_action']}",
        f"- **短线打板接力**：**{card['relay_action']}**（涨停 `{card['zt_count']}` / 跌停 `{card['dt_count']}` / 最高 `{card['max_lb']}` 板 / 大面率 `{card['big_loss_pct']:.1f}%`；{card['relay_reason']}）",
        f"- **市场快照**：创业板 `{card['chinext_ret_pct']:+.2f}%` | 中证1000 `{card['zz1000_ret_pct']:+.2f}%` | 沪深300 `{card['hs300_ret_pct']:+.2f}%` | 涨跌 `{card['up_cnt']}:{card['dn_cnt']}` | 成交约 `{card['tot_amt_yi']:.0f}亿`",
    ]
    if payload["live"] is not None:
        lines.append(f"- **上日收盘确认 ({eod['date']})**：{eod['dual_directive']} | `Health_MO={eod['health_mo']}` | `s_MB10={eod['s_mb10']}` | `ACSI={eod['acsi']}`")
    md = "\n".join(lines) + "\n"
    (LIVE_DIR / "daily_briefing.md").write_text(md, encoding="utf-8")
    return md


def send_webhook(briefing_md: str, payload: dict) -> None:
    url = os.environ.get("WEBHOOK_URL", "").strip()
    if not url:
        print("[NOTIFY] WEBHOOK_URL not set; skipped push.", flush=True)
        return
    card = payload["live"] or payload["eod"]
    title = f"A股情绪日报 | {card['dual_directive']} ({card['date']})"
    if "feishu" in url or "larksuite" in url:
        body = {"msg_type": "text", "content": {"text": briefing_md}}
    elif "dingtalk" in url:
        body = {"msgtype": "markdown", "markdown": {"title": title, "text": briefing_md}}
    elif "qyapi.weixin.qq.com" in url:
        body = {"msgtype": "markdown", "markdown": {"content": briefing_md}}
    else:
        body = {"title": title, "text": briefing_md, "data": card}
    try:
        r = requests.post(url, json=body, timeout=10)
        print(f"[NOTIFY] Webhook status={r.status_code}", flush=True)
    except Exception as e:
        print(f"[NOTIFY] Failed: {e}", flush=True)


STANDALONE_BROWSER_THEME = """<style>
  :root {
    --background: #0b0f17; --content: #131a26; --card: #111827; --border: #243045;
    --foreground: #f1f5f9; --muted-foreground: #94a3b8; --code-foreground: #7dd3fc;
    --primary: #3b82f6; --primary-foreground: #ffffff;
    --success: #22c55e; --warning: #f59e0b; --error: #ef4444;
    --chart-1: #3b82f6; --chart-2: #10b981; --chart-3: #06b6d4;
    --chart-4: #f59e0b; --chart-5: #a855f7; --chart-6: #ec4899;
  }
  html.light {
    --background: #f8fafc; --content: #f1f5f9; --card: #ffffff; --border: #e2e8f0;
    --foreground: #0f172a; --muted-foreground: #64748b; --code-foreground: #0369a1;
    --primary: #2563eb; --primary-foreground: #ffffff;
    --success: #16a34a; --warning: #d97706; --error: #dc2626;
    --chart-1: #2563eb; --chart-2: #059669; --chart-3: #0891b2;
    --chart-4: #d97706; --chart-5: #9333ea; --chart-6: #db2777;
  }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Noto Sans SC", sans-serif; font-size: 14px; }
</style>
<script>
  if (localStorage.getItem("ashare_theme") === "light") document.documentElement.classList.add("light");
  window.addEventListener("DOMContentLoaded", function() {
    var b = document.getElementById("btn-theme-toggle");
    if (!b) return;
    b.classList.remove("hidden");
    b.addEventListener("click", function() {
      var isLight = document.documentElement.classList.toggle("light");
      localStorage.setItem("ashare_theme", isLight ? "light" : "dark");
      window.dispatchEvent(new Event("themechange"));
    });
  });
</script>"""


def render_html_dashboards(payload: dict) -> list[Path]:
    written: list[Path] = []
    if not TEMPLATE_PATH.exists():
        print(f"[WARN] Template {TEMPLATE_PATH} not found yet; skipping HTML build.", flush=True)
        return written
    tpl = TEMPLATE_PATH.read_text(encoding="utf-8")
    json_blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if "__LIVE_PAYLOAD_JSON__" not in tpl:
        raise ValueError("dashboard_template.html missing __LIVE_PAYLOAD_JSON__ placeholder")
    html_external = (
        tpl.replace("<!-- __STANDALONE_THEME_CSS__ -->", STANDALONE_BROWSER_THEME)
        .replace("__LIVE_PAYLOAD_JSON__", json_blob)
    )
    for out_path in [LIVE_DIR / "index.html", DOCS_DIR / "index.html"]:
        out_path.write_text(html_external, encoding="utf-8")
        written.append(out_path)
    return written


def run_update(fast: bool = False, commit_intraday: bool = False, notify: bool = False) -> dict:
    t0 = time.time()
    now_bjt = datetime.now(BJT)
    print(f"=== A-Share Daily Sentiment Update ({now_bjt.strftime('%Y-%m-%d %H:%M:%S BJT')}) ===", flush=True)

    stk_df = pd.read_csv(STOCK_LIST_PATH, dtype=str)
    all_codes = stk_df["tc_code"].tolist() + list(INDEX_MAP.values())
    print(f"[1/5] Fetching live quotes for {len(all_codes)} symbols via Tencent qt...", flush=True)
    quotes = fetch_tencent_quotes(all_codes)
    idx_quotes = {tc: quotes[tc] for tc in INDEX_MAP.values() if tc in quotes}

    # Determine market date & quote time from HS300 / SH Comp
    ref_q = idx_quotes.get("sh000300") or idx_quotes.get("sh000001")
    if not ref_q or not ref_q["date"]:
        raise RuntimeError("Failed to fetch index reference quote from Tencent qt.gtimg.cn")
    snap_date = ref_q["date"]
    quote_time = ref_q["time"] or "15:00:00"

    st_hist = pd.read_csv(ST_DAILY_PATH)
    last_confirmed_date = str(st_hist["date"].iloc[-1])

    # Load per-stock limit-up state
    prev_state = json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else {}
    is_new_day = snap_date > last_confirmed_date
    is_post_close = (quote_time >= "15:00:00") or (not is_new_day)
    should_commit = is_new_day and (is_post_close or commit_intraday)

    print(
        f"  Quote snapshot: date={snap_date} time={quote_time} | last_confirmed={last_confirmed_date} "
        f"| mode={'COMMIT_EOD' if should_commit else ('INTRADAY_PREVIEW' if is_new_day else 'UP_TO_DATE')}",
        flush=True,
    )

    # 2. Refresh slow ACSI feeds (unless --fast)
    if fast:
        ak_status = {"mode": "skipped (--fast)"}
    else:
        print("[2/5] Refreshing ACSI slow sentiment sub-feeds (non-blocking 18s timeout)...", flush=True)
        ak_status = refresh_akshare_feeds(timeout_per_feed=18)
        print(f"  ACSI feeds: {ak_status}", flush=True)

    # 3. Compute today's breadth + limit-up/down snapshot
    print("[3/5] Computing full-market B/M/MO & limit-up/down metrics...", flush=True)
    st_row, bm_row, new_state, board_details = compute_snapshot_bars(quotes, stk_df, prev_state, snap_date)
    print(
        f"  {snap_date}: active={st_row['total_stocks']} | B={bm_row['B']:.1%}, M={bm_row['M']:.1%}, "
        f"M-B={bm_row['M_minus_B']:+.1%} | ZT={st_row['zt_count']}, DT={st_row['dt_count']}, "
        f"ZB={st_row['zb_count']}, MaxLB={st_row['max_lb']}, BigLoss={st_row['big_loss_rate']:.1%}",
        flush=True,
    )

    if should_commit:
        print(f"  Committing confirmed EOD bar for {snap_date}...", flush=True)
        upsert_csv_row(ST_DAILY_PATH, st_row)
        upsert_csv_row(BM_DAILY_PATH, bm_row)
        upsert_index_bars(idx_quotes, snap_date)
        STATE_PATH.write_text(json.dumps(new_state, ensure_ascii=False, indent=2), encoding="utf-8")
        last_confirmed_date = snap_date

    # 4. Build unified causal panels (Confirmed EOD + optional Live Intraday Preview)
    print("[4/5] Recomputing unified ACSI + Health_MO + s_MB10 + STSI signals...", flush=True)
    acsi_eod, df_eod = compute_unified_panel()
    acsi_live, df_live = (None, None)
    if is_new_day and not should_commit:
        acsi_live, df_live = compute_unified_panel(
            overlay_st_row=st_row,
            overlay_bm_row=bm_row,
            overlay_idx_quotes=idx_quotes,
            overlay_date=snap_date,
        )

    meta = {
        "updated_at_bjt": now_bjt.strftime("%Y-%m-%d %H:%M:%S BJT"),
        "snap_date": snap_date,
        "quote_time": quote_time,
        "last_confirmed_date": last_confirmed_date,
        "has_intraday": df_live is not None,
        "elapsed_sec": round(time.time() - t0, 2),
        "active_stocks": int(st_row["total_stocks"]),
        "ak_status": ak_status,
    }

    payload = build_payload(acsi_eod, df_eod, acsi_live, df_live, meta, board_details)

    # 5. Save outputs & render dashboards
    print("[5/5] Writing outputs and rebuilding dashboards...", flush=True)
    (LIVE_DIR / "live_payload.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Save unified daily signal CSV
    save_cols = [
        "hs300_close", "zz1000_close", "chinext_close",
        "acsi", "acsi_pos", "health_mo", "MO", "s_MO", "s_MB10", "M_minus_B", "B", "M",
        "stsi", "phase", "zt_count", "dt_count", "zb_rate", "max_lb", "big_loss_rate",
        "pos_mo", "w_core", "relay_ok",
    ]
    export_df = (df_live if df_live is not None else df_eod)[save_cols].loc["2021-07-05":]
    export_df.to_csv(LIVE_DIR / "daily_signals.csv", index_label="date")

    briefing = write_daily_briefing(payload)
    written_html = render_html_dashboards(payload)
    if notify:
        send_webhook(briefing, payload)

    print(f"\n[DONE] Updated in {time.time()-t0:.2f}s | HTML built: {[str(p) for p in written_html]}", flush=True)
    print("\n" + briefing, flush=True)
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Daily A-share sentiment incremental updater")
    ap.add_argument("--fast", action="store_true", help="Skip slow AkShare ACSI sub-feeds (~3-4s run)")
    ap.add_argument("--commit-intraday", action="store_true", help="Force commit intraday snapshot as EOD")
    ap.add_argument("--notify", action="store_true", help="Push daily briefing to WEBHOOK_URL")
    args = ap.parse_args()
    run_update(fast=args.fast, commit_intraday=args.commit_intraday, notify=args.notify)
