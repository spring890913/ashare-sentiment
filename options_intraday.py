"""Real-Time A-Share ETF Options Intraday Monitor & Breadth Divergence Engine.

Fetches in ~4.5 seconds per cycle:
  1. Full-market A-share breadth & limit-up pulse (5,200+ stocks via Tencent qt)
     -> live B, M, M-B, ZT/DT/ZB, max_lb, big_loss_rate
  2. 1-minute intraday Price & exact VWAP curves for 5 core option ETFs via Tencent:
     - 创业板ETF (sz159915), 中证500ETF (sh510500), 沪深300ETF (sh510300),
       上证50ETF (sh510050), 科创50ETF (sh588000)
  3. Live option chains for all 5 ETFs via Sina (IPv4 forced for reliability),
     auto-rolling from near-month to next-month when DTE <= 5 days:
     -> Black-Scholes ATM IV, live intraday ΔIV, 25D Skew, Vol/OI PCR,
        Implied Daily Move (IV/√252) vs Realized Intraday Range, and
        recommended ATM & ITM-1 Call/Put contracts with full Greeks.
  4. Combines pre-market daily sentiment regime (STSI, Health_MO, s_MB10, QVIX)
     with live intraday structure to emit actionable options alerts.

Usage:
  .venv/bin/python options_intraday.py --once                # single live snapshot
  .venv/bin/python options_intraday.py --loop --interval 60  # live 60s monitor
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
import json
import math
import os
from pathlib import Path
import socket
import time
import warnings

import numpy as np
import pandas as pd
import requests
from scipy.optimize import brentq
from scipy.stats import norm
import urllib3.util.connection as urllib3_cn

# Force IPv4 so Sina edge endpoints never hit unreachable IPv6 routes
urllib3_cn.allowed_gai_family = lambda: socket.AF_INET
warnings.filterwarnings("ignore")

from daily_update import (
    fetch_tencent_quotes,
    compute_snapshot_bars,
    STOCK_LIST_PATH,
    STATE_PATH,
    LIVE_DIR,
    DOCS_DIR,
    BJT,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OPT_TEMPLATE_PATH = BASE_DIR / "options_monitor_template.html"

RISK_FREE_R = 0.02
CONTRACT_MULT = 10000  # standard A-share ETF option = 10,000 shares / contract

OPTION_ETFS = [
    {"code": "159915", "tc": "sz159915", "name": "创业板ETF", "style": "核心成长", "idx": "chinext"},
    {"code": "510500", "tc": "sh510500", "name": "中证500ETF", "style": "中小盘弹性", "idx": "zz500"},
    {"code": "510300", "tc": "sh510300", "name": "沪深300ETF", "style": "蓝筹均衡", "idx": "hs300"},
    {"code": "510050", "tc": "sh510050", "name": "上证50ETF", "style": "大盘权重", "idx": "sse50"},
    {"code": "588000", "tc": "sh588000", "name": "科创50ETF", "style": "硬科技高波", "idx": "chinext"},
]


# ---- Black-Scholes Exact IV & Greeks ----
def bs_price(S: float, K: float, T: float, r: float, sigma: float, cp: str) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, S - K) if cp == "C" else max(0.0, K - S)
    sq = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sq)
    d2 = d1 - sigma * sq
    df = math.exp(-r * T)
    if cp == "C":
        return float(S * norm.cdf(d1) - K * df * norm.cdf(d2))
    return float(K * df * norm.cdf(-d2) - S * norm.cdf(-d1))


def bs_iv(price: float, S: float, K: float, T: float, r: float, cp: str) -> float | None:
    if price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    df = math.exp(-r * T)
    intrinsic = max(0.0, S - K * df) if cp == "C" else max(0.0, K * df - S)
    if price <= intrinsic + 1e-6:
        return None
    try:
        lo_val = bs_price(S, K, T, r, 0.01, cp) - price
        hi_val = bs_price(S, K, T, r, 3.00, cp) - price
        if lo_val * hi_val > 0:
            return None
        return float(brentq(lambda sig: bs_price(S, K, T, r, sig, cp) - price, 0.01, 3.00, xtol=1e-5, maxiter=60))
    except Exception:
        return None


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, cp: str) -> dict:
    if not sigma or sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta_day": 0.0}
    sq = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sq)
    d2 = d1 - sigma * sq
    pdf = float(norm.pdf(d1))
    df = math.exp(-r * T)
    if cp == "C":
        delta = float(norm.cdf(d1))
        theta_ann = -(S * pdf * sigma) / (2.0 * sq) - r * K * df * float(norm.cdf(d2))
    else:
        delta = float(norm.cdf(d1) - 1.0)
        theta_ann = -(S * pdf * sigma) / (2.0 * sq) + r * K * df * float(norm.cdf(-d2))
    gamma = pdf / (S * sigma * sq)
    vega = S * pdf * sq * 0.01  # price change per +1% IV
    return {
        "delta": round(delta, 3),
        "gamma": round(gamma, 4),
        "vega": round(vega, 4),
        "theta_day": round(theta_ann / 365.0, 5),
    }


def _sf(v: str, default: float = 0.0) -> float:
    try:
        return float(v) if v else default
    except Exception:
        return default


def fetch_etf_minute_bars(tc_code: str) -> dict:
    """Fetch today's 1-minute prices & exact cumulative VWAP from Tencent."""
    try:
        r = requests.get(
            f"https://web.ifzq.gtimg.cn/appstock/app/minute/query?code={tc_code}",
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        node = r.json().get("data", {}).get(tc_code, {})
        raw_bars = node.get("data", {}).get("data", [])
        qt = node.get("qt", {}).get(tc_code, [])
        prev_close = _sf(qt[4]) if len(qt) > 4 else 0.0
        cur_date = str(node.get("data", {}).get("date", ""))
        if len(cur_date) == 8:
            cur_date = f"{cur_date[:4]}-{cur_date[4:6]}-{cur_date[6:8]}"

        times, prices, vwaps = [], [], []
        day_hi, day_lo = -1e9, 1e9
        for row in raw_bars:
            parts = row.split()
            if len(parts) < 4:
                continue
            tm = f"{parts[0][:2]}:{parts[0][2:4]}"
            px = float(parts[1])
            cvol = float(parts[2])  # lots (100 shares)
            camt = float(parts[3])  # yuan
            vwap = (camt / (cvol * 100.0)) if cvol > 0 else px
            # Sanity guard in case ETF vol unit is 1 share
            if px > 0 and (vwap < px * 0.5 or vwap > px * 1.5):
                vwap = (camt / cvol) if cvol > 0 else px
            if px > 0 and (vwap < px * 0.5 or vwap > px * 1.5):
                vwap = px
            times.append(tm)
            prices.append(round(px, 4))
            vwaps.append(round(vwap, 4))
            if px > day_hi:
                day_hi = px
            if px < day_lo:
                day_lo = px

        if not prices:
            return {"ok": False}

        last_px = prices[-1]
        last_vwap = vwaps[-1]
        open_px = prices[0]
        if prev_close <= 0:
            prev_close = open_px
        ret_pct = (last_px / prev_close - 1.0) * 100.0
        o2c_pct = (last_px / open_px - 1.0) * 100.0
        amp_pct = ((day_hi - day_lo) / prev_close) * 100.0
        vwap_diff_pct = (last_px / last_vwap - 1.0) * 100.0

        # Intraday structure: last 15-min slope & position in day range
        slope15_pct = ((prices[-1] / prices[max(0, len(prices) - 16)]) - 1.0) * 100.0
        pos_in_range = ((last_px - day_lo) / max(day_hi - day_lo, 1e-6)) * 100.0

        # Downsample 1m curve slightly if >120 pts (keep every 2nd + last 20 exact)
        n = len(times)
        keep = sorted(set(list(range(0, max(0, n - 20), 2)) + list(range(max(0, n - 20), n))))
        return {
            "ok": True,
            "date": cur_date,
            "prev_close": round(prev_close, 4),
            "open": round(open_px, 4),
            "last": round(last_px, 4),
            "high": round(day_hi, 4),
            "low": round(day_lo, 4),
            "vwap": round(last_vwap, 4),
            "ret_pct": round(ret_pct, 2),
            "o2c_pct": round(o2c_pct, 2),
            "amp_pct": round(amp_pct, 2),
            "vwap_diff_pct": round(vwap_diff_pct, 2),
            "slope15_pct": round(slope15_pct, 2),
            "pos_in_range": round(pos_in_range, 1),
            "bars": {
                "t": [times[i] for i in keep],
                "p": [prices[i] for i in keep],
                "v": [vwaps[i] for i in keep],
            },
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def get_active_option_months(sess: requests.Session) -> list[str]:
    """Return ordered unique active YYMM contract months (e.g. ['2611', '2612', '2703'])."""
    try:
        r = sess.get(
            "https://stock.finance.sina.com.cn/futures/api/openapi.php/"
            "StockOptionService.getStockName?exchange=null&cate=50ETF",
            timeout=6,
        ).json()
        raw = r["result"]["data"]["contractMonth"]
        seen, out = set(), []
        for m in raw:
            ym = m.replace("-", "")[2:6]
            if len(ym) == 4 and ym not in seen:
                seen.add(ym)
                out.append(ym)
        if out:
            return out
    except Exception:
        pass
    # Fallback: current & next calendar month
    now = datetime.now(BJT)
    m1 = now.strftime("%y%m")
    ny = now.year + (1 if now.month == 12 else 0)
    nm = 1 if now.month == 12 else now.month + 1
    return [m1, f"{ny % 100:02d}{nm:02d}"]


def fetch_etf_option_chain(
    sess: requests.Session,
    etf_code: str,
    spot: float,
    prev_spot: float,
    months: list[str],
) -> dict:
    """Fetch live option chain from Sina, auto-roll if near-month DTE <= 5, compute exact BS IV & Greeks."""
    def _load_month(ym: str) -> tuple[list[dict], int, str]:
        r1 = sess.get(
            f"https://hq.sinajs.cn/list=OP_UP_{etf_code}{ym},OP_DOWN_{etf_code}{ym}",
            timeout=6,
        )
        lines = r1.text.strip().splitlines()
        codes = []
        for ln in lines:
            if '="' in ln:
                body = ln.split('="')[1].rstrip('";')
                codes.extend([c for c in body.split(",") if c.startswith("CON_OP_")])
        if not codes:
            return [], 0, ""

        contracts = []
        dte = 0
        expiry = ""
        # Batch 60 codes per request
        for i in range(0, len(codes), 60):
            ch = codes[i : i + 60]
            r2 = sess.get(f"https://hq.sinajs.cn/list={','.join(ch)}", timeout=6)
            r2.encoding = "gbk"
            for ln in r2.text.strip().splitlines():
                if '="' not in ln:
                    continue
                tag = ln.split("=")[0].replace("var hq_str_CON_OP_", "").strip()
                f = ln.split('="')[1].rstrip('";').split(",")
                if len(f) < 48:
                    continue
                bid, last, ask = _sf(f[1]), _sf(f[2]), _sf(f[3])
                oi, chg_pct, strike = int(_sf(f[5])), _sf(f[6]), _sf(f[7])
                prev_c, vol = _sf(f[8]), int(_sf(f[41]))
                name = f[37]
                std_flag = f[43]  # 'M' standard vs 'A'/'B' dividend-adjusted
                cp = f[45]        # 'C' or 'P'
                exp_str = f[46]
                days_left = int(_sf(f[47]))
                if strike <= 0 or cp not in ("C", "P"):
                    continue
                dte = days_left
                expiry = exp_str
                mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0 and ask >= bid) else last
                spread_pct = ((ask - bid) / mid * 100.0) if (bid > 0 and ask > 0 and mid > 0) else 0.0
                contracts.append({
                    "id": tag,
                    "name": name,
                    "cp": cp,
                    "std": std_flag == "M",
                    "strike": strike,
                    "bid": bid,
                    "ask": ask,
                    "last": last,
                    "mid": mid,
                    "prev_close": prev_c,
                    "chg_pct": round(chg_pct, 1),
                    "vol": vol,
                    "oi": oi,
                    "spread_pct": round(spread_pct, 2),
                })
        return contracts, dte, expiry

    near_ym = months[0]
    next_ym = months[1] if len(months) > 1 else months[0]
    near_list, near_dte, near_exp = _load_month(near_ym)

    rolled = False
    active_ym, active_list, active_dte, active_exp = near_ym, near_list, near_dte, near_exp
    if (near_dte <= 5 or not near_list) and next_ym != near_ym:
        next_list, next_dte, next_exp = _load_month(next_ym)
        if next_list:
            rolled = True
            active_ym, active_list, active_dte, active_exp = next_ym, next_list, next_dte, next_exp

    if not active_list or spot <= 0:
        return {"ok": False}

    # Prefer standard 'M' contracts if available
    std_only = [c for c in active_list if c["std"]]
    pool = std_only if len(std_only) >= 6 else active_list

    # Compute BS IV & Greeks for every contract in active month
    T_now = max(float(active_dte), 0.35) / 365.0
    T_prev = (float(active_dte) + 1.0) / 365.0
    for c in pool:
        iv_now = bs_iv(c["mid"], spot, c["strike"], T_now, RISK_FREE_R, c["cp"])
        iv_prev = bs_iv(c["prev_close"], prev_spot, c["strike"], T_prev, RISK_FREE_R, c["cp"]) if prev_spot > 0 else None
        c["iv"] = round(iv_now * 100.0, 2) if iv_now else None
        c["iv_prev"] = round(iv_prev * 100.0, 2) if iv_prev else None
        c["iv_chg"] = round((iv_now - iv_prev) * 100.0, 2) if (iv_now and iv_prev) else None
        g = bs_greeks(spot, c["strike"], T_now, RISK_FREE_R, iv_now or 0.25, c["cp"])
        c.update(g)
        c["theta_rmb"] = round(g["theta_day"] * CONTRACT_MULT, 1)  # RMB/day/contract
        c["lev"] = round(abs(g["delta"]) * spot / max(c["mid"], 1e-4), 1)

    calls = sorted([c for c in pool if c["cp"] == "C"], key=lambda x: x["strike"])
    puts = sorted([c for c in pool if c["cp"] == "P"], key=lambda x: x["strike"])

    # Volume & OI PCR across ALL contracts (near + active for full-market sentiment, or active month)
    tot_c_vol = sum(c["vol"] for c in active_list if c["cp"] == "C")
    tot_p_vol = sum(c["vol"] for c in active_list if c["cp"] == "P")
    tot_c_oi = sum(c["oi"] for c in active_list if c["cp"] == "C")
    tot_p_oi = sum(c["oi"] for c in active_list if c["cp"] == "P")
    pcr_vol = round(tot_p_vol / max(tot_c_vol, 1), 2)
    pcr_oi = round(tot_p_oi / max(tot_c_oi, 1), 2)

    # Find ATM strike (closest to spot)
    strikes = sorted({c["strike"] for c in calls} & {p["strike"] for p in puts})
    if not strikes:
        return {"ok": False}
    atm_k = min(strikes, key=lambda k: abs(k - spot))
    atm_c = next((c for c in calls if c["strike"] == atm_k), None)
    atm_p = next((p for p in puts if p["strike"] == atm_k), None)

    # ATM IV & live ΔIV (average of ATM Call & Put)
    ivs = [x["iv"] for x in (atm_c, atm_p) if x and x["iv"] is not None]
    iv_chgs = [x["iv_chg"] for x in (atm_c, atm_p) if x and x["iv_chg"] is not None]
    atm_iv = round(float(np.mean(ivs)), 2) if ivs else None
    iv_chg = round(float(np.mean(iv_chgs)), 2) if iv_chgs else 0.0
    implied_daily_pct = round(atm_iv / math.sqrt(252.0), 2) if atm_iv else None

    # 25-Delta Skew: Put(~ -0.25 Delta) IV - Call(~ +0.25 Delta) IV
    c25_pool = [c for c in calls if c["iv"] is not None and 0.15 <= c["delta"] <= 0.38]
    p25_pool = [p for p in puts if p["iv"] is not None and -0.38 <= p["delta"] <= -0.15]
    c25 = min(c25_pool, key=lambda x: abs(x["delta"] - 0.25)) if c25_pool else None
    p25 = min(p25_pool, key=lambda x: abs(x["delta"] + 0.25)) if p25_pool else None
    skew_25d = round(p25["iv"] - c25["iv"], 2) if (c25 and p25) else None

    # Recommended intraday buyer contracts: ATM + ITM-1 (Delta ~0.50-0.68)
    k_idx = strikes.index(atm_k)
    itm_c_k = strikes[max(0, k_idx - 1)]
    itm_p_k = strikes[min(len(strikes) - 1, k_idx + 1)]
    rec_calls = [c for c in calls if c["strike"] in (itm_c_k, atm_k)]
    rec_puts = [p for p in puts if p["strike"] in (atm_k, itm_p_k)]

    # Compact T-quote chain around ATM (ATM +/- 3 strikes)
    lo_i = max(0, k_idx - 3)
    hi_i = min(len(strikes), k_idx + 4)
    t_rows = []
    c_map = {c["strike"]: c for c in calls}
    p_map = {p["strike"]: p for p in puts}
    for k in strikes[lo_i:hi_i]:
        cc, pp = c_map.get(k), p_map.get(k)
        t_rows.append({
            "k": k,
            "is_atm": k == atm_k,
            "c": {
                "id": cc["id"], "name": cc["name"], "last": cc["last"], "chg": cc["chg_pct"],
                "iv": cc["iv"], "iv_chg": cc["iv_chg"], "delta": cc["delta"],
                "theta": cc["theta_rmb"], "vol": cc["vol"], "spread": cc["spread_pct"], "lev": cc["lev"],
            } if cc else None,
            "p": {
                "id": pp["id"], "name": pp["name"], "last": pp["last"], "chg": pp["chg_pct"],
                "iv": pp["iv"], "iv_chg": pp["iv_chg"], "delta": pp["delta"],
                "theta": pp["theta_rmb"], "vol": pp["vol"], "spread": pp["spread_pct"], "lev": pp["lev"],
            } if pp else None,
        })

    return {
        "ok": True,
        "near_ym": near_ym,
        "near_dte": near_dte,
        "near_exp": near_exp,
        "rolled": rolled,
        "active_ym": active_ym,
        "active_dte": active_dte,
        "active_exp": active_exp,
        "atm_k": atm_k,
        "atm_iv": atm_iv,
        "iv_chg": iv_chg,
        "implied_daily_pct": implied_daily_pct,
        "skew_25d": skew_25d,
        "pcr_vol": pcr_vol,
        "pcr_oi": pcr_oi,
        "tot_vol": tot_c_vol + tot_p_vol,
        "rec_calls": [
            {
                "id": x["id"], "name": x["name"], "k": x["strike"], "last": x["last"],
                "iv": x["iv"], "delta": x["delta"], "theta": x["theta_rmb"], "lev": x["lev"],
                "spread": x["spread_pct"], "vol": x["vol"],
            }
            for x in rec_calls
        ],
        "rec_puts": [
            {
                "id": x["id"], "name": x["name"], "k": x["strike"], "last": x["last"],
                "iv": x["iv"], "delta": x["delta"], "theta": x["theta_rmb"], "lev": x["lev"],
                "spread": x["spread_pct"], "vol": x["vol"],
            }
            for x in rec_puts
        ],
        "t_chain": t_rows,
    }


def load_premarket_playbook() -> dict:
    """Load confirmed EOD regime from results_live/live_payload.json (or compute fallback)."""
    p = LIVE_DIR / "live_payload.json"
    eod = {}
    if p.exists():
        eod = json.loads(p.read_text(encoding="utf-8")).get("eod", {})

    qvix = None
    qp = DATA_DIR / "qvix_50etf.csv"
    if qp.exists():
        qdf = pd.read_csv(qp)
        qvix = round(float(pd.to_numeric(qdf["close"], errors="coerce").dropna().iloc[-1]), 2)

    stsi = float(eod.get("stsi", 50.0))
    h_mo = float(eod.get("health_mo", 50.0))
    s_mb10 = float(eod.get("s_mb10", 50.0))
    bl = float(eod.get("big_loss_pct", 25.0))
    mo_pct = float(eod.get("mo_pct", 0.0))

    # Determine Pre-Market Options Script
    if stsi <= 20.0:
        regime = "冰点高波偏多 · 防降波剧本"
        stance = "偏多（牛市 Call 价差 / 企稳卖浅虚 Put）"
        tone = "green"
        reason = (
            f"昨收 STSI={stsi:.1f}≤20（冰点期），历史次日创业板日内均值 +0.42%、振幅 2.79%（大单边概率 24.1%），"
            f"但伴随降波（QVIX均降 -0.26）。优先用「买实值Call + 卖虚值Call」对冲降波，避开裸买虚值末日Call。"
        )
    elif stsi >= 75.0:
        regime = "高潮分歧 · 升波买方剧本"
        stance = "买方顺势突破（严禁裸卖跨式）"
        tone = "red"
        reason = f"昨收 STSI={stsi:.1f}≥75（高潮过热），历史唯一稳定升波区（次日 QVIX +0.21），日内易现放量分歧单边，只做买方顺势不做裸卖。"
    elif (qvix is not None and qvix >= 25.0) and h_mo <= 40.0:
        regime = "高波退潮 · 卖方收割降波剧本"
        stance = "偏卖方 / 熊市价差（严禁追高买Put）"
        tone = "amber"
        reason = f"QVIX={qvix:.1f}≥25 且 Health_MO={h_mo:.1f}≤40，历史次日隐波暴跌 -3.97 点，开盘恐慌冲高后优先卖方结构收割 IV 回落。"
    elif 40.0 < h_mo < 60.0:
        regime = "中性震荡 · 卖方收租剧本"
        stance = "日内宽跨 / 铁鹰收租（慎追单边买方）"
        tone = "blue"
        reason = f"Health_MO={h_mo:.1f} 处于中性区，历史次日日内振幅收敛（2.06%），无放量宽度突破前不追单边买方。"
    elif h_mo >= 60.0:
        regime = "右侧强势 · 回踩低吸剧本"
        stance = "顺势低吸平值/实值 Call（忌追高）"
        tone = "green"
        reason = f"Health_MO={h_mo:.1f}≥60，但历史次日 O→C 胜率仅 44.4%，切忌开盘盲追，须等盘中回踩 VWAP + 实时 B/M 同步抬升再买 Call。"
    else:
        regime = "弱势退潮 · 逢反抽逢高空剧本"
        stance = "防隔夜低开后反抽，破 VWAP 买 Put / 熊市价差"
        tone = "amber"
        reason = f"Health_MO={h_mo:.1f}≤40，跌势大头常在隔夜释放（均跳空 -0.17%），日内等反抽乏力（B<30%）再做熊市价差或买平值 Put。"

    if s_mb10 >= 65.0:
        style_long = "创业板ETF (159915) / 科创50ETF (588000)"
        style_short = "中证500ETF (510500) / 中证1000(MO)"
    elif s_mb10 <= 35.0:
        style_long = "中证500ETF (510500) / 中证1000(MO)"
        style_short = "创业板ETF (159915)"
    else:
        style_long = "均衡（随盘中实时 M-B 灵活切换）"
        style_short = "均衡（随盘中实时 M-B 灵活切换）"

    return {
        "eod_date": eod.get("date", ""),
        "stsi": stsi,
        "health_mo": h_mo,
        "s_mb10": s_mb10,
        "mo_pct": mo_pct,
        "big_loss_pct": bl,
        "qvix": qvix,
        "regime": regime,
        "stance": stance,
        "tone": tone,
        "reason": reason,
        "style_long": style_long,
        "style_short": style_short,
    }


def detect_intraday_alerts(
    playbook: dict,
    breadth: dict,
    etfs: list[dict],
    history: list[dict],
) -> list[dict]:
    """Generate real-time actionable options intraday alerts."""
    alerts: list[dict] = []
    b_pct = breadth["b_pct"]
    m_pct = breadth["m_pct"]
    mb_pct = breadth["mb_pct"]

    # 1. Expiry week warning (near-month DTE <= 5)
    near_dte = next((e["opt"].get("near_dte") for e in etfs if e["opt"].get("ok")), None)
    if near_dte is not None and near_dte <= 5:
        near_ym = next((e["opt"].get("near_ym") for e in etfs if e["opt"].get("ok")), "")
        act_ym = next((e["opt"].get("active_ym") for e in etfs if e["opt"].get("ok")), "")
        alerts.append({
            "level": "warn",
            "tag": "末日行权周风控",
            "title": f"当月合约 ({near_ym}) 仅剩 {near_dte} 天到期，已自动切换主力监控至下月 ({act_ym})",
            "detail": "最后 5 个交易日虚值合约 Theta 衰减加速且 Gamma 剧烈跳变，日内交易请优先使用下月平值/浅实值合约。",
        })

    # 2. Style Resonance Alert (live M-B vs premarket s_MB10)
    cy = next((e for e in etfs if e["code"] == "159915"), None)
    zz5 = next((e for e in etfs if e["code"] == "510500"), None)
    if cy and zz5 and cy["min"].get("ok") and zz5["min"].get("ok"):
        cy_vw = cy["min"]["vwap_diff_pct"]
        if mb_pct >= 8.0 and playbook["s_mb10"] >= 65.0:
            alerts.append({
                "level": "purple",
                "tag": "资金集中共振 · 选标指引",
                "title": f"实时 M-B = {mb_pct:+.1f}% 与 s_MB10 ({playbook['s_mb10']:.1f}) 强共振 → 资金向核心成长集中",
                "detail": f"创业板ETF 当前 {cy['min']['ret_pct']:+.2f}%（距VWAP {cy_vw:+.2f}%），日内做多首选 159915 认购价差，做空/对冲首选 510500/中证1000。",
            })
        elif mb_pct <= -5.0 and playbook["s_mb10"] <= 35.0:
            alerts.append({
                "level": "cyan",
                "tag": "小盘扩散共振 · 选标指引",
                "title": f"实时 M-B = {mb_pct:+.1f}% 与 s_MB10 ({playbook['s_mb10']:.1f}) 强共振 → 中小盘全面占优",
                "detail": f"中证500ETF 当前 {zz5['min']['ret_pct']:+.2f}%，日内做多首选 510500 / 中证1000 认购，回避追高创业板。",
            })

    # 3. Ice-Point Intraday Reversal Confirmation
    if playbook["stsi"] <= 20.0 and cy and cy["min"].get("ok"):
        if cy["min"]["vwap_diff_pct"] > 0.3 and cy["min"]["o2c_pct"] > 1.0 and breadth["big_loss_pct"] < 40.0:
            alerts.append({
                "level": "green",
                "tag": "冰点强反转确认",
                "title": f"冰点次日放量站稳 VWAP：创业板日内低开高走 O→C {cy['min']['o2c_pct']:+.2f}%，大面率降至 {breadth['big_loss_pct']:.1f}%",
                "detail": f"昨收 STSI={playbook['stsi']:.1f} 冰点叠加核按钮出清（今涨停 {breadth['zt']} / 跌停 {breadth['dt']}），关注回踩 VWAP ({cy['min']['vwap']:.3f}) 时的实值 Call 或牛市价差机会；若 IV 已下行避免裸追虚值。",
            })

    # 4. Fake Rally (Index above VWAP while breadth is weak)
    for e in etfs:
        if not e["min"].get("ok"):
            continue
        if e["min"]["vwap_diff_pct"] >= 0.4 and b_pct < 30.0 and m_pct < 42.0:
            alerts.append({
                "level": "red",
                "tag": "假拉升 / 权重护盘预警",
                "title": f"{e['name']} 位于 VWAP 上方 ({e['min']['vwap_diff_pct']:+.2f}%)，但全市场仅 {b_pct:.1f}% 股票上涨",
                "detail": f"当前 B={b_pct:.1f}%、M={m_pct:.1f}% 广度严重背离，属于少数权重拉指数，慎追 Call；一旦放量跌破 VWAP ({e['min']['vwap']:.3f}) 易现快速回落。",
            })
            break

    # 5. Live IV Surge / IV Crush Alerts across the 5 ETFs
    iv_surges = [e for e in etfs if e["opt"].get("ok") and (e["opt"].get("iv_chg") or 0) >= 1.5]
    iv_crushes = [e for e in etfs if e["opt"].get("ok") and (e["opt"].get("iv_chg") or 0) <= -1.5]
    if iv_surges:
        names = "、".join(f"{e['name']}({e['opt']['iv_chg']:+.1f}%)" for e in iv_surges)
        alerts.append({
            "level": "red",
            "tag": "实时升波预警 (Vega+)",
            "title": f"检测到主力隐波显著抬升：{names}",
            "detail": "买方处于 Vega 顺风区，切忌裸卖跨式/单腿扛单；顺势方向腿可叠加 Gamma 放大收益。",
        })
    if iv_crushes:
        names = "、".join(f"{e['name']}({e['opt']['iv_chg']:+.1f}%)" for e in iv_crushes)
        alerts.append({
            "level": "amber",
            "tag": "实时降波 / IV Crush 预警",
            "title": f"检测到主力隐波回落：{names}",
            "detail": "裸买虚值期权面临 Vega 杀伤，方向看多/看空请改用「垂直价差（买实卖虚）」对冲降波，或顺势做卖方收租。",
        })

    # 6. Range Exhaustion Check (Realized intraday amp > 1.6x Implied Daily Move)
    exhausted = [
        e for e in etfs
        if e["min"].get("ok") and e["opt"].get("ok") and e["opt"].get("implied_daily_pct")
        and e["min"]["amp_pct"] >= 1.6 * e["opt"]["implied_daily_pct"]
    ]
    if exhausted:
        names = "、".join(
            f"{e['name']}(振幅{e['min']['amp_pct']:.1f}% vs 隐含日波{e['opt']['implied_daily_pct']:.1f}%)"
            for e in exhausted
        )
        alerts.append({
            "level": "blue",
            "tag": "日内振幅超预期 · 买方分批止盈",
            "title": f"实际振幅已超期权隐含日波 1.6 倍：{names}",
            "detail": "Gamma 红利已充分兑现，日内买方建议分批止盈或上移移动止损，避免尾盘获利回吐。",
        })

    # 7. Breadth Intraday Trend (if we have >=3 snapshots today)
    if len(history) >= 3:
        b_delta = history[-1]["b_pct"] - history[-3]["b_pct"]
        m_delta = history[-1]["m_pct"] - history[-3]["m_pct"]
        if b_delta >= 8.0 and m_delta >= 8.0:
            alerts.append({
                "level": "green",
                "tag": "盘中宽度加速走强",
                "title": f"近几个监测点全 A 宽度同步拉升：B {b_delta:+.1f}% / M {m_delta:+.1f}%",
                "detail": "属于全市场共振做多信号，买方顺势多单可继续持有跟踪 VWAP。",
            })
        elif b_delta <= -8.0 and m_delta <= -8.0:
            alerts.append({
                "level": "red",
                "tag": "盘中宽度加速转弱",
                "title": f"近几个监测点全 A 宽度同步滑坡：B {b_delta:+.1f}% / M {m_delta:+.1f}%",
                "detail": "个股资金加速撤离，多单及时止盈，关注跌破 VWAP 品种的平值 Put 机会。",
            })

    if not alerts:
        alerts.append({
            "level": "blue",
            "tag": "盘面平稳",
            "title": "当前未触发极端异动，按盘前剧本与 VWAP 纪律耐心等待结构突破",
            "detail": f"全A B={b_pct:.1f}% / M={m_pct:.1f}% / M-B={mb_pct:+.1f}%，严守买方时间止损与 14:55 前平仓纪律。",
        })
    return alerts


def append_and_load_history(snap_date: str, point: dict) -> list[dict]:
    log_path = LIVE_DIR / f"intraday_{snap_date}.jsonl"
    hist: list[dict] = []
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                hist.append(json.loads(line))
            except Exception:
                pass
    # Deduplicate same HH:MM timestamp
    if not hist or hist[-1].get("t") != point["t"]:
        hist.append(point)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(point, ensure_ascii=False) + "\n")
    else:
        hist[-1] = point
    return hist[-240:]


def render_options_html(payload: dict) -> list[Path]:
    if not OPT_TEMPLATE_PATH.exists():
        return []
    from daily_update import STANDALONE_BROWSER_THEME

    tpl = OPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    html_ext = (
        tpl.replace("<!-- __STANDALONE_THEME_CSS__ -->", STANDALONE_BROWSER_THEME)
        .replace("__OPTIONS_PAYLOAD_JSON__", blob)
    )
    out = []
    for p in [LIVE_DIR / "options_intraday.html", DOCS_DIR / "options.html"]:
        p.write_text(html_ext, encoding="utf-8")
        out.append(p)
    return out


def run_options_snapshot(notify: bool = False) -> dict:
    t0 = time.time()
    now_bjt = datetime.now(BJT)

    stk_df = pd.read_csv(STOCK_LIST_PATH, dtype=str)
    prev_state = json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else {}
    etf_tc_codes = [e["tc"] for e in OPTION_ETFS]
    all_codes = stk_df["tc_code"].tolist() + etf_tc_codes + ["sh000300", "sh000852", "sz399006"]

    # Parallelize: (A) 5,200-stock Tencent snapshot, (B) 5 ETF 1-min VWAP bars, (C) active option months
    sina_sess = requests.Session()
    sina_sess.headers.update({"Referer": "https://stock.finance.sina.com.cn/", "User-Agent": "Mozilla/5.0"})

    with ThreadPoolExecutor(max_workers=8) as ex:
        fut_qt = ex.submit(fetch_tencent_quotes, all_codes, 80, 6)
        fut_months = ex.submit(get_active_option_months, sina_sess)
        fut_mins = {e["code"]: ex.submit(fetch_etf_minute_bars, e["tc"]) for e in OPTION_ETFS}

        quotes = fut_qt.result()
        months = fut_months.result()
        min_map = {code: f.result() for code, f in fut_mins.items()}

    ref_q = quotes.get("sh000300") or quotes.get("sz159915")
    snap_date = ref_q["date"] if ref_q and ref_q["date"] else now_bjt.strftime("%Y-%m-%d")
    quote_time = ref_q["time"] if ref_q and ref_q["time"] else now_bjt.strftime("%H:%M:%S")

    st_row, bm_row, _, _ = compute_snapshot_bars(quotes, stk_df, prev_state, snap_date)
    breadth = {
        "active_stocks": int(st_row["total_stocks"]),
        "up_cnt": int(bm_row["up_cnt"]),
        "dn_cnt": int(bm_row["dn_cnt"]),
        "b_pct": round(float(bm_row["B"]) * 100.0, 1),
        "m_pct": round(float(bm_row["M"]) * 100.0, 1),
        "mb_pct": round(float(bm_row["M_minus_B"]) * 100.0, 1),
        "tot_amt_yi": round(float(bm_row["total_amt_yi"]), 0),
        "zt": int(st_row["zt_count"]),
        "dt": int(st_row["dt_count"]),
        "zb": int(st_row["zb_count"]),
        "zb_rate_pct": round(float(st_row["zb_rate"]) * 100.0, 1),
        "max_lb": int(st_row["max_lb"]),
        "big_loss_pct": round(float(st_row["big_loss_rate"]) * 100.0, 1),
        "prev_zt_ret_pct": round(float(st_row["prev_zt_ret"]) * 100.0, 2),
    }

    # Fetch live option chains for all 5 ETFs in parallel
    def _build_etf_entry(cfg: dict) -> dict:
        code = cfg["code"]
        m_data = min_map[code]
        q_data = quotes.get(cfg["tc"], {})
        spot = m_data["last"] if m_data.get("ok") else q_data.get("close", 0.0)
        prev_spot = m_data["prev_close"] if m_data.get("ok") else q_data.get("prev_close", spot)
        opt_data = fetch_etf_option_chain(sina_sess, code, spot, prev_spot, months)
        return {
            "code": code,
            "tc": cfg["tc"],
            "name": cfg["name"],
            "style": cfg["style"],
            "min": m_data,
            "opt": opt_data,
        }

    with ThreadPoolExecutor(max_workers=5) as ex:
        etf_entries = list(ex.map(_build_etf_entry, OPTION_ETFS))

    playbook = load_premarket_playbook()

    # Log intraday point for live breadth & IV chart progression
    cy_e = etf_entries[0]
    pt = {
        "t": quote_time[:5],
        "b_pct": breadth["b_pct"],
        "m_pct": breadth["m_pct"],
        "mb_pct": breadth["mb_pct"],
        "zt": breadth["zt"],
        "dt": breadth["dt"],
        "cy_px": cy_e["min"].get("last") if cy_e["min"].get("ok") else None,
        "cy_iv": cy_e["opt"].get("atm_iv") if cy_e["opt"].get("ok") else None,
    }
    history = append_and_load_history(snap_date, pt)
    alerts = detect_intraday_alerts(playbook, breadth, etf_entries, history)

    payload = {
        "meta": {
            "updated_at_bjt": now_bjt.strftime("%Y-%m-%d %H:%M:%S BJT"),
            "snap_date": snap_date,
            "quote_time": quote_time,
            "elapsed_sec": round(time.time() - t0, 2),
            "months": months,
            "is_trading": ("09:25:00" <= quote_time <= "15:02:00") and (now_bjt.weekday() < 5),
        },
        "playbook": playbook,
        "breadth": breadth,
        "alerts": alerts,
        "etfs": etf_entries,
        "history": history,
    }

    (LIVE_DIR / "options_intraday.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (LIVE_DIR / "options_intraday.min.json").write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    built = render_options_html(payload)

    print(
        f"[OPT-MON {now_bjt.strftime('%H:%M:%S BJT')}] {snap_date} {quote_time} ({payload['meta']['elapsed_sec']}s) "
        f"| B={breadth['b_pct']}% M={breadth['m_pct']}% M-B={breadth['mb_pct']:+.1f}% ZT={breadth['zt']} DT={breadth['dt']} BL={breadth['big_loss_pct']}%",
        flush=True,
    )
    for e in etf_entries:
        m = e["min"]
        o = e["opt"]
        if m.get("ok") and o.get("ok"):
            print(
                f"  {e['name']:<8} {m['last']:.3f} ({m['ret_pct']:+.2f}%, vsVWAP {m['vwap_diff_pct']:+.2f}%, amp {m['amp_pct']:.2f}%) "
                f"| {o['active_ym']}({o['active_dte']}d) ATM-IV={o['atm_iv']}% ({o['iv_chg']:+.2f}%) Skew={o['skew_25d']} PCR={o['pcr_vol']}",
                flush=True,
            )
    print(f"  Alerts ({len(alerts)}): " + " | ".join(f"[{a['tag']}] {a['title']}" for a in alerts[:3]), flush=True)
    if built:
        print(f"  HTML updated: {[str(p) for p in built]}", flush=True)

    if notify:
        url = os.environ.get("WEBHOOK_URL", "").strip()
        if url:
            top_alerts = "\n".join(f"- **{a['tag']}**: {a['title']}" for a in alerts[:4])
            msg = (
                f"### 期权日内监控 ({snap_date} {quote_time})\n"
                f"- 宽度: B=`{breadth['b_pct']}%` | M=`{breadth['m_pct']}%` | M-B=`{breadth['mb_pct']:+.1f}%` | 涨跌停 `{breadth['zt']}/{breadth['dt']}`\n"
                f"{top_alerts}"
            )
            try:
                body = {"msg_type": "text", "content": {"text": msg}} if ("feishu" in url or "lark" in url) else {"msgtype": "markdown", "markdown": {"title": "期权日内预警", "text": msg}}
                requests.post(url, json=body, timeout=8)
            except Exception as ex_err:
                print(f"[NOTIFY] err: {ex_err}", flush=True)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Real-time A-share ETF Options Intraday Monitor")
    ap.add_argument("--once", action="store_true", help="Run single live snapshot and exit")
    ap.add_argument("--loop", action="store_true", help="Continuously poll during A-share trading hours")
    ap.add_argument("--interval", type=int, default=60, help="Polling interval in seconds (default: 60)")
    ap.add_argument("--notify", action="store_true", help="Push alerts to WEBHOOK_URL")
    args = ap.parse_args()

    if args.loop and not args.once:
        print(f"Starting continuous Options Intraday Monitor (every {args.interval}s during 09:25-15:02 BJT)...", flush=True)
        while True:
            now = datetime.now(BJT)
            hm = now.strftime("%H:%M")
            in_session = (now.weekday() < 5) and (("09:25" <= hm <= "11:32") or ("12:58" <= hm <= "15:02"))
            if in_session:
                try:
                    run_options_snapshot(notify=args.notify)
                except Exception as e:
                    print(f"[ERR] {e}", flush=True)
                time.sleep(max(15, args.interval))
            else:
                print(f"[IDLE {now.strftime('%H:%M:%S BJT')}] Outside A-share session; sleeping 60s...", flush=True)
                time.sleep(60)
    else:
        run_options_snapshot(notify=args.notify)


if __name__ == "__main__":
    main()
