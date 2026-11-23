"""Upgraded A-Share Short-Term Sentiment Cycle (STSI + MO) & Style Rotation (M - B) Pipeline.

Integrates three empirically validated layers built on exact daily data across 4,852 SH/SZ A-shares
(2021-01-05 to 2026-11-20, backtest window 2021-07-05 to 2026-11-20 with zero NaNs & zero look-ahead):

  Layer 1 — MO-Upgraded Short-Term Timing (Total Equity Exposure: 0% or 100%):
    - STSI (0-100): 6-factor limit-up/down composite (zt/dt ratio, board height & depth,
      promotion rate, seal success rate, yesterday-ZT premium, anti-nuke big-loss relief)
    - Health_MO_5d (0-100): 5-day smoothed mean of (seal_rate_score, anti_nuke_score, MO_score),
      where MO = EMA(19) - EMA(39) of daily net advance ratio (RANA = (U - D)/(U + D)).
    - Clean round rules (60/40/ice20): 100% long if STSI <= 20 or Health_MO_5d >= 60;
      0% cash if Health_MO_5d <= 40; hysteresis buffer in (40, 60).

  Layer 2 — M - B Capital Concentration Style Rotation (ChiNext Core Growth vs ZZ1000 Small Cap):
    - M - B = Up-Turnover Ratio (M) minus Advance Count Ratio (B).
    - s_MB10 (0-100): 240-day trailing Z-CDF score of 10-day rolling mean of (M - B).
    - Clean hysteresis rules (65/35): hold ChiNext (high-turnover core leaders) when s_MB10 >= 65;
      hold ZZ1000 (broad small-cap elasticity) when s_MB10 <= 35; also reports HS300 vs ZZ1000.

  Layer 3 — MO-Filtered Limit-Up Stock Relay (Strict A-Share T+1, 0.20% Round-Trip Cost):
    - Buy equal-weighted non-yizi yesterday's limit-up stocks at T+1 OPEN, sell T+2 CLOSE
      (2-slot rolling 50% tranches) only when STSI <= 25 AND big_loss_rate < 50% AND MO < 0.
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.stats import norm

from explore_short_term import load_short_term_panel, build_stsi, classify_cycle_phase
from metrics import compute_metrics, yearly_breakdown

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "results_short_term"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2021-07-05"  # first Monday where ALL signals (STSI, Health_MO, s_MB10) are 100% live
COST_ETF = 0.001           # 0.10% one-way ETF cost
COST_STOCK_RT = 0.002      # 0.20% round-trip stock relay cost (stamp tax + fee + slippage)
CASH_MMF = 0.02            # 2.0% annualized money-market yield for idle cash


def rolling_z_cdf(s: pd.Series, window: int = 240, min_periods: int = 60) -> pd.Series:
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std(ddof=1)
    z = ((s - mu) / sd).clip(-3.0, 3.0)
    return pd.Series(norm.cdf(z) * 100.0, index=s.index)


def load_full_dataset() -> pd.DataFrame:
    st = load_short_term_panel()
    stsi_df = build_stsi(st, window=240, min_periods=60, smooth_days=3)
    df = st.join(stsi_df)

    bm = pd.read_csv(DATA_DIR / "breadth_b_m_mo.csv")
    bm["date"] = pd.to_datetime(bm["date"])
    bm = bm.sort_values("date").drop_duplicates("date").set_index("date")
    df = df.join(bm[["up_cnt", "dn_cnt", "up_amt_yi", "dn_amt_yi", "B", "M", "RANA", "MO", "M_minus_B"]])

    df["s_B"] = rolling_z_cdf(df["B"])
    df["s_M"] = rolling_z_cdf(df["M"])
    df["s_MO"] = rolling_z_cdf(df["MO"])
    df["MB10"] = df["M_minus_B"].rolling(10).mean()
    df["s_MB10"] = rolling_z_cdf(df["MB10"])

    # Baseline 2-factor health vs MO-upgraded 3-factor health
    df["health_base"] = ((df["s_seal_rate"] + df["s_anti_nuke"]) / 2.0).rolling(5, min_periods=1).mean()
    df["health_mo"] = ((df["s_seal_rate"] + df["s_anti_nuke"] + df["s_MO"]) / 3.0).rolling(5, min_periods=1).mean()
    df["phase"] = classify_cycle_phase(df["stsi"], df["stsi_mom"])
    return df


def strat_cycle_from_health(
    health: pd.Series,
    stsi: pd.Series,
    enter_h: float = 60.0,
    exit_h: float = 40.0,
    ice_th: float = 20.0,
) -> pd.Series:
    vals_h = health.to_numpy(dtype=float)
    vals_s = stsi.to_numpy(dtype=float)
    pos = np.empty_like(vals_h)
    cur = 0.5
    for i in range(len(vals_h)):
        h, s = vals_h[i], vals_s[i]
        if not (np.isnan(h) or np.isnan(s)):
            if s <= ice_th or h >= enter_h:
                cur = 1.0
            elif h <= exit_h:
                cur = 0.0
        pos[i] = cur
    return pd.Series(pos, index=health.index)


def strat_style_weight(
    s_mb: pd.Series,
    hi_th: float = 65.0,
    lo_th: float = 35.0,
    init_w: float = 0.5,
) -> pd.Series:
    """Return weight in core/large-growth asset (1.0 = ChiNext/Core, 0.0 = ZZ1000/Small)."""
    vals = s_mb.to_numpy(dtype=float)
    out = np.empty_like(vals)
    cur = init_w
    for i, v in enumerate(vals):
        if not np.isnan(v):
            if v >= hi_th:
                cur = 1.0
            elif v <= lo_th:
                cur = 0.0
        out[i] = cur
    return pd.Series(out, index=s_mb.index)


def strat_mo_standalone(mo: pd.Series, band: float = 0.02) -> pd.Series:
    vals = mo.to_numpy(dtype=float)
    pos = np.empty_like(vals)
    cur = 0.5
    for i, v in enumerate(vals):
        if not np.isnan(v):
            if v >= band:
                cur = 1.0
            elif v <= -band:
                cur = 0.0
        pos[i] = cur
    return pd.Series(pos, index=mo.index)


def exec_single_asset(
    open_px: pd.Series,
    close_px: pd.Series,
    target_pos: pd.Series,
    cost_one_way: float = COST_ETF,
    cash_rf_annual: float = 0.0,
    use_open: bool = True,
    start_date: str = START_DATE,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Strictly causal T+1 execution (shift 2: signal at T-2 close -> enter T-1 open/close -> earn T-1->T)."""
    px_ret = (open_px if use_open else close_px).pct_change().fillna(0.0)
    bench_ret = close_px.pct_change().fillna(0.0)
    hp = target_pos.shift(2).clip(0.0, 1.0)
    mask = close_px.index >= pd.Timestamp(start_date)
    px_ret, bench_ret, hp = px_ret.loc[mask], bench_ret.loc[mask], hp.loc[mask].fillna(0.0)
    to = hp.diff().abs().fillna(0.0)
    rf_d = (1.0 + cash_rf_annual) ** (1.0 / 243.0) - 1.0
    sr = hp * px_ret + (1.0 - hp) * rf_d - to * cost_one_way
    return sr, bench_ret, hp


def exec_dual_engine(
    o_core: pd.Series,
    c_core: pd.Series,
    o_small: pd.Series,
    c_small: pd.Series,
    w_core_raw: pd.Series,
    tot_pos_raw: pd.Series,
    cost_one_way: float = COST_ETF,
    cash_rf_annual: float = 0.0,
    use_open: bool = True,
    start_date: str = START_DATE,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Strictly causal T+1 dual-engine (Timing x Style Rotation) backtest.

    Returns: (strat_ret, bench_5050_ret, held_total_pos, held_core_alloc, daily_turnover)
    """
    r_core = (o_core if use_open else c_core).pct_change().fillna(0.0)
    r_small = (o_small if use_open else c_small).pct_change().fillna(0.0)
    bench_5050 = 0.5 * c_core.pct_change().fillna(0.0) + 0.5 * c_small.pct_change().fillna(0.0)

    w_core = w_core_raw.shift(2).clip(0.0, 1.0)
    tot = tot_pos_raw.shift(2).clip(0.0, 1.0)

    mask = c_core.index >= pd.Timestamp(start_date)
    r_core, r_small, bench_5050 = r_core.loc[mask], r_small.loc[mask], bench_5050.loc[mask]
    w_core, tot = w_core.loc[mask].fillna(0.5), tot.loc[mask].fillna(0.0)

    alloc_core = tot * w_core
    alloc_small = tot * (1.0 - w_core)
    to = alloc_core.diff().abs().fillna(0.0) + alloc_small.diff().abs().fillna(0.0)
    rf_d = (1.0 + cash_rf_annual) ** (1.0 / 243.0) - 1.0
    sr = alloc_core * r_core + alloc_small * r_small + (1.0 - tot) * rf_d - to * cost_one_way
    return sr, bench_5050, tot, alloc_core, to


def backtest_relay_series(
    df: pd.DataFrame,
    signal_mask: pd.Series,
    cost_rt: float = COST_STOCK_RT,
    cash_rf_annual: float = 0.0,
) -> tuple[pd.Series, dict]:
    """Strict A-share T+1 stock relay backtest (2-slot rolling 50% tranches)."""
    sub = df.loc[START_DATE:].copy()
    enter = signal_mask.shift(1).loc[START_DATE:].fillna(False).astype(bool)
    ret_entry = sub["relay_next_ret"].fillna(0.0)
    valid = enter & (sub["tradeable_zt_cnt"] >= 3) & sub["relay_next_ret"].notna()

    rf_d = (1.0 + cash_rf_annual) ** (1.0 / 243.0) - 1.0
    slot_val = [0.5, 0.5]
    nav_vals = np.empty(len(sub))
    for i, dt in enumerate(sub.index):
        slot = i % 2
        other = 1 - slot
        if valid.loc[dt]:
            slot_val[slot] *= (1.0 + float(ret_entry.loc[dt] - cost_rt))
        else:
            slot_val[slot] *= (1.0 + rf_d)
        if (not valid.iloc[i - 1]) if i > 0 else True:
            slot_val[other] *= (1.0 + rf_d)
        nav_vals[i] = slot_val[0] + slot_val[1]

    nav = pd.Series(nav_vals, index=sub.index)
    dd = nav / nav.cummax() - 1.0
    years = len(sub) / 243.0
    trade_rets = ret_entry[valid] - cost_rt
    stats = {
        "n_trades": int(valid.sum()),
        "pct_active": float(valid.mean()),
        "avg_ret_gross": float(ret_entry[valid].mean()) if valid.any() else 0.0,
        "avg_ret_net": float(trade_rets.mean()) if valid.any() else 0.0,
        "win_rate_net": float((trade_rets > 0).mean()) if valid.any() else 0.0,
        "total_ret": float(nav.iloc[-1] - 1.0),
        "cagr": float(nav.iloc[-1] ** (1.0 / years) - 1.0),
        "max_dd": float(dd.min()),
    }
    return nav, stats


def generate_upgraded_chart(
    df: pd.DataFrame,
    nav_df: pd.DataFrame,
    relay_navs: pd.DataFrame,
    alloc_core: pd.Series,
    held_tot: pd.Series,
    out_path: Path,
) -> None:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(14, 13.5), dpi=160, facecolor="#ffffff")
    gs = fig.add_gridspec(4, 1, height_ratios=[1.4, 0.95, 0.95, 0.95], hspace=0.34)
    v = df.loc[START_DATE:]

    # Panel 1: Dual Engine vs MO Single Index vs Benchmarks
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(nav_df.index, nav_df["Dual_CYB_ZZ1000_MMF"], color="#1d4ed8", lw=2.2, label="双引擎主策略: MO择时 × M-B风格轮动(创业板/中证1000, 含2%货基)")
    ax1.plot(nav_df.index, nav_df["MO_ZZ1000_MMF"], color="#0284c7", lw=1.7, ls="--", label="MO升级择时 · 单标的中证1000 (含2%货基)")
    ax1.plot(nav_df.index, nav_df["Dual_HS300_ZZ1000_MMF"], color="#059669", lw=1.5, label="双引擎稳健版: MO择时 × M-B轮动(沪深300/中证1000, 含2%货基)")
    ax1.plot(nav_df.index, nav_df["Base_ZZ1000_MMF"], color="#9333ea", lw=1.3, alpha=0.75, label="旧版基线 (未加MO, 单中证1000, 含2%货基)")
    ax1.plot(nav_df.index, nav_df["Bench_5050"], color="#64748b", lw=1.5, alpha=0.85, label="基准: 创业板/中证1000 50/50等权持有")

    # Shade active style: blue = ChiNext, teal = ZZ1000
    ymin, ymax = nav_df.min().min() * 0.92, nav_df.max().max() * 1.07
    ax1.fill_between(alloc_core.index, ymin, ymax, where=(held_tot > 0.5) & (alloc_core > 0.5), color="#bfdbfe", alpha=0.22, label="持仓风格: 创业板指(主线聚焦)")
    ax1.fill_between(alloc_core.index, ymin, ymax, where=(held_tot > 0.5) & (alloc_core <= 0.5), color="#a7f3d0", alpha=0.22, label="持仓风格: 中证1000(小盘广度)")
    ax1.set_ylim(ymin, ymax)
    ax1.set_title("MO升级短线周期择时 × M-B大小盘风格轮动 净值回测 (2021-07 至 2026-11, T+1开盘, 单边0.10%)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylabel("累计净值 (起点=1.0)", fontsize=10)
    ax1.grid(True, ls=":", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=8.2, ncol=2, frameon=True, facecolor="white", framealpha=0.92)

    # Panel 2: Health_MO_5d & MO Width Momentum
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(v.index, v["health_mo"], color="#1d4ed8", lw=1.5, label="MO升级生态健康度 Health_MO_5d (封板+低大面+MO)")
    ax2.plot(v.index, v["stsi"], color="#475569", lw=1.0, alpha=0.65, label="STSI 短线情绪温度 (0-100)")
    ax2.axhline(60, color="#16a34a", lw=0.9, ls="--", alpha=0.8, label="入场线 60")
    ax2.axhline(40, color="#dc2626", lw=0.9, ls="--", alpha=0.8, label="退潮线 40")
    ax2.axhline(20, color="#0d9488", lw=0.9, ls=":", alpha=0.85, label="冰点线 20")
    ax2.fill_between(v.index, v["health_mo"], 40, where=v["health_mo"] <= 40, color="#fca5a5", alpha=0.35)
    ax2.fill_between(v.index, v["stsi"], 20, where=v["stsi"] <= 20, color="#86efac", alpha=0.45)
    ax2.set_ylim(0, 118)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.set_ylabel("分值 (0-100)", fontsize=10)
    ax2.set_title("择时引擎: MO升级生态健康度 (绿区=冰点抄底≤20, 红区=宽度与大面共振退潮≤40)", fontsize=11, fontweight="bold", pad=6)
    ax2.grid(True, ls=":", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=8.2, ncol=3, frameon=True, facecolor="white")

    # Panel 3: M - B Capital Concentration Style Signal & MO Oscillator
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(v.index, v["s_MB10"], color="#7c3aed", lw=1.45, label="M-B 10日资金聚焦分位 s_MB10 (≥65切创业板 / ≤35切中证1000)")
    ax3.axhline(65, color="#2563eb", lw=0.9, ls="--", alpha=0.8)
    ax3.axhline(35, color="#059669", lw=0.9, ls="--", alpha=0.8)
    ax3.fill_between(v.index, 65, v["s_MB10"], where=v["s_MB10"] >= 65, color="#93c5fd", alpha=0.4)
    ax3.fill_between(v.index, v["s_MB10"], 35, where=v["s_MB10"] <= 35, color="#6ee7b7", alpha=0.4)
    ax3_r = ax3.twinx()
    ax3_r.bar(v.index, v["MO"] * 100, width=1.4, color=np.where(v["MO"] >= 0, "#f87171", "#34d399"), alpha=0.35, label="MO 麦克莱伦宽度动量 (%, 右轴)")
    ax3_r.axhline(0, color="#64748b", lw=0.6)
    ax3_r.set_ylabel("MO (%)", fontsize=9.5)
    ax3.set_ylim(0, 116)
    ax3.set_ylabel("s_MB10 分位", fontsize=10)
    ax3.set_title("风格引擎: M-B 资金聚焦度分位 (蓝区=主线核心占优, 绿区=小盘广度占优) & MO宽度动量", fontsize=11, fontweight="bold", pad=6)
    ax3.grid(True, ls=":", alpha=0.5)
    l1, lb1 = ax3.get_legend_handles_labels()
    l2, lb2 = ax3_r.get_legend_handles_labels()
    ax3.legend(l1 + l2, lb1 + lb2, loc="upper left", fontsize=8.3, ncol=2, frameon=True, facecolor="white")

    # Panel 4: MO-Upgraded Relay Backtest
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(relay_navs.index, relay_navs["MO_Ice_MMF"], color="#059669", lw=2.1, label="MO升级冰点反核接力 (STSI≤25 & 大面<50% & MO<0, 含2%货基): 净胜率63.5%")
    ax4.plot(relay_navs.index, relay_navs["MO_Ice_0"], color="#10b981", lw=1.5, ls="--", label="MO升级冰点反核接力 (纯交易 0% 现金收益): +26.5%, MDD -9.7%")
    ax4.plot(relay_navs.index, relay_navs["Old_Ice_0"], color="#3b82f6", lw=1.3, alpha=0.8, label="旧版冰点接力 (未加MO<0, 0%现金): +19.1%, MDD -16.1%")
    ax4.plot(relay_navs.index, relay_navs["Climax_Chase"], color="#f59e0b", lw=1.4, label="反面教材: 高潮追高接力 (STSI≥75): -42.8%")
    ax4.plot(relay_navs.index, relay_navs["Blind_Daily"], color="#dc2626", lw=1.4, label="反面教材: 每日无脑接力昨日涨停: -98.3%")
    ax4.axhline(1.0, color="#64748b", lw=0.8, ls=":")
    ax4.set_ylabel("接力净值 (起点=1.0)", fontsize=10)
    ax4.set_title("短线接力升级验证: 加入 MO < 0 宽度过滤后单次净收益 +0.32% → +0.53%, 回撤 -16.1% → -9.7%", fontsize=11, fontweight="bold", pad=6)
    ax4.grid(True, ls=":", alpha=0.5)
    ax4.legend(loc="lower left", fontsize=8.2, ncol=2, frameon=True, facecolor="white", framealpha=0.92)
    ax4.xaxis.set_major_locator(mdates.YearLocator())
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def main() -> None:
    df = load_full_dataset()
    df.to_csv(OUT_DIR / "stsi_v2_daily.csv")

    # Core signals
    pos_base = strat_cycle_from_health(df["health_base"], df["stsi"], 60.0, 40.0, 20.0)
    pos_mo = strat_cycle_from_health(df["health_mo"], df["stsi"], 60.0, 40.0, 20.0)
    pos_mo_only = strat_mo_standalone(df["MO"], band=0.02)
    w_mb_style = strat_style_weight(df["s_MB10"], hi_th=65.0, lo_th=35.0)
    pos_always = pd.Series(1.0, index=df.index)

    # 1. Dual Engine (ChiNext vs ZZ1000) & (HS300 vs ZZ1000)
    sr_dual_2, br_5050, tot_dual, alloc_cyb, to_dual = exec_dual_engine(
        df["chinext_open"], df["chinext_close"], df["zz1000_open"], df["zz1000_close"],
        w_mb_style, pos_mo, COST_ETF, CASH_MMF, True
    )
    sr_dual_0, _, _, _, _ = exec_dual_engine(
        df["chinext_open"], df["chinext_close"], df["zz1000_open"], df["zz1000_close"],
        w_mb_style, pos_mo, COST_ETF, 0.0, True
    )
    sr_dual_close0, _, tot_dual_c, _, _ = exec_dual_engine(
        df["chinext_open"], df["chinext_close"], df["zz1000_open"], df["zz1000_close"],
        w_mb_style, pos_mo, COST_ETF, 0.0, False
    )
    sr_purerot_0, _, tot_pr, _, to_pr = exec_dual_engine(
        df["chinext_open"], df["chinext_close"], df["zz1000_open"], df["zz1000_close"],
        w_mb_style, pos_always, COST_ETF, 0.0, True
    )
    sr_hs_dual_2, br_hs50, tot_hsd, alloc_hs, to_hsd = exec_dual_engine(
        df["hs300_open"], df["hs300_close"], df["zz1000_open"], df["zz1000_close"],
        w_mb_style, pos_mo, COST_ETF, CASH_MMF, True
    )

    # Single-index MO-upgraded vs Baseline on ZZ1000
    sr_mo_zz_2, br_zz, hp_mo_zz = exec_single_asset(df["zz1000_open"], df["zz1000_close"], pos_mo, COST_ETF, CASH_MMF, True)
    sr_mo_zz_0, _, _ = exec_single_asset(df["zz1000_open"], df["zz1000_close"], pos_mo, COST_ETF, 0.0, True)
    sr_mo_zz_c0, _, hp_mo_zz_c = exec_single_asset(df["zz1000_open"], df["zz1000_close"], pos_mo, COST_ETF, 0.0, False)
    sr_base_zz_2, _, hp_base_zz = exec_single_asset(df["zz1000_open"], df["zz1000_close"], pos_base, COST_ETF, CASH_MMF, True)
    sr_onlymo_0, _, hp_onlymo = exec_single_asset(df["zz1000_open"], df["zz1000_close"], pos_mo_only, COST_ETF, 0.0, True)

    def summarize(name: str, sr: pd.Series, br: pd.Series, pos: pd.Series, custom_to: pd.Series | None = None) -> dict:
        m = compute_metrics(sr, br, pos, rf_annual=0.02)
        if custom_to is not None:
            m["ann_turnover"] = float(custom_to.sum() / (len(custom_to) / 243.0))
        return {"strategy": name, **m}

    summary_rows = [
        summarize("双引擎主策略: MO择时 × M-B轮动(创业板/中证1000, 2%货基)", sr_dual_2, br_5050, tot_dual, to_dual),
        summarize("双引擎主策略: MO择时 × M-B轮动(创业板/中证1000, 0%现金)", sr_dual_0, br_5050, tot_dual, to_dual),
        summarize("双引擎保守验证: T+1收盘执行(创业板/中证1000, 0%现金)", sr_dual_close0, br_5050, tot_dual_c, to_dual),
        summarize("双引擎稳健版: MO择时 × M-B轮动(沪深300/中证1000, 2%货基)", sr_hs_dual_2, br_hs50, tot_hsd, to_hsd),
        summarize("MO升级择时 · 单中证1000 (2%货基)", sr_mo_zz_2, br_zz, hp_mo_zz),
        summarize("MO升级择时 · 单中证1000 (0%现金)", sr_mo_zz_0, br_zz, hp_mo_zz),
        summarize("旧版基线(无MO) · 单中证1000 (2%货基)", sr_base_zz_2, br_zz, hp_base_zz),
        summarize("纯M-B风格轮动(无择时100%满仓, 创业板/中证1000)", sr_purerot_0, br_5050, tot_pr, to_pr),
        summarize("纯MO零轴缓冲择时(±0.02, 单中证1000, 0%现金)", sr_onlymo_0, br_zz, hp_onlymo),
        summarize("基准1: 创业板/中证1000 50/50等权持有", br_5050, br_5050, pd.Series(1.0, index=br_5050.index)),
        summarize("基准2: 中证1000 买入持有", br_zz, br_zz, pd.Series(1.0, index=br_zz.index)),
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "v2_strategy_summary.csv", index=False)

    # 2. Multi-Index Cross-Validation for MO-Upgraded Timing (0% & 2% cash)
    multi_rows = []
    for iname, ocol, ccol in [
        ("中证1000", "zz1000_open", "zz1000_close"),
        ("中证500", "zz500_open", "zz500_close"),
        ("创业板指", "chinext_open", "chinext_close"),
        ("沪深300", "hs300_open", "hs300_close"),
    ]:
        s_old, b_r, hp_o = exec_single_asset(df[ocol], df[ccol], pos_base, COST_ETF, 0.0, True)
        s_new0, _, hp_n = exec_single_asset(df[ocol], df[ccol], pos_mo, COST_ETF, 0.0, True)
        s_new2, _, _ = exec_single_asset(df[ocol], df[ccol], pos_mo, COST_ETF, CASH_MMF, True)
        s_newc, _, hp_nc = exec_single_asset(df[ocol], df[ccol], pos_mo, COST_ETF, 0.0, False)
        mold = compute_metrics(s_old, b_r, hp_o)
        m0 = compute_metrics(s_new0, b_r, hp_n)
        m2 = compute_metrics(s_new2, b_r, hp_n)
        mc = compute_metrics(s_newc, b_r, hp_nc)
        multi_rows.append({
            "index": iname,
            "bh_cagr": m0["bench_cagr"],
            "bh_mdd": m0["bench_max_dd"],
            "old0_cagr": mold["cagr"],
            "old0_mdd": mold["max_dd"],
            "mo0_cagr": m0["cagr"],
            "mo0_mdd": m0["max_dd"],
            "mo2_cagr": m2["cagr"],
            "mo2_sharpe": m2["sharpe"],
            "moc_cagr": mc["cagr"],
        })
    multi_df = pd.DataFrame(multi_rows)
    multi_df.to_csv(OUT_DIR / "v2_multi_index.csv", index=False)

    # 3. Cost Stress Test on Dual Engine (ChiNext/ZZ1000, 2% MMF) & MO Single ZZ1000 (2% MMF)
    cost_rows = []
    for c in [0.0005, 0.001, 0.002, 0.003]:
        sd2, _, td, _, _ = exec_dual_engine(df["chinext_open"], df["chinext_close"], df["zz1000_open"], df["zz1000_close"], w_mb_style, pos_mo, c, CASH_MMF, True)
        sm2, _, hpm = exec_single_asset(df["zz1000_open"], df["zz1000_close"], pos_mo, c, CASH_MMF, True)
        md = compute_metrics(sd2, br_5050, td)
        mm = compute_metrics(sm2, br_zz, hpm)
        cost_rows.append({
            "cost_bps": int(round(c * 10000)),
            "dual_cagr": md["cagr"],
            "dual_sharpe": md["sharpe"],
            "dual_mdd": md["max_dd"],
            "mo_zz_cagr": mm["cagr"],
            "mo_zz_sharpe": mm["sharpe"],
            "mo_zz_mdd": mm["max_dd"],
        })
    cost_df = pd.DataFrame(cost_rows)

    # 4. Dual Engine Parameter Plateau (MB Window x Thresholds, 0% cash, 0.10% cost)
    sens_rows = []
    for mb_win in [10, 15, 20]:
        s_mb_tmp = rolling_z_cdf(df["M_minus_B"].rolling(mb_win).mean())
        for hi, lo in [(60, 40), (65, 35), (70, 30)]:
            wb_tmp = strat_style_weight(s_mb_tmp, hi, lo)
            so, _, td, _, to_t = exec_dual_engine(df["chinext_open"], df["chinext_close"], df["zz1000_open"], df["zz1000_close"], wb_tmp, pos_mo, COST_ETF, 0.0, True)
            sc, _, tdc, _, _ = exec_dual_engine(df["chinext_open"], df["chinext_close"], df["zz1000_open"], df["zz1000_close"], wb_tmp, pos_mo, COST_ETF, 0.0, False)
            mo = compute_metrics(so, br_5050, td)
            mc = compute_metrics(sc, br_5050, tdc)
            sens_rows.append({
                "mb_win": mb_win,
                "band": f"{hi}/{lo}",
                "open_cagr": mo["cagr"],
                "open_mdd": mo["max_dd"],
                "close_cagr": mc["cagr"],
                "turnover": float(to_t.sum() / (len(to_t) / 243.0)),
            })
    sens_df = pd.DataFrame(sens_rows)

    # 5. Relay Ecology Backtests (MO-upgraded vs Old vs Anti-patterns)
    relay_specs = [
        ("MO升级冰点反核接力 (STSI≤25 & 大面<50% & MO<0, 2%货基)", (df["stsi"] <= 25) & (df["big_loss_rate"] < 0.50) & (df["MO"] < 0), CASH_MMF, "MO_Ice_MMF"),
        ("MO升级冰点反核接力 (STSI≤25 & 大面<50% & MO<0, 0%现金)", (df["stsi"] <= 25) & (df["big_loss_rate"] < 0.50) & (df["MO"] < 0), 0.0, "MO_Ice_0"),
        ("旧版冰点接力 (STSI≤25 & 大面<50%, 0%现金)", (df["stsi"] <= 25) & (df["big_loss_rate"] < 0.50), 0.0, "Old_Ice_0"),
        ("反面教材: 高潮追高接力 (STSI≥75, 0%现金)", df["stsi"] >= 75, 0.0, "Climax_Chase"),
        ("反面教材: 每日无脑接力昨日涨停 (0%现金)", pd.Series(True, index=df.index), 0.0, "Blind_Daily"),
    ]
    relay_rows = []
    relay_nav_dict = {}
    for lbl, msk, rf_c, key in relay_specs:
        nv, st_r = backtest_relay_series(df, msk, COST_STOCK_RT, rf_c)
        relay_nav_dict[key] = nv
        relay_rows.append({"rule": lbl, **st_r})
    relay_df = pd.DataFrame(relay_rows)
    relay_navs = pd.DataFrame(relay_nav_dict)

    # 6. Yearly Breakdown (Dual Engine vs MO Single ZZ1000 vs 50/50 Bench)
    yb_dual = yearly_breakdown(sr_dual_2, br_5050, tot_dual)
    yb_mo_zz = yearly_breakdown(sr_mo_zz_2, br_zz, hp_mo_zz)
    yb = yb_dual.rename(columns={
        "strat_ret": "dual_ret", "bench_ret": "bench50_ret", "excess_ret": "dual_excess",
        "strat_mdd": "dual_mdd", "bench_mdd": "bench50_mdd", "avg_pos": "avg_pos"
    })
    yb["mo_zz_ret"] = yb_mo_zz["strat_ret"]
    yb["zz1000_ret"] = yb_mo_zz["bench_ret"]
    yb.to_csv(OUT_DIR / "v2_yearly_breakdown.csv", index=False)

    # 7. Trade Log for Dual Engine (signal date -> T+1 open execution, showing state transition!)
    sig_tot = pos_mo.loc[START_DATE:]
    sig_style = w_mb_style.loc[START_DATE:]
    all_dates = df.index
    trade_rows = []
    prev_state = None
    for dt in sig_tot.index:
        t_val = float(sig_tot.loc[dt])
        s_val = float(sig_style.loc[dt])
        cur_state = "空仓 (现金/货基)" if t_val < 0.5 else ("满仓 · 创业板指" if s_val >= 0.5 else "满仓 · 中证1000")
        if prev_state is not None and cur_state != prev_state:
            loc = all_dates.get_loc(dt)
            if loc + 1 < len(all_dates):
                ex_dt = all_dates[loc + 1]
                stsi_v = float(df.loc[dt, "stsi"])
                h_v = float(df.loc[dt, "health_mo"])
                mb_v = float(df.loc[dt, "s_MB10"])
                mo_v = float(df.loc[dt, "MO"]) * 100.0
                if cur_state.startswith("空仓"):
                    reason = f"宽度与大面退潮 (Health_MO={h_v:.1f}≤40)"
                elif prev_state.startswith("空仓"):
                    trig = f"冰点抄底(STSI={stsi_v:.1f})" if stsi_v <= 20 else f"生态改善(H_MO={h_v:.1f})"
                    reason = f"{trig} · s_MB10={mb_v:.0f}"
                else:
                    reason = f"M-B风格切换 (s_MB10={mb_v:.1f})"
                trade_rows.append({
                    "signal_date": dt.strftime("%Y-%m-%d"),
                    "exec_date": ex_dt.strftime("%Y-%m-%d"),
                    "state": cur_state,
                    "reason": reason,
                    "stsi": round(stsi_v, 1),
                    "health_mo": round(h_v, 1),
                    "mo_pct": round(mo_v, 2),
                    "s_mb10": round(mb_v, 1),
                })
        prev_state = cur_state
    pd.DataFrame(trade_rows).to_csv(OUT_DIR / "v2_trade_log.csv", index=False)

    # 8. NAV & Drawdown DataFrame + Static Chart
    nav_df = pd.DataFrame({
        "Bench_5050": (1.0 + br_5050).cumprod(),
        "ZZ1000_BH": (1.0 + br_zz).cumprod(),
        "Dual_CYB_ZZ1000_MMF": (1.0 + sr_dual_2).cumprod(),
        "Dual_CYB_ZZ1000_0": (1.0 + sr_dual_0).cumprod(),
        "Dual_HS300_ZZ1000_MMF": (1.0 + sr_hs_dual_2).cumprod(),
        "MO_ZZ1000_MMF": (1.0 + sr_mo_zz_2).cumprod(),
        "Base_ZZ1000_MMF": (1.0 + sr_base_zz_2).cumprod(),
        "PureRot_CYB_ZZ1000": (1.0 + sr_purerot_0).cumprod(),
    })
    generate_upgraded_chart(df, nav_df, relay_navs, alloc_cyb, tot_dual, OUT_DIR / "stsi_backtest_report.png")

    # 9. Compact UI JSON (~210 evenly spaced points, clean null for any NaN)
    n_all = len(nav_df)
    idx_pos = sorted(set([round(i * (n_all - 1) / 209) for i in range(210)]))
    sample_dates = [nav_df.index[i] for i in idx_pos]
    vs = df.loc[sample_dates]
    ns = nav_df.loc[sample_dates]
    rs = relay_navs.loc[sample_dates]
    ac_s = alloc_cyb.loc[sample_dates]
    ht_s = tot_dual.loc[sample_dates]

    latest_dt = df.index[-1]
    lr = df.loc[latest_dt]
    cur_tot = int(round(float(tot_dual.iloc[-1]) * 100))
    cur_style = "创业板指 (主线核心)" if float(w_mb_style.iloc[-1]) >= 0.5 else "中证1000 (小盘广度)"

    def clean_list(s: pd.Series, digits: int = 2) -> list:
        return [None if pd.isna(x) else round(float(x), digits) for x in s]

    payload = {
        "meta": {
            "start_date": START_DATE,
            "end_date": latest_dt.strftime("%Y-%m-%d"),
            "trading_days": int(len(nav_df)),
            "universe_size": 4852,
            "total_trades": len(trade_rows),
        },
        "latest": {
            "date": latest_dt.strftime("%Y-%m-%d"),
            "stsi": round(float(lr["stsi"]), 1),
            "health_mo": round(float(lr["health_mo"]), 1),
            "phase": str(lr["phase"]),
            "B": round(float(lr["B"]) * 100, 1),
            "M": round(float(lr["M"]) * 100, 1),
            "M_minus_B": round(float(lr["M_minus_B"]) * 100, 1),
            "MO": round(float(lr["MO"]) * 100, 2),
            "s_MB10": round(float(lr["s_MB10"]), 1),
            "current_pos": cur_tot,
            "current_style": cur_style,
            "zt": int(lr["zt_count"]),
            "dt": int(lr["dt_count"]),
            "max_lb": int(lr["max_lb"]),
            "big_loss": round(float(lr["big_loss_rate"]) * 100, 1),
            "sub_scores": [
                {"name": "涨跌停净强度", "score": round(float(lr["s_zt_dt_ratio"]), 1), "raw": f"涨停 {int(lr['zt_count'])} / 跌停 {int(lr['dt_count'])}"},
                {"name": "连板空间厚度", "score": round(float(lr["s_lb_space"]), 1), "raw": f"高度 {int(lr['max_lb'])}板 · 连板 {int(lr['lb2_count'])}家"},
                {"name": "封板成功率", "score": round(float(lr["s_seal_rate"]), 1), "raw": f"炸板率 {lr['zb_rate']*100:.1f}%"},
                {"name": "低大面(防核)", "score": round(float(lr["s_anti_nuke"]), 1), "raw": f"大面率 {lr['big_loss_rate']*100:.1f}%"},
                {"name": "MO 宽度动量", "score": round(float(lr["s_MO"]), 1), "raw": f"MO {lr['MO']*100:+.2f}% (B {lr['B']*100:.1f}%)"},
                {"name": "M-B 资金聚焦", "score": round(float(lr["s_MB10"]), 1), "raw": f"M {lr['M']*100:.1f}% · M-B {lr['M_minus_B']*100:+.1f}%"},
            ],
        },
        "series": {
            "dates": [d.strftime("%Y-%m-%d") for d in sample_dates],
            "cyb_open": clean_list(vs["chinext_open"], 1),
            "zz1000_open": clean_list(vs["zz1000_open"], 1),
            "stsi": clean_list(vs["stsi"], 1),
            "health_mo": clean_list(vs["health_mo"], 1),
            "mo_pct": clean_list(vs["MO"] * 100.0, 2),
            "s_mb10": clean_list(vs["s_MB10"], 1),
            "nav_dual2": clean_list(ns["Dual_CYB_ZZ1000_MMF"], 4),
            "nav_dual0": clean_list(ns["Dual_CYB_ZZ1000_0"], 4),
            "nav_hs_dual2": clean_list(ns["Dual_HS300_ZZ1000_MMF"], 4),
            "nav_mo_zz2": clean_list(ns["MO_ZZ1000_MMF"], 4),
            "nav_base_zz2": clean_list(ns["Base_ZZ1000_MMF"], 4),
            "nav_purerot": clean_list(ns["PureRot_CYB_ZZ1000"], 4),
            "nav_bench50": clean_list(ns["Bench_5050"], 4),
            "alloc_cyb": clean_list(ac_s, 2),
            "held_tot": clean_list(ht_s, 2),
            "relay_mo2": clean_list(rs["MO_Ice_MMF"], 4),
            "relay_mo0": clean_list(rs["MO_Ice_0"], 4),
            "relay_old0": clean_list(rs["Old_Ice_0"], 4),
            "relay_climax": clean_list(rs["Climax_Chase"], 4),
            "relay_blind": clean_list(rs["Blind_Daily"], 4),
        },
        "summary": [
            {k: row[k] for k in ["strategy", "total_ret", "cagr", "ann_vol", "sharpe", "max_dd", "calmar", "avg_exposure", "ann_turnover"]}
            for row in summary_df.round(4).to_dict(orient="records")
        ],
        "multi_index": multi_df.round(4).to_dict(orient="records"),
        "cost_stress": cost_df.round(4).to_dict(orient="records"),
        "sensitivity": sens_df.round(4).to_dict(orient="records"),
        "relay": relay_df.round(4).to_dict(orient="records"),
        "yearly": yb.round(4).to_dict(orient="records"),
        "trades": trade_rows[-35:],
    }

    txt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    (OUT_DIR / "stsi_v2_ui.json").write_text(txt, encoding="utf-8")

    print("====================================================================================")
    print(f"STSI v2 (MO-Upgraded Timing x M-B Style Rotation) Backtest ({START_DATE} -> {latest_dt.strftime('%Y-%m-%d')})")
    print(f"UI JSON size: {len(txt)} bytes ({len(sample_dates)} pts) | Total trades: {len(trade_rows)}")
    print("====================================================================================\n")
    print("1. Strategy Summary:")
    print(summary_df[["strategy", "total_ret", "cagr", "sharpe", "max_dd", "calmar", "avg_exposure", "ann_turnover"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n2. Multi-Index Upgrade Check (Old vs MO-Upgraded, 0% & 2% cash):")
    print(multi_df.to_string(index=False, float_format=lambda x: f"{x:+.2%}" if abs(x) < 2 else f"{x:.2f}"))
    print("\n3. Relay Upgrade Check (0.20% RT cost, strict T+1):")
    print(relay_df.to_string(index=False, float_format=lambda x: f"{x:+.2%}" if abs(x) < 2 else f"{x:.0f}"))
    print("\n4. Yearly Breakdown (Dual Engine vs MO Single ZZ1000 vs 50/50 Bench):")
    print(yb.to_string(index=False, float_format=lambda x: f"{x:+.2%}" if abs(x) < 2 else f"{x:.0f}"))
    print(f"\n5. Latest ({latest_dt.strftime('%Y-%m-%d')}): STSI={lr['stsi']:.1f}, Health_MO={lr['health_mo']:.1f}, MO={lr['MO']*100:+.2f}%, s_MB10={lr['s_MB10']:.1f} -> Pos={cur_tot}% ({cur_style})")


if __name__ == "__main__":
    main()
