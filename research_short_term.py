"""Research & backtest short-term A-share limit-up/down sentiment cycle strategies (2021-2026)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from explore_short_term import load_short_term_panel, build_stsi, classify_cycle_phase
from metrics import compute_metrics, yearly_breakdown

START_DATE = "2021-07-01"  # after 120d initial warm-up from 2021-01-05
COST_ETF = 0.001           # 0.10% one-way for index ETF
COST_STOCK_RT = 0.0020     # 0.20% round-trip (stamp tax 0.05% + commission + slippage) for stock relay


def exec_t1_close(
    close: pd.Series,
    target_pos: pd.Series,
    cost_one_way: float = COST_ETF,
    cash_rf_annual: float = 0.0,
    start_date: str = START_DATE,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Conservative T+1 close execution: signal at day t close -> executed at day t+1 close (shift 2)."""
    ret = close.pct_change().fillna(0.0)
    hp = target_pos.shift(2).clip(0.0, 1.0)
    mask = close.index >= pd.Timestamp(start_date)
    ret = ret.loc[mask]
    hp = hp.loc[mask].fillna(0.0)
    to = hp.diff().abs().fillna(0.0)
    rf_d = (1.0 + cash_rf_annual) ** (1.0 / 243.0) - 1.0
    sr = hp * ret + (1.0 - hp) * rf_d - to * cost_one_way
    return sr, ret, hp


def exec_t1_open(
    open_px: pd.Series,
    close_px: pd.Series,
    target_pos: pd.Series,
    cost_one_way: float = COST_ETF,
    cash_rf_annual: float = 0.0,
    start_date: str = START_DATE,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Strict T+1 open execution: signal at day t close -> executed at day t+1 open."""
    oret = open_px.pct_change().fillna(0.0)
    hp = target_pos.shift(2).clip(0.0, 1.0)
    mask = close_px.index >= pd.Timestamp(start_date)
    oret = oret.loc[mask]
    bret = close_px.pct_change().fillna(0.0).loc[mask]
    hp = hp.loc[mask].fillna(0.0)
    to = hp.diff().abs().fillna(0.0)
    rf_d = (1.0 + cash_rf_annual) ** (1.0 / 243.0) - 1.0
    sr = hp * oret + (1.0 - hp) * rf_d - to * cost_one_way
    return sr, bret, hp


def strat_hysteresis(s: pd.Series, buy_th: float, sell_th: float, init_pos: float = 0.5) -> pd.Series:
    vals = s.to_numpy(dtype=float)
    pos = np.empty_like(vals)
    cur = init_pos
    for i, v in enumerate(vals):
        if not np.isnan(v):
            if v <= buy_th:
                cur = 1.0
            elif v >= sell_th:
                cur = 0.0
        pos[i] = cur
    return pd.Series(pos, index=s.index)


def strat_ice_rebound(s: pd.Series, hold_days: int = 10, ice_th: float = 20.0) -> pd.Series:
    """Ice-point rebound tactic: when STSI drops <= ice_th (or crosses back up), hold for N days."""
    vals = s.to_numpy(dtype=float)
    pos = np.zeros_like(vals)
    rem = 0
    for i, v in enumerate(vals):
        if not np.isnan(v) and v <= ice_th:
            rem = hold_days
        if rem > 0:
            pos[i] = 1.0
            rem -= 1
    return pd.Series(pos, index=s.index)


def strat_ebb_avoidance(df: pd.DataFrame) -> pd.Series:
    """Avoid ebb/nuke regimes while keeping ice-point contrarian + healthy momentum.

    Economic logic (zero curve-fitting):
      - Core money-effect health = mean(s_seal_rate, s_anti_nuke) (封板率 + 低大面率, 5日平滑)
      - Hold 100% when:
          (a) Healthy short-term ecology: health >= 50 (无大面潮、封板顺畅), OR
          (b) Extreme ice-point reversal: STSI <= 20 (恐慌彻底出清，左侧博弈反弹)
      - Otherwise (高位分歧退潮、炸板大面蔓延): 0% cash!
    """
    health = (df["s_seal_rate"] + df["s_anti_nuke"]) / 2.0
    health_ma = health.rolling(5, min_periods=1).mean()
    # Hysteresis on health to avoid daily whipsaw: enter >= 52, exit <= 45 (unless ice point <= 20)
    vals_h = health_ma.to_numpy(dtype=float)
    vals_s = df["stsi"].to_numpy(dtype=float)
    pos = np.empty_like(vals_h)
    cur = 0.5
    for i in range(len(vals_h)):
        h, s = vals_h[i], vals_s[i]
        if np.isnan(h) or np.isnan(s):
            pos[i] = cur
            continue
        if s <= 20.0:
            cur = 1.0
        elif h >= 55.0:
            cur = 1.0
        elif h <= 45.0:
            cur = 0.0
        pos[i] = cur
    return pd.Series(pos, index=df.index)


def strat_cycle_momentum(df: pd.DataFrame) -> pd.Series:
    """Pure short-term cycle regime: long in Ice-Point (<=20), Climax/Breakout (>=75), or Healthy Recovery; flat in Ebb."""
    # Combine two empirically distinct short-term drivers:
    # 1. Ice point contrarian (STSI <= 25)
    # 2. Healthy seal & low nuke momentum (health_5d >= 55)
    # Exit when STSI in middle/high (30..75) AND health_5d < 45 (退潮/分歧期)
    health5 = ((df["s_seal_rate"] + df["s_anti_nuke"]) / 2.0).rolling(5, min_periods=1).mean()
    stsi = df["stsi"]
    pos = np.empty(len(df))
    cur = 0.5
    for i in range(len(df)):
        h, s = health5.iloc[i], stsi.iloc[i]
        if np.isnan(h) or np.isnan(s):
            pos[i] = cur
            continue
        if s <= 22.0:
            cur = 1.0
        elif s >= 78.0 and h >= 50.0:
            cur = 1.0
        elif h >= 58.0:
            cur = 1.0
        elif h <= 42.0:
            cur = 0.0
        pos[i] = cur
    return pd.Series(pos, index=df.index)


def backtest_relay_ecology(df: pd.DataFrame, signal_mask: pd.Series, cost_rt: float = COST_STOCK_RT) -> dict:
    """Strictly causal A-share T+1 stock relay backtest:

    - At day t close, check signal_mask[t].
    - If True, at day t+1 OPEN, allocate 50% capital (2-slot rolling tranches to strictly honor T+1!)
      equally across all non-yizi limit-up stocks from day t, and sell at day t+2 CLOSE.
    - Note: df['relay_next_ret'][t+1] is literally close[t+2]/open[t+1] - 1 for non-yizi stocks that hit ZT on day t!
    """
    sub = df.loc[START_DATE:].copy()
    # Signal decided at day t-1 close governs entry at day t open (whose 2-day return is relay_next_ret[t])
    enter = signal_mask.shift(1).loc[START_DATE:].fillna(False).astype(bool)
    ret_entry = sub["relay_next_ret"].fillna(0.0)
    # Exclude days with 0 tradeable stocks
    valid_entry = enter & (sub["tradeable_zt_cnt"] >= 3) & sub["relay_next_ret"].notna()

    trade_rets = (ret_entry[valid_entry] - cost_rt)
    # Construct daily-approximate NAV with 2-slot 50% tranche sizing (half of return realized on entry day, half next day)
    tranche_ret = pd.Series(0.0, index=sub.index)
    tranche_ret[valid_entry] = (ret_entry[valid_entry] - cost_rt) * 0.5
    # Exact compound over non-overlapping / 2-slot portfolio:
    # Slot A enters on even days, Slot B on odd days -> portfolio daily return ≈ 0.5*r_A + 0.5*r_B
    nav = pd.Series(1.0, index=sub.index)
    slot_cash = [0.5, 0.5]
    for idx_i, dt in enumerate(sub.index):
        slot = idx_i % 2
        if valid_entry.loc[dt]:
            r = float(ret_entry.loc[dt] - cost_rt)
            slot_cash[slot] *= (1.0 + r)
        nav.loc[dt] = slot_cash[0] + slot_cash[1]

    dd = nav / nav.cummax() - 1.0
    years = len(sub) / 243.0
    cagr = float(nav.iloc[-1] ** (1.0 / years) - 1.0)
    return {
        "n_trades": int(valid_entry.sum()),
        "pct_active": float(valid_entry.mean()),
        "avg_ret_gross": float(ret_entry[valid_entry].mean()) if valid_entry.any() else 0.0,
        "avg_ret_net": float(trade_rets.mean()) if valid_entry.any() else 0.0,
        "win_rate_net": float((trade_rets > 0).mean()) if valid_entry.any() else 0.0,
        "total_ret": float(nav.iloc[-1] - 1.0),
        "cagr": cagr,
        "max_dd": float(dd.min()),
    }


def main() -> None:
    df = load_short_term_panel()
    stsi_df = build_stsi(df, window=240, min_periods=60, smooth_days=3)
    df = df.join(stsi_df)
    df["phase"] = classify_cycle_phase(df["stsi"], df["stsi_mom"])

    print("=== 1. Index Short-Term Timing on ZZ1000 (2021-07 -> 2026-11, T+1 Close, 0.10% cost, 0% cash) ===")
    bh_r = df["zz1000_close"].pct_change().loc[START_DATE:].fillna(0.0)
    bm = compute_metrics(bh_r, bh_r, pd.Series(1.0, index=bh_r.index))
    print(f"  {'基准: 中证1000买入持有':<30s} | Tot={bm['total_ret']:+6.1%} CAGR={bm['cagr']:+6.2%} Sh={bm['sharpe']:5.2f} MDD={bm['max_dd']:7.2%} Exp=100% Turn= 0.0x")

    strats = {
        "S1a_逆向双阈值(25/75)": strat_hysteresis(df["stsi"], 25, 75),
        "S1b_逆向双阈值(30/70)": strat_hysteresis(df["stsi"], 30, 70),
        "S1c_逆向双阈值(20/80)": strat_hysteresis(df["stsi"], 20, 80),
        "S2a_冰点博弈10日(≤20)": strat_ice_rebound(df["stsi"], 10, 20),
        "S2b_冰点博弈15日(≤25)": strat_ice_rebound(df["stsi"], 15, 25),
        "S3_退潮避险+冰点抄底(健康度+冰点)": strat_ebb_avoidance(df),
        "S4_五阶段情绪周期(避退潮)": strat_cycle_momentum(df),
    }

    for name, w in strats.items():
        sr0, br, hp = exec_t1_close(df["zz1000_close"], w, COST_ETF, 0.0)
        sr2, _, _ = exec_t1_close(df["zz1000_close"], w, COST_ETF, 0.02)
        m0 = compute_metrics(sr0, br, hp)
        m2 = compute_metrics(sr2, br, hp)
        print(
            f"  {name:<30s} | Tot={m0['total_ret']:+6.1%} CAGR={m0['cagr']:+6.2%} Sh={m0['sharpe']:5.2f} MDD={m0['max_dd']:7.2%} "
            f"Exp={m0['avg_exposure']*100:4.0f}% Turn={m0['ann_turnover']:4.1f}x | (2%cash) CAGR={m2['cagr']:+6.2%} Sh={m2['sharpe']:5.2f}"
        )

    print("\n=== 2. Check T+1 OPEN Execution on ZZ1000 (0% cash) ===")
    for name in ["S1a_逆向双阈值(25/75)", "S2a_冰点博弈10日(≤20)", "S3_退潮避险+冰点抄底(健康度+冰点)", "S4_五阶段情绪周期(避退潮)"]:
        sr, br, hp = exec_t1_open(df["zz1000_open"], df["zz1000_close"], strats[name], COST_ETF, 0.0)
        m = compute_metrics(sr, br, hp)
        print(f"  {name:<30s} | Tot={m['total_ret']:+6.1%} CAGR={m['cagr']:+6.2%} Sh={m['sharpe']:5.2f} MDD={m['max_dd']:7.2%}")

    print("\n=== 3. Multi-Index Cross-Check (ZZ1000 / ChiNext / ZZ500 / HS300, T+1 Close, 0% cash) ===")
    for sname in ["S3_退潮避险+冰点抄底(健康度+冰点)", "S4_五阶段情绪周期(避退潮)", "S1a_逆向双阈值(25/75)"]:
        print(f"--- {sname} ---")
        w = strats[sname]
        for iname, col in [("中证1000", "zz1000_close"), ("创业板指", "chinext_close"), ("中证500", "zz500_close"), ("沪深300", "hs300_close")]:
            px = df[col]
            b_r = px.pct_change().loc[START_DATE:].fillna(0.0)
            bm_i = compute_metrics(b_r, b_r, pd.Series(1.0, index=b_r.index))
            sr, br, hp = exec_t1_close(px, w, COST_ETF, 0.0)
            mi = compute_metrics(sr, br, hp)
            print(f"  {iname:<6s} | B&H: CAGR={bm_i['cagr']:+6.2%} MDD={bm_i['max_dd']:7.2%} | Strat: CAGR={mi['cagr']:+6.2%} Sh={mi['sharpe']:5.2f} MDD={mi['max_dd']:7.2%} Excess={mi['excess_cagr']:+6.2%}")

    print("\n=== 4. Sub-Period Split on ZZ1000 (2021-07~2023-12 Bear vs 2024-01~2026-11 Volatile/Bull) ===")
    for sname in ["S3_退潮避险+冰点抄底(健康度+冰点)", "S4_五阶段情绪周期(避退潮)", "S2a_冰点博弈10日(≤20)"]:
        sr0, br, hp = exec_t1_close(df["zz1000_close"], strats[sname], COST_ETF, 0.0)
        m1 = compute_metrics(sr0.loc[:"2023-12-31"], br.loc[:"2023-12-31"], hp.loc[:"2023-12-31"])
        m2 = compute_metrics(sr0.loc["2024-01-01":], br.loc["2024-01-01":], hp.loc["2024-01-01":])
        print(
            f"  {sname:<30s} | 21-23: {m1['cagr']:+6.2%} (B:{m1['bench_cagr']:+6.2%}, MDD {m1['max_dd']:6.1%}/{m1['bench_max_dd']:6.1%}) "
            f"| 24-26: {m2['cagr']:+6.2%} (B:{m2['bench_cagr']:+6.2%}, MDD {m2['max_dd']:6.1%}/{m2['bench_max_dd']:6.1%})"
        )

    print("\n=== 5. Short-Term Limit-Up Relay Backtest (昨日涨停非一字开盘买 -> 次日收盘卖, 含0.20%双边成本) ===")
    relay_rules = {
        "无择时: 每日盲目接力昨日涨停": pd.Series(True, index=df.index),
        "仅在STSI冰点期(≤20)出手接力": df["stsi"] <= 20,
        "仅在STSI低位区(≤30)出手接力": df["stsi"] <= 30,
        "冰点且非大面失控(STSI≤25 & 大面率<50%)": (df["stsi"] <= 25) & (df["big_loss_rate"] < 0.50),
        "晋级率冰点(s_promo≤20)接力": df["s_promo_rate"] <= 20,
        "高潮期(STSI≥75)追高接力(反面教材)": df["stsi"] >= 75,
    }
    for rname, mask in relay_rules.items():
        res = backtest_relay_ecology(df, mask, COST_STOCK_RT)
        print(
            f"  {rname:<36s} | 出手{res['n_trades']:4d}次({res['pct_active']*100:4.1f}%) | "
            f"单次毛收益={res['avg_ret_gross']:+5.2%} 净收益={res['avg_ret_net']:+5.2%} 净胜率={res['win_rate_net']*100:4.1f}% | "
            f"累计={res['total_ret']:+6.1%} MDD={res['max_dd']:7.2%}"
        )


if __name__ == "__main__":
    main()
