"""Rigorous empirical evaluation of Breadth (B), Up-Turnover Ratio (M), and McClellan Oscillator (MO).

Tests:
  1. Descriptive stats, correlations among B, M, M-B, MO, STSI, and Health_5d
  2. Multi-horizon Spearman Rank IC (1d, 3d, 5d, 10d, 20d, 60d + next-day relay) on ZZ1000 & HS300
  3. Quintile forward return tables for B, M, M-B, and MO
  4. Standalone timing backtests for B, M, MO (T+1 open & T+1 close, 0.10% cost, 0% cash yield)
  5. Incremental ablation & Newey-West marginal t-stats when combined with existing STSI / Health_5d
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

from explore_short_term import load_short_term_panel, build_stsi
from run_short_term_backtest import (
    START_DATE,
    COST_ETF,
    COST_STOCK_RT,
    strat_cycle_main,
    exec_t1_open,
    exec_t1_close,
    backtest_relay_series,
)
from metrics import compute_metrics

DATA_DIR = Path(__file__).resolve().parent / "data"


def rolling_z_cdf(s: pd.Series, window: int = 240, min_periods: int = 60) -> pd.Series:
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std(ddof=1)
    z = ((s - mu) / sd).clip(-3.0, 3.0)
    return pd.Series(norm.cdf(z) * 100.0, index=s.index)


def ols_newey_west_tstat(y: pd.Series, X: pd.DataFrame, max_lag: int = 10) -> pd.DataFrame:
    """Compute OLS coefficients and Newey-West HAC t-statistics (handling overlapping return autocorrelation)."""
    df = pd.concat([y.rename("y"), X], axis=1, sort=False).dropna()
    yv = df["y"].to_numpy(dtype=float)
    Xv = np.column_stack([np.ones(len(df)), df[X.columns].to_numpy(dtype=float)])
    n, k = Xv.shape
    xtx_inv = np.linalg.inv(Xv.T @ Xv)
    beta = xtx_inv @ (Xv.T @ yv)
    resid = yv - Xv @ beta

    # Newey-West Bartlett kernel HAC covariance
    S = np.zeros((k, k))
    scores = Xv * resid[:, None]
    S += scores.T @ scores
    for lag in range(1, max_lag + 1):
        w = 1.0 - lag / (max_lag + 1.0)
        gamma = scores[lag:].T @ scores[:-lag]
        S += w * (gamma + gamma.T)
    cov = xtx_inv @ S @ xtx_inv
    se = np.sqrt(np.diag(cov))
    tstat = beta / se
    cols = ["const"] + list(X.columns)
    return pd.DataFrame({"beta": beta, "se": se, "t_stat": tstat}, index=cols)


def main() -> None:
    st = load_short_term_panel()
    stsi_df = build_stsi(st, window=240, min_periods=60, smooth_days=3)
    df = st.join(stsi_df)
    df["health_5d"] = ((df["s_seal_rate"] + df["s_anti_nuke"]) / 2.0).rolling(5, min_periods=1).mean()

    bm = pd.read_csv(DATA_DIR / "breadth_b_m_mo.csv")
    bm["date"] = pd.to_datetime(bm["date"])
    bm = bm.sort_values("date").drop_duplicates("date").set_index("date")

    df = df.join(bm[["up_cnt", "dn_cnt", "up_amt_yi", "dn_amt_yi", "B", "M", "RANA", "EMA19", "EMA39", "MO", "M_minus_B"]])

    # Smoothed / derived versions (strictly causal)
    df["B_5d"] = df["B"].rolling(5).mean()
    df["M_5d"] = df["M"].rolling(5).mean()
    df["MB_5d"] = df["M_minus_B"].rolling(5).mean()
    df["MO_chg3"] = df["MO"] - df["MO"].shift(3)
    # Money-weighted McClellan Oscillator on (2M - 1)
    m_net = 2.0 * df["M"] - 1.0
    df["MO_M"] = m_net.ewm(span=19, adjust=False).mean() - m_net.ewm(span=39, adjust=False).mean()

    # Standardized [0, 100] trailing scores for clean integration tests
    df["s_B"] = rolling_z_cdf(df["B"])
    df["s_B5"] = rolling_z_cdf(df["B_5d"])
    df["s_M"] = rolling_z_cdf(df["M"])
    df["s_M5"] = rolling_z_cdf(df["M_5d"])
    df["s_MB"] = rolling_z_cdf(df["M_minus_B"])
    df["s_MO"] = rolling_z_cdf(df["MO"])

    ev = df.loc[START_DATE:].copy()
    print(f"Evaluation window: {ev.index.min().date()} -> {ev.index.max().date()} ({len(ev)} trading days)")

    print("\n=== 1. Descriptive Stats & Latest Values (2026-11-20) ===")
    stat_cols = ["B", "M", "M_minus_B", "MO", "MO_M", "stsi", "health_5d"]
    desc = pd.DataFrame({
        c: [
            ev[c].mean(), ev[c].std(), ev[c].min(),
            ev[c].quantile(0.10), ev[c].quantile(0.25), ev[c].median(),
            ev[c].quantile(0.75), ev[c].quantile(0.90), ev[c].max(), ev[c].iloc[-1]
        ] for c in stat_cols
    }, index=["mean", "std", "min", "p10", "p25", "p50", "p75", "p90", "max", "latest_1120"])
    print(desc.round(4).to_string())

    print("\n=== 2. Correlation Matrix (B, M, M-B, MO vs STSI & Health_5d) ===")
    corr_cols = ["B", "M", "M_minus_B", "B_5d", "M_5d", "MO", "MO_M", "stsi", "health_5d", "s_zt_dt_ratio", "s_anti_nuke"]
    print(ev[corr_cols].corr().round(3).to_string())

    print("\n=== 3. Spearman Rank IC vs Forward Returns (strictly T+1 close -> T+1+H, 2021-07 to 2026-11) ===")
    cand_cols = ["B", "M", "M_minus_B", "B_5d", "M_5d", "MB_5d", "MO", "MO_M", "MO_chg3", "stsi", "health_5d"]
    for idx_label, col_px in [("中证1000", "zz1000_close"), ("沪深300", "hs300_close"), ("创业板指", "chinext_close")]:
        c1 = df[col_px].shift(-1)
        fwds = {
            "1d": df[col_px].shift(-2) / c1 - 1.0,
            "3d": df[col_px].shift(-4) / c1 - 1.0,
            "5d": df[col_px].shift(-6) / c1 - 1.0,
            "10d": df[col_px].shift(-11) / c1 - 1.0,
            "20d": df[col_px].shift(-21) / c1 - 1.0,
            "60d": df[col_px].shift(-61) / c1 - 1.0,
        }
        if idx_label == "中证1000":
            fwds["relay_next"] = df["relay_next_ret"].shift(-1)
        rows = []
        for fc in cand_cols:
            r = {"factor": fc}
            for hn, fs in fwds.items():
                al = pd.concat([ev[fc], fs], axis=1, sort=False).dropna()
                r[hn] = al.iloc[:, 0].corr(al.iloc[:, 1], method="spearman")
            rows.append(r)
        print(f"\n--- Target: {idx_label} ---")
        print(pd.DataFrame(rows).set_index("factor").round(4).to_string())

    print("\n=== 4. Quintile Forward Returns on ZZ1000 (B, M, M-B, MO) ===")
    c1 = df["zz1000_close"].shift(-1)
    f_5d = df["zz1000_close"].shift(-6) / c1 - 1.0
    f_10d = df["zz1000_close"].shift(-11) / c1 - 1.0
    f_20d = df["zz1000_close"].shift(-21) / c1 - 1.0
    f_60d = df["zz1000_close"].shift(-61) / c1 - 1.0
    f_rel = df["relay_next_ret"].shift(-1)

    for fac in ["B", "M", "M_minus_B", "MO"]:
        q = pd.DataFrame({
            "f": ev[fac], "5d": f_5d, "10d": f_10d, "20d": f_20d, "60d": f_60d, "relay": f_rel
        }).loc[START_DATE:].dropna(subset=["f"])
        q["bin"] = pd.qcut(q["f"], 5, labels=["Q1(最低)", "Q2", "Q3", "Q4", "Q5(最高)"])
        grp = q.groupby("bin", observed=False)[["5d", "10d", "20d", "60d", "relay"]].mean()
        b_range = q.groupby("bin", observed=False)["f"].agg(["min", "max"])
        out_q = pd.concat([b_range, grp], axis=1)
        print(f"\n--- Quintiles of {fac} (ZZ1000) ---")
        print(out_q.to_string(float_format=lambda x: f"{x:+.2%}" if abs(x) < 0.35 else f"{x:.3f}"))

    print("\n=== 5. Newey-West HAC Marginal Predictive Power (Controlling for STSI & Health_5d) ===")
    for h_name, f_y, lag in [("ZZ1000 10d", f_10d, 10), ("ZZ1000 20d", f_20d, 20), ("ZZ1000 60d", f_60d, 40), ("Next Relay", f_rel, 5)]:
        X_std = ev[["stsi", "health_5d", "B", "M_minus_B", "MO"]].copy()
        X_std = (X_std - X_std.mean()) / X_std.std()
        res = ols_newey_west_tstat(f_y, X_std, max_lag=lag)
        print(f"\n--- Marginal Regression: {h_name} (standardized X, Newey-West lag={lag}) ---")
        print(res.round(4).to_string())

    print("\n=== 6. Standalone Backtests for B, M, MO on ZZ1000 & HS300 (T+1 OPEN, 0.10% cost, 0% cash) ===")
    def hyst(s: pd.Series, lo: float, hi: float, contrarian: bool = True) -> pd.Series:
        vals = s.to_numpy(dtype=float)
        pos = np.empty_like(vals)
        cur = 0.5
        for i, v in enumerate(vals):
            if not np.isnan(v):
                if contrarian:
                    if v <= lo: cur = 1.0
                    elif v >= hi: cur = 0.0
                else:
                    if v >= hi: cur = 1.0
                    elif v <= lo: cur = 0.0
            pos[i] = cur
        return pd.Series(pos, index=s.index)

    standalone = {
        "B_顺势(B_5d>0.55买/<0.45卖)": hyst(df["B_5d"], 0.45, 0.55, contrarian=False),
        "B_逆向(B_5d<0.35买/>0.65卖)": hyst(df["B_5d"], 0.35, 0.65, contrarian=True),
        "M_顺势(M_5d>0.55买/<0.45卖)": hyst(df["M_5d"], 0.45, 0.55, contrarian=False),
        "M_逆向(M_5d<0.35买/>0.65卖)": hyst(df["M_5d"], 0.35, 0.65, contrarian=True),
        "M-B_资金聚焦(MB_5d>0.02买/<-0.02卖)": hyst(df["MB_5d"], -0.02, 0.02, contrarian=False),
        "MO_经典零轴顺势(MO>0买/<0卖)": (df["MO"] > 0).astype(float),
        "MO_缓冲零轴顺势(>+0.02买/<-0.02卖)": hyst(df["MO"], -0.02, 0.02, contrarian=False),
        "MO_逆向超卖超买(<-0.06买/>+0.06卖)": hyst(df["MO"], -0.06, 0.06, contrarian=True),
        "MO_动量拐点(MO>0或MO<-0.06抄底)": ((df["MO"] > 0) | (df["MO"] < -0.06)).astype(float),
    }

    for idx_label, ocol, ccol in [("中证1000", "zz1000_open", "zz1000_close"), ("沪深300", "hs300_open", "hs300_close")]:
        bh_r = df[ccol].pct_change().loc[START_DATE:].fillna(0.0)
        bm_i = compute_metrics(bh_r, bh_r, pd.Series(1.0, index=bh_r.index))
        print(f"\n--- Standalone Timing: {idx_label} (B&H CAGR={bm_i['cagr']:+.2%}, MDD={bm_i['max_dd']:.1%}) ---")
        for name, w in standalone.items():
            sr_o, br_o, hp_o = exec_t1_open(df[ocol], df[ccol], w, COST_ETF, 0.0)
            sr_c, _, hp_c = exec_t1_close(df[ccol], w, COST_ETF, 0.0)
            mo = compute_metrics(sr_o, br_o, hp_o)
            mc = compute_metrics(sr_c, br_o, hp_c)
            print(f"  {name:<34s} | Open: CAGR={mo['cagr']:+6.2%} Sh={mo['sharpe']:5.2f} MDD={mo['max_dd']:6.1%} Turn={mo['ann_turnover']:4.1f}x | Close: {mc['cagr']:+6.2%}")

    print("\n=== 7. Incremental Ablation: Adding B / M / MO to Existing STSI Cycle Strategy (ZZ1000, T+1 OPEN, 0% cash) ===")
    w_base = strat_cycle_main(df, 60.0, 40.0, 20.0, 5)

    # Variant A: Replace or augment Health_5d with M_5d (上涨成交额占比)
    h_plus_m = ((df["s_seal_rate"] + df["s_anti_nuke"] + df["s_M"]) / 3.0).rolling(5, min_periods=1).mean()
    # Variant B: Augment Health_5d with B (上涨家数占比)
    h_plus_b = ((df["s_seal_rate"] + df["s_anti_nuke"] + df["s_B"]) / 3.0).rolling(5, min_periods=1).mean()
    # Variant C: Augment Health_5d with MO (宽度动量)
    h_plus_mo = ((df["s_seal_rate"] + df["s_anti_nuke"] + df["s_MO"]) / 3.0).rolling(5, min_periods=1).mean()
    # Variant D: MO regime filter on base strategy (when MO < -0.05 deep oversold -> also trigger ice; or when MO > 0 allow holding)
    def run_custom(h_ser: pd.Series, stsi_ser: pd.Series, enter_h=60.0, exit_h=40.0, ice_th=20.0, extra_buy=None, extra_veto=None) -> pd.Series:
        pos = np.empty(len(df))
        cur = 0.5
        for i in range(len(df)):
            h, s = h_ser.iloc[i], stsi_ser.iloc[i]
            if np.isnan(h) or np.isnan(s):
                pos[i] = cur; continue
            if s <= ice_th or (extra_buy is not None and bool(extra_buy.iloc[i])):
                cur = 1.0
            elif h >= enter_h:
                cur = 1.0
            elif h <= exit_h:
                cur = 0.0
            if extra_veto is not None and bool(extra_veto.iloc[i]):
                cur = 0.0
            pos[i] = cur
        return pd.Series(pos, index=df.index)

    ablations = {
        "原版基线: STSI(60/40/冰点20)": w_base,
        "+B 融入健康度 (封板+低大面+B)": run_custom(h_plus_b, df["stsi"]),
        "+M 融入健康度 (封板+低大面+M)": run_custom(h_plus_m, df["stsi"]),
        "+MO 融入健康度 (封板+低大面+MO)": run_custom(h_plus_mo, df["stsi"]),
        "+B 极端普跌冰点补充 (B≤15%抄底)": run_custom(df["health_5d"], df["stsi"], extra_buy=(df["B"] <= 0.15)),
        "+MO 超卖冰点补充 (MO≤-0.06抄底)": run_custom(df["health_5d"], df["stsi"], extra_buy=(df["MO"] <= -0.06)),
        "+MO 趋势过滤 (仅MO>0或冰点做多)": run_custom(df["health_5d"], df["stsi"], extra_veto=((df["MO"] < 0) & (df["stsi"] > 20))),
        "+M-B 资金聚焦过滤 (退潮仅在M<B退出)": None,
    }
    # Custom M-B exit filter: don't panic-exit if M_5d > B_5d + 0.02 (money concentrating in leaders)
    pos_mb = np.empty(len(df)); cur = 0.5
    for i in range(len(df)):
        h, s, mb = df["health_5d"].iloc[i], df["stsi"].iloc[i], df["MB_5d"].iloc[i]
        if np.isnan(h) or np.isnan(s): pos_mb[i] = cur; continue
        if s <= 20.0 or h >= 60.0: cur = 1.0
        elif h <= 40.0 and (np.isnan(mb) or mb <= 0.02): cur = 0.0
        pos_mb[i] = cur
    ablations["+M-B 资金聚焦过滤 (退潮仅在M<B退出)"] = pd.Series(pos_mb, index=df.index)

    for name, w in ablations.items():
        sr_o, br_o, hp_o = exec_t1_open(df["zz1000_open"], df["zz1000_close"], w, COST_ETF, 0.0)
        sr_c, _, hp_c = exec_t1_close(df["zz1000_close"], w, COST_ETF, 0.0)
        sr_hs, br_hs, hp_hs = exec_t1_open(df["hs300_open"], df["hs300_close"], w, COST_ETF, 0.0)
        mo = compute_metrics(sr_o, br_o, hp_o)
        mc = compute_metrics(sr_c, br_o, hp_c)
        mhs = compute_metrics(sr_hs, br_hs, hp_hs)
        print(
            f"  {name:<34s} | ZZ1000 Open:{mo['cagr']:+6.2%}(Sh{mo['sharpe']:4.2f},MDD{mo['max_dd']:6.1%},T{mo['ann_turnover']:4.1f}x) "
            f"| Close:{mc['cagr']:+6.2%} | HS300:{mhs['cagr']:+6.2%}"
        )

    print("\n=== 8. Relay Ablation: Does B / M / MO Improve Stock Relay Filtering? (0.20% RT cost, 0% cash) ===")
    base_mask = (df["stsi"] <= 25) & (df["big_loss_rate"] < 0.50)
    relay_tests = {
        "原版: STSI≤25 & 大面<50%": base_mask,
        "纯B冰点: B≤20%": df["B"] <= 0.20,
        "纯M冰点: M≤25%": df["M"] <= 0.25,
        "纯MO超卖: MO≤-0.06": df["MO"] <= -0.06,
        "原版 + B≤40% (非普涨次日)": base_mask & (df["B"] <= 0.40),
        "原版 + MO<0 (宽度偏弱期反核)": base_mask & (df["MO"] < 0),
        "原版 + M>B (资金向涨股集中)": base_mask & (df["M_minus_B"] > 0),
        "原版 + M<B (缩量普跌情绪冰点)": base_mask & (df["M_minus_B"] < 0),
    }
    for rname, msk in relay_tests.items():
        _, st_r = backtest_relay_series(df, msk, COST_STOCK_RT, 0.0)
        print(
            f"  {rname:<32s} | n={st_r['n_trades']:3d} | gross={st_r['avg_ret_gross']:+5.2%} net={st_r['avg_ret_net']:+5.2%} "
            f"win={st_r['win_rate_net']*100:4.1f}% | tot={st_r['total_ret']:+6.1%} MDD={st_r['max_dd']:6.1%}"
        )


if __name__ == "__main__":
    main()
