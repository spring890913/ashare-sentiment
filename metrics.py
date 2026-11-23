"""Performance evaluation and risk metrics for daily A-share timing strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 243.0  # A-share average trading days per calendar year


def compute_metrics(
    strat_ret: pd.Series,
    bench_ret: pd.Series,
    position: pd.Series,
    rf_annual: float = 0.02,
) -> dict[str, float]:
    """Compute comprehensive annualized risk/return & trading metrics."""
    df = pd.concat(
        {"s": strat_ret, "b": bench_ret, "pos": position}, axis=1
    ).dropna()
    s = df["s"]
    b = df["b"]
    pos = df["pos"]

    n = len(s)
    years = n / TRADING_DAYS_PER_YEAR

    nav_s = (1.0 + s).cumprod()
    nav_b = (1.0 + b).cumprod()

    total_ret = float(nav_s.iloc[-1] - 1.0)
    bench_total = float(nav_b.iloc[-1] - 1.0)

    cagr = float(nav_s.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0
    bench_cagr = float(nav_b.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0

    ann_vol = float(s.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    bench_vol = float(b.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))

    rf_daily = (1.0 + rf_annual) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    excess_rf = s - rf_daily
    sharpe = (
        float(excess_rf.mean() / s.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        if s.std(ddof=1) > 0
        else 0.0
    )
    bench_sharpe = (
        float((b - rf_daily).mean() / b.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        if b.std(ddof=1) > 0
        else 0.0
    )

    downside = np.minimum(excess_rf, 0.0)
    downside_std = float(np.sqrt(np.mean(downside**2)) * np.sqrt(TRADING_DAYS_PER_YEAR))
    sortino = float((cagr - rf_annual) / downside_std) if downside_std > 0 else 0.0

    dd_s = nav_s / nav_s.cummax() - 1.0
    dd_b = nav_b / nav_b.cummax() - 1.0
    max_dd = float(dd_s.min())
    bench_max_dd = float(dd_b.min())

    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0
    bench_calmar = float(bench_cagr / abs(bench_max_dd)) if bench_max_dd < 0 else 0.0

    # Excess vs benchmark
    active = s - b
    te = float(active.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    ir = float((cagr - bench_cagr) / te) if te > 0 else 0.0

    # Trading / position statistics
    pos_changes = pos.diff().abs().fillna(0.0)
    ann_turnover = float(pos_changes.sum() / years) if years > 0 else 0.0
    avg_exposure = float(pos.mean())
    trade_days = int((pos_changes > 1e-6).sum())

    # Monthly win rate vs benchmark & positive month rate
    monthly_s = (1.0 + s).resample("ME").prod() - 1.0
    monthly_b = (1.0 + b).resample("ME").prod() - 1.0
    monthly_win_vs_bench = float((monthly_s > monthly_b).mean())
    monthly_pos_rate = float((monthly_s > 0).mean())

    return {
        "years": round(years, 2),
        "total_ret": total_ret,
        "bench_total_ret": bench_total,
        "cagr": cagr,
        "bench_cagr": bench_cagr,
        "excess_cagr": cagr - bench_cagr,
        "ann_vol": ann_vol,
        "bench_vol": bench_vol,
        "sharpe": sharpe,
        "bench_sharpe": bench_sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "bench_max_dd": bench_max_dd,
        "calmar": calmar,
        "bench_calmar": bench_calmar,
        "info_ratio": ir,
        "avg_exposure": avg_exposure,
        "ann_turnover": ann_turnover,
        "rebalance_days": trade_days,
        "monthly_pos_rate": monthly_pos_rate,
        "monthly_beat_bench": monthly_win_vs_bench,
    }


def yearly_breakdown(
    strat_ret: pd.Series, bench_ret: pd.Series, position: pd.Series
) -> pd.DataFrame:
    """Compute calendar-year return, max drawdown, exposure, and excess return."""
    df = pd.concat({"s": strat_ret, "b": bench_ret, "pos": position}, axis=1).dropna()
    rows = []
    for year, g in df.groupby(df.index.year):
        nav_s = (1.0 + g["s"]).cumprod()
        nav_b = (1.0 + g["b"]).cumprod()
        r_s = float(nav_s.iloc[-1] - 1.0)
        r_b = float(nav_b.iloc[-1] - 1.0)
        mdd_s = float((nav_s / nav_s.cummax() - 1.0).min())
        mdd_b = float((nav_b / nav_b.cummax() - 1.0).min())
        rows.append(
            {
                "year": int(year),
                "strat_ret": r_s,
                "bench_ret": r_b,
                "excess_ret": r_s - r_b,
                "strat_mdd": mdd_s,
                "bench_mdd": mdd_b,
                "avg_pos": float(g["pos"].mean()),
            }
        )
    return pd.DataFrame(rows)
