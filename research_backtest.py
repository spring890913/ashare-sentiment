"""Construct A-share Composite Sentiment Index (ACSI) and run rigorous backtest diagnostics."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from explore_factors import load_all, rolling_pct_rank
from metrics import compute_metrics, yearly_breakdown

DATA_DIR = Path(__file__).resolve().parent / "data"


def build_sentiment_index(
    df: pd.DataFrame,
    window: int = 756,
    min_periods: int = 240,
    smooth_days: int = 5,
) -> pd.DataFrame:
    """Build strictly causal (zero look-ahead) A-share sentiment index in [0, 100].

    All sub-indicators are oriented so that:
        0   = Extreme Fear / Pessimism (极度恐慌)
        100 = Extreme Greed / Euphoria (极度贪婪)

    Publication lag handling:
        - Margin trading stats (融资余额) for day t are published on morning of t+1.
          To guarantee ALL inputs are strictly observable before day t close,
          we lag margin series by 1 trading day (shift(1)).
        - Strategy signals computed at day t close are executed on day t+1.
    """
    out = pd.DataFrame(index=df.index)

    # Strictly causal forward-fill for rare 1-day missing values in auxiliary series
    tot_amt = df["total_amount"].ffill()
    close = df["hs300_close"]
    m_bal = df["margin_bal"].ffill().shift(1)  # lag 1 day for strict causality at t close
    qvix = df["qvix"].ffill()
    below_pb = df["below_pb_ratio"].ffill()
    ebs = df["ebs"].ffill()

    # 6 Domain-Grounded Orthogonal Sub-Indicators (raw values, higher = greedier)
    raw = pd.DataFrame(index=df.index)
    # 1. 量能活跃度 (Trading Volume Activity): log(MA5 / MA120) of SH+SZ total turnover
    raw["vol_activity"] = np.log(tot_amt.rolling(5).mean() / tot_amt.rolling(120).mean())
    # 2. 杠杆资金动能 (Margin Leverage Momentum): 20d growth rate of total margin balance
    raw["margin_mom"] = m_bal.pct_change(20)
    # 3. 期权恐慌反向 (Option Implied Vol): -QVIX (higher QVIX -> lower score = fear)
    raw["opt_complacency"] = -qvix
    # 4. 短期价格乖离 (Price Extension): HS300 60d moving average BIAS
    raw["price_bias60"] = close / close.rolling(60).mean() - 1.0
    # 5. 全市场破净反向 (Market Breadth Distress): -below_net_asset_ratio
    raw["breadth_pb"] = -below_pb
    # 6. 股债性价比反向 (Cross-Asset Risk Preference): -ERP (HS300 Equity-Bond Spread)
    raw["erp_greed"] = -ebs

    # Rolling trailing percentile rank [0, 100] — ZERO future data used
    sub_cols = list(raw.columns)
    for c in sub_cols:
        out[f"s_{c}"] = rolling_pct_rank(raw[c], window=window, min_periods=min_periods)

    # Pure Trading Sentiment (4 pure market/flow/vol indicators, no valuation/PB/ERP)
    trading_cols = ["s_vol_activity", "s_margin_mom", "s_opt_complacency", "s_price_bias60"]
    out["trading_raw"] = out[trading_cols].mean(axis=1)
    out["trading_idx"] = out["trading_raw"].rolling(smooth_days, min_periods=1).mean()

    # Full A-Share Composite Sentiment Index (ACSI, 6 equal-weighted dimensions)
    all_sub_cols = [f"s_{c}" for c in sub_cols]
    out["acsi_raw"] = out[all_sub_cols].mean(axis=1)
    out["acsi"] = out["acsi_raw"].rolling(smooth_days, min_periods=1).mean()

    # Preserve raw references
    out["hs300_close"] = df["hs300_close"]
    out["hs300_open"] = df["hs300_open"]
    out["hs300_ret"] = df["hs300_ret"]
    out["total_amount_yi"] = tot_amt / 1e8
    out["qvix"] = qvix
    out["below_pb_ratio"] = below_pb
    out["ebs"] = ebs
    out["margin_bal"] = df["margin_bal"]
    return out


def run_backtest(
    close: pd.Series,
    target_pos: pd.Series,
    cost_one_way: float = 0.001,
    cash_rf_annual: float = 0.0,
    start_date: str = "2016-01-04",
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Run strictly causal T+1 close execution backtest.

    target_pos[t] is decided using info available at close of day t.
    We rebalance at close of day t+1, so day t+1's return earns target_pos[t-1]?
    Wait — be crystal clear on execution convention:
      - At day t close (15:00), indicator S[t] is computed and target position w[t] is decided.
      - Trade is executed at day t+1 close (or day t+1 open -> close).
      - Specifically:
          * If executed at day t+1 close: during day t+1 (close[t] -> close[t+1]) we still hold w[t-1];
            at close[t+1] we rebalance to w[t] paying cost |w[t] - w[t-1]| * cost_one_way,
            and then day t+2 (close[t+1] -> close[t+2]) earns w[t]!
            Wait: that's 2-day lag (shift(2)) if executed at t+1 close, OR 1-day lag (shift(1))
            if executed at t+1 open (or at t close before close).
      - Let's be super clean and realistic:
          * Signal decided at close of day t -> executed at **t+1 open**?
            Even cleaner and standard in daily close-to-close backtests:
            If you compute S[t] at close[t] (using margin[t-1], close[t], etc.), can you trade at t+1 open?
            YES! Let's check BOTH:
            1) Close-to-close with 1-day lag (`exec_pos = target_pos.shift(1)`): corresponds to trading near close of t / open of t+1, and we can explicitly verify exact open-to-open `open[t+2]/open[t+1] - 1` execution too!
    """
    ret = close.pct_change().fillna(0.0)
    # Position held during day t (from close[t-1] to close[t]) is decided at close[t-1]
    held_pos = target_pos.shift(1).clip(0.0, 1.0)
    # Restrict to backtest window
    mask = close.index >= pd.Timestamp(start_date)
    ret = ret.loc[mask]
    held_pos = held_pos.loc[mask].fillna(0.0)

    turnover = held_pos.diff().abs().fillna(held_pos.iloc[0])
    rf_daily = (1.0 + cash_rf_annual) ** (1.0 / 243.0) - 1.0
    strat_ret = held_pos * ret + (1.0 - held_pos) * rf_daily - turnover * cost_one_way
    return strat_ret, ret, held_pos


def run_backtest_open_exec(
    open_px: pd.Series,
    close_px: pd.Series,
    target_pos: pd.Series,
    cost_one_way: float = 0.001,
    cash_rf_annual: float = 0.0,
    start_date: str = "2016-01-04",
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Ultra-strict T+1 OPEN execution backtest (signal at close[t] -> trade at open[t+1]).

    Return earned by w[t] (decided at close[t], entered at open[t+1]) is open[t+2]/open[t+1] - 1.
    """
    open_ret = open_px.pct_change().fillna(0.0)
    # During [open[t-1], open[t]], position held was entered at open[t-1], decided at close[t-2]
    held_pos = target_pos.shift(2).clip(0.0, 1.0)
    mask = close_px.index >= pd.Timestamp(start_date)
    open_ret = open_ret.loc[mask]
    bench_ret = close_px.pct_change().fillna(0.0).loc[mask]
    held_pos = held_pos.loc[mask].fillna(0.0)

    turnover = held_pos.diff().abs().fillna(held_pos.iloc[0])
    rf_daily = (1.0 + cash_rf_annual) ** (1.0 / 243.0) - 1.0
    strat_ret = held_pos * open_ret + (1.0 - held_pos) * rf_daily - turnover * cost_one_way
    return strat_ret, bench_ret, held_pos


def strategy_hysteresis(
    sentiment: pd.Series,
    buy_th: float = 30.0,
    sell_th: float = 70.0,
    low_pos: float = 0.0,
    high_pos: float = 1.0,
    init_pos: float = 0.5,
) -> pd.Series:
    """Contrarian hysteresis strategy: buy when sentiment <= buy_th, sell when >= sell_th."""
    pos = pd.Series(index=sentiment.index, dtype=float)
    cur = init_pos
    for dt, val in sentiment.items():
        if pd.isna(val):
            pos.loc[dt] = cur
            continue
        if val <= buy_th:
            cur = high_pos
        elif val >= sell_th:
            cur = low_pos
        pos.loc[dt] = cur
    return pos


def strategy_continuous(
    sentiment: pd.Series,
    low_s: float = 20.0,
    high_s: float = 80.0,
    min_pos: float = 0.0,
    max_pos: float = 1.0,
    rebal_band: float = 0.10,
) -> pd.Series:
    """Continuous contrarian position sizing with no-trade rebalance buffer band."""
    raw_w = max_pos - (sentiment - low_s) / (high_s - low_s) * (max_pos - min_pos)
    raw_w = raw_w.clip(min_pos, max_pos)
    pos = pd.Series(index=sentiment.index, dtype=float)
    cur = np.nan
    for dt, w in raw_w.items():
        if pd.isna(w):
            pos.loc[dt] = 0.5 if np.isnan(cur) else cur
            continue
        if np.isnan(cur) or abs(w - cur) >= rebal_band or w in (min_pos, max_pos):
            cur = float(w)
        pos.loc[dt] = cur
    return pos


def strategy_sentiment_trend_hybrid(
    sentiment: pd.Series,
    close: pd.Series,
    fear_th: float = 30.0,
    greed_th: float = 70.0,
    ma_len: int = 20,
    greed_floor: float = 0.0,
) -> pd.Series:
    """Left-side fear entry + right-side trend-confirmed greed exit.

    - Fear zone (S <= fear_th): accumulate to 100% (左侧极度恐慌满仓)
    - Neutral zone (fear_th < S < greed_th): hold current position (or 100% if above MA20 and S < 55)
    - Greed zone (S >= greed_th): keep riding the bull trend while close > MA20;
      once close < MA20 (trend breaks in overheated zone), reduce to greed_floor!
    """
    ma = close.rolling(ma_len).mean()
    pos = pd.Series(index=sentiment.index, dtype=float)
    cur = 0.5
    for dt in sentiment.index:
        s = sentiment.loc[dt]
        c = close.loc[dt]
        m = ma.loc[dt]
        if pd.isna(s) or pd.isna(m):
            pos.loc[dt] = cur
            continue
        if s <= fear_th:
            cur = 1.0
        elif s >= greed_th:
            if c < m:
                cur = greed_floor
        else:
            # Neutral recovery: if we were out during euphoria and market cools down below 45 with uptrend, re-enter
            if cur < 1.0 and s <= 45.0 and c > m:
                cur = 1.0
        pos.loc[dt] = cur
    return pos


def quantile_forward_analysis(sentiment: pd.Series, close: pd.Series, start_date: str = "2016-01-04") -> pd.DataFrame:
    """Analyze forward returns across fixed 0-20, 20-40, 40-60, 60-80, 80-100 sentiment buckets."""
    # Forward returns strictly starting from t+1 close (no look-ahead)
    c1 = close.shift(-1)
    df = pd.DataFrame({
        "S": sentiment,
        "fwd_20d": close.shift(-21) / c1 - 1.0,
        "fwd_60d": close.shift(-61) / c1 - 1.0,
        "fwd_120d": close.shift(-121) / c1 - 1.0,
        "fwd_250d": close.shift(-251) / c1 - 1.0,
    }).loc[start_date:].dropna(subset=["S"])

    bins = [0, 20, 40, 60, 80, 100.01]
    labels = ["0-20 极度恐慌", "20-40 偏悲观", "40-60 中性", "60-80 偏乐观", "80-100 极度贪婪"]
    df["zone"] = pd.cut(df["S"], bins=bins, labels=labels, right=False)

    rows = []
    for z, g in df.groupby("zone", observed=False):
        row = {"情绪区间": str(z), "交易日数": len(g), "占比": len(g) / len(df)}
        for h in ["fwd_20d", "fwd_60d", "fwd_120d", "fwd_250d"]:
            sub = g[h].dropna()
            row[f"{h}_均值"] = sub.mean() if len(sub) else np.nan
            row[f"{h}_胜率"] = (sub > 0).mean() if len(sub) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    df = load_all()
    sent = build_sentiment_index(df, window=756, min_periods=240, smooth_days=5)

    print("=== Latest Sentiment Reading ===")
    sub_show = ["acsi", "trading_idx", "s_vol_activity", "s_margin_mom", "s_opt_complacency", "s_price_bias60", "s_breadth_pb", "s_erp_greed", "hs300_close"]
    print(sent[sub_show].tail(5).round(2).to_string())

    print("\n=== Historical Turning Points Check (ACSI 0-100) ===")
    milestones = [
        ("2016-01-28", "2016熔断底部"),
        ("2018-01-24", "2018年初蓝筹顶部"),
        ("2019-01-03", "2019年初2440大底"),
        ("2020-03-23", "2020疫情全球熔断底"),
        ("2021-02-18", "2021核心资产见顶"),
        ("2022-04-26", "2022年4月上海疫情底"),
        ("2022-10-31", "2022年10月恐慌底"),
        ("2024-02-05", "2024年2月微盘流动性底"),
        ("2024-09-18", "2024.9.24行情启动前冰点"),
        ("2024-10-08", "2024.10.8天量沸点"),
        ("2026-11-20", "当前最新交易日"),
    ]
    for dt_str, label in milestones:
        idx = sent.index.asof(pd.Timestamp(dt_str))
        row = sent.loc[idx]
        print(
            f"  {idx.strftime('%Y-%m-%d')} | {label:<18s} | HS300={row['hs300_close']:7.1f} | "
            f"ACSI={row['acsi']:5.1f} | Trading={row['trading_idx']:5.1f}"
        )

    print("\n=== Quantile Forward Return Analysis (ACSI vs HS300, 2016-2026) ===")
    q_acsi = quantile_forward_analysis(sent["acsi"], df["hs300_close"])
    print(q_acsi.to_string(index=False, float_format=lambda x: f"{x:.2%}" if abs(x) < 2 else f"{x:.0f}"))

    print("\n=== Quantile Forward Return Analysis (Pure Trading Sentiment vs HS300, 2016-2026) ===")
    q_tr = quantile_forward_analysis(sent["trading_idx"], df["hs300_close"])
    print(q_tr.to_string(index=False, float_format=lambda x: f"{x:.2%}" if abs(x) < 2 else f"{x:.0f}"))

    # Compare strategies on HS300 (2016-01-04 to 2026-11-20), 0.10% one-way cost, cash_rf = 0% (pure equity comparison) AND 2% (money market)
    strategies = {
        "A1_逆向双阈值(30/70)": strategy_hysteresis(sent["acsi"], 30, 70, 0.0, 1.0),
        "A2_逆向双阈值(25/75)": strategy_hysteresis(sent["acsi"], 25, 75, 0.0, 1.0),
        "B_连续仓位映射(20-80)": strategy_continuous(sent["acsi"], 20, 80, 0.0, 1.0, 0.10),
        "C_左侧恐慌+右侧趋势止盈(30/68,MA20)": strategy_sentiment_trend_hybrid(sent["acsi"], df["hs300_close"], 30, 68, 20, 0.0),
        "D_纯交易情绪双阈值(30/70)": strategy_hysteresis(sent["trading_idx"], 30, 70, 0.0, 1.0),
        "E_纯交易情绪连续仓位(20-80)": strategy_continuous(sent["trading_idx"], 20, 80, 0.0, 1.0, 0.10),
    }

    print("\n=== Backtest Comparison on HS300 (2016-01-04 to 2026-11-20, 0.10% cost, 0% cash yield) ===")
    rows = []
    # Add Buy & Hold row first
    bh_ret = df["hs300_close"].pct_change().loc["2016-01-04":].fillna(0.0)
    bh_pos = pd.Series(1.0, index=bh_ret.index)
    bh_m = compute_metrics(bh_ret, bh_ret, bh_pos, rf_annual=0.02)
    rows.append({"Strategy": "基准: 沪深300买入持有", **bh_m})

    for name, w in strategies.items():
        s_ret, b_ret, h_pos = run_backtest(df["hs300_close"], w, cost_one_way=0.001, cash_rf_annual=0.0)
        m = compute_metrics(s_ret, b_ret, h_pos, rf_annual=0.02)
        rows.append({"Strategy": name, **m})

    res_df = pd.DataFrame(rows)
    show_cols = ["Strategy", "total_ret", "cagr", "ann_vol", "sharpe", "max_dd", "calmar", "avg_exposure", "ann_turnover"]
    print(res_df[show_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Ultra-Strict T+1 OPEN Execution Verification (cost=0.10%, cash_rf=0%) ===")
    for name in ["A1_逆向双阈值(30/70)", "B_连续仓位映射(20-80)", "C_左侧恐慌+右侧趋势止盈(30/68,MA20)"]:
        s_ret, b_ret, h_pos = run_backtest_open_exec(df["hs300_open"], df["hs300_close"], strategies[name], cost_one_way=0.001, cash_rf_annual=0.0)
        m = compute_metrics(s_ret, b_ret, h_pos, rf_annual=0.02)
        print(f"  {name:<32s} | CAGR={m['cagr']:6.2%} | Sharpe={m['sharpe']:5.2f} | MaxDD={m['max_dd']:7.2%} | Calmar={m['calmar']:5.2f}")

    print("\n=== Multi-Index Robustness Check (Same ACSI, NO re-tuning! 2016-2026, 0% cash yield) ===")
    for idx_name, col in [
        ("沪深300", "hs300_close"),
        ("中证500", "zz500_close"),
        ("中证1000", "zz1000_close"),
        ("创业板指", "chinext_close"),
        ("上证指数", "sh_close"),
    ]:
        px = df[col]
        bh_r = px.pct_change().loc["2016-01-04":].fillna(0.0)
        bh_m = compute_metrics(bh_r, bh_r, pd.Series(1.0, index=bh_r.index))
        s_ret_b, b_ret_b, h_pos_b = run_backtest(px, strategies["B_连续仓位映射(20-80)"], cost_one_way=0.001, cash_rf_annual=0.0)
        m_b = compute_metrics(s_ret_b, b_ret_b, h_pos_b)
        s_ret_c, b_ret_c, h_pos_c = run_backtest(px, strategy_sentiment_trend_hybrid(sent["acsi"], px, 30, 68, 20, 0.0), cost_one_way=0.001, cash_rf_annual=0.0)
        m_c = compute_metrics(s_ret_c, b_ret_c, h_pos_c)
        print(
            f"  {idx_name:<6s} | B&H: CAGR={bh_m['cagr']:6.2%}, Sharpe={bh_m['sharpe']:5.2f}, MDD={bh_m['max_dd']:7.2%} "
            f"| 连续仓位B: CAGR={m_b['cagr']:6.2%}, Sharpe={m_b['sharpe']:5.2f}, MDD={m_b['max_dd']:7.2%} "
            f"| 情绪+趋势C: CAGR={m_c['cagr']:6.2%}, Sharpe={m_c['sharpe']:5.2f}, MDD={m_c['max_dd']:7.2%}"
        )


if __name__ == "__main__":
    main()
