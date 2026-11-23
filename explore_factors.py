"""Exploratory analysis of A-share sentiment sub-indicators and forward return diagnostics."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_all() -> pd.DataFrame:
    def read_idx(name: str) -> pd.DataFrame:
        p = DATA_DIR / f"idx_{name}.csv"
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_csv(p)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").drop_duplicates("date").set_index("date")
        return df

    hs300 = read_idx("hs300")
    sh = read_idx("sh_comp")
    sz = read_idx("sz_comp")
    sz_cz = read_idx("sz_成指")
    sse50 = read_idx("sse50")
    zz500 = read_idx("zz500")
    zz1000 = read_idx("zz1000")
    chinext = read_idx("chinext")
    zz_all = read_idx("zz_all")
    zz_div = read_idx("zz_div")

    df = pd.DataFrame(index=hs300.index)
    df["hs300_open"] = hs300["open"]
    df["hs300_close"] = hs300["close"]
    df["hs300_amount"] = hs300["amount"]
    df["hs300_ret"] = hs300["close"].pct_change()

    df["sh_close"] = sh["close"]
    df["sh_amount"] = sh["amount"]
    df["sz_amount"] = sz["amount"] if not sz.empty else sz_cz["amount"]
    df["total_amount"] = df["sh_amount"] + df["sz_amount"]  # 元

    for name, idx_df in [
        ("sse50", sse50),
        ("zz500", zz500),
        ("zz1000", zz1000),
        ("chinext", chinext),
        ("zz_all", zz_all),
        ("zz_div", zz_div),
    ]:
        if not idx_df.empty:
            df[f"{name}_close"] = idx_df["close"]
            df[f"{name}_open"] = idx_df["open"]
            df[f"{name}_amount"] = idx_df["amount"]

    # Margin info (published T+1 morning; align carefully)
    margin = pd.read_csv(DATA_DIR / "margin_info.csv")
    margin["date"] = pd.to_datetime(margin["日期"])
    margin = margin.sort_values("date").drop_duplicates("date").set_index("date")
    for col, alias in [
        ("融资余额", "margin_bal"),          # 亿元
        ("融券余额", "short_bal"),           # 亿元
        ("融资买入额", "margin_buy"),        # 亿元
        ("参与交易的投资者数量", "margin_traders"),
        ("平均维持担保比例", "margin_maint_ratio"),
    ]:
        if col in margin.columns:
            df[alias] = pd.to_numeric(margin[col], errors="coerce")

    # QVIX (50ETF implied volatility)
    qvix = pd.read_csv(DATA_DIR / "qvix_50etf.csv")
    qvix["date"] = pd.to_datetime(qvix["date"])
    qvix = qvix.sort_values("date").drop_duplicates("date").set_index("date")
    df["qvix"] = pd.to_numeric(qvix["close"], errors="coerce")

    # Below net asset ratio (破净率)
    pb = pd.read_csv(DATA_DIR / "below_net_asset.csv")
    pb["date"] = pd.to_datetime(pb["date"])
    pb = pb.sort_values("date").drop_duplicates("date").set_index("date")
    df["below_pb_ratio"] = pd.to_numeric(pb["below_net_asset_ratio"], errors="coerce")

    # Equity-Bond Spread (沪深300股债利差)
    ebs = pd.read_csv(DATA_DIR / "ebs_hs300.csv")
    ebs["date"] = pd.to_datetime(ebs["日期"])
    ebs = ebs.sort_values("date").drop_duplicates("date").set_index("date")
    df["ebs"] = pd.to_numeric(ebs["股债利差"], errors="coerce")

    return df


def rolling_pct_rank(s: pd.Series, window: int = 756, min_periods: int = 240) -> pd.Series:
    """Strictly trailing rolling percentile rank in [0, 100] (no look-ahead)."""
    return s.rolling(window=window, min_periods=min_periods).apply(
        lambda x: float(np.sum(x <= x[-1]) - 1) / max(len(x) - 1, 1) * 100.0,
        raw=True,
    )


def main() -> None:
    df = load_all()
    print("Combined shape:", df.shape, "date range:", df.index.min().date(), "->", df.index.max().date())
    print("\nRecent 5 rows sanity check (units):")
    cols_check = [
        "hs300_close", "sh_amount", "sz_amount", "total_amount",
        "margin_bal", "margin_buy", "margin_maint_ratio", "qvix", "below_pb_ratio", "ebs"
    ]
    print(df[cols_check].tail(5).to_string())

    # Check margin_buy / total_amount ratio (convert margin_buy from 亿元 to 元)
    df["margin_buy_ratio"] = (df["margin_buy"] * 1e8) / df["total_amount"]
    print("\nmargin_buy_ratio quantiles (2015+):")
    print(df.loc["2015-01-01":, "margin_buy_ratio"].quantile([0.01, 0.1, 0.5, 0.9, 0.99]))

    # Forward-fill occasional single-day missing values in auxiliary series (only past -> present)
    aux_cols = ["margin_bal", "margin_buy", "margin_maint_ratio", "qvix", "below_pb_ratio", "ebs", "margin_buy_ratio"]
    print("\nMissing counts since 2015-03-01:")
    print(df.loc["2015-03-01":, aux_cols].isna().sum())

    # Strictly lag margin by 1 day so at close of day t, ALL features are 100% known before close of day t
    # (and execution occurs on day t+1!)
    m_buy_ratio = df["margin_buy_ratio"].ffill().shift(1)
    m_bal = df["margin_bal"].ffill().shift(1)
    m_maint = df["margin_maint_ratio"].ffill().shift(1)
    m_traders = df["margin_traders"].ffill().shift(1)
    qvix = df["qvix"].ffill()
    below_pb = df["below_pb_ratio"].ffill()
    ebs = df["ebs"].ffill()
    tot_amt = df["total_amount"]
    close = df["hs300_close"]

    # Construct candidate raw sentiment factors (oriented so HIGHER = MORE GREEDY / OPTIMISTIC)
    factors = pd.DataFrame(index=df.index)
    # 1. 量能活跃度: 5日均成交额 / 120日均成交额 (对数)
    factors["vol_activity"] = np.log(tot_amt.rolling(5).mean() / tot_amt.rolling(120).mean())
    # 2. 杠杆进攻意愿: 融资买入占两市成交额比例 (5日均值)
    factors["margin_buy_share"] = m_buy_ratio.rolling(5).mean()
    # 3. 杠杆扩张速度: 融资余额 20日增速
    factors["margin_bal_chg20"] = m_bal.pct_change(20)
    # 4. 担保比例水位 (高担保比例=盈利充裕乐观，低担保比例=逼近平仓恐慌)
    factors["margin_maint"] = m_maint
    # 4b. 两融交易者活跃度: 5日均值 / 120日均值
    factors["margin_trader_act"] = np.log(m_traders.rolling(5).mean() / m_traders.rolling(120).mean())
    # 5. 价格趋势乖离度: 沪深300 相对 60日均线乖离率
    factors["bias60"] = close / close.rolling(60).mean() - 1.0
    # 6. 价格动量: 沪深300 120日收益率
    factors["mom120"] = close.pct_change(120)
    # 7. 风格风险偏好: 创业板指 / 上证50 20日相对涨幅 (或 中证500/上证50)
    if "chinext_close" in df.columns and "sse50_close" in df.columns:
        factors["risk_appetite_cyb50"] = (df["chinext_close"] / df["sse50_close"]).pct_change(20)
    if "zz1000_close" in df.columns and "sse50_close" in df.columns:
        factors["risk_appetite_1000_50"] = (df["zz1000_close"] / df["sse50_close"]).pct_change(20)
    if "chinext_close" in df.columns and "zz_div_close" in df.columns:
        factors["risk_appetite_cyb_div"] = (df["chinext_close"] / df["zz_div_close"]).pct_change(20)
    if "zz1000_close" in df.columns and "zz_div_close" in df.columns:
        factors["risk_appetite_1000_div"] = (df["zz1000_close"] / df["zz_div_close"]).pct_change(20)
    # 8. 破净率 (反向: 破净越多越悲观 -> 取负)
    factors["neg_below_pb"] = -below_pb
    # 9. 股债利差 ERP (反向: ERP越高股票越便宜越悲观 -> 取负)
    factors["neg_ebs"] = -ebs
    # 10. 50ETF波指 QVIX (测试原始反向 vs 定向波指)
    factors["neg_qvix"] = -qvix
    # 定向波动率情绪: 下跌时高波动=恐慌(极低分)，上涨时高波动=狂热(极高分)
    ret20_sign = np.tanh(close.pct_change(20) / 0.05)  # smooth sign in [-1, 1]
    factors["signed_qvix"] = qvix * ret20_sign
    # 波动率相对20日已实现波动率或简单20日波动率
    rv20 = df["hs300_ret"].rolling(20).std() * np.sqrt(252)
    factors["neg_rv20"] = -rv20

    # Convert each factor to 3-year (756d) trailing rolling percentile [0, 100]
    pct_factors = pd.DataFrame({c: rolling_pct_rank(factors[c], window=756, min_periods=240) for c in factors.columns})

    # Forward returns of HS300 (execution starting t+1 close -> t+1+H close, zero look-ahead)
    # Actually forward return from t+1 close to t+1+H close:
    c_next = close.shift(-1)
    fwd_rets = {
        "fwd_5d": close.shift(-6) / c_next - 1.0,
        "fwd_20d": close.shift(-21) / c_next - 1.0,
        "fwd_60d": close.shift(-61) / c_next - 1.0,
        "fwd_120d": close.shift(-121) / c_next - 1.0,
        "fwd_250d": close.shift(-251) / c_next - 1.0,
    }

    eval_df = pct_factors.loc["2016-01-01":].copy()
    print("\n=== Factor Spearman Rank IC vs HS300 Forward Returns (2016-01 to 2026-11) ===")
    print("Note: NEGATIVE IC means CONTRARIAN indicator (low sentiment -> high future return)")
    ic_rows = []
    for col in pct_factors.columns:
        row = {"factor": col}
        for h_name, f_ret in fwd_rets.items():
            aligned = pd.concat([eval_df[col], f_ret], axis=1).dropna()
            row[h_name] = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman")
        ic_rows.append(row)
    ic_table = pd.DataFrame(ic_rows).set_index("factor")
    print(ic_table.round(4).to_string())

    print("\n=== Factor Correlation Matrix (2016-2026) ===")
    print(eval_df.corr().round(2).to_string())


if __name__ == "__main__":
    main()
