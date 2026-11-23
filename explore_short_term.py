"""Exploratory diagnostics for A-share Short-Term Limit-Up/Down Sentiment Cycle (2021-2026)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_short_term_panel() -> pd.DataFrame:
    st = pd.read_csv(DATA_DIR / "short_term_daily.csv")
    st["date"] = pd.to_datetime(st["date"])
    st = st.sort_values("date").drop_duplicates("date").set_index("date")

    for name in ["hs300", "zz500", "zz1000", "chinext", "sh_comp"]:
        p = DATA_DIR / f"idx_{name}.csv"
        if p.exists():
            idx = pd.read_csv(p)
            idx["date"] = pd.to_datetime(idx["date"])
            idx = idx.sort_values("date").drop_duplicates("date").set_index("date")
            st[f"{name}_open"] = idx["open"]
            st[f"{name}_close"] = idx["close"]
            st[f"{name}_ret"] = idx["close"].pct_change()
    return st


def build_stsi(
    df: pd.DataFrame,
    window: int = 240,
    min_periods: int = 60,
    smooth_days: int = 3,
) -> pd.DataFrame:
    """Construct strictly causal Short-Term Sentiment Index (STSI, 0-100) from 6 limit-up/down dimensions."""
    raw = pd.DataFrame(index=df.index)
    # 1. 涨跌停净强度: log((涨停+5)/(跌停+5))
    raw["zt_dt_ratio"] = np.log((df["zt_count"] + 5.0) / (df["dt_count"] + 5.0))
    # 2. 连板高度与梯队厚度: 0.5*log1p(最高连板) + 0.5*log1p(连板家数)
    raw["lb_space"] = 0.5 * np.log1p(df["max_lb"]) + 0.5 * np.log1p(df["lb2_count"])
    # 3. 连板晋级率
    raw["promo_rate"] = df["promo_rate"]
    # 4. 封板成功率 (1 - 炸板率)
    raw["seal_rate"] = 1.0 - df["zb_rate"]
    # 5. 昨日涨停今日溢价率 (赚钱效应)
    raw["prev_zt_ret"] = df["prev_zt_ret"]
    # 6. 大面率反向 (亏钱效应出清)
    raw["anti_nuke"] = -df["big_loss_rate"]

    z_subs = pd.DataFrame(index=df.index)
    out = pd.DataFrame(index=df.index)
    for c in raw.columns:
        mu = raw[c].rolling(window, min_periods=min_periods).mean()
        sd = raw[c].rolling(window, min_periods=min_periods).std(ddof=1)
        z = ((raw[c] - mu) / sd).clip(-3.0, 3.0)
        z_subs[c] = z
        out[f"s_{c}"] = pd.Series(norm.cdf(z) * 100.0, index=df.index)

    comp_z = z_subs.mean(axis=1)
    c_mu = comp_z.rolling(window, min_periods=min_periods).mean()
    c_sd = comp_z.rolling(window, min_periods=min_periods).std(ddof=1)
    stsi_raw = pd.Series(
        norm.cdf(((comp_z - c_mu) / c_sd).clip(-3.0, 3.0)) * 100.0, index=df.index
    )
    out["stsi_raw"] = stsi_raw
    out["stsi"] = stsi_raw.rolling(smooth_days, min_periods=1).mean()
    # Short-term emotion velocity (momentum): today's raw vs 5d EMA
    out["stsi_ema5"] = stsi_raw.ewm(span=5, adjust=False).mean()
    out["stsi_mom"] = out["stsi"] - out["stsi"].shift(3)
    return out


def classify_cycle_phase(stsi: pd.Series, stsi_mom: pd.Series) -> pd.Series:
    """Classify each trading day into the 5 classic A-share short-term emotion cycle phases."""
    phases = []
    for s, m in zip(stsi, stsi_mom):
        if pd.isna(s) or pd.isna(m):
            phases.append("预热期")
        elif s <= 20:
            phases.append("1_冰点期")
        elif s >= 80:
            phases.append("4_高潮期")
        elif m >= 0:
            phases.append("2_修复启动" if s < 50 else "3_发酵主升")
        else:
            phases.append("5_分歧退潮")
    return pd.Series(phases, index=stsi.index)


def main() -> None:
    df = load_short_term_panel()
    print(f"Loaded short_term_daily: {len(df)} days ({df.index.min().date()} -> {df.index.max().date()})")
    print("\nSummary statistics of raw short-term metrics (2021-2026):")
    cols = ["zt_count", "dt_count", "zb_count", "zb_rate", "max_lb", "lb2_count", "promo_rate", "prev_zt_ret", "big_loss_rate", "up_ratio", "relay_next_ret"]
    print(df[cols].describe().loc[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(4).to_string())

    stsi_df = build_stsi(df, window=240, min_periods=60, smooth_days=3)
    df = df.join(stsi_df)
    df["phase"] = classify_cycle_phase(df["stsi"], df["stsi_mom"])

    eval_df = df.loc["2021-07-01":].copy()

    # 1. Check Rank IC of each sub-factor + STSI vs ZZ1000 & ChiNext & Relay next-day returns
    print("\n=== Spearman Rank IC vs ZZ1000 Forward Returns (strictly T+1 close -> T+1+H, 2021-07 to 2026-11) ===")
    c1 = df["zz1000_close"].shift(-1)
    fwd = {
        "1d(T+1->T+2)": df["zz1000_close"].shift(-2) / c1 - 1.0,
        "3d": df["zz1000_close"].shift(-4) / c1 - 1.0,
        "5d": df["zz1000_close"].shift(-6) / c1 - 1.0,
        "10d": df["zz1000_close"].shift(-11) / c1 - 1.0,
        "20d": df["zz1000_close"].shift(-21) / c1 - 1.0,
        "next_relay_o2c": df["relay_next_ret"].shift(-1),  # signal at T close -> buy non-yizi prev_zt at T+1 open -> sell T+2 close!
        "next_day_zt_ret": df["prev_zt_ret"].shift(-1),    # tomorrow's limit-up premium
    }
    fac_cols = ["stsi_raw", "stsi", "stsi_mom", "s_zt_dt_ratio", "s_lb_space", "s_promo_rate", "s_seal_rate", "s_prev_zt_ret", "s_anti_nuke"]
    ic_rows = []
    for fc in fac_cols:
        r = {"factor": fc}
        for hn, fs in fwd.items():
            al = pd.concat([eval_df[fc], fs], axis=1, sort=False).dropna()
            r[hn] = al.iloc[:, 0].corr(al.iloc[:, 1], method="spearman")
        ic_rows.append(r)
    print(pd.DataFrame(ic_rows).set_index("factor").round(4).to_string())

    # 2. Quantile forward returns on ZZ1000, ChiNext, and Relay Strategy
    print("\n=== STSI Quantile Forward Returns (2021-07 to 2026-11) ===")
    qdf = pd.DataFrame({
        "stsi": eval_df["stsi"],
        "stsi_raw": eval_df["stsi_raw"],
        "zz1000_1d": fwd["1d(T+1->T+2)"],
        "zz1000_5d": fwd["5d"],
        "zz1000_10d": fwd["10d"],
        "zz1000_20d": fwd["20d"],
        "cyb_10d": df["chinext_close"].shift(-11) / df["chinext_close"].shift(-1) - 1.0,
        "relay_next": fwd["next_relay_o2c"],
        "next_zt_prem": fwd["next_day_zt_ret"],
    }).loc["2021-07-01":].dropna(subset=["stsi"])

    qdf["zone"] = pd.cut(qdf["stsi"], [0, 20, 40, 60, 80, 100.01], labels=["0-20 冰点", "20-40 偏弱", "40-60 震荡", "60-80 偏强", "80-100 高潮"])
    g_mean = qdf.groupby("zone", observed=False)[["zz1000_1d", "zz1000_5d", "zz1000_10d", "zz1000_20d", "cyb_10d", "relay_next", "next_zt_prem"]].mean()
    g_cnt = qdf.groupby("zone", observed=False).size().rename("days")
    print(pd.concat([g_cnt, g_mean], axis=1).to_string(float_format=lambda x: f"{x:+.2%}" if abs(x) < 2 else f"{x:.0f}"))

    # 3. 5-Phase Cycle Breakdown
    print("\n=== 5-Phase Short-Term Emotion Cycle Forward Returns ===")
    qdf["phase"] = eval_df["phase"]
    p_mean = qdf.groupby("phase")[["zz1000_1d", "zz1000_5d", "zz1000_10d", "zz1000_20d", "relay_next", "next_zt_prem"]].mean()
    p_cnt = qdf.groupby("phase").size().rename("days")
    print(pd.concat([p_cnt, p_mean], axis=1).to_string(float_format=lambda x: f"{x:+.2%}" if abs(x) < 2 else f"{x:.0f}"))

    # 4. Famous short-term turning points check
    print("\n=== Famous Short-Term Turning Points ===")
    milestones = [
        ("2022-04-26", "2022.4跌停潮冰点"),
        ("2023-03-20", "2023 AI主升发酵"),
        ("2024-02-05", "2024.2微盘股核按钮冰点"),
        ("2024-09-18", "924行情前冰点"),
        ("2024-10-08", "924天量涨停高潮"),
        ("2024-11-08", "2024.11妖股高标高潮"),
        ("2025-04-07", "2025.4关税冲击冰点"),
        ("2026-11-20", "当前最新交易日"),
    ]
    for dstr, lbl in milestones:
        idx = df.index.asof(pd.Timestamp(dstr))
        r = df.loc[idx]
        print(
            f"  {idx.strftime('%Y-%m-%d')} | {lbl:<18s} | STSI={r['stsi']:5.1f} ({r['phase']}) | "
            f"涨停={int(r['zt_count']):3d} 跌停={int(r['dt_count']):3d} 炸板={r['zb_rate']*100:4.1f}% "
            f"高度={int(r['max_lb']):2d}板 昨涨停={r['prev_zt_ret']*100:+5.2f}% 大面={r['big_loss_rate']*100:4.1f}%"
        )


if __name__ == "__main__":
    main()
