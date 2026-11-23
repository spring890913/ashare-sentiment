"""Fetch and cache historical A-share market and sentiment datasets."""

from __future__ import annotations

from pathlib import Path
import time
import akshare as ak
import baostock as bs
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2012-01-01"
END_DATE = "2026-11-22"

# (alias, baostock_code, akshare_em_code)
INDICES = [
    ("hs300", "sh.000300", "sh000300"),      # 沪深300
    ("sh_comp", "sh.000001", "sh000001"),    # 上证指数
    ("sz_comp", "sz.399106", "sz399106"),    # 深证综指 (全深市成交额)
    ("sse50", "sh.000016", "sh000016"),      # 上证50
    ("zz500", "sh.000905", "sh000905"),      # 中证500
    ("zz1000", "sh.000852", "sh000852"),     # 中证1000
    ("chinext", "sz.399006", "sz399006"),    # 创业板指
    ("zz_all", "sh.000985", "sh000985"),     # 中证全指
    ("zz_div", "sh.000922", "sh000922"),     # 中证红利
]


def fetch_indices() -> None:
    need_bs = []
    for name, bs_code, em_code in INDICES:
        out = DATA_DIR / f"idx_{name}.csv"
        if out.exists() and out.stat().st_size > 10000:
            print(f"[CACHE] {name}: already cached ({out.stat().st_size} bytes)", flush=True)
            continue
        # Try fast AkShare Eastmoney first
        t0 = time.time()
        try:
            df = ak.index_zh_daily_em(symbol=em_code, start_date="20120101", end_date="20261122")
            if df is not None and len(df) > 100:
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                for col in ["open", "high", "low", "close", "volume", "amount"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                df["pctChg"] = df["close"].pct_change() * 100.0
                df.to_csv(out, index=False)
                print(
                    f"[OK-EM] {name} ({em_code}): {len(df)} rows, "
                    f"{df['date'].iloc[0]} -> {df['date'].iloc[-1]} ({time.time()-t0:.1f}s)",
                    flush=True,
                )
                continue
        except Exception as e:
            print(f"[WARN-EM] {name} ({em_code}) failed: {e}; will use baostock", flush=True)
        need_bs.append((name, bs_code))

    if need_bs:
        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"Baostock login failed: {lg.error_msg}")
        try:
            for name, code in need_bs:
                t0 = time.time()
                rs = bs.query_history_k_data_plus(
                    code,
                    "date,open,high,low,close,volume,amount,pctChg",
                    start_date=START_DATE,
                    end_date=END_DATE,
                    frequency="d",
                )
                rows = []
                while rs.error_code == "0" and rs.next():
                    rows.append(rs.get_row_data())
                if not rows:
                    print(f"[WARN-BS] {name} ({code}): no rows", flush=True)
                    continue
                df = pd.DataFrame(rows, columns=rs.fields)
                for col in ["open", "high", "low", "close", "volume", "amount", "pctChg"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                out = DATA_DIR / f"idx_{name}.csv"
                df.to_csv(out, index=False)
                print(
                    f"[OK-BS] {name} ({code}): {len(df)} rows, "
                    f"{df['date'].iloc[0]} -> {df['date'].iloc[-1]} ({time.time()-t0:.1f}s)",
                    flush=True,
                )
        finally:
            bs.logout()


def fetch_akshare_sentiment() -> None:
    tasks = [
        ("margin_info.csv", "日期", lambda: ak.stock_margin_account_info()),
        ("qvix_50etf.csv", "date", lambda: ak.index_option_50etf_qvix()),
        ("below_net_asset.csv", "date", lambda: ak.stock_a_below_net_asset_statistics(symbol="全部A股")),
        ("ebs_hs300.csv", "日期", lambda: ak.stock_ebs_lg()),
    ]
    for fname, date_col, fn in tasks:
        out = DATA_DIR / fname
        if out.exists() and out.stat().st_size > 10000:
            print(f"[CACHE] {fname}: already cached", flush=True)
            continue
        t0 = time.time()
        print(f"Fetching {fname}...", flush=True)
        df = fn()
        df[date_col] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
        df = df.sort_values(date_col).drop_duplicates(date_col)
        df.to_csv(out, index=False)
        print(
            f"[OK] {fname}: {len(df)} rows, "
            f"{df[date_col].iloc[0]} -> {df[date_col].iloc[-1]} ({time.time()-t0:.1f}s)",
            flush=True,
        )


if __name__ == "__main__":
    fetch_indices()
    fetch_akshare_sentiment()
    print("All data fetched successfully.", flush=True)
