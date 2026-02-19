"""
Download and process the dataset for Chapter 5: Bitcoin Regime-Switching VAR.

Sources
-------
- Bitcoin price       : Yahoo Finance via yfinance (ticker: BTC-USD)
- 3-Month LIBOR proxy : FRED series DGS3MO (3-Month Treasury constant maturity).
                        Actual USD LIBOR was discontinued in June 2023; DGS3MO
                        is the standard public proxy used in academic replications.
- 10-Year Treasury    : FRED series DGS10
- GPR index           : Caldara & Iacoviello Geopolitical Risk Index (daily).
                        Downloaded from matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls
- IER                 : 5-Year Breakeven Inflation Rate, FRED series T5YIE.
                        "IER" = Implied (Inflation) Expectations Rate.

Sample period: 2015-01-02 to 2023-04-21 (business days only, 2,035 obs)

Outputs
-------
  msvar_data.csv      Raw daily data matching the original column schema:
                        libor, date, bond10, btc, gpr, ier
  msvar_processed.csv Processed model inputs (2,031 obs):
                        date, btc, d_bond10, d_libor
                      where:
                        btc      = daily log return of Bitcoin
                        d_bond10 = 5-day overlapping change in 10Y Treasury yield
                        d_libor  = 5-day overlapping change in 3-Month LIBOR proxy

Usage
-----
    pip install yfinance pandas requests openpyxl
    python download_data.py
"""

import io
import sys

import pandas as pd
import numpy as np
import requests
import yfinance as yf


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

START = "2015-01-01"
END = "2023-04-22"  # yfinance end is exclusive; FRED end is inclusive

FRED_API = "https://fred.stlouisfed.org/graph/fredgraph.csv"
GPR_URL = (
    "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def fetch_fred(series_id: str, col_name: str) -> pd.Series:
    """Download a FRED series and return it as a named daily Series."""
    url = f"{FRED_API}?id={series_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), parse_dates=["DATE"], index_col="DATE")
    # FRED uses "." for missing values
    df = df.replace(".", pd.NA)
    series = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    series.name = col_name
    return series


def fetch_btc() -> pd.Series:
    """Download BTC-USD closing prices from Yahoo Finance."""
    ticker = yf.Ticker("BTC-USD")
    hist = ticker.history(start=START, end=END, auto_adjust=True)
    prices = hist["Close"]
    prices.index = prices.index.tz_localize(None).normalize()
    prices.name = "btc"
    return prices


def fetch_gpr() -> pd.Series:
    """Download Caldara-Iacoviello daily GPR index from matteoiacoviello.com."""
    resp = requests.get(GPR_URL, timeout=60)
    resp.raise_for_status()
    df = pd.read_excel(io.BytesIO(resp.content), engine="openpyxl")

    # The spreadsheet has a 'Date' column and a 'GPRD' column (daily GPR).
    # Column names can vary slightly across versions — find them defensively.
    date_col = next(c for c in df.columns if "date" in str(c).lower())
    gpr_col = next(
        c for c in df.columns
        if str(c).upper() in ("GPRD", "GPR_DAILY", "GPR DAILY", "DAILY GPR")
        or str(c).strip().upper().startswith("GPR")
    )
    df = df[[date_col, gpr_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    series = pd.to_numeric(df[gpr_col], errors="coerce")
    series.name = "gpr"
    series.index = pd.DatetimeIndex(series.index).normalize()
    return series


# ---------------------------------------------------------------------------
# Build msvar_data.csv
# ---------------------------------------------------------------------------

def build_raw(output_path: str = "msvar_data.csv") -> pd.DataFrame:
    """Download all series, align on business-day index, save raw CSV."""
    print("Downloading BTC-USD from Yahoo Finance ...")
    btc = fetch_btc()

    print("Downloading DGS3MO (3-Month Treasury / LIBOR proxy) from FRED ...")
    libor = fetch_fred("DGS3MO", "libor")

    print("Downloading DGS10 (10-Year Treasury) from FRED ...")
    bond10 = fetch_fred("DGS10", "bond10")

    print("Downloading T5YIE (5-Year Breakeven Inflation / IER) from FRED ...")
    ier = fetch_fred("T5YIE", "ier")

    print("Downloading GPR daily index from matteoiacoviello.com ...")
    gpr = fetch_gpr()

    # Align all series on BTC trading days (BTC trades every day including
    # weekends, but FRED/GPR are business-day series). Use BTC index as the
    # master; forward-fill weekends for FRED/GPR (max 3 days).
    idx = btc.index
    df = pd.DataFrame(index=idx)
    df["btc"] = btc

    for series in (libor, bond10, ier, gpr):
        aligned = series.reindex(idx, method="ffill", limit=3)
        df[series.name] = aligned

    # Restrict to sample period and drop rows where any series is missing
    df = df.loc[START:END].dropna()

    # Format the date column to match the original Stata-style format: DDmmmYYYY
    # e.g. "02jan2015". This matches the original msvar_data.csv exactly.
    month_abbr = {
        1: "jan", 2: "feb", 3: "mar", 4: "apr", 5: "may", 6: "jun",
        7: "jul", 8: "aug", 9: "sep", 10: "oct", 11: "nov", 12: "dec",
    }
    df.index.name = "date_idx"
    df.reset_index(inplace=True)
    df["date"] = df["date_idx"].apply(
        lambda d: f"{d.day:02d}{month_abbr[d.month]}{d.year}"
    )

    # Reorder columns to match original schema: libor, date, bond10, btc, gpr, ier
    df = df[["libor", "date", "bond10", "btc", "gpr", "ier"]]

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")
    print(f"  Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    return df


# ---------------------------------------------------------------------------
# Build msvar_processed.csv
# ---------------------------------------------------------------------------

def build_processed(raw_df: pd.DataFrame, output_path: str = "msvar_processed.csv") -> pd.DataFrame:
    """Compute log-returns and 5-day overlapping changes; save processed CSV.

    Transformations applied:
      btc      = log(P_t / P_{t-1})              daily log return
      d_bond10 = bond10_t - bond10_{t-5}         5-day overlapping change
      d_libor  = libor_t  - libor_{t-5}          5-day overlapping change

    The 5-day window aligns with the thesis convention; the overlapping
    structure induces MA(4) autocorrelation in d_bond10 and d_libor.
    """
    # Parse dates from the Stata-style string back to proper datetime
    df = raw_df.copy()
    df["date"] = pd.to_datetime(df["date"], format="%d%b%Y")

    df["btc_log"] = np.log(df["btc"])
    df["btc_return"] = df["btc_log"].diff(1)

    df["d_bond10"] = df["bond10"].diff(5)
    df["d_libor"] = df["libor"].diff(5)

    processed = df[["date", "btc_return", "d_bond10", "d_libor"]].copy()
    processed.columns = ["date", "btc", "d_bond10", "d_libor"]

    # Drop the first 5 rows lost to differencing (matches original 2,031 obs)
    processed = processed.dropna()
    processed["date"] = processed["date"].dt.strftime("%Y-%m-%d")

    processed.to_csv(output_path, index=False)
    print(f"Saved {len(processed)} rows to {output_path}")
    print(f"  Date range: {processed['date'].iloc[0]} to {processed['date'].iloc[-1]}")
    return processed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Chapter 5 MS-VAR — Data Download Script")
    print("=" * 60)
    print()

    try:
        raw = build_raw("msvar_data.csv")
        print()
        build_processed(raw, "msvar_processed.csv")
        print()
        print("Done. Both CSV files are ready.")
    except requests.HTTPError as exc:
        print(f"HTTP error while downloading data: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
