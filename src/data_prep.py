# Automatically extracted data preparation functions
import datetime
import urllib3
import warnings
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import re

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


pd.set_option('display.max_columns', 50)
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')

# Global Analytical Constants
PREMIUM_CUTOFF = 150000  # 150k guineas: market threshold for high-value lots
EARLY_PERIOD = (2009, 2015)
MID_PERIOD = (2016, 2020)
RECENT_PERIOD = (2021, 2025)

cpi_october = {
    2009: 89.2, 2010: 91.9, 2011: 96.5, 2012: 99.1, 2013: 101.3, 2014: 102.6,
    2015: 102.6, 2016: 103.5, 2017: 106.6, 2018: 109.2, 2019: 110.8, 2020: 111.8,
    2021: 116.5, 2022: 129.4, 2023: 135.5, 2024: 138.6, 2025: 141.4  # Estimate for 2025
}
BASE_YEAR = 2024
COUNTRY_SUFFIX_RE = re.compile(r'\s*\(([A-Z]+)\)\s*$')


def extract_country_suffix(series: pd.Series) -> pd.Series:
    """Extract country codes such as (GB) without destroying the original string."""
    return series.astype('string').str.extract(COUNTRY_SUFFIX_RE, expand=False)


def strip_country_suffix(series: pd.Series) -> pd.Series:
    """Remove country suffixes such as (GB) while keeping the full entity name."""
    return series.astype('string').str.replace(COUNTRY_SUFFIX_RE, '', regex=True).str.strip()


def parse_numeric_series(series: pd.Series) -> pd.Series:
    """Parse Tattersalls-style numeric fields like 90.000 or 108.675 into floats."""
    cleaned = (
        series.astype('string')
        .str.strip()
        .replace({'-': pd.NA, 'nan': pd.NA, 'None': pd.NA, '': pd.NA}) # type: ignore
        .str.replace('.', '', regex=False)
        .str.replace(',', '.', regex=False)
    )
    return pd.to_numeric(cleaned, errors='coerce')


def title_from_canonical(series: pd.Series) -> pd.Series:
    """Convert canonical uppercase labels into a readable title-case display string."""
    return (
        series.astype('string')
        .str.strip()
        .str.replace(r'\s+', ' ', regex=True)
        .str.lower()
        .str.title()
    )


def normalize_root_entity(name: str, stopwords=None, aliases=None):
    """Conservative normalization for high-cardinality entity names."""
    if pd.isna(name):
        return None
    stopwords = stopwords or set()
    aliases = aliases or {}
    s = str(name).upper().strip()
    s = aliases.get(s, s)
    s = s.replace('&', ' AND ')
    s = re.sub(r'[^A-Z0-9\s]', ' ', s)
    tokens = [t for t in s.split() if t not in stopwords]
    s = ' '.join(tokens)
    s = re.sub(r'\s+', ' ', s).strip()
    return s if s else None


def bootstrap_ci(values, stat_func=np.median, n_boot=2000, ci=0.95, random_state=42):
    """
    Bootstrap confidence interval for a univariate statistic.

    Vectorized implementation for efficiency with large samples.

    Parameters:
    -----------
    values : array-like
        Data to bootstrap
    stat_func : callable
        Statistic function (default: np.median for robustness with heavy tails)
    n_boot : int
        Number of bootstrap samples (default: 2000)
    ci : float
        Confidence level (default: 0.95)
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    tuple : (observed_statistic, ci_lower, ci_upper)

    Notes:
    ------
    Uses percentile method. For n > 4000 with median or proportions,
    this converges to BCa method (Efron & Tibshirani, 1993).

    For small samples (n < 500) with asymmetric distributions,
    consider using scipy.stats.bootstrap with method='BCa'.
    """
    clean = pd.Series(values).dropna().astype(float).to_numpy()

    if clean.size == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(random_state)

    # Vectorized sampling: generate all bootstrap samples at once
    # Shape: (n_boot, clean.size)
    boot_samples = rng.choice(clean, size=(n_boot, clean.size), replace=True)

    # Apply statistic to each bootstrap sample
    # Optimization: use axis-specific functions when possible (avoids apply_along_axis overhead)
    if stat_func == np.median:
        boot_stats = np.median(boot_samples, axis=1)
    elif stat_func == np.mean:
        boot_stats = np.mean(boot_samples, axis=1)
    elif stat_func == np.std:
        boot_stats = np.std(boot_samples, axis=1, ddof=1)
    else:
        # Fallback for custom functions (e.g., trimmed mean, quantiles)
        boot_stats = np.apply_along_axis(stat_func, 1, boot_samples)

    observed = stat_func(clean)

    alpha = (1 - ci) / 2
    return observed, np.quantile(boot_stats, alpha), np.quantile(boot_stats, 1 - alpha)


def bootstrap_proportion_ci(boolean_values, n_boot=2000, ci=0.95, random_state=42):
    """
    Bootstrap confidence interval for a binary proportion.

    Vectorized implementation using binomial sampling optimization.

    Parameters:
    -----------
    boolean_values : array-like
        Binary data (0/1 or True/False)
    n_boot : int
        Number of bootstrap samples (default: 2000)
    ci : float
        Confidence level (default: 0.95)
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    tuple : (observed_proportion, ci_lower, ci_upper)

    Notes:
    ------
    For proportions with n > 5000, the percentile method is 
      approximately equivalent to BCa due to CLT convergence.

     Mathematical note: The bootstrap distribution of a proportion 
      is approximately symmetric when n*p and n*(1-p) > 10.
    """
    clean = pd.Series(boolean_values).dropna().astype(int).to_numpy()

    if clean.size == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(random_state)

    # Vectorized approach: generate all samples at once
    # Shape: (n_boot, clean.size)
    boot_samples = rng.choice(clean, size=(n_boot, clean.size), replace=True)

    # Calculate proportion for each bootstrap sample
    # Sum along axis=1 gives count of 1s, divide by n gives proportion
    boot_props = boot_samples.sum(axis=1) / clean.size

    observed_prop = clean.mean()

    alpha = (1 - ci) / 2
    return observed_prop, np.quantile(boot_props, alpha), np.quantile(boot_props, 1 - alpha)


def permutation_test(values_a, values_b, stat_func=np.median, n_perm=5000, random_state=42):
    """
    Two-sided permutation test for a difference in statistics.

    Tests the null hypothesis that two groups have the same distribution 
     (specifically, that their statistics are equal).

    Parameters:
    ----------
    values_a : array-like
        First group data
    values_b : array-like
        Second group data
    stat_func : callable
        Statistic function to compare (default: np.median for robustness)
    n_perm : int
        Number of permutations (default: 5000)
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    tuple : (observed_difference, p_value)

    Notes:
    -----
    The permutation test is exact (not approximate) under H₀,
     making it more robust than parametric tests for heavy-tailed data.

    Assumptions:
     - Exchangeability under H₀: if groups come from the same distribution,
        any permutation is equally likely
     - Independence within and between groups

    For heavy-tailed data with different tail behaviors,
     consider using robust statistics like median or trimmed mean.
    """
    a = pd.Series(values_a).dropna().astype(float).to_numpy()
    b = pd.Series(values_b).dropna().astype(float).to_numpy()

    if a.size == 0 or b.size == 0:
        return np.nan, np.nan

    # Observed difference in statistics between groups
    observed_diff = stat_func(a) - stat_func(b)

    # Pool both groups for permutation under H₀ (null hypothesis: no difference)
    pooled = np.concatenate([a, b])

    rng = np.random.default_rng(random_state)

    # Vectorized permutation approach would require storing all permuted arrays,
    # which is memory-intensive. Loop is acceptable here.
    perm_stats = np.empty(n_perm)
    for i in range(n_perm):
        shuffled_indices = rng.permutation(len(pooled))
        perm_a = pooled[shuffled_indices[:len(a)]]
        perm_b = pooled[shuffled_indices[len(a):]]
        perm_stats[i] = stat_func(perm_a) - stat_func(perm_b)

    # Two-sided p-value: proportion of permutations with absolute difference >= observed
    p_value = np.mean(np.abs(perm_stats) >= abs(observed_diff))

    return observed_diff, p_value


def mean_annual_share_table(df, entity_col, label_col=None):
    """
    Normalize entity prominence by annual market share instead of raw counts.

    This approach corrects for year-to-year variation in total sales volume,
    giving a more accurate picture of entity prominence over time.

    Parameters:
    ----------
    df : DataFrame
        Input data with 'sale_year' and entity_col columns
    entity_col : str
        Column name for entity identifier (e.g., 'buyer_normalized')
    label_col : str, optional
        Column name for display label (e.g., 'buyer_title')

    Returns:
    -------
    DataFrame with columns:
        - label_col (if provided): most common display name for entity
        - total_sales: sum of sales across all years
        - active_years: number of years with at least one sale
        - mean_annual_share: average share within each year
        - peak_annual_share: maximum share in any single year

    Notes:
    -----
    Mean annual share is more robust than total share when comparing entities 
      across periods with different total volumes.

    Example interpretation:
        If mean_annual_share = 0.15, the entity averaged 15% of market 
         activity in years they were active.
    """
    per_year = (
        df.groupby(['sale_year', entity_col])
        .size()
        .rename('sales')
        .reset_index()
    )

    # Calculate share within each year (corrects for volume variation)
    per_year['share_within_year'] = per_year['sales'] / per_year.groupby('sale_year')['sales'].transform('sum')

    summary = (
        per_year.groupby(entity_col)
        .agg(
            total_sales=('sales', 'sum'),
            active_years=('sale_year', 'nunique'),
            mean_annual_share=('share_within_year', 'mean'),
            peak_annual_share=('share_within_year', 'max')
        )
        .sort_values(['mean_annual_share', 'total_sales'], ascending=[False, False])
    )

    if label_col is not None:
        # Get most common label for each entity (handles minor variations)
        label_map = (
            df[[entity_col, label_col]]
            .dropna()
            .groupby(entity_col)[label_col]
            .agg(lambda x: x.value_counts().index[0])
        )
        summary.insert(0, label_col, summary.index.to_series().map(label_map))

    return summary

_BOE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-GB,en;q=0.9",
}
_CACHE_DEFAULT = Path(__file__).parent.parent / "data" / "processed" / "macro_data.parquet"


def _fetch_boe_rate(start_year: int) -> pd.Series:
    """Parse Bank Rate history from BoE public HTML table → annual mean series."""
    import requests
    from bs4 import BeautifulSoup

    r = requests.get(
        "https://www.bankofengland.co.uk/boeapps/database/Bank-Rate.asp",
        headers=_BOE_HEADERS,
        verify=False,
        timeout=20,
    )
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    df = pd.read_html(StringIO(str(soup.find("table"))))[0]
    df.columns = ["date_changed", "rate"]
    df["date_changed"] = pd.to_datetime(df["date_changed"], format="%d %b %y")
    df = df.sort_values("date_changed")

    end = pd.Timestamp.today().normalize()
    daily_idx = pd.date_range(f"{start_year}-01-01", end, freq="D")
    return (
        df.set_index("date_changed")
        .reindex(daily_idx)
        .ffill()
        .resample("YE")
        .mean()["rate"]
    )


def _fetch_gbp_eur(start_year: int) -> pd.Series:
    """Download GBP/EUR daily closes from Yahoo Finance → annual mean series."""
    import yfinance as yf

    fx = yf.download("GBPEUR=X", start=f"{start_year}-01-01", progress=False)["Close"]
    if hasattr(fx, "columns"):
        fx = fx.iloc[:, 0]
    return fx.resample("YE").mean()


def get_macro_data(
    start_year: int = 2009,
    cache_path: Path | None = None,
    max_cache_age_days: int = 7,
) -> pd.DataFrame:
    """Return annual macroeconomic indicators for the Tattersalls model.

    Fetches live data from official sources and caches to disk so notebook
    reruns don't trigger network calls unnecessarily.

    Sources
    -------
    - GBP/EUR : Yahoo Finance (yfinance), daily close → annual mean
    - BoE Base Rate : bankofengland.co.uk/boeapps/database/Bank-Rate.asp
      (public HTML table, forward-filled daily → annual mean)

    Parameters
    ----------
    start_year : int
        First year to include (default 2009).
    cache_path : Path | None
        Parquet file for disk cache. Defaults to data/processed/macro_data.parquet.
    max_cache_age_days : int
        Days before cache is considered stale and refreshed (default 7).

    Returns
    -------
    pd.DataFrame
        Columns: sale_year (int), gbp_eur_rate (float), boe_base_rate (float).
        One row per year from start_year to most recent available.
    """
    cache = Path(cache_path) if cache_path else _CACHE_DEFAULT

    # Return cache if fresh
    if cache.exists():
        age = datetime.datetime.now() - datetime.datetime.fromtimestamp(cache.stat().st_mtime)
        if age.days < max_cache_age_days:
            df = pd.read_parquet(cache)
            return df[df["sale_year"] >= start_year].reset_index(drop=True)

    print("Fetching macro data from live sources...")
    try:
        boe = _fetch_boe_rate(start_year)
        fx = _fetch_gbp_eur(start_year)
        print("  GBP/EUR : Yahoo Finance OK")
        print("  BoE Base Rate : bankofengland.co.uk OK")
    except Exception as exc:
        warnings.warn(f"Live fetch failed ({exc}). Falling back to cached/hardcoded values.")
        if cache.exists():
            df = pd.read_parquet(cache)
            return df[df["sale_year"] >= start_year].reset_index(drop=True)
        raise RuntimeError("No cache available and live fetch failed.") from exc

    # Align on year index
    years = boe.index.year.intersection(fx.index.year)
    df = pd.DataFrame({
        "sale_year": years,
        "gbp_eur_rate": fx[fx.index.year.isin(years)].values.round(4),
        "boe_base_rate": boe[boe.index.year.isin(years)].values.round(4),
    })

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df[df["sale_year"] >= start_year].reset_index(drop=True)
