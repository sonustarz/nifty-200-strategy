

"""
Nifty 200 Live Momentum Dashboard
=================================

A Streamlit dashboard that:

1. Reads a Nifty 200 constituent CSV containing Symbol and Sector.
2. Downloads adjusted daily price history from Yahoo Finance.
3. Applies the strategy's market filter, stock filters and momentum ranking.
4. Builds an equal-weight target portfolio of up to 20 stocks.
5. Enforces a maximum 10% per stock and 30% per sector.
6. Compares the target portfolio with current holdings.
7. Produces BUY, BUY MORE, HOLD, SELL PARTIAL and SELL ALL actions.
8. Shows when the strategy says those actions should be executed.

Strategy timing
---------------
Official signal:
    Last completed trading day of the month.

Execution:
    Next trading day, normally near the market open.

The "Provisional latest daily close" mode is only a live watchlist. It should
not be treated as the official monthly rebalance signal.

Required universe CSV
---------------------
Symbol,Sector
RELIANCE,Oil Gas & Consumable Fuels
TCS,Information Technology
HDFCBANK,Financial Services

Optional holdings CSV
---------------------
Symbol,Quantity,AveragePrice
RELIANCE,10,2800
TCS,3,3900

Install
-------
pip install --upgrade streamlit pandas numpy yfinance plotly

Run
---
streamlit run nifty200_live_dashboard.py
"""


from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional
import math

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# =============================================================================
# USER SETTINGS
# =============================================================================

# Change this to your Nifty 200 constituent CSV path.
TICKER_FILE = Path("ind_nifty200listWithSector.csv")

# Optional current holdings file. The dashboard also provides a file uploader.
HOLDINGS_FILE = Path("current_holdings_template.csv")

INDEX_YAHOO_SYMBOL = "^CNX200"

# Enough history for 252-day volatility, 12-month momentum and 200-day SMA.
HISTORY_PERIOD = "3y"
DATA_INTERVAL = "1d"

STARTING_CAPITAL = 100_000.0
TARGET_STOCKS = 20
RANK_EXIT_THRESHOLD = 30
MAX_STOCK_WEIGHT = 0.10
MAX_SECTOR_WEIGHT = 0.30

SMA_FAST = 50
SMA_SLOW = 200
HIGH_52W = 252
MOMENTUM_SKIP = 21
MOMENTUM_6M_LOOKBACK = 126
MOMENTUM_12M_LOOKBACK = 252
VOLATILITY_LOOKBACK = 252
TRADING_DAYS_PER_YEAR = 252

# Yahoo calls are cached for 15 minutes to reduce repeated requests/rate limits.
CACHE_TTL_SECONDS = 15 * 60
DOWNLOAD_CHUNK_SIZE = 40


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def find_column(
    columns: Iterable[object],
    candidates: Iterable[str],
) -> Optional[object]:
    """Find a matching column name, ignoring case and surrounding spaces."""
    lookup = {str(column).strip().lower(): column for column in columns}

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lookup:
            return lookup[key]

    return None


def normalise_local_symbol(value: object) -> str:
    """Return a clean local NSE symbol without the .NS suffix."""
    symbol = str(value).strip().upper()

    if symbol.endswith(".NS"):
        symbol = symbol[:-3]

    return symbol


def to_yahoo_symbol(local_symbol: str) -> str:
    """Convert an NSE symbol to Yahoo Finance format."""
    symbol = str(local_symbol).strip().upper()

    if symbol.endswith(".NS") or symbol.startswith("^"):
        return symbol

    return f"{symbol}.NS"


def safe_number(value: object) -> float:
    """Convert a value to float; return NaN when conversion is impossible."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan

    return result if np.isfinite(result) else np.nan


def inr(value: object) -> str:
    """Format a value in Indian rupees."""
    number = safe_number(value)

    if pd.isna(number):
        return ""

    return f"₹{number:,.0f}"


def percentage(value: object, decimals: int = 1) -> str:
    """Format a decimal as a percentage."""
    number = safe_number(value)

    if pd.isna(number):
        return ""

    return f"{number * 100:.{decimals}f}%"


def clean_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean one ticker's Yahoo Finance OHLCV dataframe."""
    if frame is None or frame.empty:
        return pd.DataFrame()

    result = frame.copy()

    result.index = pd.to_datetime(
        result.index,
        errors="coerce",
        utc=True,
    )
    result = result.loc[~result.index.isna()].copy()
    result.index = result.index.tz_convert(None).normalize()

    result = result[~result.index.duplicated(keep="last")]
    result.sort_index(inplace=True)

    # Flatten any remaining one-level tuple-like columns.
    result.columns = [
        column[0] if isinstance(column, tuple) else column
        for column in result.columns
    ]

    rename_map = {}
    for column in result.columns:
        clean_name = str(column).strip()
        lower_name = clean_name.lower()

        if lower_name == "adj close":
            rename_map[column] = "Adj Close"
        elif lower_name == "open":
            rename_map[column] = "Open"
        elif lower_name == "high":
            rename_map[column] = "High"
        elif lower_name == "low":
            rename_map[column] = "Low"
        elif lower_name == "close":
            rename_map[column] = "Close"
        elif lower_name == "volume":
            rename_map[column] = "Volume"

    result.rename(columns=rename_map, inplace=True)

    if "Close" not in result.columns and "Adj Close" in result.columns:
        result["Close"] = result["Adj Close"]

    numeric_columns = [
        column
        for column in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        if column in result.columns
    ]

    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    if "Close" in result.columns:
        result.dropna(subset=["Close"], inplace=True)

    return result


def extract_ticker_frame(
    raw: pd.DataFrame,
    yahoo_symbol: str,
    total_tickers: int,
) -> pd.DataFrame:
    """Extract one ticker from a yfinance download result."""
    if raw is None or raw.empty:
        return pd.DataFrame()

    if not isinstance(raw.columns, pd.MultiIndex):
        if total_tickers == 1:
            return clean_price_frame(raw)

        return pd.DataFrame()

    # Common yfinance layout: level 0 = field, level 1 = ticker.
    for level in range(raw.columns.nlevels):
        level_values = raw.columns.get_level_values(level)

        if yahoo_symbol in level_values:
            extracted = raw.xs(
                yahoo_symbol,
                axis=1,
                level=level,
                drop_level=True,
            )
            return clean_price_frame(extracted)

    return pd.DataFrame()


# =============================================================================
# INPUT FILES
# =============================================================================

def read_universe(path: Path) -> pd.DataFrame:
    """
    Load a Nifty 200 file.

    Required:
        Symbol or Ticker column
        Sector or Industry column
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Universe file not found:\n{path}\n\n"
            "Create a CSV with Symbol and Sector columns."
        )

    raw = pd.read_csv(path)

    if raw.empty:
        raise ValueError("The universe CSV is empty.")

    symbol_column = find_column(
        raw.columns,
        ["Symbol", "Ticker", "Security Symbol"],
    )
    sector_column = find_column(
        raw.columns,
        ["Sector", "Industry", "Industry Name", "Sector Name"],
    )

    if symbol_column is None:
        raise ValueError(
            "The universe CSV needs a Symbol or Ticker column.\n"
            f"Available columns: {list(raw.columns)}"
        )

    if sector_column is None:
        raise ValueError(
            "The universe CSV needs a Sector or Industry column because the "
            "strategy limits each sector to 30%.\n"
            f"Available columns: {list(raw.columns)}"
        )

    result = raw[[symbol_column, sector_column]].copy()
    result.columns = ["Symbol", "Sector"]

    result["Symbol"] = result["Symbol"].map(normalise_local_symbol)
    result["Sector"] = (
        result["Sector"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    result = result[
        (result["Symbol"] != "")
        & (~result["Symbol"].isin(["NAN", "NONE", "NULL"]))
    ].copy()

    if (result["Sector"] == "").any():
        missing = result.loc[result["Sector"] == "", "Symbol"].tolist()

        raise ValueError(
            "Sector is missing for these symbols:\n"
            + ", ".join(missing[:30])
        )

    result.drop_duplicates(subset=["Symbol"], keep="last", inplace=True)
    result["YahooSymbol"] = result["Symbol"].map(to_yahoo_symbol)
    result.reset_index(drop=True, inplace=True)

    return result


def read_holdings_from_source(source) -> pd.DataFrame:
    """
    Read an optional holdings CSV.

    Required:
        Symbol

    Optional:
        Quantity
        AveragePrice
    """
    if source is None:
        return pd.DataFrame(
            columns=["Symbol", "Quantity", "AveragePrice"]
        )

    raw = pd.read_csv(source)

    if raw.empty:
        return pd.DataFrame(
            columns=["Symbol", "Quantity", "AveragePrice"]
        )

    symbol_column = find_column(
        raw.columns,
        ["Symbol", "Ticker", "Security Symbol"],
    )

    if symbol_column is None:
        raise ValueError(
            "The holdings CSV needs a Symbol or Ticker column."
        )

    quantity_column = find_column(
        raw.columns,
        ["Quantity", "Qty", "Shares", "Units"],
    )
    average_price_column = find_column(
        raw.columns,
        ["AveragePrice", "Average Price", "AvgPrice", "Avg Price"],
    )

    result = pd.DataFrame()
    result["Symbol"] = raw[symbol_column].map(normalise_local_symbol)

    if quantity_column is None:
        result["Quantity"] = np.nan
    else:
        result["Quantity"] = pd.to_numeric(
            raw[quantity_column],
            errors="coerce",
        )

    if average_price_column is None:
        result["AveragePrice"] = np.nan
    else:
        result["AveragePrice"] = pd.to_numeric(
            raw[average_price_column],
            errors="coerce",
        )

    result = result[
        (result["Symbol"] != "")
        & (~result["Symbol"].isin(["NAN", "NONE", "NULL"]))
    ].copy()

    result.drop_duplicates(subset=["Symbol"], keep="last", inplace=True)
    result.reset_index(drop=True, inplace=True)

    return result


# =============================================================================
# YAHOO FINANCE DATA
# =============================================================================

@st.cache_data(
    ttl=CACHE_TTL_SECONDS,
    show_spinner=False,
)
def download_price_history(
    yahoo_symbols: tuple[str, ...],
    history_period: str,
    interval: str,
) -> tuple[Dict[str, pd.DataFrame], list[str]]:
    """
    Download multiple symbols in chunks.

    Returns:
        dictionary: Yahoo symbol -> cleaned dataframe
        list of symbols that returned no usable data
    """
    data: Dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    symbol_list = list(yahoo_symbols)

    for start in range(0, len(symbol_list), DOWNLOAD_CHUNK_SIZE):
        chunk = symbol_list[start:start + DOWNLOAD_CHUNK_SIZE]

        try:
            raw = yf.download(
                tickers=chunk,
                period=history_period,
                interval=interval,
                auto_adjust=True,
                actions=False,
                progress=False,
                threads=True,
                group_by="column",
                repair=True,
                timeout=30,
            )
        except Exception:
            raw = pd.DataFrame()

        for symbol in chunk:
            frame = extract_ticker_frame(
                raw=raw,
                yahoo_symbol=symbol,
                total_tickers=len(chunk),
            )

            if frame.empty or "Close" not in frame.columns:
                # Retry failed symbols individually.
                try:
                    single_raw = yf.download(
                        tickers=symbol,
                        period=history_period,
                        interval=interval,
                        auto_adjust=True,
                        actions=False,
                        progress=False,
                        threads=False,
                        group_by="column",
                        repair=True,
                        timeout=30,
                    )
                except Exception:
                    single_raw = pd.DataFrame()

                frame = extract_ticker_frame(
                    raw=single_raw,
                    yahoo_symbol=symbol,
                    total_tickers=1,
                )

            if frame.empty or "Close" not in frame.columns:
                failed.append(symbol)
            else:
                data[symbol] = frame

    return data, failed


@st.cache_data(
    ttl=CACHE_TTL_SECONDS,
    show_spinner=False,
)
def download_index_history(
    yahoo_symbol: str,
    history_period: str,
    interval: str,
) -> pd.DataFrame:
    """Download and clean Nifty 200 index history."""
    raw = yf.download(
        tickers=yahoo_symbol,
        period=history_period,
        interval=interval,
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=False,
        group_by="column",
        repair=True,
        timeout=30,
    )

    frame = extract_ticker_frame(
        raw=raw,
        yahoo_symbol=yahoo_symbol,
        total_tickers=1,
    )

    if frame.empty or "Close" not in frame.columns:
        raise RuntimeError(
            f"Yahoo Finance returned no usable data for {yahoo_symbol}."
        )

    return frame


# =============================================================================
# SIGNAL CALCULATION
# =============================================================================

def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate the complete strategy indicator set."""
    result = frame.copy()
    close = result["Close"]
    daily_return = close.pct_change(fill_method=None)

    result["SMA50"] = close.rolling(
        SMA_FAST,
        min_periods=SMA_FAST,
    ).mean()

    result["SMA200"] = close.rolling(
        SMA_SLOW,
        min_periods=SMA_SLOW,
    ).mean()

    result["High252"] = close.rolling(
        HIGH_52W,
        min_periods=HIGH_52W,
    ).max()

    result["Momentum6M"] = (
        close.shift(MOMENTUM_SKIP)
        / close.shift(MOMENTUM_6M_LOOKBACK)
        - 1.0
    )

    result["Momentum12M"] = (
        close.shift(MOMENTUM_SKIP)
        / close.shift(MOMENTUM_12M_LOOKBACK)
        - 1.0
    )

    result["Volatility"] = (
        daily_return.rolling(
            VOLATILITY_LOOKBACK,
            min_periods=VOLATILITY_LOOKBACK,
        ).std(ddof=1)
        * math.sqrt(TRADING_DAYS_PER_YEAR)
    )

    valid_volatility = result["Volatility"].where(
        result["Volatility"] > 0
    )

    result["Score6M"] = (
        result["Momentum6M"] / valid_volatility
    )
    result["Score12M"] = (
        result["Momentum12M"] / valid_volatility
    )

    result["MomentumScore"] = (
        0.50 * result["Score6M"]
        + 0.50 * result["Score12M"]
    )

    result["AboveSMA200"] = (
        result["Close"] > result["SMA200"]
    )
    result["SMA50AboveSMA200"] = (
        result["SMA50"] > result["SMA200"]
    )
    result["Near52WeekHigh"] = (
        result["Close"] > 0.90 * result["High252"]
    )

    result["EntryEligible"] = (
        result["AboveSMA200"]
        & result["SMA50AboveSMA200"]
        & result["Near52WeekHigh"]
        & result["MomentumScore"].notna()
    )

    return result


def latest_completed_month_end(
    index_calendar: pd.DatetimeIndex,
) -> pd.Timestamp:
    """
    Return a conservative official signal date.

    If data includes the current calendar month, use the prior month's last
    trading date. If the latest data is from an older month, use that latest
    available month's last trading date.
    """
    calendar = pd.DatetimeIndex(index_calendar).sort_values()

    if len(calendar) == 0:
        raise ValueError("The index calendar is empty.")

    today_period = pd.Timestamp.now().to_period("M")
    latest_period = calendar.max().to_period("M")

    monthly_last = (
        pd.Series(calendar, index=calendar)
        .groupby(calendar.to_period("M"))
        .last()
    )

    if latest_period < today_period:
        return pd.Timestamp(monthly_last.iloc[-1])

    if len(monthly_last) < 2:
        raise ValueError(
            "Not enough index history to identify a completed month."
        )

    return pd.Timestamp(monthly_last.iloc[-2])


def next_index_trading_day(
    index_calendar: pd.DatetimeIndex,
    signal_date: pd.Timestamp,
) -> Optional[pd.Timestamp]:
    """Return the first index trading date after the signal date."""
    calendar = pd.DatetimeIndex(index_calendar).sort_values()
    location = calendar.searchsorted(signal_date, side="right")

    if location >= len(calendar):
        return None

    return pd.Timestamp(calendar[location])


def latest_row_on_or_before(
    frame: pd.DataFrame,
    signal_date: pd.Timestamp,
) -> Optional[pd.Series]:
    """Get the latest row on or before the requested signal date."""
    eligible = frame.loc[frame.index <= signal_date]

    if eligible.empty:
        return None

    row = eligible.iloc[-1]

    return row


def build_cross_section(
    signal_date: pd.Timestamp,
    universe: pd.DataFrame,
    price_history: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build the point-in-time ranking table."""
    rows = []

    for universe_row in universe.itertuples(index=False):
        local_symbol = universe_row.Symbol
        yahoo_symbol = universe_row.YahooSymbol
        sector = universe_row.Sector

        raw_frame = price_history.get(yahoo_symbol)

        if raw_frame is None or raw_frame.empty:
            continue

        indicator_frame = add_indicators(raw_frame)
        row = latest_row_on_or_before(
            indicator_frame,
            signal_date,
        )

        if row is None:
            continue

        score = safe_number(row.get("MomentumScore", np.nan))

        rows.append(
            {
                "Symbol": local_symbol,
                "YahooSymbol": yahoo_symbol,
                "Sector": sector,
                "PriceDate": indicator_frame.loc[
                    indicator_frame.index <= signal_date
                ].index[-1],
                "Close": safe_number(row.get("Close", np.nan)),
                "SMA50": safe_number(row.get("SMA50", np.nan)),
                "SMA200": safe_number(row.get("SMA200", np.nan)),
                "High252": safe_number(row.get("High252", np.nan)),
                "Momentum6M": safe_number(
                    row.get("Momentum6M", np.nan)
                ),
                "Momentum12M": safe_number(
                    row.get("Momentum12M", np.nan)
                ),
                "Volatility": safe_number(
                    row.get("Volatility", np.nan)
                ),
                "MomentumScore": score,
                "AboveSMA200": bool(
                    row.get("AboveSMA200", False)
                ),
                "SMA50AboveSMA200": bool(
                    row.get("SMA50AboveSMA200", False)
                ),
                "Near52WeekHigh": bool(
                    row.get("Near52WeekHigh", False)
                ),
                "EntryEligible": bool(
                    row.get("EntryEligible", False)
                ),
            }
        )

    if not rows:
        return pd.DataFrame()

    cross = pd.DataFrame(rows)

    # Rank all stocks with a valid momentum score. Entry filters are applied
    # separately so an existing holding can remain while ranked <= 30 even if
    # it no longer satisfies the entry-only SMA50/high filters.
    valid_score = cross["MomentumScore"].notna()

    cross.loc[valid_score, "Rank"] = (
        cross.loc[valid_score, "MomentumScore"]
        .rank(method="first", ascending=False)
    )

    cross["Rank"] = cross["Rank"].astype("Int64")

    cross.sort_values(
        ["Rank", "Symbol"],
        ascending=[True, True],
        na_position="last",
        inplace=True,
    )
    cross.reset_index(drop=True, inplace=True)

    return cross


# =============================================================================
# PORTFOLIO CONSTRUCTION
# =============================================================================

def select_target_symbols(
    cross: pd.DataFrame,
    current_symbols: set[str],
) -> list[str]:
    """
    Select up to 20 target stocks.

    Existing holdings are retained while:
        Close > SMA200
        Rank <= 30

    New stocks must pass all entry filters.

    A practical six-name-per-sector guard is used at the normal 20-stock,
    5%-per-stock target. Final weights are separately capped at 30% by sector.
    """
    if cross.empty:
        return []

    ranked = cross.dropna(subset=["Rank"]).copy()
    ranked["RankNumeric"] = ranked["Rank"].astype(int)

    retained = ranked[
        ranked["Symbol"].isin(current_symbols)
        & ranked["AboveSMA200"]
        & (ranked["RankNumeric"] <= RANK_EXIT_THRESHOLD)
    ].sort_values(["RankNumeric", "Symbol"])

    entrants = ranked[
        ranked["EntryEligible"]
        & (~ranked["Symbol"].isin(retained["Symbol"]))
    ].sort_values(["RankNumeric", "Symbol"])

    ordered = pd.concat(
        [retained, entrants],
        ignore_index=True,
    ).drop_duplicates(subset=["Symbol"], keep="first")

    normal_weight = min(
        1.0 / TARGET_STOCKS,
        MAX_STOCK_WEIGHT,
    )

    max_names_per_sector = max(
        1,
        int(
            math.floor(
                MAX_SECTOR_WEIGHT / normal_weight + 1e-12
            )
        ),
    )

    selected: list[str] = []
    sector_counts: Dict[str, int] = {}

    for row in ordered.itertuples(index=False):
        if len(selected) >= TARGET_STOCKS:
            break

        sector = str(row.Sector)

        if sector_counts.get(sector, 0) >= max_names_per_sector:
            continue

        selected.append(row.Symbol)
        sector_counts[sector] = (
            sector_counts.get(sector, 0) + 1
        )

    return selected


def calculate_target_weights(
    selected: list[str],
    cross: pd.DataFrame,
) -> Dict[str, float]:
    """
    Equal-weight the selected stocks.

    Limits:
        Maximum 10% per stock
        Maximum 30% per sector

    If a sector cap binds, that sector is scaled down and unused money remains
    as cash.
    """
    if not selected:
        return {}

    cross_by_symbol = cross.set_index("Symbol")
    base_weight = min(
        1.0 / len(selected),
        MAX_STOCK_WEIGHT,
    )

    weights = {
        symbol: base_weight
        for symbol in selected
    }

    sector_groups: Dict[str, list[str]] = {}

    for symbol in selected:
        sector = str(cross_by_symbol.at[symbol, "Sector"])
        sector_groups.setdefault(sector, []).append(symbol)

    for sector_symbols in sector_groups.values():
        total_sector_weight = sum(
            weights[symbol]
            for symbol in sector_symbols
        )

        if total_sector_weight > MAX_SECTOR_WEIGHT + 1e-12:
            capped_weight = (
                MAX_SECTOR_WEIGHT / len(sector_symbols)
            )

            for symbol in sector_symbols:
                weights[symbol] = min(
                    capped_weight,
                    MAX_STOCK_WEIGHT,
                )

    return weights


# =============================================================================
# ACTION GENERATION
# =============================================================================

def exit_reason(
    symbol: str,
    market_filter: bool,
    cross_by_symbol: pd.DataFrame,
    target_symbols: set[str],
) -> str:
    """Explain why an existing holding is not in the target portfolio."""
    if not market_filter:
        return (
            "Market filter negative: Nifty 200 is below its 200-day SMA"
        )

    if symbol not in cross_by_symbol.index:
        return (
            "No valid current ranking data or stock is absent from the universe"
        )

    row = cross_by_symbol.loc[symbol]

    if not bool(row["AboveSMA200"]):
        return "Stock close is below its 200-day SMA"

    rank = row["Rank"]

    if pd.isna(rank) or int(rank) > RANK_EXIT_THRESHOLD:
        return (
            f"Momentum rank is worse than {RANK_EXIT_THRESHOLD}"
        )

    if symbol not in target_symbols:
        return (
            "Removed by portfolio-size or 30% sector-concentration limit"
        )

    return "No exit condition"


def build_actions(
    cross: pd.DataFrame,
    target_weights: Dict[str, float],
    holdings: pd.DataFrame,
    capital: float,
    market_filter: bool,
    execution_text: str,
) -> pd.DataFrame:
    """
    Compare the strategy target with current holdings.

    When Quantity is supplied, the dashboard calculates estimated target and
    order quantities using the latest signal close. Actual execution price may
    differ at the next trading day's open.
    """
    columns = [
        "Action",
        "Symbol",
        "Sector",
        "Rank",
        "LatestClose",
        "CurrentQuantity",
        "TargetQuantity",
        "OrderQuantity",
        "TargetWeight",
        "TargetAmount",
        "When",
        "Reason",
    ]

    if cross.empty:
        return pd.DataFrame(columns=columns)

    cross_by_symbol = cross.set_index("Symbol")
    holdings_by_symbol = (
        holdings.set_index("Symbol")
        if not holdings.empty
        else pd.DataFrame()
    )

    current_symbols = (
        set(holdings["Symbol"])
        if not holdings.empty
        else set()
    )
    target_symbols = set(target_weights)

    rows = []

    # Target portfolio: new buys, buy more, partial sells or holds.
    for symbol, weight in target_weights.items():
        row = cross_by_symbol.loc[symbol]
        latest_close = safe_number(row["Close"])
        target_amount = capital * weight

        target_quantity = (
            math.floor(target_amount / latest_close)
            if latest_close > 0
            else np.nan
        )

        is_current = symbol in current_symbols
        current_quantity = np.nan

        if is_current and "Quantity" in holdings_by_symbol.columns:
            current_quantity = safe_number(
                holdings_by_symbol.at[symbol, "Quantity"]
            )

        if not is_current:
            action = "BUY"
            order_quantity = target_quantity
            reason = (
                "New entry: passes all filters and is selected by momentum rank"
            )

        elif pd.notna(current_quantity):
            difference = int(target_quantity - current_quantity)

            if difference > 0:
                action = "BUY MORE"
                order_quantity = difference
                reason = "Existing qualifying holding is below target weight"
            elif difference < 0:
                action = "SELL PARTIAL"
                order_quantity = abs(difference)
                reason = "Existing qualifying holding is above target weight"
            else:
                action = "HOLD"
                order_quantity = 0
                reason = (
                    "Close above SMA200 and momentum rank remains within 30"
                )
        else:
            action = "HOLD / REBALANCE"
            order_quantity = np.nan
            reason = (
                "Holding qualifies, but Quantity is missing so exact rebalance "
                "shares cannot be calculated"
            )

        rows.append(
            {
                "Action": action,
                "Symbol": symbol,
                "Sector": row["Sector"],
                "Rank": row["Rank"],
                "LatestClose": latest_close,
                "CurrentQuantity": current_quantity,
                "TargetQuantity": target_quantity,
                "OrderQuantity": order_quantity,
                "TargetWeight": weight,
                "TargetAmount": target_amount,
                "When": execution_text,
                "Reason": reason,
            }
        )

    # Full exits.
    for symbol in sorted(current_symbols - target_symbols):
        if symbol in cross_by_symbol.index:
            row = cross_by_symbol.loc[symbol]
            sector = row["Sector"]
            rank = row["Rank"]
            latest_close = safe_number(row["Close"])
        else:
            sector = ""
            rank = pd.NA
            latest_close = np.nan

        current_quantity = safe_number(
            holdings_by_symbol.at[symbol, "Quantity"]
        )

        rows.append(
            {
                "Action": "SELL ALL",
                "Symbol": symbol,
                "Sector": sector,
                "Rank": rank,
                "LatestClose": latest_close,
                "CurrentQuantity": current_quantity,
                "TargetQuantity": 0,
                "OrderQuantity": current_quantity,
                "TargetWeight": 0.0,
                "TargetAmount": 0.0,
                "When": execution_text,
                "Reason": exit_reason(
                    symbol=symbol,
                    market_filter=market_filter,
                    cross_by_symbol=cross_by_symbol,
                    target_symbols=target_symbols,
                ),
            }
        )

    actions = pd.DataFrame(rows, columns=columns)

    action_order = {
        "SELL ALL": 1,
        "SELL PARTIAL": 2,
        "BUY": 3,
        "BUY MORE": 4,
        "HOLD / REBALANCE": 5,
        "HOLD": 6,
    }

    actions["_ActionOrder"] = (
        actions["Action"]
        .map(action_order)
        .fillna(99)
    )

    actions.sort_values(
        ["_ActionOrder", "Rank", "Symbol"],
        ascending=[True, True, True],
        na_position="last",
        inplace=True,
    )
    actions.drop(columns=["_ActionOrder"], inplace=True)
    actions.reset_index(drop=True, inplace=True)

    return actions


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def styled_action_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a display-friendly action table."""
    if frame.empty:
        return frame

    display = frame.copy()

    display["LatestClose"] = display["LatestClose"].map(inr)
    display["TargetAmount"] = display["TargetAmount"].map(inr)
    display["TargetWeight"] = display["TargetWeight"].map(percentage)

    for column in [
        "CurrentQuantity",
        "TargetQuantity",
        "OrderQuantity",
    ]:
        display[column] = display[column].apply(
            lambda value: (
                ""
                if pd.isna(value)
                else f"{int(value):,}"
            )
        )

    return display


def styled_ranking_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a display-friendly ranking table."""
    if frame.empty:
        return frame

    columns = [
        "Rank",
        "Symbol",
        "Sector",
        "Close",
        "SMA50",
        "SMA200",
        "High252",
        "Momentum6M",
        "Momentum12M",
        "Volatility",
        "MomentumScore",
        "AboveSMA200",
        "SMA50AboveSMA200",
        "Near52WeekHigh",
        "EntryEligible",
    ]

    display = frame[columns].copy()

    for column in ["Close", "SMA50", "SMA200", "High252"]:
        display[column] = display[column].map(inr)

    for column in [
        "Momentum6M",
        "Momentum12M",
        "Volatility",
    ]:
        display[column] = display[column].map(percentage)

    display["MomentumScore"] = display["MomentumScore"].apply(
        lambda value: (
            ""
            if pd.isna(value)
            else f"{value:.3f}"
        )
    )

    return display


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Convert a dataframe to downloadable UTF-8 CSV bytes."""
    return frame.to_csv(index=False).encode("utf-8")


# =============================================================================
# STREAMLIT APPLICATION
# =============================================================================

st.set_page_config(
    page_title="Nifty 200 Momentum Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Nifty 200 Monthly Momentum Dashboard")

st.caption(
    "Daily Yahoo Finance screening with official month-end portfolio actions"
)

st.warning(
    "The strategy trades only after a month-end signal. "
    "A mid-month screen is provisional. Verify all prices and orders with "
    "your broker before execution."
)

with st.sidebar:
    st.header("Dashboard settings")

    universe_path_text = st.text_input(
        "Nifty 200 CSV path",
        value=str(TICKER_FILE),
        help="CSV must contain Symbol and Sector/Industry columns.",
    )

    uploaded_holdings = st.file_uploader(
        "Upload current holdings CSV",
        type=["csv"],
        help=(
            "Recommended columns: Symbol, Quantity, AveragePrice. "
            "Without holdings, the dashboard can show buys but cannot "
            "identify your personal exits."
        ),
    )

    use_local_holdings = st.checkbox(
        "Use local holdings file when no upload is provided",
        value=HOLDINGS_FILE.exists(),
    )

    capital = st.number_input(
        "Portfolio capital (INR)",
        min_value=10_000.0,
        value=float(STARTING_CAPITAL),
        step=10_000.0,
    )

    signal_mode = st.radio(
        "Signal mode",
        options=[
            "Official: latest completed month-end",
            "Provisional: latest available daily close",
        ],
        index=0,
        help=(
            "Official mode follows the strategy. Provisional mode is a "
            "watchlist and should normally not be traded before month-end."
        ),
    )

    if st.button(
        "Refresh Yahoo data",
        type="primary",
        use_container_width=True,
    ):
        st.cache_data.clear()
        st.rerun()

    st.caption(
        f"Yahoo data cache: {CACHE_TTL_SECONDS // 60} minutes"
    )


# Load local inputs.
try:
    universe = read_universe(Path(universe_path_text))
except Exception as error:
    st.error(str(error))
    st.stop()

try:
    if uploaded_holdings is not None:
        holdings = read_holdings_from_source(uploaded_holdings)
    elif use_local_holdings and HOLDINGS_FILE.exists():
        holdings = read_holdings_from_source(HOLDINGS_FILE)
    else:
        holdings = read_holdings_from_source(None)
except Exception as error:
    st.error(f"Holdings file error: {error}")
    st.stop()


# Fetch data.
progress_message = st.empty()

with st.spinner(
    f"Fetching Nifty 200 data for {len(universe)} symbols..."
):
    try:
        index_history = download_index_history(
            yahoo_symbol=INDEX_YAHOO_SYMBOL,
            history_period=HISTORY_PERIOD,
            interval=DATA_INTERVAL,
        )

        price_history, failed_symbols = download_price_history(
            yahoo_symbols=tuple(universe["YahooSymbol"]),
            history_period=HISTORY_PERIOD,
            interval=DATA_INTERVAL,
        )
    except Exception as error:
        st.error(f"Yahoo Finance download failed: {error}")
        st.stop()


index_indicators = add_indicators(index_history)

if signal_mode.startswith("Official"):
    signal_date = latest_completed_month_end(
        index_indicators.index
    )
    is_provisional = False
else:
    signal_date = pd.Timestamp(index_indicators.index.max())
    is_provisional = True

index_row = latest_row_on_or_before(
    index_indicators,
    signal_date,
)

if index_row is None:
    st.error("No index data is available for the selected signal date.")
    st.stop()

index_close = safe_number(index_row["Close"])
index_sma200 = safe_number(index_row["SMA200"])

market_filter = bool(
    pd.notna(index_sma200)
    and index_close > index_sma200
)

execution_date = next_index_trading_day(
    index_indicators.index,
    signal_date,
)

if is_provisional:
    execution_text = (
        "Do not trade yet; reassess after the current month-end close"
    )
elif execution_date is not None:
    execution_text = (
        f"{execution_date.date()} near market open"
    )
else:
    execution_text = (
        "Next NSE trading day after the signal date, near market open"
    )


cross = build_cross_section(
    signal_date=signal_date,
    universe=universe,
    price_history=price_history,
)

if cross.empty:
    st.error(
        "No stocks had enough usable history to calculate the strategy."
    )
    st.stop()

current_symbols = (
    set(holdings["Symbol"])
    if not holdings.empty
    else set()
)

if market_filter:
    selected = select_target_symbols(
        cross=cross,
        current_symbols=current_symbols,
    )
    target_weights = calculate_target_weights(
        selected=selected,
        cross=cross,
    )
else:
    selected = []
    target_weights = {}

actions = build_actions(
    cross=cross,
    target_weights=target_weights,
    holdings=holdings,
    capital=float(capital),
    market_filter=market_filter,
    execution_text=execution_text,
)


# Headline metrics.
eligible_count = int(cross["EntryEligible"].sum())
invested_weight = sum(target_weights.values())
cash_weight = max(0.0, 1.0 - invested_weight)

metric_columns = st.columns(6)

metric_columns[0].metric(
    "Market filter",
    "ON" if market_filter else "OFF",
)

metric_columns[1].metric(
    "Signal date",
    str(signal_date.date()),
)

metric_columns[2].metric(
    "Nifty 200 close",
    f"{index_close:,.1f}",
)

metric_columns[3].metric(
    "Nifty 200 SMA200",
    f"{index_sma200:,.1f}" if pd.notna(index_sma200) else "N/A",
)

metric_columns[4].metric(
    "Eligible stocks",
    eligible_count,
)

metric_columns[5].metric(
    "Target holdings",
    len(target_weights),
)

if is_provisional:
    st.info(
        "This is a provisional daily screen. The official strategy action "
        "is determined using the completed month-end closing data."
    )
elif market_filter:
    st.success(
        f"Market filter is positive. Execute confirmed rebalance orders: "
        f"{execution_text}."
    )
else:
    st.error(
        f"Market filter is negative. Strategy action: sell all holdings "
        f"{execution_text} and remain in cash."
    )

if failed_symbols:
    failed_local = [
        normalise_local_symbol(symbol)
        for symbol in failed_symbols
    ]
    st.warning(
        f"No usable data for {len(failed_local)} symbols: "
        + ", ".join(failed_local[:25])
    )


# Tabs.
action_tab, eligible_tab, ranking_tab, portfolio_tab, chart_tab = st.tabs(
    [
        "Action plan",
        "Entry candidates",
        "Full ranking",
        "Target portfolio",
        "Stock chart",
    ]
)


with action_tab:
    st.subheader("Actionable strategy output")

    if holdings.empty:
        st.info(
            "No holdings file is loaded. BUY candidates are available, but "
            "SELL actions require your current holdings."
        )

    sell_actions = actions[
        actions["Action"].isin(["SELL ALL", "SELL PARTIAL"])
    ]

    buy_actions = actions[
        actions["Action"].isin(["BUY", "BUY MORE"])
    ]

    hold_actions = actions[
        actions["Action"].isin(["HOLD", "HOLD / REBALANCE"])
    ]

    st.markdown("### Sell actions")
    if sell_actions.empty:
        st.write("No sell action is currently generated.")
    else:
        st.dataframe(
            styled_action_table(sell_actions),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### Buy actions")
    if not market_filter:
        st.write(
            "No buys are allowed because the Nifty 200 market filter is OFF."
        )
    elif buy_actions.empty:
        st.write("No new buy action is currently generated.")
    else:
        st.dataframe(
            styled_action_table(buy_actions),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### Hold or rebalance actions")
    if hold_actions.empty:
        st.write("No continuing holdings are identified.")
    else:
        st.dataframe(
            styled_action_table(hold_actions),
            use_container_width=True,
            hide_index=True,
        )

    st.download_button(
        "Download action plan CSV",
        data=to_csv_bytes(actions),
        file_name=(
            f"nifty200_actions_{signal_date.date()}.csv"
        ),
        mime="text/csv",
    )


with eligible_tab:
    st.subheader("Stocks passing all new-entry filters")

    eligible = cross[cross["EntryEligible"]].copy()

    if eligible.empty:
        st.write("No stock currently passes every entry filter.")
    else:
        st.dataframe(
            styled_ranking_table(eligible),
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "A stock passing the filters is not automatically bought. "
        "It must also be selected within the target portfolio after ranking "
        "and sector constraints."
    )

    st.download_button(
        "Download eligible stocks CSV",
        data=to_csv_bytes(eligible),
        file_name=(
            f"eligible_stocks_{signal_date.date()}.csv"
        ),
        mime="text/csv",
    )


with ranking_tab:
    st.subheader("Cross-sectional momentum ranking")

    st.dataframe(
        styled_ranking_table(cross),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "The rank is calculated for all stocks with a valid momentum score. "
        "Entry filters are shown separately."
    )

    st.download_button(
        "Download full ranking CSV",
        data=to_csv_bytes(cross),
        file_name=(
            f"momentum_ranking_{signal_date.date()}.csv"
        ),
        mime="text/csv",
    )


with portfolio_tab:
    st.subheader("Target portfolio")

    if not target_weights:
        st.write("Target portfolio: 100% cash.")
    else:
        target_rows = []

        cross_by_symbol = cross.set_index("Symbol")

        for symbol, weight in target_weights.items():
            row = cross_by_symbol.loc[symbol]
            target_amount = float(capital) * weight
            close = safe_number(row["Close"])

            target_rows.append(
                {
                    "Symbol": symbol,
                    "Sector": row["Sector"],
                    "Rank": row["Rank"],
                    "Weight": weight,
                    "TargetAmount": target_amount,
                    "EstimatedSharesAtSignalClose": (
                        math.floor(target_amount / close)
                        if close > 0
                        else np.nan
                    ),
                    "SignalClose": close,
                }
            )

        target_table = pd.DataFrame(target_rows)

        display_target = target_table.copy()
        display_target["Weight"] = display_target["Weight"].map(
            percentage
        )
        display_target["TargetAmount"] = (
            display_target["TargetAmount"].map(inr)
        )
        display_target["SignalClose"] = (
            display_target["SignalClose"].map(inr)
        )

        st.dataframe(
            display_target,
            use_container_width=True,
            hide_index=True,
        )

        sector_table = (
            target_table.groupby("Sector", as_index=False)["Weight"]
            .sum()
            .sort_values("Weight", ascending=False)
        )
        sector_table["Weight"] = sector_table["Weight"].map(
            percentage
        )

        st.markdown("### Sector allocation")
        st.dataframe(
            sector_table,
            use_container_width=True,
            hide_index=True,
        )

    allocation_columns = st.columns(3)

    allocation_columns[0].metric(
        "Invested target",
        percentage(invested_weight),
    )
    allocation_columns[1].metric(
        "Cash target",
        percentage(cash_weight),
    )
    allocation_columns[2].metric(
        "Target capital",
        inr(capital),
    )

    st.caption(
        "Estimated share quantities use the signal close. Actual order "
        "quantity should be recalculated using the next trading day's "
        "available execution price."
    )


with chart_tab:
    st.subheader("Price and trend chart")

    chart_symbol = st.selectbox(
        "Select stock",
        options=list(cross["Symbol"]),
    )

    yahoo_symbol = to_yahoo_symbol(chart_symbol)
    chart_data = price_history.get(yahoo_symbol)

    if chart_data is None or chart_data.empty:
        st.write("No chart data is available.")
    else:
        chart_indicators = add_indicators(chart_data)

        chart_window = chart_indicators.loc[
            chart_indicators.index
            >= chart_indicators.index.max() - pd.Timedelta(days=550),
            ["Close", "SMA50", "SMA200"],
        ].copy()

        st.line_chart(
            chart_window,
            use_container_width=True,
        )

        selected_row = cross.set_index("Symbol").loc[chart_symbol]

        detail_columns = st.columns(5)
        detail_columns[0].metric(
            "Rank",
            (
                int(selected_row["Rank"])
                if pd.notna(selected_row["Rank"])
                else "N/A"
            ),
        )
        detail_columns[1].metric(
            "Close",
            inr(selected_row["Close"]),
        )
        detail_columns[2].metric(
            "6M momentum",
            percentage(selected_row["Momentum6M"]),
        )
        detail_columns[3].metric(
            "12M momentum",
            percentage(selected_row["Momentum12M"]),
        )
        detail_columns[4].metric(
            "Entry eligible",
            "YES" if selected_row["EntryEligible"] else "NO",
        )


st.divider()

st.caption(
    "Data source: Yahoo Finance through yfinance. Signals are generated from "
    "adjusted daily OHLC data. This tool is for strategy research and "
    "decision support; it does not place broker orders."
)
