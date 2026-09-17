"""
Technical Indicators (Stateless Pure Calculations)
"""

import numpy as np
import pandas as pd


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculates Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Computes Wilder's Average True Range (ATR).
    Expects DataFrame with columns: ['High', 'Low', 'Close']
    """
    high = df['High']
    low = df['Low']
    prev_close = df['Close'].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    return atr


def compute_rvol(volume_series: pd.Series, window: int = 20) -> pd.Series:
    """
    Computes Relative Volume (RVOL) against a rolling rolling average.
    """
    rolling_vol_mean = volume_series.rolling(window=window).mean()
    rvol = volume_series / rolling_vol_mean.replace(0, np.nan)
    return rvol.fillna(1.0)
