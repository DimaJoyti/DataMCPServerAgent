"""
Technical Indicators for Algorithmic Trading Strategies

Comprehensive collection of technical indicators used in trading strategies.
Implements efficient calculations using pandas and numpy.
"""

from typing import Dict

import numpy as np
import pandas as pd


class TechnicalIndicators:
    """Collection of technical indicators for trading strategies."""

    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return data.rolling(window=period).mean()

    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return data.ewm(span=period).mean()

    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def macd(
        data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Dict[str, pd.Series]:
        """MACD (Moving Average Convergence Divergence)."""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line

        return {"macd": macd_line, "signal": signal_line, "histogram": histogram}

    @staticmethod
    def bollinger_bands(
        data: pd.Series, period: int = 20, std_dev: float = 2
    ) -> Dict[str, pd.Series]:
        """Bollinger Bands."""
        sma = TechnicalIndicators.sma(data, period)
        std = data.rolling(window=period).std()

        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        return {"upper": upper_band, "middle": sma, "lower": lower_band}

    @staticmethod
    def stochastic(
        high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3
    ) -> Dict[str, pd.Series]:
        """Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()

        return {"k": k_percent, "d": d_percent}

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return atr

    @staticmethod
    def williams_r(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Williams %R."""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()

        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))

        return williams_r

    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Commodity Channel Index."""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x)))
        )

        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)

        return cci

    @staticmethod
    def adx(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> Dict[str, pd.Series]:
        """Average Directional Index."""
        # Calculate True Range
        tr = TechnicalIndicators.atr(high, low, close, 1)

        # Calculate Directional Movement
        dm_plus = np.where((high.diff() > low.diff().abs()) & (high.diff() > 0), high.diff(), 0)
        dm_minus = np.where(
            (low.diff().abs() > high.diff()) & (low.diff() < 0), low.diff().abs(), 0
        )

        dm_plus = pd.Series(dm_plus, index=high.index)
        dm_minus = pd.Series(dm_minus, index=low.index)

        # Smooth the values
        tr_smooth = tr.rolling(window=period).mean()
        dm_plus_smooth = dm_plus.rolling(window=period).mean()
        dm_minus_smooth = dm_minus.rolling(window=period).mean()

        # Calculate DI+ and DI-
        di_plus = 100 * (dm_plus_smooth / tr_smooth)
        di_minus = 100 * (dm_minus_smooth / tr_smooth)

        # Calculate DX and ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=period).mean()

        return {"adx": adx, "di_plus": di_plus, "di_minus": di_minus}

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume."""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]

        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]

        return obv

    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price."""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap

    @staticmethod
    def fibonacci_retracement(high: float, low: float) -> Dict[str, float]:
        """Fibonacci Retracement Levels."""
        diff = high - low

        return {
            "0.0": high,
            "23.6": high - 0.236 * diff,
            "38.2": high - 0.382 * diff,
            "50.0": high - 0.5 * diff,
            "61.8": high - 0.618 * diff,
            "78.6": high - 0.786 * diff,
            "100.0": low,
        }

    @staticmethod
    def pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
        """Pivot Points."""
        pivot = (high + low + close) / 3

        return {
            "pivot": pivot,
            "r1": 2 * pivot - low,
            "r2": pivot + (high - low),
            "r3": high + 2 * (pivot - low),
            "s1": 2 * pivot - high,
            "s2": pivot - (high - low),
            "s3": low - 2 * (high - pivot),
        }

    @staticmethod
    def z_score(data: pd.Series, period: int = 20) -> pd.Series:
        """Z-Score for mean reversion strategies."""
        rolling_mean = data.rolling(window=period).mean()
        rolling_std = data.rolling(window=period).std()
        z_score = (data - rolling_mean) / rolling_std
        return z_score

    @staticmethod
    def correlation(data1: pd.Series, data2: pd.Series, period: int = 20) -> pd.Series:
        """Rolling correlation between two series."""
        return data1.rolling(window=period).corr(data2)

    @staticmethod
    def cointegration_test(data1: pd.Series, data2: pd.Series) -> Dict[str, float]:
        """Simple cointegration test for pairs trading."""
        # This is a simplified version - in production, use statsmodels
        from scipy import stats

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(data1, data2)

        # Calculate residuals
        residuals = data2 - (slope * data1 + intercept)

        # ADF test would be performed here in production
        # For now, return basic statistics
        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value**2,
            "p_value": p_value,
            "residuals_mean": residuals.mean(),
            "residuals_std": residuals.std(),
        }

    @staticmethod
    def calculate_all_indicators(
        df: pd.DataFrame,
        price_col: str = "close",
        high_col: str = "high",
        low_col: str = "low",
        volume_col: str = "volume",
    ) -> pd.DataFrame:
        """Calculate all technical indicators for a DataFrame."""
        result = df.copy()

        # Moving averages
        result["sma_20"] = TechnicalIndicators.sma(df[price_col], 20)
        result["sma_50"] = TechnicalIndicators.sma(df[price_col], 50)
        result["ema_12"] = TechnicalIndicators.ema(df[price_col], 12)
        result["ema_26"] = TechnicalIndicators.ema(df[price_col], 26)

        # Momentum indicators
        result["rsi"] = TechnicalIndicators.rsi(df[price_col])

        # MACD
        macd_data = TechnicalIndicators.macd(df[price_col])
        result["macd"] = macd_data["macd"]
        result["macd_signal"] = macd_data["signal"]
        result["macd_histogram"] = macd_data["histogram"]

        # Bollinger Bands
        bb_data = TechnicalIndicators.bollinger_bands(df[price_col])
        result["bb_upper"] = bb_data["upper"]
        result["bb_middle"] = bb_data["middle"]
        result["bb_lower"] = bb_data["lower"]

        # Volatility
        if all(col in df.columns for col in [high_col, low_col]):
            result["atr"] = TechnicalIndicators.atr(df[high_col], df[low_col], df[price_col])

        # Volume indicators
        if volume_col in df.columns:
            result["obv"] = TechnicalIndicators.obv(df[price_col], df[volume_col])
            if all(col in df.columns for col in [high_col, low_col]):
                result["vwap"] = TechnicalIndicators.vwap(
                    df[high_col], df[low_col], df[price_col], df[volume_col]
                )

        # Z-Score for mean reversion
        result["z_score"] = TechnicalIndicators.z_score(df[price_col])

        return result
