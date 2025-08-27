import os
import pandas as pd
import numpy as np

class TechnicalFeatureEngineer:
    def __init__(self, csv_path):
        self.df_raw = pd.read_csv(csv_path, parse_dates=["date"])
        self.df_long = self._melt_to_long_format()

    def _melt_to_long_format(self):
        # If already long format, skip melting
        if {"symbol", "close"}.issubset(self.df_raw.columns):
            return self.df_raw.sort_values(by=["symbol", "date"]).copy()
        
        # Otherwise assume it's wide format and melt
        df_long = self.df_raw.melt(id_vars=["date"], var_name="symbol", value_name="close")
        return df_long.sort_values(by=["symbol", "date"])
# ================================================
# Moving Averages
# ================================================
    def sma(self, windows: list):
        """Add Simple Moving Averages (SMA) for each window size in list."""
        for window in windows:
            col_name = f"sma_{window}"
            self.df_long[col_name] = (
                self.df_long.groupby("symbol")["close"]
                .transform(lambda x: x.rolling(window=window, min_periods=window).mean())
            )

    def ema(self, windows: list):
        """Add Exponential Moving Averages (EMA) for each window size in list."""
        for window in windows:
            col_name = f"ema_{window}"
            self.df_long[col_name] = (
                self.df_long.groupby("symbol")["close"]
                .transform(lambda x: x.ewm(span=window, min_periods=window, adjust=False).mean())
            )

    
# ================================================
# Trend Indicators
# ================================================
    def rsi(self, window: int):
        """
        Add RSI (Relative Strength Index) using close prices only.
        
        Formula:
            1. Change = current_close - previous_close
            2. Gain = max(change, 0), Loss = abs(min(change, 0))
            3. Avg Gain = rolling mean of Gain over `window`
            4. Avg Loss = rolling mean of Loss over `window`
            5. RS = Avg Gain / Avg Loss
            6. RSI = 100 - (100 / (1 + RS))
        """
        col_name = f"rsi_{window}"
        
        def compute_rsi(x):
            delta = x.diff()  # price change
            gain = delta.clip(lower=0)  # positive gains only
            loss = -delta.clip(upper=0)  # negative losses only (in positive form)
            avg_gain = gain.rolling(window, min_periods=window).mean()
            avg_loss = loss.rolling(window, min_periods=window).mean()
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))  # RSI calculation

        self.df_long[col_name] = (
            self.df_long.groupby("symbol")["close"]
            .transform(compute_rsi)
        )

    def macd(self, fast=12, slow=26, signal=9):
        """
        Add MACD and MACD signal line using close prices only.

        Formula:
            1. EMA_fast = EMA(close, span=fast)
            2. EMA_slow = EMA(close, span=slow)
            3. MACD = EMA_fast - EMA_slow
            4. Signal Line = EMA(MACD, span=signal)
        """
        group = self.df_long.groupby("symbol")["close"]
        
        # Fast and slow EMAs
        ema_fast = group.transform(lambda x: x.ewm(span=fast, min_periods=fast, adjust=False).mean())
        ema_slow = group.transform(lambda x: x.ewm(span=slow, min_periods=slow, adjust=False).mean())
        
        # MACD line: difference between fast and slow EMAs
        macd = ema_fast - ema_slow

        # Signal line: EMA of the MACD line
        signal_line = macd.groupby(self.df_long["symbol"]).transform(
            lambda x: x.ewm(span=signal, min_periods=signal, adjust=False).mean()
        )

        self.df_long["macd"] = macd
        self.df_long["macd_signal"] = signal_line

    def bollinger_bands(self, window: int = 20, num_std: float = 2.0):
        """
        Add Bollinger Bands (upper, lower, middle) to the DataFrame.

        Formula:
            Middle Band = SMA(window)
            Upper Band = SMA + num_std * rolling_std
            Lower Band = SMA - num_std * rolling_std
        """
        group = self.df_long.groupby("symbol")["close"]

        middle_band = group.transform(lambda x: x.rolling(window=window, min_periods=window).mean())
        rolling_std = group.transform(lambda x: x.rolling(window=window, min_periods=window).std())
        
        self.df_long[f"bb_middle_{window}"] = middle_band
        self.df_long[f"bb_upper_{window}"] = middle_band + num_std * rolling_std
        self.df_long[f"bb_lower_{window}"] = middle_band - num_std * rolling_std

# ================================================
# Momentum/Reverting Indicators
# ================================================
    def log_return(self):
        """
        Add daily log return: log(close_t / close_t-1) for each symbol.
        """
        self.df_long["log_return"] = (
            self.df_long.groupby("symbol")["close"]
            .transform(lambda x: np.log(x / x.shift(1)))
        )

    def cumulative_log_return(self, windows: list):
        """
        Add cumulative log returns over multiple recent windows.

        Formula:
            cumulative_log_return = log(close) - log(close.shift(window))
        """
        for window in windows:
            col_name = f"cumulative_{window}d_log_return"
            self.df_long[col_name] = (
                self.df_long.groupby("symbol")["close"]
                .transform(lambda x: np.log(x / x.shift(window)))
            )

    def ma_crossover_signal(self, fast: int = 5, slow: int = 60):
        """
        Add moving average crossover trading signal using pre-defined SMA logic.

        Signal:
            +1 → Buy signal (fast crosses above slow)
            -1 → Sell signal (fast crosses below slow)
            NaN → If SMA is not available (e.g., due to insufficient data)
            0 → No crossover
        """
        # Ensure SMAs are computed
        self.sma([fast, slow])

        fast_col = f"sma_{fast}"
        slow_col = f"sma_{slow}"
        signal_col = f"sma_crossover_{fast}_{slow}"

        # Compute MA difference and lagged difference by symbol
        self.df_long[signal_col] = np.nan  # Initialize with NaNs
        for symbol, group in self.df_long.groupby("symbol"):
            ma_diff = group[fast_col] - group[slow_col]
            ma_diff_prev = ma_diff.shift(1)

            signal = pd.Series(0, index=ma_diff.index)
            signal[(ma_diff > 0) & (ma_diff_prev <= 0)] = 1
            signal[(ma_diff < 0) & (ma_diff_prev >= 0)] = -1

            # Mask where either SMA is missing
            signal[group[fast_col].isna() | group[slow_col].isna()] = np.nan

            self.df_long.loc[signal.index, signal_col] = signal

# ================================================
# Volatility Indicators
# ================================================
    def price_volatility_std(self, windows: list):
        """
        Add rolling standard deviation of close prices (price volatility).
        """
        for window in windows:
            col_name = f"vol_std_{window}"
            self.df_long[col_name] = (
                self.df_long.groupby("symbol")["close"]
                .transform(lambda x: x.rolling(window=window, min_periods=window).std())
            )
    
    def log_return_volatility(self, windows: list):
        """
        Add rolling standard deviation of log returns (return volatility).
        """
        for window in windows:
            col_name = f"log_return_vol_{window}"
            self.df_long[col_name] = (
                self.df_long.groupby("symbol")["close"]
                .transform(lambda x: np.log(x / x.shift(1)))
                .groupby(self.df_long["symbol"])
                .transform(lambda x: x.rolling(window=window, min_periods=window).std())
            )

    
    def get_features(self):
        return self.df_long


if __name__ == "__main__":

    price_data = TechnicalFeatureEngineer("filled_close_price_matrix_test.csv")

    # Add features
    price_data.log_return()
    price_data.sma([5, 10, 20, 30, 60])
    price_data.ema([10, 20, 50])
    price_data.rsi(14)
    price_data.macd()
    price_data.bollinger_bands()
    price_data.cumulative_log_return([3, 5, 10, 20])
    price_data.ma_crossover_signal(5, 60)
    price_data.price_volatility_std([5])
    price_data.log_return_volatility([5, 10])

    # Get and export features
    features_df = price_data.get_features()
    features_df.to_csv("features_output_test.csv", index=False)
    print(features_df.head(50))