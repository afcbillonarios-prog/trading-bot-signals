import pandas as pd
import numpy as np
import ta
from typing import Dict, Tuple

class TechnicalIndicators:
    @staticmethod
    def calculate_ema(df: pd.DataFrame, window: int, column: str = 'close') -> pd.Series:
        """Calculate Exponential Moving Average"""
        return ta.trend.ema_indicator(df[column], window=window)
    
    @staticmethod
    def calculate_rsi(df: pd.DataFrame, window: int = 14, column: str = 'close') -> pd.Series:
        """Calculate Relative Strength Index"""
        return ta.momentum.rsi(df[column], window=window)
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        return ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=window)
    
    @staticmethod
    def calculate_volume_delta(df: pd.DataFrame) -> pd.Series:
        """Calculate volume delta (buy volume - sell volume approximation)"""
        # Simple approximation: if close > open, volume is buying pressure
        # if close < open, volume is selling pressure
        df = df.copy()
        df['volume_delta'] = np.where(df['close'] > df['open'], df['volume'], -df['volume'])
        return df['volume_delta']
    
    @staticmethod
    def calculate_candle_body_ratio(df: pd.DataFrame) -> pd.Series:
        """Calculate candle body to total range ratio"""
        df = df.copy()
        body = abs(df['close'] - df['open'])
        total_range = df['high'] - df['low']
        # Avoid division by zero
        df['body_ratio'] = np.where(total_range > 0, body / total_range, 0)
        return df['body_ratio']
    
    @staticmethod
    def calculate_wick_ratio(df: pd.DataFrame) -> pd.Series:
        """Calculate upper and lower wick ratio"""
        df = df.copy()
        body_top = np.maximum(df['open'], df['close'])
        body_bottom = np.minimum(df['open'], df['close'])
        
        upper_wick = df['high'] - body_top
        lower_wick = body_bottom - df['low']
        total_range = df['high'] - df['low']
        
        # Avoid division by zero
        df['upper_wick_ratio'] = np.where(total_range > 0, upper_wick / total_range, 0)
        df['lower_wick_ratio'] = np.where(total_range > 0, lower_wick / total_range, 0)
        
        return df['upper_wick_ratio'], df['lower_wick_ratio']
    
    @staticmethod
    def calculate_trend_strength(df: pd.DataFrame, ema_short: int = 20, ema_long: int = 50) -> pd.Series:
        """Calculate trend strength based on EMA crossover and distance"""
        df = df.copy()
        ema_short_series = TechnicalIndicators.calculate_ema(df, ema_short)
        ema_long_series = TechnicalIndicators.calculate_ema(df, ema_long)
        
        # Normalized distance between EMAs
        df['ema_distance'] = (ema_short_series - ema_long_series) / ema_long_series
        # Trend strength is the absolute distance, signed by direction
        df['trend_strength'] = df['ema_distance']
        return df['trend_strength']
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to dataframe"""
        df = df.copy()
        
        # EMA
        df['ema20'] = TechnicalIndicators.calculate_ema(df, 20)
        df['ema50'] = TechnicalIndicators.calculate_ema(df, 50)
        
        # RSI
        df['rsi'] = TechnicalIndicators.calculate_rsi(df, 14)
        
        # ATR
        df['atr'] = TechnicalIndicators.calculate_atr(df, 14)
        
        # Volume delta
        df['volume_delta'] = TechnicalIndicators.calculate_volume_delta(df)
        
        # Candle body ratio
        df['body_ratio'] = TechnicalIndicators.calculate_candle_body_ratio(df)
        
        # Wick ratios
        df['upper_wick_ratio'], df['lower_wick_ratio'] = TechnicalIndicators.calculate_wick_ratio(df)
        
        # Trend strength
        df['trend_strength'] = TechnicalIndicators.calculate_trend_strength(df)
        
        return df

# Example usage
if __name__ == "__main__":
    # Sample data for testing
    data = {
        'open': [100, 102, 101, 103, 105],
        'high': [105, 106, 104, 107, 108],
        'low': [99, 101, 100, 102, 104],
        'close': [103, 104, 102, 106, 107],
        'volume': [1000, 1200, 900, 1100, 1300]
    }
    df = pd.DataFrame(data)
    df_with_indicators = TechnicalIndicators.add_all_indicators(df)
    print(df_with_indicators)