import vectorbt as vbt
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, initial_capital: float = 10000, fee: float = 0.001):
        """
        Initialize backtesting engine
        
        Args:
            initial_capital: Starting capital in USD
            fee: Trading fee as percentage (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.fee = fee
        
    def generate_signals(
        self, 
        data: pd.DataFrame, 
        strategy_params: Dict = None
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate trading signals based on strategy
        
        Args:
            data: DataFrame with OHLCV data and indicators
            strategy_params: Strategy parameters
            
        Returns:
            Tuple of (entries, exits) as boolean Series
        """
        if strategy_params is None:
            strategy_params = {
                'ema_fast': 20,
                'ema_slow': 50,
                'rsi_buy': 55,
                'rsi_sell': 45,
                'min_confidence': 0.7
            }
        
        # Calculate indicators if not present
        if 'ema20' not in data.columns:
            from indicators.technical_indicators import TechnicalIndicators
            data = TechnicalIndicators.add_all_indicators(data)
        
        # Generate basic signals based on EMA and RSI
        ema_fast = strategy_params['ema_fast']
        ema_slow = strategy_params['ema_slow']
        rsi_buy = strategy_params['rsi_buy']
        rsi_sell = strategy_params['rsi_sell']
        
        # Buy conditions: EMA fast > EMA slow AND RSI > threshold
        buy_condition = (
            (data[f'ema{ema_fast}'] > data[f'ema{ema_slow}']) &
            (data['rsi'] > rsi_buy)
        )
        
        # Sell conditions: EMA fast < EMA slow AND RSI < threshold
        sell_condition = (
            (data[f'ema{ema_fast}'] < data[f'ema{ema_slow}']) &
            (data['rsi'] < rsi_sell)
        )
        
        return buy_condition, sell_condition
    
    def run_backtest(
        self, 
        data: pd.DataFrame, 
        initial_capital: float = None,
        fee: float = None,
        strategy_params: Dict = None,
        sl_atr_multiplier: float = 1.0,
        tp_atr_multiplier: float = 2.0
    ) -> Dict:
        """
        Run backtest using vectorbt
        
        Args:
            data: DataFrame with OHLCV data
            initial_capital: Starting capital
            fee: Trading fee
            strategy_params: Strategy parameters
            sl_atr_multiplier: Stop loss ATR multiplier
            tp_atr_multiplier: Take profit ATR multiplier
            
        Returns:
            Dictionary with backtest results
        """
        if initial_capital is None:
            initial_capital = self.initial_capital
        if fee is None:
            fee = self.fee
            
        # Generate signals
        entries, exits = self.generate_signals(data, strategy_params)
        
        # Calculate ATR for stop loss and take profit
        if 'atr' not in data.columns:
            from indicators.technical_indicators import TechnicalIndicators
            data = TechnicalIndicators.add_all_indicators(data)
        
        # For simplicity in this example, we'll use fixed SL/TP based on ATR
        # In a more sophisticated implementation, we'd use dynamic SL/TP
        sl_points = data['atr'] * sl_atr_multiplier
        tp_points = data['atr'] * tp_atr_multiplier
        
        # Create portfolio using vectorbt
        pf = vbt.Portfolio.from_signals(
            data['close'],
            entries=entries,
            exits=exits,
            init_cash=initial_capital,
            fees=fee,
            sl_stop=sl_points,  # Stop loss in price points
            tp_stop=tp_points,  # Take profit in price points
            freq='5T'  # 5 minutes frequency
        )
        
        # Calculate performance metrics
        stats = pf.stats()
        
        # Extract key metrics
        results = {
            'total_return': stats['Total Return [%]'],
            'annual_return': stats['Annualized Return [%]'],
            'max_drawdown': stats['Max Drawdown [%]'],
            'win_rate': stats['Win Rate [%]'],
            'profit_factor': stats['Profit Factor'],
            'sharpe_ratio': stats['Sharpe Ratio'],
            'sortino_ratio': stats['Sortino Ratio'],
            'calmar_ratio': stats['Calmar Ratio'],
            'total_trades': stats['Total Trades'],
            'win_trades': stats['Winning Trades'],
            'loss_trades': stats['Losing Trades'],
            'avg_win': stats['Avg Win [%]'],
            'avg_loss': stats['Avg Loss [%]'],
            'expectancy': stats['Expectancy [%]'],
            'portfolio': pf
        }
        
        logger.info(f"Backtest completed: {results['total_trades']} trades, "
                   f"{results['win_rate']:.2f}% win rate, "
                   f"{results['total_return']:.2f}% total return")
        
        return results
    
    def walk_forward_analysis(
        self,
        data: pd.DataFrame,
        window_size: int = 1000,  # Number of periods for training/test window
        step_size: int = 200,     # Number of periods to step forward
        **kwargs
    ) -> Dict:
        """
        Perform walk-forward analysis to test strategy robustness
        
        Args:
            data: DataFrame with OHLCV data
            window_size: Size of each window for testing
            step_size: Step size for moving window
            **kwargs: Additional arguments to pass to run_backtest
            
        Returns:
            Dictionary with walk-forward results
        """
        results_list = []
        
        # Iterate through data with walk-forward windows
        for start_idx in range(0, len(data) - window_size, step_size):
            end_idx = start_idx + window_size
            window_data = data.iloc[start_idx:end_idx]
            
            # Run backtest on this window
            window_result = self.run_backtest(window_data, **kwargs)
            window_result['window_start'] = start_idx
            window_result['window_end'] = end_idx
            results_list.append(window_result)
        
        # Aggregate results
        if results_list:
            agg_results = {
                'avg_total_return': np.mean([r['total_return'] for r in results_list]),
                'avg_annual_return': np.mean([r['annual_return'] for r in results_list]),
                'avg_max_drawdown': np.mean([r['max_drawdown'] for r in results_list]),
                'avg_win_rate': np.mean([r['win_rate'] for r in results_list]),
                'avg_profit_factor': np.mean([r['profit_factor'] for r in results_list]),
                'avg_sharpe_ratio': np.mean([r['sharpe_ratio'] for r in results_list]),
                'total_windows': len(results_list),
                'results_per_window': results_list
            }
            
            logger.info(f"Walk-forward analysis completed over {len(results_list)} windows")
            return agg_results
        else:
            logger.warning("No windows processed in walk-forward analysis")
            return {}

# Example usage
if __name__ == "__main__":
    # This would normally load your data
    # For demonstration, we'll create sample data
    import yfinance as yf
    
    logging.basicConfig(level=logging.INFO)
    
    # Download sample data (replace with your data source)
    try:
        btc_data = yf.download('BTC-USD', start='2023-01-01', end='2023-12-31', interval='5m')
        btc_data.columns = [col.lower() for col in btc_data.columns]  # Standardize column names
        
        # Run backtest
        engine = BacktestEngine(initial_capital=10000, fee=0.001)
        results = engine.run_backtest(btc_data)
        
        print("Backtest Results:")
        for key, value in results.items():
            if key != 'portfolio':  # Skip the portfolio object
                print(f"{key}: {value}")
                
    except Exception as e:
        logger.error(f"Error in backtesting example: {e}")
        print("Please install yfinance or provide your own data for testing")