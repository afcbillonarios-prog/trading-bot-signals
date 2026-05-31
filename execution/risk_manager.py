import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, max_risk_per_trade: float = 0.01, max_daily_drawdown: float = 0.05):
        """
        Initialize risk manager
        
        Args:
            max_risk_per_trade: Maximum risk per trade as fraction of capital (0.01 = 1%)
            max_daily_drawdown: Maximum daily drawdown as fraction of capital (0.05 = 5%)
        """
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_drawdown = max_daily_drawdown
        self.daily_pnl = 0.0
        self.capital = 10000.0  # Default starting capital
        self.trade_history = []
        
    def calculate_position_size(
        self, 
        entry_price: float, 
        stop_loss: float, 
        capital: float = None,
        risk_per_trade: float = None
    ) -> Dict:
        """
        Calculate position size based on risk parameters
        
        Args:
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            capital: Available capital (uses self.capital if None)
            risk_per_trade: Risk per trade (uses self.max_risk_per_trade if None)
            
        Returns:
            Dictionary with position size details
        """
        if capital is None:
            capital = self.capital
        if risk_per_trade is None:
            risk_per_trade = self.max_risk_per_trade
        
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit <= 0:
            logger.warning("Invalid stop loss: risk per unit is zero or negative")
            return {
                'position_size': 0,
                'risk_amount': 0,
                'risk_percentage': 0,
                'max_loss': 0
            }
        
        # Calculate position size
        risk_amount = capital * risk_per_trade
        position_size = risk_amount / risk_per_unit
        
        # Calculate actual risk
        actual_risk = position_size * risk_per_unit
        actual_risk_percentage = actual_risk / capital
        
        result = {
            'position_size': position_size,
            'risk_amount': actual_risk,
            'risk_percentage': actual_risk_percentage * 100,
            'max_loss': actual_risk,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'risk_per_unit': risk_per_unit
        }
        
        logger.debug(f"Position size calculated: {result}")
        return result
    
    def calculate_stop_loss_take_profit(
        self, 
        entry_price: float, 
        atr_value: float, 
        direction: str,
        sl_multiplier: float = 1.0,
        tp_multiplier: float = 2.0
    ) -> Dict:
        """
        Calculate stop loss and take profit levels based on ATR
        
        Args:
            entry_price: Entry price
            atr_value: ATR value for volatility
            direction: 'long' or 'short'
            sl_multiplier: ATR multiplier for stop loss
            tp_multiplier: ATR multiplier for take profit
            
        Returns:
            Dictionary with SL and TP levels
        """
        if direction.lower() == 'long':
            stop_loss = entry_price - (atr_value * sl_multiplier)
            take_profit = entry_price + (atr_value * tp_multiplier)
        elif direction.lower() == 'short':
            stop_loss = entry_price + (atr_value * sl_multiplier)
            take_profit = entry_price - (atr_value * tp_multiplier)
        else:
            raise ValueError("Direction must be 'long' or 'short'")
        
        result = {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_price': entry_price,
            'atr_value': atr_value,
            'sl_multiplier': sl_multiplier,
            'tp_multiplier': tp_multiplier
        }
        
        logger.debug(f"SL/TP calculated: {result}")
        return result
    
    def check_daily_drawdown_limit(self, current_pnl: float) -> bool:
        """
        Check if daily drawdown limit has been exceeded
        
        Args:
            current_pnl: Current P&L for the day
            
        Returns:
            True if trading should be halted, False otherwise
        """
        self.daily_pnl = current_pnl
        drawdown_percentage = abs(self.daily_pnl) / self.capital if self.daily_pnl < 0 else 0
        
        if drawdown_percentage > self.max_daily_drawdown:
            logger.warning(f"Daily drawdown limit exceeded: {drawdown_percentage*100:.2f}%")
            return True
        return False
    
    def update_capital(self, pnl: float):
        """
        Update capital after a trade
        
        Args:
            pnl: Profit/loss from the trade
        """
        self.capital += pnl
        self.daily_pnl += pnl
        self.trade_history.append({
            'pnl': pnl,
            'capital': self.capital,
            'daily_pnl': self.daily_pnl
        })
        logger.info(f"Capital updated: {self.capital:.2f} (P&L: {pnl:.2f})")
    
    def reset_daily_stats(self):
        """Reset daily P&L statistics (call at start of each trading day)"""
        self.daily_pnl = 0.0
        logger.info("Daily statistics reset")
    
    def get_risk_metrics(self) -> Dict:
        """
        Get current risk metrics
        
        Returns:
            Dictionary with risk metrics
        """
        return {
            'current_capital': self.capital,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_percentage': (self.daily_pnl / self.capital) * 100,
            'max_risk_per_trade': self.max_risk_per_trade * 100,
            'max_daily_drawdown': self.max_daily_drawdown * 100,
            'total_trades': len(self.trade_history)
        }

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize risk manager
    risk_manager = RiskManager(max_risk_per_trade=0.01, max_daily_drawdown=0.05)
    
    # Example: Calculate position size for a long BTC trade
    entry_price = 60000.0
    atr_value = 1000.0  # Example ATR value
    
    # Calculate SL/TP
    sl_tp = risk_manager.calculate_stop_loss_take_profit(
        entry_price=entry_price,
        atr_value=atr_value,
        direction='long',
        sl_multiplier=1.0,
        tp_multiplier=2.0
    )
    
    print("SL/TP Levels:")
    print(f"Entry: {sl_tp['entry_price']}")
    print(f"Stop Loss: {sl_tp['stop_loss']}")
    print(f"Take Profit: {sl_tp['take_profit']}")
    
    # Calculate position size
    position_info = risk_manager.calculate_position_size(
        entry_price=entry_price,
        stop_loss=sl_tp['stop_loss'],
        capital=10000.0
    )
    
    print("\nPosition Size:")
    print(f"Position Size: {position_info['position_size']:.6f} BTC")
    print(f"Risk Amount: ${position_info['risk_amount']:.2f}")
    print(f"Risk Percentage: {position_info['risk_percentage']:.2f}%")
    
    # Check daily drawdown
    current_pnl = -300.0  # Example daily loss
    should_halt = risk_manager.check_daily_drawdown_limit(current_pnl)
    print(f"\nShould halt trading: {should_halt}")