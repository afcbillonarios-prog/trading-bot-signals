import logging
import requests
import json
from typing import Dict, Optional
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class TelegramNotifier:
    def __init__(self, bot_token: str = None, chat_id: str = None):
        """
        Initialize Telegram notifier
        
        Args:
            bot_token: Telegram bot token (can also be set via TELEGRAM_BOT_TOKEN env var)
            chat_id: Telegram chat ID (can also be set via TELEGRAM_CHAT_ID env var)
        """
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        
        if not self.bot_token:
            logger.warning("Telegram bot token not provided. Notifications will be disabled.")
        if not self.chat_id:
            logger.warning("Telegram chat ID not provided. Notifications will be disabled.")
        
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}" if self.bot_token else None
    
    def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Send a message to Telegram chat
        
        Args:
            message: Message text to send
            parse_mode: Parse mode for message formatting (HTML or Markdown)
            
        Returns:
            True if message sent successfully, False otherwise
        """
        if not self.bot_token or not self.chat_id:
            logger.debug("Telegram credentials not configured, skipping notification")
            return False
        
        url = f"{self.base_url}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': parse_mode
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            if result.get('ok'):
                logger.info("Telegram notification sent successfully")
                return True
            else:
                logger.error(f"Telegram API error: {result.get('description')}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending Telegram notification: {e}")
            return False
    
    def send_trading_signal(self, signal: Dict) -> bool:
        """
        Send a formatted trading signal notification
        
        Args:
            signal: Dictionary containing signal information
            
        Returns:
            True if notification sent successfully, False otherwise
        """
        # Format the signal as a readable message
        direction_emoji = "🟢" if signal['direction'] == 'buy' else "🔴"
        direction_text = "BUY" if signal['direction'] == 'buy' else "SELL"
        
        message = f"""
{direction_emoji} <b>{direction_text} SIGNAL</b> {direction_emoji}

<b>Symbol:</b> {signal['symbol']}
<b>Direction:</b> {direction_text}
<b>Confidence:</b> {signal['confidence']:.2%}
<b>Entry Price:</b> ${signal['entry_price']:,.2f}
<b>Stop Loss:</b> ${signal['stop_loss']:,.2f}
<b>Take Profit:</b> ${signal['take_profit']:,.2f}
<b>Position Size:</b> {signal['position_size']:.6f}
<b>Risk Amount:</b> ${signal['risk_amount']:,.2f}

<i>Time:</i> {signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC')}

<b>Indicators:</b>
• EMA20: ${signal['indicators']['ema20']:,.2f}
• EMA50: ${signal['indicators']['ema50']:,.2f}
• RSI: {signal['indicators']['rsi']:.2f}
• ATR: ${signal['indicators']['atr']:,.2f}
        """.strip()
        
        return self.send_message(message)
    
    def send_performance_update(self, performance_data: Dict) -> bool:
        """
        Send a performance update notification
        
        Args:
            performance_data: Dictionary containing performance metrics
            
        Returns:
            True if notification sent successfully, False otherwise
        """
        message = f"""
📊 <b>TRADING BOT PERFORMANCE UPDATE</b> 📊

<b>Period:</b> {performance_data.get('period', 'Daily')}
<b>Total Trades:</b> {performance_data.get('total_trades', 0)}
<b>Win Rate:</b> {performance_data.get('win_rate', 0):.2%}
<b>Profit Factor:</b> {performance_data.get('profit_factor', 0):.2f}
<b>Total Return:</b> {performance_data.get('total_return', 0):.2%}
<b>Max Drawdown:</b> {performance_data.get('max_drawdown', 0):.2%}
<b>Sharpe Ratio:</b> {performance_data.get('sharpe_ratio', 0):.2f}

<b>Current Capital:</b> ${performance_data.get('current_capital', 0):,.2f}
<b>Daily P&L:</b> ${performance_data.get('daily_pnl', 0):,.2f}
        """.strip()
        
        return self.send_message(message)
    
    def send_error_alert(self, error_message: str) -> bool:
        """
        Send an error alert notification
        
        Args:
            error_message: Error message to send
            
        Returns:
            True if notification sent successfully, False otherwise
        """
        message = f"""
🚨 <b>TRADING BOT ERROR ALERT</b> 🚨

<b>Error:</b> {error_message}
<b>Time:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}

Please check the bot logs for more details.
        """.strip()
        
        return self.send_message(message)
    
    def send_startup_notification(self) -> bool:
        """
        Send a notification when the bot starts up
        
        Returns:
            True if notification sent successfully, False otherwise
        """
        message = f"""
🚀 <b>TRADING BOT STARTED</b> 🚀

The trading bot is now online and monitoring markets.

<b>Symbols:</b> {', '.join(['XBT/USD', 'XAU/USD'])}
<b>Timeframe:</b> 5 minutes
<b>Strategy:</b> EMA + RSI + ML Filter
<b>Risk:</b> 1% per trade

Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
        """.strip()
        
        return self.send_message(message)
    
    def send_shutdown_notification(self) -> bool:
        """
        Send a notification when the bot shuts down
        
        Returns:
            True if notification sent successfully, False otherwise
        """
        message = f"""
🛑 <b>TRADING BOT STOPPED</b> 🛑

The trading bot has been shut down.

Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
        """.strip()
        
        return self.send_message(message)

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize notifier (will use environment variables if not provided)
    notifier = TelegramNotifier()
    
    # Send a test message
    test_signal = {
        'symbol': 'XBT/USD',
        'direction': 'buy',
        'confidence': 0.75,
        'entry_price': 60000.00,
        'stop_loss': 59000.00,
        'take_profit': 62000.00,
        'position_size': 0.01,
        'risk_amount': 100.00,
        'timestamp': datetime.utcnow(),
        'indicators': {
            'ema20': 60500.00,
            'ema50': 60000.00,
            'rsi': 65.5,
            'atr': 1000.00
        }
    }
    
    # Only send if credentials are configured
    if notifier.bot_token and notifier.chat_id:
        notifier.send_trading_signal(test_signal)
    else:
        print("Telegram credentials not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables.")
        print("Example:")
        print("export TELEGRAM_BOT_TOKEN='123456789:ABCDEFghijklmnopqrstuvwxyz'")
        print("export TELEGRAM_CHAT_ID='987654321'")