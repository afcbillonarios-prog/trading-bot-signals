import json
import os
from datetime import datetime, timedelta
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class SignalLogger:
    def __init__(self, signals_file: str = "signals.json", max_signals: int = 50):
        """
        Initialize signal logger
        
        Args:
            signals_file: Path to signals JSON file
            max_signals: Maximum number of signals to keep in file
        """
        self.signals_file = signals_file
        self.max_signals = max_signals
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Ensure signals file exists with empty array"""
        if not os.path.exists(self.signals_file):
            with open(self.signals_file, 'w') as f:
                json.dump({
                    "last_updated": datetime.utcnow().isoformat() + "Z",
                    "signals": []
                }, f, indent=2)
            logger.info(f"Created signals file: {self.signals_file}")
    
    def _load_signals(self) -> Dict:
        """Load signals from JSON file"""
        try:
            with open(self.signals_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load signals file: {e}. Creating new one.")
            self._ensure_file_exists()
            return self._load_signals()
    
    def _save_signals(self, data: Dict):
        """Save signals to JSON file"""
        try:
            with open(self.signals_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved signals to {self.signals_file}")
        except Exception as e:
            logger.error(f"Error saving signals file: {e}")
    
    def add_signal(self, signal: Dict):
        """
        Add a new signal to the signals file
        
        Args:
            signal: Signal dictionary to add
        """
        data = self._load_signals()
        
        # Add timestamp if not present
        if 'timestamp' not in signal:
            signal['timestamp'] = datetime.utcnow().isoformat() + "Z"
        
        # Ensure signal has required fields for display
        required_fields = ['id', 'symbol', 'direction', 'confidence', 'entry_price', 
                          'stop_loss', 'take_profit', 'position_size', 'risk_amount']
        
        # Generate ID if not present
        if 'id' not in signal:
            signal['id'] = int(datetime.utcnow().timestamp() * 1000)
        
        # Add signal to beginning of list (most recent first)
        data['signals'].insert(0, signal)
        
        # Keep only the most recent signals
        if len(data['signals']) > self.max_signals:
            data['signals'] = data['signals'][:self.max_signals]
        
        # Update timestamp
        data['last_updated'] = datetime.utcnow().isoformat() + "Z"
        
        # Save to file
        self._save_signals(data)
        logger.info(f"Added new signal for {signal['symbol']} ({signal['direction']})")
    
    def get_recent_signals(self, limit: int = 10) -> List[Dict]:
        """
        Get recent signals
        
        Args:
            limit: Maximum number of signals to return
            
        Returns:
            List of signal dictionaries
        """
        data = self._load_signals()
        return data['signals'][:limit]

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    logger = SignalLogger()
    
    # Test signal
    test_signal = {
        'symbol': 'XBT/USD',
        'direction': 'BUY',
        'confidence': 0.75,
        'entry_price': 60250.50,
        'stop_loss': 59250.50,
        'take_profit': 62250.50,
        'position_size': 0.0166,
        'risk_amount': 100.00,
        'indicators': {
            'ema20': 60300.25,
            'ema50': 60100.75,
            'rsi': 62.3,
            'atr': 1000.0
        }
    }
    
    logger.add_signal(test_signal)
    print("Test signal added to signals.json")