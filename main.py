import asyncio
import logging
import signal
import sys
from datetime import datetime, time
import pandas as pd
from typing import Dict, Optional

# Import our modules
from data.kraken_websocket import KrakenWebSocket
from models.train_model import MLModelTrainer
from backtesting.backtest_engine import BacktestEngine
from execution.risk_manager import RiskManager
from indicators.technical_indicators import TechnicalIndicators

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, config: Dict = None):
        """
        Initialize the trading bot
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        self.running = False
        
        # Initialize components
        self.data_collector = KrakenWebSocket(
            symbols=self.config['symbols'],
            timeframe=self.config['timeframe']
        )
        
        self.ml_model = MLModelTrainer(model_type=self.config['model_type'])
        self.backtest_engine = BacktestEngine(
            initial_capital=self.config['initial_capital'],
            fee=self.config['fee']
        )
        self.risk_manager = RiskManager(
            max_risk_per_trade=self.config['max_risk_per_trade'],
            max_daily_drawdown=self.config['max_daily_drawdown']
        )
        
        # State variables
        self.current_data = {symbol: pd.DataFrame() for symbol in self.config['symbols']}
        self.models_trained = {symbol: False for symbol in self.config['symbols']}
        self.last_signal_time = {symbol: None for symbol in self.config['symbols']}
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _default_config(self) -> Dict:
        """Default configuration for the trading bot"""
        return {
            'symbols': ['XBT/USD', 'XAU/USD'],  # BTC/USD, XAU/USD on Kraken
            'timeframe': '5m',
            'model_type': 'xgboost',  # or 'lstm'
            'initial_capital': 10000.0,
            'fee': 0.001,  # 0.1% trading fee
            'max_risk_per_trade': 0.01,  # 1% risk per trade
            'max_daily_drawdown': 0.05,  # 5% max daily drawdown
            'min_confidence': 0.7,  # Minimum ML confidence to take signal
            'retrain_interval_hours': 24,  # Retrain model every 24 hours
            'backtest_window_days': 30,  # Days of data for backtesting
            'trade_hours': {
                'start': time(8, 0),  # 8:00 AM UTC
                'end': time(16, 0)    # 4:00 PM UTC (London/NY overlap)
            }
        }
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def _is_trading_hours(self) -> bool:
        """Check if current time is within trading hours"""
        now = datetime.utcnow().time()
        start_time = self.config['trade_hours']['start']
        end_time = self.config['trade_hours']['end']
        
        # Handle case where trading hours cross midnight (not applicable in our config)
        if start_time <= end_time:
            return start_time <= now <= end_time
        else:
            return now >= start_time or now <= end_time
    
    async def _update_data(self, symbol: str):
        """Update data for a symbol from the WebSocket collector"""
        # Get latest data from collector
        df = self.data_collector.get_dataframe(symbol)
        if not df.empty:
            self.current_data[symbol] = df
            
            # Add technical indicators
            self.current_data[symbol] = TechnicalIndicators.add_all_indicators(
                self.current_data[symbol]
            )
            
            logger.debug(f"Updated {symbol} data: {len(df)} candles")
    
    async def _train_model_if_needed(self, symbol: str):
        """Train ML model if needed based on interval"""
        if not self.models_trained[symbol]:
            logger.info(f"Training initial model for {symbol}")
            await self._train_model(symbol)
            self.models_trained[symbol] = True
        # In a production system, we'd also check time since last training
    
    async def _train_model(self, symbol: str):
        """Train ML model for a symbol"""
        try:
            df = self.current_data[symbol]
            if len(df) < 100:  # Need sufficient data
                logger.warning(f"Not enough data to train model for {symbol}")
                return
            
            # Train model
            self.ml_model = MLModelTrainer(model_type=self.config['model_type'])
            results = self.ml_model.train(df)
            
            # Save model
            model_path = f"models/{symbol.replace('/', '_')}_model"
            self.ml_model.save_model(model_path)
            
            logger.info(f"Model trained and saved for {symbol}. "
                       f"Accuracy: {results['classification_report']['accuracy']:.4f}")
                       
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
    
    async def _generate_signal(self, symbol: str) -> Optional[Dict]:
        """Generate trading signal for a symbol"""
        try:
            df = self.current_data[symbol]
            if len(df) < 50:  # Need sufficient data for indicators
                return None
            
            # Get latest data point
            latest = df.iloc[-1]
            
            # Basic strategy conditions
            ema20 = latest['ema20']
            ema50 = latest['ema50']
            rsi = latest['rsi']
            atr = latest['atr']
            
            # Determine signal direction
            signal_direction = None
            confidence = 0.0
            
            # Buy conditions
            if ema20 > ema50 and rsi > 55:
                signal_direction = 'buy'
                # Calculate basic confidence based on indicator strength
                ema_strength = (ema20 - ema50) / ema50
                rsi_strength = (rsi - 50) / 50  # Normalize RSI above 50
                confidence = min(0.9, 0.5 + (ema_strength * 10) + (rsi_strength * 0.5))
            
            # Sell conditions
            elif ema20 < ema50 and rsi < 45:
                signal_direction = 'sell'
                # Calculate basic confidence based on indicator strength
                ema_strength = (ema50 - ema20) / ema20
                rsi_strength = (50 - rsi) / 50  # Normalize RSI below 50
                confidence = min(0.9, 0.5 + (ema_strength * 10) + (rsi_strength * 0.5))
            
            # If we have a signal direction, enhance with ML
            if signal_direction and self.models_trained[symbol]:
                try:
                    # Get ML prediction
                    ml_signal, ml_confidence = self.ml_model.predict(df.tail(20))  # Use recent data
                    
                    # Convert ML signal to our format (-1=sell, 0=hold, 1=buy)
                    if ml_signal == 1 and signal_direction == 'buy':
                        # ML agrees with buy signal
                        confidence = (confidence + ml_confidence) / 2
                    elif ml_signal == -1 and signal_direction == 'sell':
                        # ML agrees with sell signal
                        confidence = (confidence + ml_confidence) / 2
                    else:
                        # ML disagrees, reduce confidence or reject signal
                        confidence *= 0.5
                        if confidence < self.config['min_confidence']:
                            signal_direction = None
                except Exception as e:
                    logger.warning(f"ML prediction failed for {symbol}: {e}")
                    # Continue with technical signal only
            
            # Check if confidence meets minimum threshold
            if signal_direction and confidence >= self.config['min_confidence']:
                # Calculate stop loss and take profit
                sl_tp = self.risk_manager.calculate_stop_loss_take_profit(
                    entry_price=latest['close'],
                    atr_value=atr,
                    direction='long' if signal_direction == 'buy' else 'short',
                    sl_multiplier=1.0,
                    tp_multiplier=2.0
                )
                
                # Calculate position size
                position_info = self.risk_manager.calculate_position_size(
                    entry_price=latest['close'],
                    stop_loss=sl_tp['stop_loss'],
                    capital=self.risk_manager.capital
                )
                
                return {
                    'symbol': symbol,
                    'direction': signal_direction,
                    'confidence': confidence,
                    'entry_price': latest['close'],
                    'stop_loss': sl_tp['stop_loss'],
                    'take_profit': sl_tp['take_profit'],
                    'position_size': position_info['position_size'],
                    'risk_amount': position_info['risk_amount'],
                    'timestamp': latest.name,
                    'indicators': {
                        'ema20': ema20,
                        'ema50': ema50,
                        'rsi': rsi,
                        'atr': atr
                    }
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    async def _execute_trade(self, signal: Dict):
        """Execute a trade based on signal (placeholder for actual execution)"""
        # In a real implementation, this would connect to exchange API
        # For now, we'll just log the signal
        logger.info(f"EXECUTING TRADE: {signal}")
        
        # Here you would integrate with exchange API (Kraken, Binance, etc.)
        # or MT5 bridge for actual order execution
        
        # Example of what real execution might look like:
        # exchange = ccxt.kraken({
        #     'apiKey': self.config['api_key'],
        #     'secret': self.config['api_secret'],
        #     'enableRateLimit': True,
        # })
        # 
        # order = exchange.create_order(
        #     symbol=signal['symbol'],
        #     type='market',
        #     side=signal['direction'],
        #     amount=signal['position_size'],
        #     params={
        #         'stop_loss': signal['stop_loss'],
        #         'take_profit': signal['take_profit']
        #     }
        # )
    
    async def run(self):
        """Main trading loop"""
        logger.info("Starting trading bot...")
        self.running = True
        
        # Start data collection in background
        data_task = asyncio.create_task(self.data_collector.connect())
        
        # Give WebSocket time to connect and gather initial data
        await asyncio.sleep(5)
        
        try:
            while self.running:
                # Check if we're in trading hours
                if not self._is_trading_hours():
                    logger.info("Outside trading hours, waiting...")
                    await asyncio.sleep(60)  # Check again in a minute
                    continue
                
                # Update data for each symbol
                for symbol in self.config['symbols']:
                    await self._update_data(symbol)
                    
                    # Train model if needed
                    await self._train_model_if_needed(symbol)
                    
                    # Generate signal
                    signal = await self._generate_signal(symbol)
                    
                    if signal:
                        logger.info(f"Signal generated for {symbol}: {signal['direction']} "
                                   f"at {signal['entry_price']} with confidence {signal['confidence']:.2f}")
                        
                        # Execute trade
                        await self._execute_trade(signal)
                        
                        # Update last signal time
                        self.last_signal_time[symbol] = datetime.utcnow()
                
                # Wait before next iteration (respect 5-minute timeframe)
                await asyncio.sleep(30)  # Check every 30 seconds
                
        except Exception as e:
            logger.error(f"Error in main trading loop: {e}")
        finally:
            # Clean up
            self.running = False
            data_task.cancel()
            try:
                await data_task
            except asyncio.CancelledError:
                pass
            logger.info("Trading bot stopped")

def main():
    """Entry point for the trading bot"""
    bot = TradingBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()