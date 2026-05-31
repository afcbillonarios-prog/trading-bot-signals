import sqlalchemy as sa
from sqlalchemy import create_column, Table, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

Base = declarative_base()

class OHLCVData(Base):
    """OHLCV data table for storing historical price data"""
    __tablename__ = 'ohlcv_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    # Technical indicators (optional, can be computed on the fly)
    ema20 = Column(Float)
    ema50 = Column(Float)
    rsi = Column(Float)
    atr = Column(Float)
    
    def __repr__(self):
        return f"<OHLCVData(symbol='{self.symbol}', timestamp='{self.timestamp}', close={self.close})>"

class TradingSignals(Base):
    """Table for storing generated trading signals"""
    __tablename__ = 'trading_signals'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    direction = Column(String(10), nullable=False)  # 'buy' or 'sell'
    confidence = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    take_profit = Column(Float, nullable=False)
    position_size = Column(Float, nullable=False)
    risk_amount = Column(Float, nullable=False)
    executed = Column(Boolean, default=False)
    profit_loss = Column(Float)  # Updated after trade closes
    
    def __repr__(self):
        return f"<TradingSignal(symbol='{self.symbol}', timestamp='{self.timestamp}', direction='{self.direction}')>"

class ModelPerformance(Base):
    """Table for storing ML model performance metrics"""
    __tablename__ = 'model_performance'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    model_type = Column(String(20), nullable=False)  # 'xgboost', 'lstm', etc.
    timestamp = Column(DateTime, nullable=False, index=True)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    training_samples = Column(Integer)
    
    def __repr__(self):
        return f"<ModelPerformance(symbol='{self.symbol}', model_type='{self.model_type}', accuracy={self.accuracy})>"

class DatabaseManager:
    def __init__(self, database_url: str = None):
        """
        Initialize database manager
        
        Args:
            database_url: SQLAlchemy database URL (e.g., 'postgresql://user:pass@localhost/dbname')
        """
        if database_url is None:
            # Default to local PostgreSQL instance
            database_url = os.getenv(
                'DATABASE_URL', 
                'postgresql://postgres:password@localhost:5432/trading_bot'
            )
        
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        logger.info(f"Database manager initialized with URL: {database_url}")
    
    def create_tables(self):
        """Create all tables in the database"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    def get_session(self):
        """Get a new database session"""
        return self.SessionLocal()
    
    def save_ohlcv_data(self, symbol: str, df):
        """
        Save OHLCV data to database
        
        Args:
            symbol: Trading symbol (e.g., 'XBT/USD')
            df: DataFrame with OHLCV data (index should be timestamp)
        """
        session = self.get_session()
        try:
            # Reset index to get timestamp as column
            df_reset = df.reset_index()
            
            # Convert to list of dictionaries for bulk insert
            records = []
            for _, row in df_reset.iterrows():
                record = {
                    'symbol': symbol,
                    'timestamp': row['timestamp'],
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume'])
                }
                
                # Add technical indicators if available
                if 'ema20' in row:
                    record['ema20'] = float(row['ema20']) if not pd.isna(row['ema20']) else None
                if 'ema50' in row:
                    record['ema50'] = float(row['ema50']) if not pd.isna(row['ema50']) else None
                if 'rsi' in row:
                    record['rsi'] = float(row['rsi']) if not pd.isna(row['rsi']) else None
                if 'atr' in row:
                    record['atr'] = float(row['atr']) if not pd.isna(row['atr']) else None
                
                records.append(record)
            
            # Bulk insert
            session.execute(OHLCVData.__table__.insert(), records)
            session.commit()
            logger.info(f"Saved {len(records)} OHLCV records for {symbol}")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving OHLCV data for {symbol}: {e}")
            raise
        finally:
            session.close()
    
    def get_ohlcv_data(self, symbol: str, limit: int = None) -> pd.DataFrame:
        """
        Retrieve OHLCV data from database
        
        Args:
            symbol: Trading symbol
            limit: Maximum number of records to retrieve (None for all)
            
        Returns:
            DataFrame with OHLCV data
        """
        session = self.get_session()
        try:
            query = session.query(OHLCVData).filter(OHLCVData.symbol == symbol)
            
            if limit:
                query = query.order_by(OHLCVData.timestamp.desc()).limit(limit)
            else:
                query = query.order_by(OHLCVData.timestamp.asc())
            
            results = query.all()
            
            if not results:
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for row in results:
                data.append({
                    'timestamp': row.timestamp,
                    'open': row.open,
                    'high': row.high,
                    'low': row.low,
                    'close': row.close,
                    'volume': row.volume,
                    'ema20': row.ema20,
                    'ema50': row.ema50,
                    'rsi': row.rsi,
                    'atr': row.atr
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving OHLCV data for {symbol}: {e}")
            raise
        finally:
            session.close()
    
    def save_trading_signal(self, signal_data: dict):
        """
        Save a trading signal to database
        
        Args:
            signal_data: Dictionary containing signal information
        """
        session = self.get_session()
        try:
            signal = TradingSignals(
                symbol=signal_data['symbol'],
                timestamp=signal_data['timestamp'],
                direction=signal_data['direction'],
                confidence=signal_data['confidence'],
                entry_price=signal_data['entry_price'],
                stop_loss=signal_data['stop_loss'],
                take_profit=signal_data['take_profit'],
                position_size=signal_data['position_size'],
                risk_amount=signal_data['risk_amount']
            )
            
            session.add(signal)
            session.commit()
            logger.info(f"Saved trading signal for {signal_data['symbol']}")
            return signal.id
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving trading signal: {e}")
            raise
        finally:
            session.close()
    
    def save_model_performance(self, symbol: str, model_type: str, metrics: dict):
        """
        Save model performance metrics to database
        
        Args:
            symbol: Trading symbol
            model_type: Type of model ('xgboost', 'lstm', etc.)
            metrics: Dictionary with performance metrics
        """
        session = self.get_session()
        try:
            performance = ModelPerformance(
                symbol=symbol,
                model_type=model_type,
                timestamp=datetime.utcnow(),
                accuracy=metrics.get('accuracy'),
                precision=metrics.get('precision'),
                recall=metrics.get('recall'),
                f1_score=metrics.get('f1_score'),
                training_samples=metrics.get('training_samples')
            )
            
            session.add(performance)
            session.commit()
            logger.info(f"Saved model performance for {symbol} ({model_type})")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving model performance: {e}")
            raise
        finally:
            session.close()

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize database manager
    db_manager = DatabaseManager()
    
    # Create tables
    db_manager.create_tables()
    
    # Example: Save some sample data
    # In practice, this would come from your data collector
    import pandas as pd
    import numpy as np
    
    # Sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='5T')
    sample_data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(100, 1000, 100),
        'ema20': np.random.randn(100).cumsum() + 100,
        'ema50': np.random.randn(100).cumsum() + 100,
        'rsi': np.random.rand(100) * 100,
        'atr': np.random.rand(100) * 2 + 1
    }, index=dates)
    
    # Save sample data
    db_manager.save_ohlcv_data('XBT/USD', sample_data)
    
    # Retrieve data
    retrieved_data = db_manager.get_ohlcv_data('XBT/USD', limit=10)
    print("Retrieved data:")
    print(retrieved_data.head())