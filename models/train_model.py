import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import logging
from typing import Tuple, Dict, Any
import json

logger = logging.getLogger(__name__)

class MLModelTrainer:
    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialize ML model trainer
        
        Args:
            model_type: 'xgboost', 'lstm', or 'lightgbm'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for ML model
        
        Args:
            df: DataFrame with OHLCV and technical indicators
            
        Returns:
            DataFrame with selected features
        """
        # Select features for the model
        feature_cols = [
            'rsi',
            'ema20',
            'ema50',
            'atr',
            'volume_delta',
            'body_ratio',
            'upper_wick_ratio',
            'lower_wick_ratio',
            'trend_strength'
        ]
        
        # Add price change features
        df['price_change'] = df['close'].pct_change()
        df['price_change_5'] = df['close'].pct_change(5)
        
        # Add volume features
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Add volatility features
        df['volatility'] = df['close'].rolling(window=20).std()
        
        # Update feature columns
        self.feature_columns = feature_cols + [
            'price_change', 'price_change_5', 
            'volume_ratio', 'volatility'
        ]
        
        # Select only existing columns
        available_features = [col for col in self.feature_columns if col in df.columns]
        return df[available_features].copy()
    
    def create_labels(self, df: pd.DataFrame, future_periods: int = 3, threshold: float = 0.005) -> pd.Series:
        """
        Create labels for supervised learning
        
        Args:
            df: DataFrame with price data
            future_periods: Number of periods to look ahead for labeling
            threshold: Minimum price change to consider as signal (0.5%)
            
        Returns:
            Series with labels: 1 (buy), -1 (sell), 0 (hold)
        """
        # Calculate future returns
        future_return = df['close'].shift(-future_periods) / df['close'] - 1
        
        # Create labels
        labels = pd.Series(0, index=df.index)  # Default to hold
        labels[future_return > threshold] = 1   # Buy signal
        labels[future_return < -threshold] = -1 # Sell signal
        
        return labels
    
    def train_xgboost(self, X_train, y_train, X_val, y_val) -> xgb.XGBClassifier:
        """Train XGBoost model"""
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=3,  # buy, sell, hold
            random_state=42,
            n_jobs=-1
        )
        
        # Convert labels from [-1, 0, 1] to [0, 1, 2] for XGBoost
        y_train_mapped = y_train + 1
        y_val_mapped = y_val + 1
        
        model.fit(
            X_train, y_train_mapped,
            eval_set=[(X_val, y_val_mapped)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        return model
    
    def train_lstm(self, X_train, y_train, X_val, y_val, sequence_length: int = 20) -> keras.Model:
        """Train LSTM model"""
        # Reshape data for LSTM: [samples, time steps, features]
        def create_sequences(X, y, seq_length):
            xs, ys = [], []
            for i in range(len(X) - seq_length):
                xs.append(X[i:(i + seq_length)])
                ys.append(y[i + seq_length])
            return np.array(xs), np.array(ys)
        
        X_train_seq, y_train_seq = create_sequences(X_train.values, y_train.values, sequence_length)
        X_val_seq, y_val_seq = create_sequences(X_val.values, y_val.values, sequence_length)
        
        # Convert labels to categorical
        y_train_cat = keras.utils.to_categorical(y_train_seq + 1, num_classes=3)
        y_val_cat = keras.utils.to_categorical(y_val_seq + 1, num_classes=3)
        
        # Build LSTM model
        model = keras.Sequential([
            layers.LSTM(50, return_sequences=True, input_shape=(sequence_length, X_train.shape[1])),
            layers.Dropout(0.2),
            layers.LSTM(50, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(25),
            layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = model.fit(
            X_train_seq, y_train_cat,
            batch_size=32,
            epochs=50,
            validation_data=(X_val_seq, y_val_cat),
            verbose=1
        )
        
        return model
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train ML model
        
        Args:
            df: DataFrame with OHLCV data and indicators
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training {self.model_type} model...")
        
        # Prepare features
        features_df = self.prepare_features(df)
        
        # Create labels
        labels = self.create_labels(df)
        
        # Align features and labels (drop NaN values)
        combined = pd.concat([features_df, labels.rename('label')], axis=1)
        combined.dropna(inplace=True)
        
        X = combined[self.feature_columns]
        y = combined['label']
        
        # Split data using time series split to avoid lookahead bias
        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, val_index in tscv.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model based on type
        if self.model_type == 'xgboost':
            self.model = self.train_xgboost(
                X_train_scaled, y_train, X_val_scaled, y_val
            )
        elif self.model_type == 'lstm':
            self.model = self.train_lstm(
                X_train_scaled, y_train, X_val_scaled, y_val
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Evaluate model
        if self.model_type == 'xgboost':
            y_pred = self.model.predict(X_val_scaled)
            y_pred_labels = y_pred - 1  # Convert back to [-1, 0, 1]
        else:  # LSTM
            y_pred_prob = self.model.predict(X_val_scaled.reshape(-1, 1, X_val_scaled.shape[1]))
            y_pred = np.argmax(y_pred_prob, axis=1) - 1  # Convert to [-1, 0, 1]
            y_pred_labels = y_pred
        
        # Calculate metrics
        report = classification_report(y_val, y_pred_labels, output_dict=True)
        
        logger.info(f"Model training completed. Accuracy: {report['accuracy']:.4f}")
        
        return {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'classification_report': report,
            'model_type': self.model_type
        }
    
    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """
        Make prediction on new data
        
        Args:
            df: DataFrame with latest data point
            
        Returns:
            Tuple of (signal, confidence) where signal is -1, 0, 1 and confidence is 0-1
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Prepare features
        features_df = self.prepare_features(df)
        
        # Get latest row
        latest_features = features_df.iloc[-1:][self.feature_columns]
        
        # Scale features
        latest_scaled = self.scaler.transform(latest_features)
        
        # Make prediction
        if self.model_type == 'xgboost':
            pred_proba = self.model.predict_proba(latest_scaled)[0]
            pred_class = self.model.predict(latest_scaled)[0] - 1  # Convert to [-1, 0, 1]
            confidence = max(pred_proba)
        else:  # LSTM
            # Reshape for LSTM: [1, 1, features]
            lstm_input = latest_scaled.reshape((1, 1, latest_scaled.shape[1]))
            pred_proba = self.model.predict(lstm_input)[0]
            pred_class = np.argmax(pred_proba) - 1  # Convert to [-1, 0, 1]
            confidence = max(pred_proba)
        
        return int(pred_class), float(confidence)
    
    def save_model(self, filepath: str):
        """Save trained model and scaler"""
        if self.model_type == 'xgboost':
            joblib.dump(self.model, f"{filepath}_model.pkl")
        else:  # LSTM
            self.model.save(f"{filepath}_model.h5")
        
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'feature_columns': self.feature_columns
        }
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model and scaler"""
        # Load metadata
        with open(f"{filepath}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.model_type = metadata['model_type']
        self.feature_columns = metadata['feature_columns']
        
        # Load model
        if self.model_type == 'xgboost':
            self.model = joblib.load(f"{filepath}_model.pkl")
        else:  # LSTM
            self.model = keras.models.load_model(f"{filepath}_model.h5")
        
        # Load scaler
        self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        
        logger.info(f"Model loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    # This would normally load your data
    # For demonstration, we'll create sample data
    logging.basicConfig(level=logging.INFO)
    
    # Sample data creation (replace with actual data loading)
    dates = pd.date_range('2023-01-01', periods=1000, freq='5T')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 102,
        'low': np.random.randn(1000).cumsum() + 98,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(100, 1000, 1000)
    })
    df.set_index('timestamp', inplace=True)
    
    # Add technical indicators
    from indicators.technical_indicators import TechnicalIndicators
    df = TechnicalIndicators.add_all_indicators(df)
    
    # Train model
    trainer = MLModelTrainer(model_type='xgboost')
    results = trainer.train(df)
    
    # Save model
    trainer.save_model('models/xgboost_model')
    
    # Make prediction
    signal, confidence = trainer.predict(df.tail(1))
    print(f"Signal: {signal}, Confidence: {confidence}")