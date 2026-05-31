import asyncio
import json
import websockets
import pandas as pd
from datetime import datetime
import logging

class KrakenWebSocket:
    def __init__(self, symbols=['XBT/USD', 'XAU/USD'], timeframe='5m'):
        self.symbols = symbols
        self.timeframe = timeframe
        self.ws_url = "wss://ws.kraken.com"
        self.data = {symbol: [] for symbol in symbols}
        self.logger = logging.getLogger(__name__)
        
    async def connect(self):
        async with websockets.connect(self.ws_url) as websocket:
            # Subscribe to ticker and ohlc for each symbol
            for symbol in self.symbols:
                # Subscribe to OHLC data for the specified timeframe
                subscribe_msg = {
                    "event": "subscribe",
                    "pair": [symbol],
                    "subscription": {
                        "name": "ohlc",
                        "interval": int(self.timeframe.replace('m', ''))  # Convert '5m' to 5
                    }
                }
                await websocket.send(json.dumps(subscribe_msg))
                self.logger.info(f"Subscribed to {symbol} OHLC data")
                
                # Also subscribe to ticker for real-time price (optional)
                ticker_sub = {
                    "event": "subscribe",
                    "pair": [symbol],
                    "subscription": {
                        "name": "ticker"
                    }
                }
                await websocket.send(json.dumps(ticker_sub))
                self.logger.info(f"Subscribed to {symbol} ticker data")
            
            # Listen for messages
            async for message in websocket:
                await self.handle_message(message, websocket)
    
    async def handle_message(self, message, websocket):
        try:
            data = json.loads(message)
            
            # Handle subscription status
            if isinstance(data, dict) and data.get('event') == 'subscriptionStatus':
                if data.get('status') == 'subscribed':
                    self.logger.info(f"Subscribed to {data.get('channel')} for {data.get('pair')}")
                elif data.get('status') == 'error':
                    self.logger.error(f"Subscription error: {data.get('errorMessage')}")
                return
            
            # Handle OHLC data
            if isinstance(data, list) and len(data) > 1 and isinstance(data[1], dict) and 'ohlc' in data[1]:
                symbol = data[3]  # Pair symbol
                ohlc_data = data[1]['ohlc']
                # Each OHLC entry is [time, open, high, low, close, vwap, volume, count]
                for ohlc in ohlc_data:
                    timestamp = datetime.fromtimestamp(ohlc[0])
                    ohlc_dict = {
                        'timestamp': timestamp,
                        'open': float(ohlc[1]),
                        'high': float(ohlc[2]),
                        'low': float(ohlc[3]),
                        'close': float(ohlc[4]),
                        'vwap': float(ohlc[5]),
                        'volume': float(ohlc[6]),
                        'count': int(ohlc[7])
                    }
                    self.data[symbol].append(ohlc_dict)
                    # Keep only last 1000 candles to avoid memory issues
                    if len(self.data[symbol]) > 1000:
                        self.data[symbol] = self.data[symbol][-1000:]
                
                self.logger.debug(f"Received OHLC data for {symbol}: {len(ohlc_data)} candles")
            
            # Handle ticker data (optional, for real-time price)
            elif isinstance(data, list) and len(data) > 1 and isinstance(data[1], dict) and 'a' in data[1]:
                symbol = data[3]
                ticker_data = data[1]
                # ticker_data contains: a[ask price, ask whole lot volume, ask lot volume],
                # b[bid price, bid whole lot volume, bid lot volume], c[last trade price, last trade volume],
                # v[volume today, volume last 24h], p[average price today, average price last 24h],
                # t[number of trades today, number of trades last 24h], l[low today, low last 24h],
                # h[high today, high last 24h], o[open price today, open price last 24h]
                self.logger.debug(f"Ticker update for {symbol}: {ticker_data.get('c', [])[0]}")
                
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
    
    def get_dataframe(self, symbol):
        """Convert stored data to pandas DataFrame"""
        if symbol not in self.data or len(self.data[symbol]) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.data[symbol])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        return df
    
    def get_latest_candle(self, symbol):
        """Get the most recent completed candle"""
        df = self.get_dataframe(symbol)
        if len(df) > 0:
            return df.iloc[-1]
        return None

# Example usage
async def main():
    logging.basicConfig(level=logging.INFO)
    ws = KrakenWebSocket(['XBT/USD', 'XAU/USD'], '5m')
    await ws.connect()

if __name__ == "__main__":
    asyncio.run(main())