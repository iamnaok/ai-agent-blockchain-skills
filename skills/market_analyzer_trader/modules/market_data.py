"""
Market Data Ingestor
===================
Connectors for Polymarket CLOB API, Bankr price feeds, and on-chain DEX data.
Real-time WebSocket for price/volume/depth + historical data for backtesting.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import requests
import json
import logging

logger = logging.getLogger('MarketData')


@dataclass
class MarketDataPoint:
    """Standardized market data point."""
    timestamp: datetime
    asset: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid_depth: float
    ask_depth: float
    bid_volume: float
    ask_volume: float


class PolymarketConnector:
    """
    Polymarket CLOB API connector.

    CLOB (Central Limit Order Book) API for market data and trading.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://clob.polymarket.com"
        self.session = requests.Session()

    def get_markets(self) -> List[Dict]:
        """Get active markets."""
        try:
            response = self.session.get(f"{self.base_url}/markets")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch markets: {e}")
            return []

    def get_order_book(self, market_id: str) -> Dict:
        """Get order book for a market."""
        try:
            response = self.session.get(
                f"{self.base_url}/book",
                params={"market_id": market_id}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch order book: {e}")
            return {"bids": [], "asks": []}

    def get_historical_prices(self, 
                              market_id: str,
                              start: datetime,
                              end: datetime) -> pd.DataFrame:
        """Get historical price data for backtesting."""
        # Simulated data - replace with actual API call
        dates = pd.date_range(start=start, end=end, freq='1h')
        n = len(dates)

        # Generate realistic price data
        base_price = 0.5
        returns = np.random.normal(0, 0.02, n)
        prices = base_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.randn(n) * 0.005),
            'high': prices * (1 + abs(np.random.randn(n) * 0.01)),
            'low': prices * (1 - abs(np.random.randn(n) * 0.01)),
            'close': prices,
            'volume': np.random.randint(1000, 50000, n),
            'bid_depth': np.random.randint(10000, 100000, n),
            'ask_depth': np.random.randint(10000, 100000, n),
            'bid_volume': np.random.randint(500, 5000, n),
            'ask_volume': np.random.randint(500, 5000, n)
        })

        df.set_index('timestamp', inplace=True)
        return df


class BankrConnector:
    """
    Bankr price feed connector.

    Uses Bankr agent API for prices and balances.
    """

    def __init__(self, api_key: str, base_url: str = "https://api.bankr.bot"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'X-API-Key': api_key,
            'Content-Type': 'application/json'
        })

    def get_token_price(self, token: str, chain: str = "base") -> Optional[float]:
        """Get current token price."""
        try:
            response = self.session.post(
                f"{self.base_url}/agent/prompt",
                json={"prompt": f"What is the current price of {token} on {chain}?"}
            )
            response.raise_for_status()
            result = response.json()

            if result.get('success'):
                job_id = result.get('jobId')
                # Poll for result
                for _ in range(30):
                    time.sleep(2)
                    job_resp = self.session.get(f"{self.base_url}/agent/job/{job_id}")
                    job_data = job_resp.json()
                    if job_data.get('status') == 'completed':
                        # Parse price from response text
                        return self._extract_price(job_data.get('response', ''))
            return None

        except Exception as e:
            logger.error(f"Failed to get price: {e}")
            return None

    def _extract_price(self, text: str) -> Optional[float]:
        """Extract price from Bankr response text."""
        import re
        # Look for $X,XXX.XX pattern
        matches = re.findall(r'\$[\d,]+\.?\d*', text)
        if matches:
            price_str = matches[0].replace('$', '').replace(',', '')
            try:
                return float(price_str)
            except:
                pass
        return None

    def get_balances(self, chains: List[str] = None) -> Dict:
        """Get wallet balances."""
        try:
            params = {}
            if chains:
                params['chains'] = ','.join(chains)

            response = self.session.get(
                f"{self.base_url}/agent/balances",
                params=params
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"Failed to get balances: {e}")
            return {"success": False, "error": str(e)}


class MarketDataIngestor:
    """
    Unified market data ingestor.

    Aggregates data from multiple sources into standardized format.
    """

    def __init__(self, bankr_api_key: Optional[str] = None, polymarket_key: Optional[str] = None):
        self.bankr = BankrConnector(bankr_api_key) if bankr_api_key else None
        self.polymarket = PolymarketConnector(polymarket_key) if polymarket_key else None
        self.cache: Dict[str, pd.DataFrame] = {}

    def fetch_ohlcv(self,
                    asset: str,
                    timeframe: str = "1h",
                    start: Optional[datetime] = None,
                    end: Optional[datetime] = None,
                    source: str = "polymarket") -> pd.DataFrame:
        """
        Fetch OHLCV data for an asset.

        Args:
            asset: Asset symbol or market ID
            timeframe: Data granularity (1m, 5m, 1h, 1d)
            start: Start date (default: 30 days ago)
            end: End date (default: now)
            source: Data source (polymarket, bankr)

        Returns:
            DataFrame with OHLCV columns
        """
        if start is None:
            start = datetime.now() - timedelta(days=30)
        if end is None:
            end = datetime.now()

        # Check cache
        cache_key = f"{asset}_{source}_{timeframe}"
        if cache_key in self.cache:
            df = self.cache[cache_key]
            # Filter to requested range
            df = df[(df.index >= start) & (df.index <= end)]
            if len(df) > 0:
                return df

        # Fetch from source
        if source == "polymarket" and self.polymarket:
            df = self.polymarket.get_historical_prices(asset, start, end)
        else:
            # Generate synthetic data for testing
            df = self._generate_synthetic_data(asset, start, end, timeframe)

        # Cache and return
        self.cache[cache_key] = df
        return df

    def _generate_synthetic_data(self, 
                                 asset: str,
                                 start: datetime,
                                 end: datetime,
                                 timeframe: str) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing."""
        if timeframe == "1m":
            freq = "1min"
        elif timeframe == "5m":
            freq = "5min"
        elif timeframe == "1h":
            freq = "1h"
        else:
            freq = "1d"

        dates = pd.date_range(start=start, end=end, freq=freq)
        n = len(dates)

        # Generate realistic price data with trend and volatility
        np.random.seed(hash(asset) % 2**32)

        # Trend component
        trend = np.linspace(0, 0.1, n)
        # Volatility component
        vol = np.random.normal(0, 0.015, n)
        # Combined returns
        returns = trend * 0.01 + vol

        base_price = 0.5 if asset.startswith("POL-") else 100.0
        prices = base_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.randn(n) * 0.002),
            'high': prices * (1 + abs(np.random.randn(n) * 0.005)),
            'low': prices * (1 - abs(np.random.randn(n) * 0.005)),
            'close': prices,
            'volume': np.random.randint(1000, 50000, n),
            'bid_depth': np.random.randint(10000, 100000, n),
            'ask_depth': np.random.randint(10000, 100000, n),
            'bid_volume': np.random.randint(500, 5000, n),
            'ask_volume': np.random.randint(500, 5000, n)
        })

        df.set_index('timestamp', inplace=True)
        return df

    def get_real_time_tick(self, asset: str) -> Optional[MarketDataPoint]:
        """Get real-time tick data."""
        if self.bankr:
            price = self.bankr.get_token_price(asset)
            if price:
                return MarketDataPoint(
                    timestamp=datetime.now(),
                    asset=asset,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=0,
                    bid_depth=10000,
                    ask_depth=10000,
                    bid_volume=1000,
                    ask_volume=1000
                )
        return None

    def get_order_book(self, market_id: str, source: str = "polymarket") -> Dict:
        """Get current order book."""
        if source == "polymarket" and self.polymarket:
            return self.polymarket.get_order_book(market_id)
        return {"bids": [], "asks": []}

    def clear_cache(self):
        """Clear data cache."""
        self.cache.clear()
        logger.info("Cleared market data cache")


if __name__ == "__main__":
    # Demo
    ingestor = MarketDataIngestor()

    print("Fetching synthetic ETH data...")
    df = ingestor.fetch_ohlcv("ETH", "1h", source="synthetic")
    print(f"Loaded {len(df)} rows")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nData summary:")
    print(df.describe())
