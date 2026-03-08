"""
Hyperliquid Connector
=====================
Live DEX data connector for Hyperliquid perpetuals.
Fetches price, orderbook, trades, funding, and position data.

API Docs: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api
"""

import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import requests
import logging

logger = logging.getLogger('Hyperliquid')


@dataclass
class HLMarketData:
    """Standardized Hyperliquid market data."""
    asset: str
    timestamp: datetime
    mark_price: float
    index_price: float
    oracle_price: float
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    funding_rate: float
    open_interest: float
    volume_24h: float
    price_change_24h: float
    high_24h: float
    low_24h: float


@dataclass
class HLOrderBook:
    """Hyperliquid L2 order book."""
    asset: str
    timestamp: datetime
    bids: List[Tuple[float, float]]  # (price, size)
    asks: List[Tuple[float, float]]  # (price, size)
    spread: float
    mid_price: float


@dataclass
class HLTrade:
    """Recent trade."""
    timestamp: datetime
    price: float
    size: float
    side: str  # 'buy' or 'sell'


class HyperliquidConnector:
    """
    Hyperliquid DEX API connector.

    Provides:
    - Real-time price data
    - Order book depth
    - Recent trades
    - Funding rates
    - Liquidation data
    """

    BASE_URL = "https://api.hyperliquid.xyz"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })

    def _post(self, endpoint: str, payload: Dict) -> Dict:
        """POST request to Hyperliquid API."""
        url = f"{self.BASE_URL}{endpoint}"
        try:
            response = self.session.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Hyperliquid API error: {e}")
            return {}

    def get_all_mids(self) -> Dict[str, float]:
        """Get mid prices for all assets."""
        payload = {"type": "allMids"}
        result = self._post("/info", payload)
        return result if isinstance(result, dict) else {}

    def get_market_data(self, asset: str) -> Optional[HLMarketData]:
        """
        Get comprehensive market data for an asset.

        Args:
            asset: Asset symbol (e.g., 'BTC', 'ETH', 'SOL')

        Returns:
            HLMarketData or None if error
        """
        # Get meta and asset context
        payload = {"type": "metaAndAssetCtxs"}
        result = self._post("/info", payload)

        if not result or len(result) < 2:
            return None

        meta = result[0]
        asset_ctxs = result[1]

        # Find asset index
        try:
            asset_idx = meta['universe'].index({'name': asset, 'maxLeverage': 50})
        except (ValueError, KeyError):
            # Try case insensitive
            for i, a in enumerate(meta.get('universe', [])):
                if a.get('name', '').upper() == asset.upper():
                    asset_idx = i
                    break
            else:
                logger.error(f"Asset {asset} not found in Hyperliquid markets")
                return None

        # Get asset context
        ctx = asset_ctxs[asset_idx]

        # Get orderbook for bid/ask
        ob = self.get_orderbook(asset, depth=1)

        return HLMarketData(
            asset=asset,
            timestamp=datetime.utcnow(),
            mark_price=float(ctx.get('markPx', 0)),
            index_price=float(ctx.get('indexPx', 0)),
            oracle_price=float(ctx.get('oraclePx', ctx.get('markPx', 0))),
            bid=ob.bids[0][0] if ob and ob.bids else float(ctx.get('markPx', 0)),
            ask=ob.asks[0][0] if ob and ob.asks else float(ctx.get('markPx', 0)),
            bid_size=ob.bids[0][1] if ob and ob.bids else 0,
            ask_size=ob.asks[0][1] if ob and ob.asks else 0,
            funding_rate=float(ctx.get('funding', 0)),
            open_interest=float(ctx.get('openInterest', 0)),
            volume_24h=float(ctx.get('dayNtlVlm', 0)),
            price_change_24h=float(ctx.get('dayReturn', 0)),
            high_24h=float(ctx.get('highPx24hr', 0)),
            low_24h=float(ctx.get('lowPx24hr', 0))
        )

    def get_orderbook(self, asset: str, depth: int = 20) -> Optional[HLOrderBook]:
        """
        Get L2 order book for an asset.

        Args:
            asset: Asset symbol
            depth: Number of levels to fetch (max 100)

        Returns:
            HLOrderBook or None
        """
        payload = {
            "type": "l2Book",
            "coin": asset
        }
        result = self._post("/info", payload)

        if not result or 'levels' not in result:
            return None

        levels = result['levels']
        if len(levels) < 2:
            return None

        bids_raw = levels[0]
        asks_raw = levels[1]

        # Parse (px, sz) tuples
        bids = [(float(b['px']), float(b['sz'])) for b in bids_raw[:depth]]
        asks = [(float(a['px']), float(a['sz'])) for a in asks_raw[:depth]]

        if not bids or not asks:
            return None

        mid = (bids[0][0] + asks[0][0]) / 2
        spread = asks[0][0] - bids[0][0]

        return HLOrderBook(
            asset=asset,
            timestamp=datetime.utcnow(),
            bids=bids,
            asks=asks,
            spread=spread,
            mid_price=mid
        )

    def get_recent_trades(self, asset: str, limit: int = 100) -> List[HLTrade]:
        """
        Get recent trades for an asset.

        Args:
            asset: Asset symbol
            limit: Number of trades (max 1000)

        Returns:
            List of HLTrade
        """
        payload = {
            "type": "recentTrades",
            "coin": asset
        }
        result = self._post("/info", payload)

        if not result or not isinstance(result, list):
            return []

        trades = []
        for t in result[:limit]:
            trades.append(HLTrade(
                timestamp=datetime.fromtimestamp(float(t.get('time', 0)) / 1000),
                price=float(t.get('px', 0)),
                size=float(t.get('sz', 0)),
                side='buy' if t.get('side') == 'B' else 'sell'
            ))

        return trades

    def get_funding_rate(self, asset: str) -> Dict:
        """Get current and predicted funding rate."""
        market_data = self.get_market_data(asset)
        if not market_data:
            return {}

        return {
            'asset': asset,
            'current_funding': market_data.funding_rate,
            'annualized': market_data.funding_rate * 365 * 3,  # 8-hour periods
            'timestamp': market_data.timestamp
        }

    def get_liquidations(self, asset: str, limit: int = 100) -> List[Dict]:
        """Get recent liquidations."""
        # Note: Hyperliquid doesn't have a direct liquidations endpoint
        # This would need to be derived from trade data with large sizes
        # For now, return empty list
        logger.warning("Liquidation data requires WebSocket or trade filtering")
        return []

    def get_open_interest(self, asset: str) -> float:
        """Get open interest in USD."""
        market_data = self.get_market_data(asset)
        return market_data.open_interest if market_data else 0

    def analyze_liquidity(self, asset: str) -> Dict:
        """
        Analyze order book liquidity.

        Returns:
            Dict with liquidity metrics
        """
        ob = self.get_orderbook(asset, depth=50)
        if not ob:
            return {}

        # Calculate depth at different price levels
        bid_depth = sum(b[1] for b in ob.bids)
        ask_depth = sum(a[1] for a in ob.asks)

        # Spread as percentage
        spread_pct = (ob.spread / ob.mid_price) * 100 if ob.mid_price > 0 else 0

        # Imbalance
        total_depth = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0

        return {
            'asset': asset,
            'mid_price': ob.mid_price,
            'spread': ob.spread,
            'spread_pct': spread_pct,
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'imbalance': imbalance,
            'timestamp': ob.timestamp,
            'liquidity_quality': 'good' if spread_pct < 0.05 else 'moderate' if spread_pct < 0.1 else 'poor'
        }


def main():
    """Test the Hyperliquid connector."""
    connector = HyperliquidConnector()

    print("=== Hyperliquid Connector Test ===\n")

    # Test with BTC
    asset = "BTC"

    print(f"Fetching market data for {asset}...")
    market = connector.get_market_data(asset)

    if market:
        print(f"\n{asset} Market Data:")
        print(f"  Mark Price: ${market.mark_price:,.2f}")
        print(f"  Index Price: ${market.index_price:,.2f}")
        print(f"  Bid/Ask: ${market.bid:,.2f} / ${market.ask:,.2f}")
        print(f"  Spread: ${market.ask - market.bid:,.2f} ({((market.ask-market.bid)/market.mark_price)*100:.4f}%)")
        print(f"  Funding Rate: {market.funding_rate:.6f} ({market.funding_rate*100:.4f}%)")
        print(f"  24h Volume: ${market.volume_24h:,.0f}")
        print(f"  24h Change: {market.price_change_24h*100:.2f}%")
        print(f"  Open Interest: ${market.open_interest:,.0f}")

        print(f"\nLiquidity Analysis:")
        liq = connector.analyze_liquidity(asset)
        if liq:
            print(f"  Spread: {liq['spread_pct']:.4f}%")
            print(f"  Imbalance: {liq['imbalance']:.2%} (positive = more bids)")
            print(f"  Quality: {liq['liquidity_quality']}")
    else:
        print(f"Failed to fetch {asset} data")

    # Test with ETH
    print(f"\n{'='*50}\n")
    asset = "ETH"

    print(f"Fetching market data for {asset}...")
    market = connector.get_market_data(asset)

    if market:
        print(f"\n{asset} Market Data:")
        print(f"  Mark Price: ${market.mark_price:,.2f}")
        print(f"  Funding Rate: {market.funding_rate:.6f}")
        print(f"  24h Volume: ${market.volume_24h:,.0f}")
    else:
        print(f"Failed to fetch {asset} data")


if __name__ == "__main__":
    main()
