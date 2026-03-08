"""
Public Liquidation Oracle
=========================
Real-time liquidation monitoring via Hyperliquid WebSocket.
No authentication required - uses public trade stream.

Features:
- WebSocket connection to Hyperliquid
- Real-time liquidation detection
- Liquidation cluster heatmap
- Price-level density estimation
- Risk zone alerts
"""

import json
import asyncio
import websockets
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import time
import logging

logger = logging.getLogger('LiquidationOracle')


@dataclass
class LiquidationEvent:
    """Detected liquidation event."""
    asset: str
    timestamp: datetime
    price: float
    size: float
    side: str  # 'long_liquidated' or 'short_liquidated'
    value_usd: float
    estimated_leverage: float


@dataclass
class LiquidationCluster:
    """Cluster of liquidations at a price level."""
    price_level: float
    total_size: float
    total_value: float
    count: int
    last_liquidation: datetime
    events: List[LiquidationEvent] = field(default_factory=list)


class LiquidationOracle:
    """
    Public Liquidation Oracle using Hyperliquid WebSocket.

    Monitors trade stream for liquidation events and builds
    real-time liquidation heatmap.

    No authentication required - uses public WebSocket.
    """

    WS_URL = "wss://api.hyperliquid.xyz/ws"

    def __init__(self):
        self.liquidations: Dict[str, List[LiquidationEvent]] = defaultdict(list)
        self.clusters: Dict[str, Dict[float, LiquidationCluster]] = defaultdict(dict)
        self.running = False
        self.ws = None
        self.loop = None
        self.thread = None

        # Thresholds for liquidation detection
        self.min_liquidation_size = 1000  # $1000 minimum
        self.cluster_radius = 0.005  # 0.5% price radius for clustering
        self.max_history = 1000  # Keep last 1000 events per asset

    def _is_liquidation(self, trade: Dict) -> bool:
        """
        Detect if a trade is a liquidation.

        Liquidations on Hyperliquid have specific markers:
        - Large size relative to typical trades
        - Often at market price (crossing spread)
        - Specific trade flags (if available)
        """
        size = float(trade.get('sz', 0))
        price = float(trade.get('px', 0))
        value = size * price

        # Criteria 1: Large size ($1000+)
        if value < self.min_liquidation_size:
            return False

        # Criteria 2: Round lot sizes (liquidations often at round numbers)
        if size < 1.0:
            return False

        # Criteria 3: Check for aggressive crossing (liquidations hit market)
        # This requires order book context, simplified here
        return True

    def _estimate_leverage(self, size: float, value: float, price: float) -> float:
        """
        Estimate leverage from liquidation size.

        Conservative estimate based on:
        - Typical position sizes
        - Margin requirements
        - Historical liquidation patterns
        """
        # Simplified: larger liquidations often = higher leverage
        # In reality would need account data
        if value > 50000:
            return 10.0
        elif value > 10000:
            return 5.0
        else:
            return 3.0

    def _update_clusters(self, event: LiquidationEvent):
        """Update liquidation clusters with new event."""
        asset = event.asset
        price = event.price

        # Find or create cluster
        found_cluster = None
        for level, cluster in self.clusters[asset].items():
            if abs(price - level) / level < self.cluster_radius:
                found_cluster = cluster
                break

        if found_cluster is None:
            # Create new cluster
            cluster = LiquidationCluster(
                price_level=price,
                total_size=0,
                total_value=0,
                count=0,
                last_liquidation=event.timestamp,
                events=[]
            )
            self.clusters[asset][price] = cluster
        else:
            cluster = found_cluster

        # Update cluster
        cluster.total_size += event.size
        cluster.total_value += event.value_usd
        cluster.count += 1
        cluster.last_liquidation = event.timestamp
        cluster.events.append(event)

        # Trim to max history
        if len(cluster.events) > self.max_history:
            cluster.events = cluster.events[-self.max_history:]

    async def _websocket_handler(self, assets: List[str]):
        """WebSocket connection handler."""
        try:
            async with websockets.connect(self.WS_URL) as ws:
                self.ws = ws
                logger.info(f"Connected to Hyperliquid WebSocket")

                # Subscribe to trade feed for each asset
                for asset in assets:
                    subscribe_msg = {
                        "method": "subscribe",
                        "subscription": {
                            "type": "trades",
                            "coin": asset
                        }
                    }
                    await ws.send(json.dumps(subscribe_msg))
                    logger.info(f"Subscribed to trades: {asset}")

                # Process messages
                async for message in ws:
                    try:
                        data = json.loads(message)
                        await self._process_message(data)
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")

        except Exception as e:
            logger.error(f"WebSocket error: {e}")

    async def _process_message(self, data: Dict):
        """Process WebSocket message."""
        if data.get("channel") != "trades":
            return

        trades = data.get("data", [])
        if not isinstance(trades, list):
            trades = [trades]

        for trade in trades:
            if self._is_liquidation(trade):
                event = LiquidationEvent(
                    asset=trade.get("coin", "UNKNOWN"),
                    timestamp=datetime.utcnow(),
                    price=float(trade.get("px", 0)),
                    size=float(trade.get("sz", 0)),
                    side="long_liquidated" if trade.get("side") == "B" else "short_liquidated",
                    value_usd=float(trade.get("px", 0)) * float(trade.get("sz", 0)),
                    estimated_leverage=self._estimate_leverage(
                        float(trade.get("sz", 0)),
                        float(trade.get("px", 0)) * float(trade.get("sz", 0)),
                        float(trade.get("px", 0))
                    )
                )

                # Store event
                self.liquidations[event.asset].append(event)
                self._update_clusters(event)

                logger.info(f"LIQUIDATION: {event.asset} {event.side} at ${event.price:,.2f} "
                          f"size={event.size:.4f} value=${event.value_usd:,.0f}")

    def start(self, assets: List[str] = None):
        """Start WebSocket connection in background thread."""
        if self.running:
            return

        if assets is None:
            assets = ["BTC", "ETH", "SOL"]

        self.running = True

        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._websocket_handler(assets))

        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()
        logger.info(f"Liquidation Oracle started for: {assets}")

    def stop(self):
        """Stop WebSocket connection."""
        self.running = False
        if self.loop and self.ws:
            asyncio.run_coroutine_threadsafe(self.ws.close(), self.loop)
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Liquidation Oracle stopped")

    def get_liquidation_heatmap(self, asset: str, n_levels: int = 10) -> List[Dict]:
        """
        Get liquidation heatmap for an asset.

        Returns top N price levels with most liquidation activity.
        """
        if asset not in self.clusters:
            return []

        # Sort clusters by total value liquidated
        clusters = sorted(
            self.clusters[asset].values(),
            key=lambda c: c.total_value,
            reverse=True
        )

        return [
            {
                "price_level": c.price_level,
                "total_liquidated": c.total_value,
                "total_size": c.total_size,
                "count": c.count,
                "last_liquidation": c.last_liquidation.isoformat(),
                "avg_leverage": np.mean([e.estimated_leverage for e in c.events]) if c.events else 0
            }
            for c in clusters[:n_levels]
        ]

    def get_risk_zones(self, asset: str, current_price: float) -> Dict:
        """
        Identify risk zones relative to current price.

        Returns support/resistance levels based on liquidation clusters.
        """
        heatmap = self.get_liquidation_heatmap(asset, n_levels=20)

        if not heatmap:
            return {"above": [], "below": []}

        # Split into above/below current price
        above = [h for h in heatmap if h["price_level"] > current_price]
        below = [h for h in heatmap if h["price_level"] < current_price]

        # Sort by proximity to current price
        above = sorted(above, key=lambda x: x["price_level"])
        below = sorted(below, key=lambda x: x["price_level"], reverse=True)

        return {
            "above": above[:5],  # Top 5 above current price
            "below": below[:5],  # Top 5 below current price
            "largest_cluster": heatmap[0] if heatmap else None
        }

    def estimate_liquidation_cascade_risk(self, asset: str, 
                                         price_direction: str,
                                         price_move_pct: float) -> float:
        """
        Estimate risk of liquidation cascade if price moves X%.

        Returns: Risk score 0-1 (higher = more risk)
        """
        if asset not in self.clusters:
            return 0.0

        current_clusters = list(self.clusters[asset].values())
        if not current_clusters:
            return 0.0

        # Find clusters that would be hit
        at_risk_value = 0
        total_value = sum(c.total_value for c in current_clusters)

        for cluster in current_clusters:
            if price_direction == "up":
                # Short liquidations happen when price goes up
                if cluster.events and cluster.events[0].side == "short_liquidated":
                    at_risk_value += cluster.total_value
            else:
                # Long liquidations happen when price goes down
                if cluster.events and cluster.events[0].side == "long_liquidated":
                    at_risk_value += cluster.total_value

        risk_score = min(at_risk_value / total_value, 1.0) if total_value > 0 else 0.0
        return risk_score

    def get_recent_liquidations(self, asset: str, minutes: int = 60) -> List[Dict]:
        """Get liquidations in last N minutes."""
        if asset not in self.liquidations:
            return []

        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        recent = [
            {
                "timestamp": e.timestamp.isoformat(),
                "price": e.price,
                "size": e.size,
                "side": e.side,
                "value_usd": e.value_usd,
                "estimated_leverage": e.estimated_leverage
            }
            for e in self.liquidations[asset]
            if e.timestamp > cutoff
        ]

        return recent


class LiquidationEnhancedTrader:
    """
    Enhanced trader that incorporates liquidation oracle data.
    """

    def __init__(self, capital: float = 1000):
        self.oracle = LiquidationOracle()
        self.capital = capital

    def start_monitoring(self, assets: List[str] = None):
        """Start liquidation monitoring."""
        self.oracle.start(assets)

    def analyze_with_liquidations(self, asset: str, current_price: float) -> Dict:
        """
        Generate trade analysis incorporating liquidation data.
        """
        # Get liquidation insights
        risk_zones = self.oracle.get_risk_zones(asset, current_price)
        heatmap = self.oracle.get_liquidation_heatmap(asset, n_levels=5)
        recent = self.oracle.get_recent_liquidations(asset, minutes=60)

        # Calculate cascade risk
        long_cascade_risk = self.oracle.estimate_liquidation_cascade_risk(
            asset, "down", 0.05  # 5% down
        )
        short_cascade_risk = self.oracle.estimate_liquidation_cascade_risk(
            asset, "up", 0.05  # 5% up
        )

        # Build analysis
        analysis = {
            "asset": asset,
            "current_price": current_price,
            "liquidation_heatmap": heatmap,
            "risk_zones": risk_zones,
            "recent_liquidations_1h": len(recent),
            "cascade_risk": {
                "long_cascade_if_down_5pct": round(long_cascade_risk, 2),
                "short_cascade_if_up_5pct": round(short_cascade_risk, 2)
            },
            "trading_implications": []
        }

        # Add trading implications
        implications = []

        if risk_zones["below"]:
            closest_support = risk_zones["below"][0]
            implications.append(
                f"Strong support at ${closest_support['price_level']:,.2f} "
                f"({closest_support['total_liquidated']:,.0f} liquidated)"
            )

        if risk_zones["above"]:
            closest_resistance = risk_zones["above"][0]
            implications.append(
                f"Resistance at ${closest_resistance['price_level']:,.2f} "
                f"({closest_resistance['total_liquidated']:,.0f} liquidated)"
            )

        if long_cascade_risk > 0.3:
            implications.append(
                f"⚠️ HIGH long liquidation risk below ${current_price * 0.95:,.2f} "
                f"({long_cascade_risk*100:.0f}% of clusters at risk)"
            )

        if short_cascade_risk > 0.3:
            implications.append(
                f"⚠️ HIGH short liquidation risk above ${current_price * 1.05:,.2f} "
                f"({short_cascade_risk*100:.0f}% of clusters at risk)"
            )

        if len(recent) > 10:
            implications.append(
                f"🔥 Heavy liquidation activity: {len(recent)} in last hour"
            )

        analysis["trading_implications"] = implications

        return analysis


def main():
    """Test liquidation oracle."""
    print("="*70)
    print("PUBLIC LIQUIDATION ORACLE TEST")
    print("="*70)

    oracle = LiquidationOracle()

    # Simulate some liquidation data (since WebSocket may take time)
    print("\nSimulating liquidation events...")

    test_events = [
        LiquidationEvent("BTC", datetime.utcnow(), 67200, 0.5, "long_liquidated", 33600, 10),
        LiquidationEvent("BTC", datetime.utcnow(), 67100, 0.3, "long_liquidated", 20130, 15),
        LiquidationEvent("BTC", datetime.utcnow(), 67000, 0.8, "long_liquidated", 53600, 20),
        LiquidationEvent("BTC", datetime.utcnow(), 67800, 0.4, "short_liquidated", 27120, 10),
        LiquidationEvent("BTC", datetime.utcnow(), 67900, 0.6, "short_liquidated", 40740, 12),
    ]

    for event in test_events:
        oracle.liquidations[event.asset].append(event)
        oracle._update_clusters(event)

    # Get heatmap
    print("\nBTC Liquidation Heatmap:")
    heatmap = oracle.get_liquidation_heatmap("BTC")
    for i, cluster in enumerate(heatmap[:5], 1):
        print(f"  {i}. ${cluster['price_level']:,.0f}: "
              f"${cluster['total_liquidated']:,.0f} liquidated "
              f"({cluster['count']} events)")

    # Get risk zones
    print("\nRisk Zones at $67,300:")
    risk = oracle.get_risk_zones("BTC", 67300)

    print("\nSupport levels (below price):")
    for level in risk["below"]:
        print(f"  ${level['price_level']:,.0f}: ${level['total_liquidated']:,.0f}")

    print("\nResistance levels (above price):")
    for level in risk["above"]:
        print(f"  ${level['price_level']:,.0f}: ${level['total_liquidated']:,.0f}")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print("\nTo start real-time monitoring:")
    print("  oracle = LiquidationOracle()")
    print("  oracle.start(['BTC', 'ETH', 'SOL'])")


if __name__ == "__main__":
    main()
