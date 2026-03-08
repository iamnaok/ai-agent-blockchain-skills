"""
Execution Engine v1.1
VWAP/TWAP execution with order flow delta and liquidity scoring.

Based on: Professional Quant Trader research
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger('ExecutionEngine')


@dataclass
class ExecutionSlice:
    """Single execution slice for TWAP/VWAP."""
    timestamp: datetime
    volume: float
    price: float
    side: str  # 'buy' or 'sell'
    liquidity_score: float


@dataclass
class OrderBook:
    """Order book snapshot."""
    bids: List[Tuple[float, float]]  # (price, volume)
    asks: List[Tuple[float, float]]  # (price, volume)
    timestamp: datetime

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else float('inf')

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    def get_bid_volume(self, levels: int = 5) -> float:
        """Get total bid volume up to N levels."""
        return sum(v for _, v in self.bids[:levels])

    def get_ask_volume(self, levels: int = 5) -> float:
        """Get total ask volume up to N levels."""
        return sum(v for _, v in self.asks[:levels])


class ExecutionEngine:
    """
    Optimal execution engine with VWAP/TWAP, order flow delta, and liquidity scoring.

    Key insights from research:
    - VWAP deviation: Buy below VWAP, sell above VWAP
    - Order flow delta > 0.2 predicts 60%+ short-term price increase
    - Liquidity score > 0.7 minimizes adverse selection
    - Equal-weight clips optimal for prediction markets
    """

    def __init__(self,
                 max_slippage: float = 0.005,  # 0.5% max slippage
                 vwap_window: int = 20,  # 20-period VWAP
                 delta_threshold: float = 0.2,  # Min delta for entry
                 liquidity_threshold: float = 0.7,  # Min liquidity score
                 clip_count: int = 4):  # Number of execution clips
        """
        Args:
            max_slippage: Maximum acceptable slippage (0.5%)
            vwap_window: Window for VWAP calculation (20 periods)
            delta_threshold: Minimum order flow delta to trade (0.2)
            liquidity_threshold: Minimum liquidity score (0.7)
            clip_count: Number of clips for TWAP/VWAP execution
        """
        self.max_slippage = max_slippage
        self.vwap_window = vwap_window
        self.delta_threshold = delta_threshold
        self.liquidity_threshold = liquidity_threshold
        self.clip_count = clip_count

    def calculate_vwap(self, prices: List[float], volumes: List[float]) -> float:
        """
        Calculate Volume-Weighted Average Price.

        VWAP = Σ(Price × Volume) / Σ(Volume)
        """
        if len(prices) != len(volumes) or len(prices) == 0:
            return 0.0

        prices_arr = np.array(prices[-self.vwap_window:])
        volumes_arr = np.array(volumes[-self.vwap_window:])

        total_value = np.sum(prices_arr * volumes_arr)
        total_volume = np.sum(volumes_arr)

        return total_value / total_volume if total_volume > 0 else 0.0

    def calculate_vwap_deviation(self, 
                                 current_price: float, 
                                 prices: List[float], 
                                 volumes: List[float]) -> float:
        """
        Calculate VWAP deviation signal.

        Returns:
            Negative: Price below VWAP (buy signal)
            Positive: Price above VWAP (sell signal)
        """
        vwap = self.calculate_vwap(prices, volumes)
        if vwap == 0:
            return 0.0

        return (current_price - vwap) / vwap

    def calculate_order_flow_delta(self,
                                    bid_volumes: List[float],
                                    ask_volumes: List[float]) -> float:
        """
        Calculate order flow delta.

        Delta = (BidVol - AskVol) / TotalVol

        Research shows delta > 0.2 predicts 60%+ short-term price increase.
        """
        if len(bid_volumes) != len(ask_volumes) or len(bid_volumes) == 0:
            return 0.0

        total_bid_vol = np.sum(bid_volumes[-self.vwap_window:])
        total_ask_vol = np.sum(ask_volumes[-self.vwap_window:])
        total_vol = total_bid_vol + total_ask_vol

        if total_vol == 0:
            return 0.0

        delta = (total_bid_vol - total_ask_vol) / total_vol
        return delta

    def calculate_liquidity_score(self, order_book: OrderBook) -> float:
        """
        Calculate liquidity score.

        Score = 1 - (slippage / max_slippage)

        Research: Score > 0.7 minimizes adverse selection.
        Markets with score < 0.6 have 3x higher adverse selection risk.
        """
        mid = order_book.mid_price
        spread = order_book.spread

        if mid == 0 or spread == 0:
            return 1.0  # Perfect liquidity if no spread

        # Calculate expected slippage for small trade
        slippage = spread / (2 * mid)  # Half-spread as slippage proxy

        # Score based on relative slippage
        score = 1.0 - (slippage / self.max_slippage)
        return np.clip(score, 0.0, 1.0)

    def should_execute(self,
                       side: str,
                       current_price: float,
                       prices: List[float],
                       volumes: List[float],
                       bid_volumes: List[float],
                       ask_volumes: List[float],
                       order_book: OrderBook) -> Tuple[bool, str]:
        """
        Determine if conditions are right for execution.

        Args:
            side: 'buy' or 'sell'
            current_price: Current market price
            prices: Historical prices
            volumes: Historical volumes
            bid_volumes: Historical bid volumes
            ask_volumes: Historical ask volumes
            order_book: Current order book

        Returns:
            (should_execute, reason)
        """
        checks = []

        # 1. Check VWAP deviation
        vwap_dev = self.calculate_vwap_deviation(current_price, prices, volumes)

        if side == 'buy':
            if vwap_dev < 0:
                checks.append(f"PASS: Price ${current_price:.4f} below VWAP ({vwap_dev*100:.2f}%)")
            else:
                checks.append(f"FAIL: Price ${current_price:.4f} above VWAP ({vwap_dev*100:.2f}%)")
        else:  # sell
            if vwap_dev > 0:
                checks.append(f"PASS: Price ${current_price:.4f} above VWAP ({vwap_dev*100:.2f}%)")
            else:
                checks.append(f"FAIL: Price ${current_price:.4f} below VWAP ({vwap_dev*100:.2f}%)")

        # 2. Check order flow delta
        delta = self.calculate_order_flow_delta(bid_volumes, ask_volumes)

        if side == 'buy':
            if delta >= self.delta_threshold:
                checks.append(f"PASS: Delta {delta:.3f} >= {self.delta_threshold} (bullish)")
            else:
                checks.append(f"FAIL: Delta {delta:.3f} < {self.delta_threshold}")
        else:
            if delta <= -self.delta_threshold:
                checks.append(f"PASS: Delta {delta:.3f} <= -{self.delta_threshold} (bearish)")
            else:
                checks.append(f"FAIL: Delta {delta:.3f} > -{self.delta_threshold}")

        # 3. Check liquidity
        liq_score = self.calculate_liquidity_score(order_book)

        if liq_score >= self.liquidity_threshold:
            checks.append(f"PASS: Liquidity {liq_score:.3f} >= {self.liquidity_threshold}")
        else:
            checks.append(f"FAIL: Liquidity {liq_score:.3f} < {self.liquidity_threshold}")

        # Determine if all critical checks pass
        failures = [c for c in checks if c.startswith('FAIL')]
        should_execute = len(failures) == 0

        reason = "; ".join(checks)

        if should_execute:
            logger.info(f"✓ Execution approved: {side} @ ${current_price:.4f}")
        else:
            logger.warning(f"✗ Execution blocked: {side}")

        return should_execute, reason

    def generate_twap_slices(self,
                             total_quantity: float,
                             total_duration_minutes: int = 30) -> List[ExecutionSlice]:
        """
        Generate TWAP (Time-Weighted Average Price) execution slices.

        Equal intervals, equal quantities.
        Research: Equal-weight clips optimal for prediction markets.
        """
        interval_minutes = total_duration_minutes / self.clip_count
        quantity_per_slice = total_quantity / self.clip_count

        slices = []
        now = datetime.now()

        for i in range(self.clip_count):
            slice_time = now + timedelta(minutes=i * interval_minutes)
            slices.append(ExecutionSlice(
                timestamp=slice_time,
                volume=quantity_per_slice,
                price=0.0,  # Will be filled at execution time
                side='buy',  # Default, caller should set
                liquidity_score=0.0
            ))

        return slices

    def generate_vwap_slices(self,
                               total_quantity: float,
                               historical_volumes: List[float],
                               total_duration_minutes: int = 30) -> List[ExecutionSlice]:
        """
        Generate VWAP execution slices.

        Quantity proportional to historical volume distribution.
        """
        if len(historical_volumes) < self.clip_count:
            # Fall back to TWAP if insufficient history
            return self.generate_twap_slices(total_quantity, total_duration_minutes)

        # Use last N periods for volume distribution
        recent_volumes = np.array(historical_volumes[-self.clip_count:])
        volume_weights = recent_volumes / np.sum(recent_volumes)

        slices = []
        now = datetime.now()
        interval_minutes = total_duration_minutes / self.clip_count

        for i in range(self.clip_count):
            slice_time = now + timedelta(minutes=i * interval_minutes)
            slice_qty = total_quantity * volume_weights[i]

            slices.append(ExecutionSlice(
                timestamp=slice_time,
                volume=slice_qty,
                price=0.0,
                side='buy',
                liquidity_score=0.0
            ))

        return slices

    def estimate_slippage(self,
                          order_size: float,
                          order_book: OrderBook,
                          side: str = 'buy') -> float:
        """
        Estimate slippage for an order.

        Uses order book depth to calculate price impact.
        """
        remaining_size = order_size
        total_cost = 0.0

        if side == 'buy':
            levels = order_book.asks
        else:
            levels = order_book.bids

        for price, volume in levels:
            if remaining_size <= 0:
                break

            take = min(remaining_size, volume)
            total_cost += take * price
            remaining_size -= take

        if order_size == 0:
            return 0.0

        avg_fill_price = total_cost / (order_size - remaining_size) if (order_size - remaining_size) > 0 else 0
        mid_price = order_book.mid_price

        if mid_price == 0:
            return 0.0

        slippage = abs(avg_fill_price - mid_price) / mid_price
        return slippage

    def get_optimal_entry_time(self,
                                side: str,
                                prices: List[float],
                                volumes: List[float],
                                bid_volumes: List[float],
                                ask_volumes: List[float],
                                max_wait_minutes: int = 10) -> Tuple[datetime, float, str]:
        """
        Calculate optimal entry time based on VWAP deviation and order flow.

        Returns:
            (optimal_time, expected_price, reason)
        """
        current_price = prices[-1] if prices else 0
        vwap_dev = self.calculate_vwap_deviation(current_price, prices, volumes)
        delta = self.calculate_order_flow_delta(bid_volumes, ask_volumes)

        if side == 'buy':
            if vwap_dev < -0.001:  # Already below VWAP
                return datetime.now(), current_price, "Immediate: Price below VWAP"

            # Wait for price to drop below VWAP or for bullish delta
            expected_time = datetime.now() + timedelta(minutes=min(max_wait_minutes, 5))
            expected_price = current_price * (1 - abs(vwap_dev)) if vwap_dev > 0 else current_price

            if delta >= self.delta_threshold:
                return expected_time, expected_price, f"Wait {max_wait_minutes}min: Bullish delta {delta:.3f}"

            return expected_time, expected_price, f"Wait {max_wait_minutes}min: VWAP deviation {vwap_dev*100:.2f}%"

        else:  # sell
            if vwap_dev > 0.001:  # Already above VWAP
                return datetime.now(), current_price, "Immediate: Price above VWAP"

            expected_time = datetime.now() + timedelta(minutes=min(max_wait_minutes, 5))
            expected_price = current_price * (1 + abs(vwap_dev)) if vwap_dev < 0 else current_price

            if delta <= -self.delta_threshold:
                return expected_time, expected_price, f"Wait {max_wait_minutes}min: Bearish delta {delta:.3f}"

            return expected_time, expected_price, f"Wait {max_wait_minutes}min: VWAP deviation {vwap_dev*100:.2f}%"


if __name__ == "__main__":
    # Demo
    engine = ExecutionEngine()

    # Simulated data
    prices = [100, 101, 99, 98, 97, 96, 95, 96, 97, 98, 99, 100, 101]
    volumes = [1000, 1200, 800, 900, 1100, 1300, 1500, 1400, 1200, 1000, 900, 1100, 1300]
    bid_vols = [600, 700, 400, 500, 600, 800, 900, 800, 700, 600, 500, 600, 700]
    ask_vols = [400, 500, 400, 400, 500, 500, 600, 600, 500, 400, 400, 500, 600]

    current = prices[-1]

    # VWAP calculation
    vwap = engine.calculate_vwap(prices, volumes)
    vwap_dev = engine.calculate_vwap_deviation(current, prices, volumes)
    delta = engine.calculate_order_flow_delta(bid_vols, ask_vols)

    print(f"Current: ${current}")
    print(f"VWAP: ${vwap:.2f}")
    print(f"VWAP Deviation: {vwap_dev*100:.2f}%")
    print(f"Order Flow Delta: {delta:.3f}")

    # Order book
    ob = OrderBook(
        bids=[(99.5, 500), (99.0, 800), (98.5, 1200)],
        asks=[(100.5, 600), (101.0, 900), (101.5, 1500)],
        timestamp=datetime.now()
    )

    liq = engine.calculate_liquidity_score(ob)
    print(f"Liquidity Score: {liq:.3f}")

    # Check if should buy
    should_buy, reason = engine.should_execute(
        'buy', current, prices, volumes, bid_vols, ask_vols, ob
    )
    print(f"\nBuy Decision: {should_buy}")
    print(f"Reason: {reason}")
