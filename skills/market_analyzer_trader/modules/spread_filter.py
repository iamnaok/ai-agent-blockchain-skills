"""
APEX Spread Filter
==================
Block trades when spread exceeds threshold.

Typical: reject if spread > 0.3%
Protects against adverse execution in illiquid markets.
"""

from typing import Dict, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SpreadCheck:
    """Spread check result."""
    spread_pct: float
    allowed: bool
    threshold: float
    timestamp: datetime


class SpreadFilter:
    """
    Spread-based trade filter.

    Prevents execution when spread is too wide,
    indicating illiquidity or adverse conditions.
    """

    def __init__(self, threshold: float = 0.003, adaptive: bool = True):
        """
        Args:
            threshold: Max allowed spread (default 0.3%)
            adaptive: Adjust threshold based on market regime
        """
        self.threshold = threshold
        self.adaptive = adaptive
        self.base_threshold = threshold
        self.rejection_count = 0
        self.pass_count = 0

    def calculate_spread(self, 
                        best_bid: float, 
                        best_ask: float,
                        mid_price: float = None) -> float:
        """
        Calculate spread percentage.

        Args:
            best_bid: Best bid price
            best_ask: Best ask price
            mid_price: Optional mid price (calculated if not provided)

        Returns:
            Spread as percentage
        """
        if mid_price is None:
            mid_price = (best_bid + best_ask) / 2

        spread = (best_ask - best_bid) / mid_price
        return spread

    def check_spread(self,
                    best_bid: float,
                    best_ask: float,
                    volatility: float = None) -> SpreadCheck:
        """
        Check if spread is acceptable.

        Args:
            best_bid: Best bid price
            best_ask: Best ask price
            volatility: Optional volatility for adaptive threshold

        Returns:
            SpreadCheck with result
        """
        spread = self.calculate_spread(best_bid, best_ask)

        # Adaptive threshold based on volatility
        threshold = self.threshold
        if self.adaptive and volatility:
            # In high vol, allow slightly wider spread
            threshold = min(self.threshold * (1 + volatility), 0.01)  # Max 1%

        allowed = spread <= threshold

        if allowed:
            self.pass_count += 1
        else:
            self.rejection_count += 1

        return SpreadCheck(
            spread_pct=spread,
            allowed=allowed,
            threshold=threshold,
            timestamp=datetime.now()
        )

    def should_trade(self, 
                    spread_check: SpreadCheck,
                    side: str = None) -> Tuple[bool, str]:
        """Determine if trade should proceed."""
        if spread_check.allowed:
            return True, f"Spread {spread_check.spread_pct*100:.3f}% OK"
        else:
            return False, f"Spread {spread_check.spread_pct*100:.3f}% exceeds {spread_check.threshold*100:.3f}%"

    def get_stats(self) -> Dict:
        """Get filter statistics."""
        total = self.pass_count + self.rejection_count
        return {
            'threshold': self.threshold,
            'rejection_count': self.rejection_count,
            'pass_count': self.pass_count,
            'rejection_rate': self.rejection_count / total if total > 0 else 0
        }


if __name__ == "__main__":
    print("APEX Spread Filter Demo")
    print("=" * 60)

    filter = SpreadFilter(threshold=0.003)  # 0.3%

    # Test spreads
    test_cases = [
        ("Normal", 2999.5, 3000.5, 3000),    # 0.033% - OK
        ("Wide", 2985, 3015, 3000),         # 1.0% - BLOCK
        ("Tight", 2999.9, 3000.1, 3000),   # 0.007% - OK
        ("Extreme", 2950, 3050, 3000),     # 3.33% - BLOCK
    ]

    for name, bid, ask, mid in test_cases:
        result = filter.check_spread(bid, ask, mid)
        status = "PASS" if result.allowed else "BLOCK"
        print(f"{name}: spread={result.spread_pct*100:.3f}% | {status}")

    print(f"\nStats: {filter.get_stats()}")
