"""
Signal Outcome Tracker
======================
Tracks trading recommendations and validates outcomes against TP/SL.

Features:
- Records Long/Short recommendations with SL/TP levels
- Tracks price movements to determine TP_HIT or SL_HIT
- Calculates win rate per signal type
- Provides recommendations every 4 hours based on accumulated data
"""

import json
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger('SignalOutcomeTracker')


class Outcome(Enum):
    PENDING = "pending"
    TP_HIT = "tp_hit"
    SL_HIT = "sl_hit"
    NEITHER = "neither"
    EXPIRED = "expired"


class Direction(Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass
class SignalRecord:
    """A recorded trading recommendation."""
    id: str
    timestamp: str
    asset: str
    direction: str  # long, short, neutral
    entry_price: float
    stop_loss: float
    take_profit: float
    signal_type: str  # e.g., "liquidation_cluster", "crowd_extreme"
    confidence: float
    outcome: str = Outcome.PENDING.value
    resolved_at: Optional[str] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict) -> 'SignalRecord':
        return SignalRecord(**d)


class SignalOutcomeTracker:
    """
    Tracks signal recommendations and validates outcomes.

    Usage:
        tracker = SignalOutcomeTracker()

        # Record a recommendation
        signal_id = tracker.record_signal(
            asset="ETH",
            direction="long",
            entry_price=3000,
            stop_loss=2850,
            take_profit=3300,
            signal_type="liquidation_cluster",
            confidence=0.75
        )

        # Check outcomes (call periodically)
        tracker.check_outcomes(current_prices)

        # Get win rate
        win_rate = tracker.get_win_rate("liquidation_cluster")
    """

    def __init__(self, data_file: str = "/a0/usr/workdir/signal_performance.json"):
        self.data_file = data_file
        self.signals: List[SignalRecord] = []
        self.by_signal: Dict[str, Dict] = {}
        self.by_regime: Dict[str, Dict] = {}
        self.load()

    def load(self):
        """Load signal history from file."""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.signals = [SignalRecord.from_dict(s) for s in data.get('signals', [])]
                    self.by_signal = data.get('by_signal', {})
                    self.by_regime = data.get('by_regime', {})
                logger.info(f"Loaded {len(self.signals)} historical signals")
            except Exception as e:
                logger.error(f"Failed to load signal history: {e}")
                self._init_empty()
        else:
            self._init_empty()

    def _init_empty(self):
        """Initialize empty tracking structure."""
        self.signals = []
        self.by_signal = {}
        self.by_regime = {}

    def save(self):
        """Save signal history to file."""
        data = {
            'signals': [s.to_dict() for s in self.signals],
            'by_signal': self.by_signal,
            'by_regime': self.by_regime,
            'summary': {
                'total_signals': len(self.signals),
                'total_resolved': len([s for s in self.signals if s.outcome != Outcome.PENDING.value]),
                'overall_win_rate': self._calculate_overall_win_rate()
            },
            'last_updated': datetime.now().isoformat()
        }
        try:
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.signals)} signals to {self.data_file}")
        except Exception as e:
            logger.error(f"Failed to save signal history: {e}")

    def record_signal(self,
                     asset: str,
                     direction: str,
                     entry_price: float,
                     stop_loss: float,
                     take_profit: float,
                     signal_type: str,
                     confidence: float) -> str:
        """
        Record a new trading recommendation.

        Returns:
            Signal ID for tracking
        """
        signal_id = f"{asset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{signal_type}"

        signal = SignalRecord(
            id=signal_id,
            timestamp=datetime.now().isoformat(),
            asset=asset,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_type=signal_type,
            confidence=confidence
        )

        self.signals.append(signal)

        # Update per-signal stats
        if signal_type not in self.by_signal:
            self.by_signal[signal_type] = {
                'total': 0, 'tp_hits': 0, 'sl_hits': 0, 'neither': 0,
                'win_rate': 0.0, 'avg_profit': 0.0, 'avg_loss': 0.0
            }
        self.by_signal[signal_type]['total'] += 1

        self.save()
        logger.info(f"Recorded {direction} signal for {asset} at ${entry_price:.2f}")
        return signal_id

    def check_outcomes(self, current_prices: Dict[str, float]):
        """
        Check pending signals against current prices.

        Args:
            current_prices: Dict of {asset: current_price}
        """
        for signal in self.signals:
            if signal.outcome != Outcome.PENDING.value:
                continue

            asset = signal.asset
            if asset not in current_prices:
                continue

            current_price = current_prices[asset]

            # Check if TP or SL hit
            if signal.direction == Direction.LONG.value:
                if current_price >= signal.take_profit:
                    self._resolve_signal(signal, Outcome.TP_HIT, current_price)
                elif current_price <= signal.stop_loss:
                    self._resolve_signal(signal, Outcome.SL_HIT, current_price)
            elif signal.direction == Direction.SHORT.value:
                if current_price <= signal.take_profit:
                    self._resolve_signal(signal, Outcome.TP_HIT, current_price)
                elif current_price >= signal.stop_loss:
                    self._resolve_signal(signal, Outcome.SL_HIT, current_price)

            # Check expiration (24 hours)
            signal_time = datetime.fromisoformat(signal.timestamp)
            if datetime.now() - signal_time > timedelta(hours=24):
                self._resolve_signal(signal, Outcome.EXPIRED, current_price)

    def _resolve_signal(self, signal: SignalRecord, outcome: Outcome, exit_price: float):
        """Resolve a signal with its outcome."""
        signal.outcome = outcome.value
        signal.resolved_at = datetime.now().isoformat()
        signal.exit_price = exit_price

        # Calculate PnL
        if signal.direction == Direction.LONG.value:
            signal.pnl = (exit_price - signal.entry_price) / signal.entry_price
        else:
            signal.pnl = (signal.entry_price - exit_price) / signal.entry_price

        # Update stats
        stats = self.by_signal.get(signal.signal_type, {})
        if outcome == Outcome.TP_HIT:
            stats['tp_hits'] = stats.get('tp_hits', 0) + 1
        elif outcome == Outcome.SL_HIT:
            stats['sl_hits'] = stats.get('sl_hits', 0) + 1
        else:
            stats['neither'] = stats.get('neither', 0) + 1

        # Recalculate win rate
        resolved = stats.get('tp_hits', 0) + stats.get('sl_hits', 0)
        if resolved > 0:
            stats['win_rate'] = stats['tp_hits'] / resolved

        self.by_signal[signal.signal_type] = stats
        self.save()

        logger.info(f"Signal {signal.id}: {outcome.value} at ${exit_price:.2f} (PnL: {signal.pnl*100:+.2f}%)")

    def get_win_rate(self, signal_type: str) -> float:
        """Get win rate for a specific signal type."""
        stats = self.by_signal.get(signal_type, {})
        return stats.get('win_rate', 0.0)

    def get_recommendation_weight(self, signal_type: str) -> float:
        """Get dynamic weight based on historical performance."""
        win_rate = self.get_win_rate(signal_type)

        # Weight calculation: 
        # - < 40% win rate: 0.5 weight (reduce)
        # - 40-60% win rate: 1.0 weight (normal)
        # - > 60% win rate: 1.5 weight (increase)
        if win_rate < 0.40:
            return 0.5
        elif win_rate > 0.60:
            return 1.5
        return 1.0

    def get_signal_stats(self, signal_type: str) -> Dict:
        """Get full stats for a signal type."""
        return self.by_signal.get(signal_type, {
            'total': 0, 'tp_hits': 0, 'sl_hits': 0, 'neither': 0,
            'win_rate': 0.0
        })

    def _calculate_overall_win_rate(self) -> float:
        """Calculate overall win rate across all signals."""
        total_tp = sum(s.get('tp_hits', 0) for s in self.by_signal.values())
        total_sl = sum(s.get('sl_hits', 0) for s in self.by_signal.values())
        if total_tp + total_sl == 0:
            return 0.0
        return total_tp / (total_tp + total_sl)

    def get_summary(self) -> Dict:
        """Get summary of all tracking data."""
        return {
            'total_signals': len(self.signals),
            'pending': len([s for s in self.signals if s.outcome == Outcome.PENDING.value]),
            'resolved': len([s for s in self.signals if s.outcome != Outcome.PENDING.value]),
            'tp_hits': len([s for s in self.signals if s.outcome == Outcome.TP_HIT.value]),
            'sl_hits': len([s for s in self.signals if s.outcome == Outcome.SL_HIT.value]),
            'overall_win_rate': self._calculate_overall_win_rate(),
            'by_signal': self.by_signal
        }


if __name__ == "__main__":
    # Demo
    tracker = SignalOutcomeTracker()

    # Record a sample signal
    signal_id = tracker.record_signal(
        asset="ETH",
        direction="long",
        entry_price=3000,
        stop_loss=2850,
        take_profit=3300,
        signal_type="liquidation_cluster",
        confidence=0.75
    )

    print(f"Recorded signal: {signal_id}")
    print(f"Summary: {tracker.get_summary()}")
