"""
APEX Edge Scorer
================
Edge scoring with learned weights based on signal accuracy.

Formula: Edge = Σ(signal_score × learned_weight)
Signal strength mapping: STRONG=20, MODERATE=10, WEAK=5

Weight update:
- Accuracy >60%: weight → min(1.5, current × 1.1)
- Accuracy <40%: weight → max(0.3, current × 0.9)
- Re-check every 100 trades
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import os


@dataclass
class SignalPerformance:
    """Track individual signal performance."""
    name: str
    correct: int = 0
    total: int = 0
    weight: float = 1.0
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.5

    def record(self, was_correct: bool):
        self.total += 1
        if was_correct:
            self.correct += 1

    def update_weight(self):
        """Update weight based on accuracy."""
        acc = self.accuracy
        if acc > 0.60:
            self.weight = min(1.5, self.weight * 1.1)
        elif acc < 0.40:
            self.weight = max(0.3, self.weight * 0.9)
        self.last_updated = datetime.now()


class EdgeScorer:
    """
    APEX Edge Scorer with learning.

    Signal strength mapping:
    - STRONG: ±20 points
    - MODERATE: ±10 points  
    - WEAK: ±5 points
    """

    def __init__(self, memory_path: str = None):
        self.weights: Dict[str, float] = {}
        self.performance: Dict[str, SignalPerformance] = {}
        self.memory_path = memory_path or '/a0/usr/workdir/bot_memory.json'
        self._load_memory()

        # Initialize all 20 signals with default weight 1.0
        default_signals = [
            'cvd_divergence', 'cvd_momentum', 'liquidation_cluster', 'volume_absorption',
            'funding_arb', 'liquidity_imbalance', 'price_discrepancy', 'funding_persistent',
            'crowd_extreme', 'taker_flow', 'order_flow_imbalance', 'gamma_wall',
            'kalman_trend', 'kalman_deviation', 'liq_vel_exhaustion', 'liq_vel_accel',
            'whale_delta', 'particle_filter', 'rsi', 'macd'
        ]

        for sig in default_signals:
            if sig not in self.performance:
                self.performance[sig] = SignalPerformance(name=sig, weight=1.0)

    def _load_memory(self):
        """Load learned weights from memory."""
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, 'r') as f:
                    data = json.load(f)
                    for name, perf in data.get('signal_performance', {}).items():
                        self.performance[name] = SignalPerformance(**perf)
            except Exception as e:
                print(f"Warning: Could not load memory: {e}")

    def _save_memory(self):
        """Save learned weights to memory."""
        data = {
            'signal_performance': {
                name: {
                    'name': p.name,
                    'correct': p.correct,
                    'total': p.total,
                    'weight': p.weight,
                    'last_updated': p.last_updated.isoformat()
                }
                for name, p in self.performance.items()
            }
        }
        try:
            os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
            with open(self.memory_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save memory: {e}")

    def _signal_to_strength(self, value: float) -> float:
        """Convert signal value to strength points."""
        abs_val = abs(value)
        if abs_val > 0.7:
            return 20.0 * np.sign(value)  # STRONG
        elif abs_val > 0.3:
            return 10.0 * np.sign(value)  # MODERATE
        else:
            return 5.0 * np.sign(value)   # WEAK

    def compute_edge(self, signals: Dict[str, float]) -> Tuple[float, Dict]:
        """
        Calculate composite edge score.

        Args:
            signals: Dict of signal_name -> signal_value (-1 to 1)

        Returns:
            (edge_score, breakdown)
        """
        total_weighted = 0.0
        total_weight = 0.0
        breakdown = {}

        for name, value in signals.items():
            if name not in self.performance:
                continue

            perf = self.performance[name]
            strength = self._signal_to_strength(value)
            weighted = strength * perf.weight

            total_weighted += weighted
            total_weight += abs(perf.weight)

            breakdown[name] = {
                'raw_value': value,
                'strength': strength,
                'weight': perf.weight,
                'weighted': weighted,
                'accuracy': perf.accuracy
            }

        # Normalize edge score to -1 to 1
        if total_weight > 0:
            edge = np.tanh(total_weighted / 100)  # Normalize
        else:
            edge = 0.0

        return edge, breakdown

    def record_outcome(self, signals: Dict[str, float], outcome: bool, threshold: float = 0.3):
        """
        Record signal outcomes for learning.

        Args:
            signals: Dict of signal_name -> signal_value
            outcome: True if signal was correct
            threshold: Minimum signal strength to record
        """
        for name, value in signals.items():
            if abs(value) >= threshold and name in self.performance:
                self.performance[name].record(outcome)

        # Update weights every 100 trades
        total_trades = sum(p.total for p in self.performance.values())
        if total_trades % 100 == 0:
            self.update_weights()

    def update_weights(self):
        """Update all signal weights based on accuracy."""
        print(f"Updating weights at {datetime.now()}")
        for name, perf in self.performance.items():
            old_weight = perf.weight
            perf.update_weight()
            if perf.weight != old_weight:
                print(f"  {name}: {old_weight:.2f} → {perf.weight:.2f} (acc: {perf.accuracy:.2%})")

        self._save_memory()

    def get_top_signals(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top performing signals by accuracy."""
        sorted_sigs = sorted(
            self.performance.items(),
            key=lambda x: x[1].accuracy,
            reverse=True
        )
        return [(name, p.accuracy) for name, p in sorted_sigs[:n]]

    def get_summary(self) -> Dict:
        """Get performance summary."""
        total_trades = sum(p.total for p in self.performance.values())
        avg_accuracy = sum(p.accuracy for p in self.performance.values()) / len(self.performance) if self.performance else 0

        return {
            'total_trades': total_trades,
            'avg_accuracy': avg_accuracy,
            'top_signals': self.get_top_signals(5),
            'all_signals': {name: {'accuracy': p.accuracy, 'weight': p.weight} 
                          for name, p in self.performance.items()}
        }


if __name__ == "__main__":
    print("APEX Edge Scorer Demo")
    print("=" * 60)

    scorer = EdgeScorer()

    # Example signals
    signals = {
        'cvd_divergence': 0.8,
        'cvd_momentum': 0.4,
        'liquidation_cluster': 0.9,
        'particle_filter': 0.6,
        'rsi': -0.3,
        'macd': 0.5
    }

    edge, breakdown = scorer.compute_edge(signals)
    print(f"\nEdge Score: {edge:.3f}")
    print(f"Direction: {'BUY' if edge > 0.3 else 'SELL' if edge < -0.3 else 'HOLD'}")

    print(f"\nTop Contributing Signals:")
    sorted_breakdown = sorted(breakdown.items(), key=lambda x: abs(x[1]['weighted']), reverse=True)
    for name, data in sorted_breakdown[:5]:
        print(f"  {name}: raw={data['raw_value']:+.2f}, strength={data['strength']:+.1f}, "
              f"weight={data['weight']:.2f}, weighted={data['weighted']:+.1f}")
