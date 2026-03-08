"""
APEX Learning Engine
==================
Track signal accuracy and update weights.

Features:
- Signal accuracy tracking in bot_memory
- Accuracy-based weight updates
- Brier score tracking
- Periodic re-evaluation
"""

import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict
import numpy as np


@dataclass
class SignalOutcome:
    """Record of signal and outcome."""
    timestamp: str
    signal_name: str
    signal_value: float
    prediction: str  # 'up', 'down', 'flat'
    actual: str      # 'up', 'down', 'flat'
    was_correct: bool
    pnl: float       # Actual P&L

    def to_dict(self):
        return asdict(self)


@dataclass
class SignalStats:
    """Statistics for a signal."""
    name: str
    total: int = 0
    correct: int = 0
    brier_score: float = 0.0
    weight: float = 1.0
    last_updated: str = None

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now().isoformat()

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.5

    def update_weight(self):
        """Update weight based on accuracy."""
        acc = self.accuracy
        if acc > 0.60:
            self.weight = min(1.5, self.weight * 1.1)
        elif acc < 0.40:
            self.weight = max(0.3, self.weight * 0.9)
        self.last_updated = datetime.now().isoformat()

    def to_dict(self):
        return {
            'name': self.name,
            'total': self.total,
            'correct': self.correct,
            'accuracy': self.accuracy,
            'brier_score': self.brier_score,
            'weight': self.weight,
            'last_updated': self.last_updated
        }


class LearningEngine:
    """
    Learning engine for signal optimization.

    Tracks outcomes, calculates Brier scores,
    and updates signal weights based on accuracy.
    """

    def __init__(self, memory_path: str = None):
        self.memory_path = memory_path or '/a0/usr/workdir/bot_memory.json'
        self.outcomes: List[SignalOutcome] = []
        self.stats: Dict[str, SignalStats] = {}
        self.retrain_interval = 100  # Retrain every N outcomes
        self._load_memory()

    def _load_memory(self):
        """Load from persistent storage."""
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, 'r') as f:
                    data = json.load(f)
                    self.outcomes = [SignalOutcome(**o) for o in data.get('outcomes', [])]
                    for name, stats in data.get('signal_stats', {}).items():
                        self.stats[name] = SignalStats(**stats)
            except Exception as e:
                print(f"Could not load memory: {e}")

    def _save_memory(self):
        """Save to persistent storage."""
        try:
            os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
            with open(self.memory_path, 'w') as f:
                json.dump({
                    'outcomes': [o.to_dict() for o in self.outcomes[-5000:]],  # Keep last 5k
                    'signal_stats': {name: stats.to_dict() for name, stats in self.stats.items()},
                    'last_saved': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            print(f"Could not save memory: {e}")

    def record_outcome(self,
                      signal_name: str,
                      signal_value: float,
                      prediction: str,
                      actual_return: float,
                      pnl: float = 0):
        """
        Record signal outcome.

        Args:
            signal_name: Name of the signal
            signal_value: Signal strength (-1 to 1)
            prediction: 'up', 'down', or 'flat'
            actual_return: Actual price change
            pnl: Actual P&L from trade
        """
        # Determine actual direction
        if actual_return > 0.001:
            actual = 'up'
        elif actual_return < -0.001:
            actual = 'down'
        else:
            actual = 'flat'

        # Check if correct
        was_correct = prediction == actual

        # Create outcome
        outcome = SignalOutcome(
            timestamp=datetime.now().isoformat(),
            signal_name=signal_name,
            signal_value=signal_value,
            prediction=prediction,
            actual=actual,
            was_correct=was_correct,
            pnl=pnl
        )

        self.outcomes.append(outcome)

        # Update stats
        if signal_name not in self.stats:
            self.stats[signal_name] = SignalStats(name=signal_name)

        stats = self.stats[signal_name]
        stats.total += 1
        if was_correct:
            stats.correct += 1

        # Update Brier score
        prob = (signal_value + 1) / 2  # Convert -1,1 to 0,1
        actual_prob = 1.0 if actual == 'up' else 0.0 if actual == 'down' else 0.5
        brier = (prob - actual_prob) ** 2
        stats.brier_score = (stats.brier_score * (stats.total - 1) + brier) / stats.total

        # Periodic retrain
        if len(self.outcomes) % self.retrain_interval == 0:
            self.update_all_weights()

    def update_all_weights(self):
        """Update all signal weights."""
        print(f"[{datetime.now()}] Updating weights...")
        for name, stats in self.stats.items():
            old_weight = stats.weight
            stats.update_weight()
            if stats.weight != old_weight:
                print(f"  {name}: {old_weight:.2f} → {stats.weight:.2f} "
                      f"(acc: {stats.accuracy:.1%})")

        self._save_memory()

    def get_signal_performance(self, 
                               signal_name: str = None) -> Dict:
        """Get performance metrics."""
        if signal_name:
            if signal_name in self.stats:
                return self.stats[signal_name].to_dict()
            return {}

        return {
            name: stats.to_dict()
            for name, stats in self.stats.items()
        }

    def get_top_signals(self, 
                       min_trades: int = 10,
                       n: int = 5) -> List[Tuple[str, float]]:
        """Get top performing signals."""
        valid = [(name, s.accuracy) for name, s in self.stats.items() 
                if s.total >= min_trades]
        return sorted(valid, key=lambda x: x[1], reverse=True)[:n]

    def get_brier_scores(self) -> Dict[str, float]:
        """Get Brier scores for all signals."""
        return {name: s.brier_score for name, s in self.stats.items()}

    def get_summary(self) -> Dict:
        """Get learning engine summary."""
        total_outcomes = len(self.outcomes)
        total_signals = len(self.stats)

        if total_outcomes == 0:
            return {
                'total_outcomes': 0,
                'total_signals': 0,
                'avg_accuracy': 0
            }

        avg_acc = np.mean([s.accuracy for s in self.stats.values()]) if self.stats else 0

        return {
            'total_outcomes': total_outcomes,
            'total_signals': total_signals,
            'avg_accuracy': avg_acc,
            'top_performers': self.get_top_signals(n=5),
            'brier_scores': self.get_brier_scores()
        }


if __name__ == "__main__":
    print("APEX Learning Engine Demo")
    print("=" * 60)

    engine = LearningEngine()

    # Simulate outcomes
    np.random.seed(42)
    signals = ['cvd_divergence', 'liquidation_cluster', 'crowd_extreme']

    print(f"\nRecording outcomes...")
    for i in range(150):
        signal = np.random.choice(signals)
        # Simulate different accuracies
        if signal == 'liquidation_cluster':
            correct = np.random.random() < 0.74  # 74% accuracy
        elif signal == 'cvd_divergence':
            correct = np.random.random() < 0.60
        else:
            correct = np.random.random() < 0.45

        engine.record_outcome(
            signal_name=signal,
            signal_value=np.random.uniform(-1, 1),
            prediction='up' if np.random.random() > 0.5 else 'down',
            actual_return=0.01 if correct else -0.01,
            pnl=10 if correct else -10
        )

    print(f"\nPerformance Summary:")
    summary = engine.get_summary()
    print(f"  Total outcomes: {summary['total_outcomes']}")
    print(f"  Avg accuracy: {summary['avg_accuracy']:.1%}")

    print(f"\nTop Signals:")
    for name, acc in summary['top_performers']:
        stats = engine.get_signal_performance(name)
        print(f"  {name}: {acc:.1%} acc, {stats['brier_score']:.3f} Brier, weight={stats['weight']:.2f}")
