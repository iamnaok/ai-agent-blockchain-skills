"""
Adaptive Thresholds Module for Non-Stationary Market Conditions

This module implements rolling window statistics and adaptive threshold calculation
to handle non-stationary distributions in market data.
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import os


@dataclass
class SignalStatistics:
    """Statistics for a single signal over a rolling window."""
    signal_name: str
    window_size: int
    values: deque
    timestamps: deque
    
    def __post_init__(self):
        if not isinstance(self.values, deque):
            self.values = deque(maxlen=self.window_size)
        if not isinstance(self.timestamps, deque):
            self.timestamps = deque(maxlen=self.window_size)
    
    def add_value(self, value: float, timestamp: datetime):
        """Add a new value to the rolling window."""
        self.values.append(value)
        self.timestamps.append(timestamp)
    
    def get_statistics(self) -> Dict[str, float]:
        """Calculate current statistics for the window."""
        if len(self.values) < 10:  # Minimum sample size
            return {
                'mean': 0.0,
                'std': 1.0,
                'min': -1.0,
                'max': 1.0,
                'percentile_25': -0.5,
                'percentile_75': 0.5,
                'skewness': 0.0,
                'kurtosis': 3.0
            }
        
        values_array = np.array(self.values)
        
        return {
            'mean': float(np.mean(values_array)),
            'std': float(np.std(values_array)) if len(values_array) > 1 else 1.0,
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'percentile_25': float(np.percentile(values_array, 25)),
            'percentile_75': float(np.percentile(values_array, 75)),
            'skewness': float(self._calculate_skewness(values_array)),
            'kurtosis': float(self._calculate_kurtosis(values_array))
        }
    
    def _calculate_skewness(self, values: np.ndarray) -> float:
        """Calculate skewness of the distribution."""
        if len(values) < 3:
            return 0.0
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0.0
        return float(np.mean(((values - mean) / std) ** 3))
    
    def _calculate_kurtosis(self, values: np.ndarray) -> float:
        """Calculate excess kurtosis of the distribution."""
        if len(values) < 4:
            return 3.0
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 3.0
        return float(np.mean(((values - mean) / std) ** 4))


class AdaptiveThresholdManager:
    """
    Manages adaptive thresholds for all signals based on rolling window statistics.
    
    This class tracks signal performance over time and adjusts thresholds
to account for changing market conditions (non-stationary distributions).
    """
    
    def __init__(self, window_size: int = 100, min_samples: int = 20):
        """
        Initialize the adaptive threshold manager.
        
        Args:
            window_size: Size of rolling window for statistics
            min_samples: Minimum samples before adapting thresholds
        """
        self.window_size = window_size
        self.min_samples = min_samples
        self.signal_stats: Dict[str, SignalStatistics] = {}
        self.base_thresholds: Dict[str, float] = {}
        self.adaptive_thresholds: Dict[str, float] = {}
        self.regime_multipliers: Dict[str, float] = {
            'low_volatility': 0.7,
            'normal': 1.0,
            'high_volatility': 1.3,
            'trending': 1.2,
            'ranging': 0.8
        }
        
    def initialize_signal(self, signal_name: str, base_threshold: float):
        """Initialize tracking for a new signal."""
        if signal_name not in self.signal_stats:
            self.signal_stats[signal_name] = SignalStatistics(
                signal_name=signal_name,
                window_size=self.window_size,
                values=deque(maxlen=self.window_size),
                timestamps=deque(maxlen=self.window_size)
            )
            self.base_thresholds[signal_name] = base_threshold
            self.adaptive_thresholds[signal_name] = base_threshold
    
    def update_signal(self, signal_name: str, value: float, timestamp: Optional[datetime] = None):
        """
        Update statistics for a signal with a new value.
        
        Args:
            signal_name: Name of the signal
            value: Signal value (typically -1 to 1 or 0 to 1)
            timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if signal_name not in self.signal_stats:
            self.initialize_signal(signal_name, 0.5)  # Default threshold
        
        self.signal_stats[signal_name].add_value(value, timestamp)
        
        # Recalculate adaptive threshold if we have enough samples
        if len(self.signal_stats[signal_name].values) >= self.min_samples:
            self._calculate_adaptive_threshold(signal_name)
    
    def _calculate_adaptive_threshold(self, signal_name: str):
        """Calculate adaptive threshold based on rolling statistics."""
        stats = self.signal_stats[signal_name].get_statistics()
        base_threshold = self.base_thresholds.get(signal_name, 0.5)
        
        # Adjust threshold based on distribution characteristics
        # Higher volatility = higher threshold (more selective)
        # Lower volatility = lower threshold (more permissive)
        
        std = stats['std']
        mean_abs = abs(stats['mean'])
        
        # Calculate volatility factor
        # If std is high, distribution is spread out, need higher threshold
        volatility_factor = 1.0 + (std - 0.3) * 0.5  # Normalize around std=0.3
        volatility_factor = max(0.5, min(1.5, volatility_factor))  # Clamp between 0.5 and 1.5
        
        # Calculate skewness adjustment
        # Positive skew = more extreme positive values, lower threshold for shorts
        # Negative skew = more extreme negative values, lower threshold for longs
        skewness = stats['skewness']
        skewness_factor = 1.0 - abs(skewness) * 0.1  # Slight reduction for skewed distributions
        skewness_factor = max(0.7, min(1.3, skewness_factor))
        
        # Combine factors
        adaptive_threshold = base_threshold * volatility_factor * skewness_factor
        
        # Smooth transition (EMA-style)
        old_threshold = self.adaptive_thresholds.get(signal_name, base_threshold)
        smoothing = 0.3  # 30% weight to new value
        adaptive_threshold = old_threshold * (1 - smoothing) + adaptive_threshold * smoothing
        
        self.adaptive_thresholds[signal_name] = adaptive_threshold
    
    def get_threshold(self, signal_name: str, regime: str = 'normal') -> float:
        """
        Get adaptive threshold for a signal, adjusted for current regime.
        
        Args:
            signal_name: Name of the signal
            regime: Current market regime (low_volatility, normal, high_volatility, trending, ranging)
            
        Returns:
            Adaptive threshold value
        """
        if signal_name not in self.adaptive_thresholds:
            return self.base_thresholds.get(signal_name, 0.5)
        
        base_adaptive = self.adaptive_thresholds[signal_name]
        regime_mult = self.regime_multipliers.get(regime, 1.0)
        
        return base_adaptive * regime_mult
    
    def get_signal_strength(self, signal_name: str, raw_value: float) -> float:
        """
        Calculate signal strength relative to historical distribution.
        
        Args:
            signal_name: Name of the signal
            raw_value: Raw signal value
            
        Returns:
            Normalized signal strength (z-score like)
        """
        if signal_name not in self.signal_stats:
            return raw_value
        
        stats = self.signal_stats[signal_name].get_statistics()
        std = stats['std']
        mean = stats['mean']
        
        if std == 0:
            return raw_value
        
        # Calculate z-score
        z_score = (raw_value - mean) / std
        
        # Convert to percentile-like score (0-1 range)
        # Using sigmoid-like transformation
        from scipy.special import expit
        strength = expit(z_score * 2)  # Scale factor of 2 for sensitivity
        
        return float(strength)
    
    def get_all_thresholds(self) -> Dict[str, float]:
        """Get all current adaptive thresholds."""
        return self.adaptive_thresholds.copy()
    
    def get_all_statistics(self) -> Dict[str, Dict]:
        """Get statistics for all signals."""
        return {
            name: stats.get_statistics()
            for name, stats in self.signal_stats.items()
        }
    
    def save_state(self, filepath: str):
        """Save current state to file."""
        state = {
            'window_size': self.window_size,
            'min_samples': self.min_samples,
            'base_thresholds': self.base_thresholds,
            'adaptive_thresholds': self.adaptive_thresholds,
            'signal_stats': {
                name: {
                    'values': list(stats.values),
                    'timestamps': [t.isoformat() for t in stats.timestamps]
                }
                for name, stats in self.signal_stats.items()
            }
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load state from file."""
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.window_size = state.get('window_size', self.window_size)
        self.min_samples = state.get('min_samples', self.min_samples)
        self.base_thresholds = state.get('base_thresholds', {})
        self.adaptive_thresholds = state.get('adaptive_thresholds', {})
        
        # Restore signal statistics
        for name, data in state.get('signal_stats', {}).items():
            self.signal_stats[name] = SignalStatistics(
                signal_name=name,
                window_size=self.window_size,
                values=deque(data['values'], maxlen=self.window_size),
                timestamps=deque(
                    [datetime.fromisoformat(t) for t in data['timestamps']],
                    maxlen=self.window_size
                )
            )


class RollingWindowNormalizer:
    """
    Normalizes signals using rolling window statistics.
    
    This handles non-stationary distributions by normalizing based on
    recent history rather than fixed parameters.
    """
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.windows: Dict[str, deque] = {}
    
    def update(self, signal_name: str, value: float) -> float:
        """
        Update window and return normalized value.
        
        Args:
            signal_name: Name of the signal
            value: Raw signal value
            
        Returns:
            Normalized value (z-score)
        """
        if signal_name not in self.windows:
            self.windows[signal_name] = deque(maxlen=self.window_size)
        
        window = self.windows[signal_name]
        window.append(value)
        
        if len(window) < 10:
            # Not enough data, return raw value scaled
            return value / 2.0
        
        # Calculate z-score
        arr = np.array(window)
        mean = np.mean(arr)
        std = np.std(arr)
        
        if std == 0:
            return 0.0
        
        z_score = (value - mean) / std
        
        # Clip to reasonable range
        return float(np.clip(z_score, -3.0, 3.0))
    
    def normalize_batch(self, signal_name: str, values: List[float]) -> List[float]:
        """Normalize a batch of values."""
        return [self.update(signal_name, v) for v in values]


# Global instance for use across modules
_adaptive_threshold_manager: Optional[AdaptiveThresholdManager] = None


def get_adaptive_threshold_manager(
    window_size: int = 100,
    min_samples: int = 20
) -> AdaptiveThresholdManager:
    """Get or create global adaptive threshold manager instance."""
    global _adaptive_threshold_manager
    if _adaptive_threshold_manager is None:
        _adaptive_threshold_manager = AdaptiveThresholdManager(
            window_size=window_size,
            min_samples=min_samples
        )
    return _adaptive_threshold_manager
