"""
Concept Drift Detection Module for Non-Stationary Markets

Detects when market conditions change significantly, triggering model retraining
or parameter adjustments.
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import os


class DriftType(Enum):
    """Types of concept drift detected."""
    SUDDEN = "sudden"          # Abrupt change
    GRADUAL = "gradual"        # Slow change over time
    INCREMENTAL = "incremental"  # Step-by-step change
    RECURRING = "recurring"    # Cyclical patterns
    NONE = "none"              # No drift detected


@dataclass
class DriftEvent:
    """Represents a detected drift event."""
    timestamp: datetime
    drift_type: DriftType
    signal_name: str
    severity: float  # 0.0 to 1.0
    description: str
    recommended_action: str
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'drift_type': self.drift_type.value,
            'signal_name': self.signal_name,
            'severity': self.severity,
            'description': self.description,
            'recommended_action': self.recommended_action
        }


class StatisticalDriftDetector:
    """
    Detects concept drift using statistical methods.
    
    Methods implemented:
    - Kolmogorov-Smirnov test (distribution change)
    - CUSUM (cumulative sum for mean shift)
    - Page-Hinkley test (change in mean)
    - ADWIN (Adaptive Windowing)
    """
    
    def __init__(
        self,
        window_size: int = 100,
        drift_threshold: float = 0.05,
        min_samples: int = 30
    ):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.min_samples = min_samples
        
        # Reference window (older data)
        self.reference_window: deque = deque(maxlen=window_size)
        # Current window (recent data)
        self.current_window: deque = deque(maxlen=window_size)
        
        # CUSUM parameters
        self.cusum_threshold = 5.0
        self.cusum_min = 0.0
        self.cusum_max = 0.0
        
        # Page-Hinkley parameters
        self.ph_threshold = 50.0
        self.ph_sum = 0.0
        self.ph_min = float('inf')
        
        self.drift_detected = False
        self.last_drift_time: Optional[datetime] = None
    
    def add_value(self, value: float, timestamp: Optional[datetime] = None):
        """Add a new value to the detector."""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Initialize reference window
        if len(self.reference_window) < self.min_samples:
            self.reference_window.append(value)
            return None
        
        # Add to current window
        self.current_window.append(value)
        
        # Check for drift if we have enough samples
        if len(self.current_window) >= self.min_samples:
            return self._check_drift(timestamp)
        
        return None
    
    def _check_drift(self, timestamp: datetime) -> Optional[DriftEvent]:
        """Check for drift using multiple methods."""
        drift_detected = False
        severity = 0.0
        drift_type = DriftType.NONE
        description = ""
        
        # Method 1: KS Test (Kolmogorov-Smirnov)
        ks_drift, ks_severity = self._ks_test()
        if ks_drift:
            drift_detected = True
            severity = max(severity, ks_severity)
            drift_type = DriftType.SUDDEN
            description = "Distribution change detected (KS test)"
        
        # Method 2: CUSUM
        cusum_drift, cusum_severity = self._cusum_test()
        if cusum_drift:
            drift_detected = True
            severity = max(severity, cusum_severity)
            if drift_type == DriftType.NONE:
                drift_type = DriftType.GRADUAL
            description += "; Mean shift detected (CUSUM)"
        
        # Method 3: Page-Hinkley
        ph_drift, ph_severity = self._page_hinkley_test()
        if ph_drift:
            drift_detected = True
            severity = max(severity, ph_severity)
            if drift_type == DriftType.NONE:
                drift_type = DriftType.INCREMENTAL
            description += "; Incremental change detected (Page-Hinkley)"
        
        if drift_detected and not self.drift_detected:
            self.drift_detected = True
            self.last_drift_time = timestamp
            
            # Determine recommended action
            if severity > 0.7:
                action = "Immediate retraining recommended"
            elif severity > 0.4:
                action = "Schedule retraining within 24 hours"
            else:
                action = "Monitor closely, adjust thresholds"
            
            return DriftEvent(
                timestamp=timestamp,
                drift_type=drift_type,
                signal_name="market",
                severity=severity,
                description=description.strip("; "),
                recommended_action=action
            )
        
        # Reset drift flag if conditions normalize
        if not drift_detected:
            self.drift_detected = False
        
        return None
    
    def _ks_test(self) -> Tuple[bool, float]:
        """Kolmogorov-Smirnov test for distribution change."""
        if len(self.reference_window) < self.min_samples or len(self.current_window) < self.min_samples:
            return False, 0.0
        
        ref = np.array(self.reference_window)
        cur = np.array(self.current_window)
        
        # Calculate empirical CDFs
        ref_sorted = np.sort(ref)
        cur_sorted = np.sort(cur)
        
        # KS statistic (simplified)
        ref_mean = np.mean(ref)
        cur_mean = np.mean(cur)
        ref_std = np.std(ref)
        cur_std = np.std(cur)
        
        if ref_std == 0 or cur_std == 0:
            return False, 0.0
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((ref_std**2 + cur_std**2) / 2)
        effect_size = abs(cur_mean - ref_mean) / pooled_std
        
        # Drift detected if effect size is large
        drift = effect_size > 0.5  # Medium to large effect
        severity = min(1.0, effect_size / 2.0)
        
        return drift, severity
    
    def _cusum_test(self) -> Tuple[bool, float]:
        """CUSUM test for mean shift."""
        if len(self.current_window) < 2:
            return False, 0.0
        
        values = list(self.current_window)
        mean = np.mean(values[:-1]) if len(values) > 1 else 0
        
        # CUSUM calculation
        for x in values:
            self.cusum_max = max(0, self.cusum_max + x - mean - 0.5)
            self.cusum_min = min(0, self.cusum_min + x - mean + 0.5)
        
        drift = (self.cusum_max > self.cusum_threshold or 
                abs(self.cusum_min) > self.cusum_threshold)
        
        severity = min(1.0, max(self.cusum_max, abs(self.cusum_min)) / self.cusum_threshold / 2)
        
        return drift, severity
    
    def _page_hinkley_test(self) -> Tuple[bool, float]:
        """Page-Hinkley test for incremental change."""
        if len(self.current_window) < 2:
            return False, 0.0
        
        values = list(self.current_window)
        mean = np.mean(values)
        
        # Page-Hinkley statistic
        for x in values:
            self.ph_sum += x - mean - 0.1  # 0.1 is the minimum change magnitude
            self.ph_min = min(self.ph_min, self.ph_sum)
        
        ph_stat = self.ph_sum - self.ph_min
        drift = ph_stat > self.ph_threshold
        severity = min(1.0, ph_stat / self.ph_threshold / 2)
        
        return drift, severity
    
    def reset(self):
        """Reset the detector after drift handling."""
        # Move current window to reference
        self.reference_window = self.current_window.copy()
        self.current_window.clear()
        
        # Reset CUSUM
        self.cusum_max = 0.0
        self.cusum_min = 0.0
        
        # Reset Page-Hinkley
        self.ph_sum = 0.0
        self.ph_min = float('inf')
        
        self.drift_detected = False


class MultiSignalDriftDetector:
    """
    Manages drift detection for multiple signals.
    """
    
    def __init__(
        self,
        window_size: int = 100,
        drift_threshold: float = 0.05
    ):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.detectors: Dict[str, StatisticalDriftDetector] = {}
        self.drift_history: List[DriftEvent] = []
        self.callbacks: List[Callable[[DriftEvent], None]] = []
    
    def register_signal(self, signal_name: str):
        """Register a signal for drift detection."""
        if signal_name not in self.detectors:
            self.detectors[signal_name] = StatisticalDriftDetector(
                window_size=self.window_size,
                drift_threshold=self.drift_threshold
            )
    
    def update(self, signal_name: str, value: float, timestamp: Optional[datetime] = None):
        """Update drift detector for a signal."""
        if signal_name not in self.detectors:
            self.register_signal(signal_name)
        
        event = self.detectors[signal_name].add_value(value, timestamp)
        
        if event:
            event.signal_name = signal_name
            self.drift_history.append(event)
            
            # Trigger callbacks
            for callback in self.callbacks:
                callback(event)
            
            return event
        
        return None
    
    def on_drift_detected(self, callback: Callable[[DriftEvent], None]):
        """Register a callback for drift events."""
        self.callbacks.append(callback)
    
    def get_drift_summary(self, hours: int = 24) -> Dict:
        """Get summary of recent drift events."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_events = [e for e in self.drift_history if e.timestamp > cutoff]
        
        return {
            'total_events': len(recent_events),
            'by_type': {
                drift_type.value: len([e for e in recent_events if e.drift_type == drift_type])
                for drift_type in DriftType
            },
            'by_signal': {
                signal: len([e for e in recent_events if e.signal_name == signal])
                for signal in set(e.signal_name for e in recent_events)
            },
            'average_severity': np.mean([e.severity for e in recent_events]) if recent_events else 0.0,
            'latest_event': recent_events[-1].to_dict() if recent_events else None
        }
    
    def should_retrain(self, signal_name: Optional[str] = None) -> bool:
        """Check if retraining is recommended."""
        cutoff = datetime.now() - timedelta(hours=24)
        
        if signal_name:
            events = [e for e in self.drift_history 
                     if e.signal_name == signal_name and e.timestamp > cutoff]
        else:
            events = [e for e in self.drift_history if e.timestamp > cutoff]
        
        # Retrain if severe drift detected
        severe_events = [e for e in events if e.severity > 0.6]
        return len(severe_events) > 0
    
    def save_state(self, filepath: str):
        """Save drift detection state."""
        state = {
            'drift_history': [e.to_dict() for e in self.drift_history[-1000:]],  # Keep last 1000
            'detector_params': {
                'window_size': self.window_size,
                'drift_threshold': self.drift_threshold
            }
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load drift detection state."""
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.drift_history = [
            DriftEvent(
                timestamp=datetime.fromisoformat(e['timestamp']),
                drift_type=DriftType(e['drift_type']),
                signal_name=e['signal_name'],
                severity=e['severity'],
                description=e['description'],
                recommended_action=e['recommended_action']
            )
            for e in state.get('drift_history', [])
        ]


class RegimeAwareNormalizer:
    """
    Normalizes signals based on detected market regime.
    Adjusts for different behaviors in trending vs ranging markets.
    """
    
    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        self.price_history: deque = deque(maxlen=lookback)
        self.volatility_history: deque = deque(maxlen=lookback)
    
    def update(self, price: float, volatility: float):
        """Update with latest market data."""
        self.price_history.append(price)
        self.volatility_history.append(volatility)
    
    def get_regime(self) -> str:
        """Detect current market regime."""
        if len(self.price_history) < self.lookback // 2:
            return 'unknown'
        
        prices = np.array(self.price_history)
        volatilities = np.array(self.volatility_history)
        
        # Calculate trend strength
        returns = np.diff(np.log(prices))
        trend_strength = abs(np.mean(returns)) / (np.std(returns) + 1e-8)
        
        # Calculate average volatility
        avg_vol = np.mean(volatilities) if len(volatilities) > 0 else 0.3
        
        # Determine regime
        if trend_strength > 0.5 and avg_vol > 0.4:
            return 'trending_volatile'
        elif trend_strength > 0.5:
            return 'trending'
        elif avg_vol > 0.5:
            return 'high_volatility'
        elif avg_vol < 0.2:
            return 'low_volatility'
        else:
            return 'ranging'
    
    def get_normalization_params(self) -> Dict[str, float]:
        """Get normalization parameters for current regime."""
        regime = self.get_regime()
        
        params = {
            'unknown': {'mean': 0.0, 'scale': 1.0},
            'trending_volatile': {'mean': 0.0, 'scale': 1.5},
            'trending': {'mean': 0.1, 'scale': 1.2},
            'high_volatility': {'mean': 0.0, 'scale': 1.8},
            'low_volatility': {'mean': 0.0, 'scale': 0.7},
            'ranging': {'mean': 0.0, 'scale': 0.9}
        }
        
        return params.get(regime, params['unknown'])
    
    def normalize(self, value: float) -> float:
        """Normalize value based on current regime."""
        params = self.get_normalization_params()
        return (value - params['mean']) / params['scale']


# Global instance
_drift_detector: Optional[MultiSignalDriftDetector] = None
_regime_normalizer: Optional[RegimeAwareNormalizer] = None


def get_drift_detector(
    window_size: int = 100,
    drift_threshold: float = 0.05
) -> MultiSignalDriftDetector:
    """Get or create global drift detector instance."""
    global _drift_detector
    if _drift_detector is None:
        _drift_detector = MultiSignalDriftDetector(
            window_size=window_size,
            drift_threshold=drift_threshold
        )
    return _drift_detector


def get_regime_normalizer(lookback: int = 50) -> RegimeAwareNormalizer:
    """Get or create global regime normalizer."""
    global _regime_normalizer
    if _regime_normalizer is None:
        _regime_normalizer = RegimeAwareNormalizer(lookback=lookback)
    return _regime_normalizer
