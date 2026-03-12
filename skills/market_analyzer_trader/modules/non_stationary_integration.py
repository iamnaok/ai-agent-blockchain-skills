"""
Non-Stationary Distribution Integration Module

Integrates adaptive thresholds and concept drift detection into the
APEX signal engine for handling non-stationary market conditions.

Phase 1 Implementation: Adaptive thresholds, rolling window normalization,
and concept drift detection.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np

# Import Phase 1 components
from .adaptive_thresholds import (
    get_adaptive_threshold_manager,
    RollingWindowNormalizer,
    AdaptiveThresholdManager
)
from .concept_drift_detector import (
    get_drift_detector,
    get_regime_normalizer,
    DriftEvent,
    DriftType
)

logger = logging.getLogger(__name__)


class NonStationarySignalProcessor:
    """
    Processes signals with non-stationary distribution handling.
    
    This class integrates:
    - Adaptive thresholds based on rolling window statistics
    - Concept drift detection for market regime changes
    - Regime-aware normalization for different market conditions
    """
    
    def __init__(
        self,
        window_size: int = 100,
        min_samples: int = 20,
        drift_threshold: float = 0.05
    ):
        """
        Initialize the non-stationary signal processor.
        
        Args:
            window_size: Size of rolling window for statistics
            min_samples: Minimum samples before adapting
            drift_threshold: Threshold for drift detection
        """
        self.window_size = window_size
        self.min_samples = min_samples
        
        # Initialize components
        self.threshold_manager = get_adaptive_threshold_manager(
            window_size=window_size,
            min_samples=min_samples
        )
        self.drift_detector = get_drift_detector(
            window_size=window_size,
            drift_threshold=drift_threshold
        )
        self.regime_normalizer = get_regime_normalizer(lookback=50)
        self.rolling_normalizer = RollingWindowNormalizer(window_size=window_size)
        
        # Track signal history for analysis
        self.signal_history: Dict[str, List[float]] = {}
        self.last_drift_event: Optional[DriftEvent] = None
        
        logger.info(f"NonStationarySignalProcessor initialized (window={window_size})")
    
    def process_signal(
        self,
        signal_name: str,
        raw_value: float,
        timestamp: Optional[datetime] = None,
        price: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process a signal with non-stationary handling.
        
        Args:
            signal_name: Name of the signal
            raw_value: Raw signal value
            timestamp: Optional timestamp
            price: Optional current price for regime detection
            volatility: Optional volatility for regime detection
            
        Returns:
            Dictionary with processed signal information
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Step 1: Update threshold manager with new value
        self.threshold_manager.update_signal(signal_name, raw_value, timestamp)
        
        # Step 2: Check for concept drift
        drift_event = self.drift_detector.update(signal_name, raw_value, timestamp)
        if drift_event:
            self.last_drift_event = drift_event
            logger.warning(
                f"Drift detected: {drift_event.drift_type.value} "
                f"for {signal_name} (severity: {drift_event.severity:.2f})"
            )
        
        # Step 3: Update regime normalizer if price/volatility provided
        if price is not None and volatility is not None:
            self.regime_normalizer.update(price, volatility)
        
        # Step 4: Get current market regime
        regime = self.regime_normalizer.get_regime()
        
        # Step 5: Calculate adaptive threshold
        adaptive_threshold = self.threshold_manager.get_threshold(signal_name, regime)
        
        # Step 6: Get signal strength (z-score like)
        signal_strength = self.threshold_manager.get_signal_strength(signal_name, raw_value)
        
        # Step 7: Apply regime-aware normalization
        normalized_value = self.regime_normalizer.normalize(raw_value)
        
        # Step 8: Apply rolling window normalization
        rolling_normalized = self.rolling_normalizer.update(signal_name, raw_value)
        
        # Step 9: Determine if signal is significant
        is_significant = abs(raw_value) >= adaptive_threshold
        
        # Step 10: Calculate confidence based on distribution
        stats = self.threshold_manager.signal_stats.get(signal_name)
        if stats and len(stats.values) >= self.min_samples:
            confidence = self._calculate_confidence(raw_value, stats.get_statistics())
        else:
            confidence = 0.5  # Neutral confidence with insufficient data
        
        return {
            'signal_name': signal_name,
            'raw_value': raw_value,
            'timestamp': timestamp,
            'adaptive_threshold': adaptive_threshold,
            'signal_strength': signal_strength,
            'regime': regime,
            'normalized_value': normalized_value,
            'rolling_normalized': rolling_normalized,
            'is_significant': is_significant,
            'confidence': confidence,
            'drift_detected': drift_event is not None,
            'drift_event': drift_event.to_dict() if drift_event else None,
            'samples_collected': len(self.threshold_manager.signal_stats.get(signal_name, []).values) if signal_name in self.threshold_manager.signal_stats else 0
        }
    
    def _calculate_confidence(self, value: float, stats: Dict[str, float]) -> float:
        """
        Calculate confidence based on how extreme the value is in the distribution.
        
        Args:
            value: Signal value
            stats: Distribution statistics
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        mean = stats['mean']
        std = stats['std']
        
        if std == 0:
            return 0.5
        
        # Calculate z-score
        z_score = abs(value - mean) / std
        
        # Convert to confidence using sigmoid
        # Higher z-score = higher confidence (more extreme = more confident)
        confidence = 1 / (1 + np.exp(-(z_score - 1.5)))
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def should_retrain(self, signal_name: Optional[str] = None) -> bool:
        """
        Check if retraining is recommended due to drift.
        
        Args:
            signal_name: Optional specific signal to check
            
        Returns:
            True if retraining is recommended
        """
        return self.drift_detector.should_retrain(signal_name)
    
    def get_drift_summary(self, hours: int = 24) -> Dict:
        """Get summary of recent drift events."""
        return self.drift_detector.get_drift_summary(hours)
    
    def get_all_thresholds(self) -> Dict[str, float]:
        """Get all current adaptive thresholds."""
        return self.threshold_manager.get_all_thresholds()
    
    def get_regime_info(self) -> Dict[str, Any]:
        """Get current regime information."""
        regime = self.regime_normalizer.get_regime()
        params = self.regime_normalizer.get_normalization_params()
        
        return {
            'current_regime': regime,
            'normalization_params': params,
            'price_history_length': len(self.regime_normalizer.price_history),
            'volatility_history_length': len(self.regime_normalizer.volatility_history)
        }
    
    def reset_signal(self, signal_name: str):
        """Reset statistics for a specific signal (e.g., after retraining)."""
        if signal_name in self.threshold_manager.signal_stats:
            self.threshold_manager.signal_stats[signal_name].values.clear()
            self.threshold_manager.signal_stats[signal_name].timestamps.clear()
            logger.info(f"Reset statistics for signal: {signal_name}")
    
    def save_state(self, filepath: str):
        """Save all component states."""
        self.threshold_manager.save_state(filepath.replace('.json', '_thresholds.json'))
        self.drift_detector.save_state(filepath.replace('.json', '_drift.json'))
        logger.info(f"Saved non-stationary processor state to {filepath}")
    
    def load_state(self, filepath: str):
        """Load all component states."""
        threshold_file = filepath.replace('.json', '_thresholds.json')
        drift_file = filepath.replace('.json', '_drift.json')
        
        if os.path.exists(threshold_file):
            self.threshold_manager.load_state(threshold_file)
        if os.path.exists(drift_file):
            self.drift_detector.load_state(drift_file)
        
        logger.info(f"Loaded non-stationary processor state from {filepath}")


# Integration helper for apex_signal_engine.py

def integrate_non_stationary_processing(
    signal_engine,
    window_size: int = 100,
    min_samples: int = 20
) -> NonStationarySignalProcessor:
    """
    Integrate non-stationary processing into the APEX signal engine.
    
    This function should be called during signal engine initialization
    to enable Phase 1 features.
    
    Args:
        signal_engine: The APEX signal engine instance
        window_size: Rolling window size
        min_samples: Minimum samples before adapting
        
    Returns:
        Configured NonStationarySignalProcessor instance
    """
    processor = NonStationarySignalProcessor(
        window_size=window_size,
        min_samples=min_samples
    )
    
    # Store reference in signal engine for access
    signal_engine.non_stationary_processor = processor
    
    logger.info("Non-stationary processing integrated into signal engine")
    
    return processor


def process_signal_with_adaptation(
    signal_engine,
    signal_name: str,
    raw_value: float,
    **kwargs
) -> Dict[str, Any]:
    """
    Process a signal with non-stationary adaptation.
    
    This is a convenience function that can be called from the signal engine
    to process signals with Phase 1 features.
    
    Args:
        signal_engine: The APEX signal engine instance
        signal_name: Name of the signal
        raw_value: Raw signal value
        **kwargs: Additional parameters (timestamp, price, volatility)
        
    Returns:
        Processed signal information
    """
    if not hasattr(signal_engine, 'non_stationary_processor'):
        # Initialize if not already done
        integrate_non_stationary_processing(signal_engine)
    
    return signal_engine.non_stationary_processor.process_signal(
        signal_name, raw_value, **kwargs
    )


# Example usage and testing
if __name__ == "__main__":
    # Test the non-stationary processor
    processor = NonStationarySignalProcessor(window_size=50, min_samples=10)
    
    # Simulate some signals
    import random
    for i in range(100):
        # Simulate changing distribution (non-stationary)
        if i < 30:
            value = random.gauss(0, 0.3)  # Low volatility
        elif i < 60:
            value = random.gauss(0.5, 0.5)  # Shifted mean, higher vol
        else:
            value = random.gauss(-0.3, 0.8)  # Different regime
        
        result = processor.process_signal(
            'test_signal',
            value,
            price=100 + i * 0.1,
            volatility=0.3 + (i % 50) / 100
        )
        
        if i % 20 == 0:
            print(f"Step {i}: value={value:.3f}, threshold={result['adaptive_threshold']:.3f}, "
                  f"regime={result['regime']}, drift={result['drift_detected']}")
    
    print("\nFinal thresholds:", processor.get_all_thresholds())
    print("Drift summary:", processor.get_drift_summary())
