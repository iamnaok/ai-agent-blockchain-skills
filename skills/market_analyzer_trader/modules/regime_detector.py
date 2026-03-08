"""
APEX Regime Detector (HMM)
=========================
Hidden Markov Model for market regime detection.

States:
- TRENDING_UP (bullish trend)
- TRENDING_DOWN (bearish trend)
- RANGE_BOUND (consolidation)
- HIGH_VOL (high volatility)

Regime multiplier:
- HIGH_VOL: 0.5× edge score (reduce exposure)
- Others: 1.0× edge score
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from scipy.stats import norm


class MarketRegime(Enum):
    """Market regime states."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE_BOUND = "range_bound"
    HIGH_VOL = "high_vol"
    UNKNOWN = "unknown"


@dataclass
class RegimeState:
    """Current regime state with confidence."""
    regime: MarketRegime
    confidence: float
    duration: int  # Bars in current regime
    multiplier: float  # Edge score multiplier
    features: Dict[str, float]  # Feature values


class HiddenMarkovModel:
    """
    Simple HMM for regime detection.

    Uses Gaussian emissions for features:
    - Returns (mean, vol)
    - Trend strength (ADX-like)
    - Volatility regime
    """

    def __init__(self, n_states: int = 4):
        self.n_states = n_states
        self.states = list(MarketRegime)[:n_states]

        # Transition matrix (simplified - equal probability)
        self.transitions = np.ones((n_states, n_states)) / n_states
        np.fill_diagonal(self.transitions, 0.7)  # Stay in state 70%
        self.transitions = self.transitions / self.transitions.sum(axis=1, keepdims=True)

        # Emission parameters (mean, std) for each feature
        self.emissions = {
            MarketRegime.TRENDING_UP: {
                'returns_mean': 0.001, 'returns_std': 0.015,
                'trend_strength': 0.7, 'vol_regime': 0.3
            },
            MarketRegime.TRENDING_DOWN: {
                'returns_mean': -0.001, 'returns_std': 0.015,
                'trend_strength': 0.7, 'vol_regime': 0.3
            },
            MarketRegime.RANGE_BOUND: {
                'returns_mean': 0.0, 'returns_std': 0.008,
                'trend_strength': 0.2, 'vol_regime': 0.5
            },
            MarketRegime.HIGH_VOL: {
                'returns_mean': 0.0, 'returns_std': 0.04,
                'trend_strength': 0.4, 'vol_regime': 1.0
            }
        }

        # Current state probabilities
        self.state_probs = np.ones(n_states) / n_states
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_duration = 0

    def _extract_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract regime features from OHLCV data."""
        returns = df['close'].pct_change().dropna()

        # Returns features
        returns_mean = returns.iloc[-20:].mean()
        returns_std = returns.iloc[-20:].std()

        # Trend strength (ADX-like: directional movement / total movement)
        highs = df['high'].iloc[-20:]
        lows = df['low'].iloc[-20:]
        up_moves = highs.diff().clip(lower=0).sum()
        down_moves = (-lows.diff()).clip(lower=0).sum()
        total_range = (highs - lows).sum()
        trend_strength = (up_moves - down_moves) / total_range if total_range > 0 else 0

        # Volatility regime
        vol_regime = returns_std / returns.std() if returns.std() > 0 else 1.0

        return {
            'returns_mean': returns_mean,
            'returns_std': returns_std,
            'trend_strength': abs(trend_strength),
            'vol_regime': vol_regime
        }

    def _emission_prob(self, features: Dict[str, float], state: MarketRegime) -> float:
        """Calculate emission probability for features given state."""
        params = self.emissions.get(state, self.emissions[MarketRegime.RANGE_BOUND])

        # Gaussian likelihood for each feature
        p_returns = norm.pdf(features['returns_mean'], 
                            params['returns_mean'], 
                            params['returns_std'])
        p_trend = norm.pdf(features['trend_strength'],
                          params['trend_strength'],
                          0.2)
        p_vol = norm.pdf(features['vol_regime'],
                        params['vol_regime'],
                        0.2)

        return p_returns * p_trend * p_vol

    def update(self, df: pd.DataFrame) -> RegimeState:
        """
        Update HMM with new data and return current regime.

        Args:
            df: OHLCV DataFrame

        Returns:
            RegimeState with current regime and multiplier
        """
        features = self._extract_features(df)

        # Calculate emission probabilities
        emissions = np.array([
            self._emission_prob(features, state)
            for state in self.states
        ])

        # Forward algorithm
        new_probs = self.state_probs @ self.transitions * emissions
        new_probs = new_probs / new_probs.sum()  # Normalize

        self.state_probs = new_probs

        # Get most likely state
        max_idx = np.argmax(self.state_probs)
        new_regime = self.states[max_idx]
        confidence = self.state_probs[max_idx]

        # Update duration
        if new_regime == self.current_regime:
            self.regime_duration += 1
        else:
            self.regime_duration = 1
            self.current_regime = new_regime

        # Multiplier: HIGH_VOL gets 0.5×
        multiplier = 0.5 if new_regime == MarketRegime.HIGH_VOL else 1.0

        return RegimeState(
            regime=new_regime,
            confidence=confidence,
            duration=self.regime_duration,
            multiplier=multiplier,
            features=features
        )


class RegimeDetector:
    """
    Market regime detector with HMM.

    Provides regime-aware edge score adjustment.
    """

    def __init__(self, use_hmm: bool = True):
        self.use_hmm = use_hmm
        self.hmm = HiddenMarkovModel() if use_hmm else None
        self.current_regime = RegimeState(
            regime=MarketRegime.UNKNOWN,
            confidence=0.0,
            duration=0,
            multiplier=1.0,
            features={}
        )
        self.regime_history: List[RegimeState] = []

    def update(self, df: pd.DataFrame) -> RegimeState:
        """Update regime detection."""
        if self.use_hmm:
            self.current_regime = self.hmm.update(df)
        else:
            # Simple rule-based fallback
            self.current_regime = self._rule_based_regime(df)

        self.regime_history.append(self.current_regime)
        return self.current_regime

    def _rule_based_regime(self, df: pd.DataFrame) -> RegimeState:
        """Fallback rule-based regime detection."""
        returns = df['close'].pct_change().iloc[-20:]
        vol = returns.std()
        trend = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1)

        if vol > returns.std() * 2:
            regime = MarketRegime.HIGH_VOL
            multiplier = 0.5
        elif abs(trend) > 0.05:
            regime = MarketRegime.TRENDING_UP if trend > 0 else MarketRegime.TRENDING_DOWN
            multiplier = 1.0
        else:
            regime = MarketRegime.RANGE_BOUND
            multiplier = 1.0

        return RegimeState(
            regime=regime,
            confidence=0.7,
            duration=len(self.regime_history),
            multiplier=multiplier,
            features={'vol': vol, 'trend': trend}
        )

    def apply_multiplier(self, edge_score: float) -> float:
        """Apply regime multiplier to edge score."""
        return edge_score * self.current_regime.multiplier

    def should_trade(self) -> Tuple[bool, str]:
        """Check if we should trade in current regime."""
        if self.current_regime.regime == MarketRegime.HIGH_VOL:
            if self.current_regime.confidence > 0.8:
                return False, "High volatility regime - reducing exposure"

        if self.current_regime.regime == MarketRegime.UNKNOWN:
            return False, "Unknown regime - waiting for clarity"

        return True, f"OK - {self.current_regime.regime.value}"

    def get_summary(self) -> Dict:
        """Get regime summary."""
        return {
            'current_regime': self.current_regime.regime.value,
            'confidence': self.current_regime.confidence,
            'duration': self.current_regime.duration,
            'multiplier': self.current_regime.multiplier,
            'features': self.current_regime.features
        }


if __name__ == "__main__":
    print("APEX Regime Detector (HMM) Demo")
    print("=" * 60)

    detector = RegimeDetector()

    # Generate synthetic regimes
    np.random.seed(42)

    for regime_name, params in [
        ("TRENDING_UP", {"trend": 0.1, "vol": 0.015}),
        ("RANGE_BOUND", {"trend": 0.0, "vol": 0.008}),
        ("HIGH_VOL", {"trend": 0.0, "vol": 0.04})
    ]:
        dates = pd.date_range('2025-01-01', periods=50, freq='1h')
        returns = np.random.normal(params["trend"]/50, params["vol"], 50)
        prices = 3000 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'open': prices * (1 + np.random.randn(50) * 0.001),
            'high': prices * (1 + abs(np.random.randn(50) * 0.003)),
            'low': prices * (1 - abs(np.random.randn(50) * 0.003)),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 50)
        }, index=dates)

        state = detector.update(df)
        print(f"\n{regime_name}:")
        print(f"  Detected: {state.regime.value}")
        print(f"  Confidence: {state.confidence:.2f}")
        print(f"  Multiplier: {state.multiplier}")
