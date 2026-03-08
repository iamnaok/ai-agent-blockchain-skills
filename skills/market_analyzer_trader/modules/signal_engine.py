"""
Signal Engine
=============
Integrates particle filter for real-time probability estimation
with technical indicators (RSI, MACD, VWAP) and order flow.

Output: Signal strength (-1 to +1) with confidence interval.
"""

import sys
sys.path.insert(0, '/a0/usr/workdir/ai-agent-blockchain-skills/skills/professional_quant_trader/modules')

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

# Import particle filter
try:
    from particle_filter import PredictionMarketParticleFilter
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "particle_filter",
        "/a0/usr/workdir/ai-agent-blockchain-skills/skills/professional_quant_trader/modules/particle_filter.py"
    )
    pf_module = importlib.util.module_from_spec(spec)
    sys.modules['particle_filter'] = pf_module
    spec.loader.exec_module(pf_module)
    PredictionMarketParticleFilter = pf_module.PredictionMarketParticleFilter

logger = logging.getLogger('SignalEngine')


@dataclass
class Signal:
    """Trading signal with confidence."""
    asset: str
    timestamp: datetime
    strength: float  # -1 to +1
    confidence: float  # 0 to 1
    raw_prob: float
    filtered_prob: float
    vwap_deviation: float
    rsi: float
    macd: float
    order_flow_delta: float
    direction: str  # 'buy', 'sell', or 'hold'


class SignalEngine:
    """
    Multi-factor signal generation engine.

    Combines:
    1. Particle filter for probability estimation
    2. Technical indicators (RSI, MACD, VWAP deviation)
    3. Order flow analysis (delta)
    """

    def __init__(self,
                 pf_particles: int = 5000,
                 pf_process_vol: float = 0.03,
                 pf_obs_noise: float = 0.05,
                 rsi_period: int = 14,
                 rsi_overbought: float = 70,
                 rsi_oversold: float = 30,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 vwap_window: int = 20,
                 delta_threshold: float = 0.2):
        """
        Args:
            pf_particles: Number of particles for filter
            pf_process_vol: Process volatility in logit space
            pf_obs_noise: Observation noise
            rsi_period: RSI lookback period
            rsi_overbought: RSI overbought threshold
            rsi_oversold: RSI oversold threshold
            macd_fast: MACD fast EMA period
            macd_slow: MACD slow EMA period
            macd_signal: MACD signal line period
            vwap_window: VWAP calculation window
            delta_threshold: Order flow delta threshold
        """
        self.pf_params = {
            'n_particles': pf_particles,
            'process_vol': pf_process_vol,
            'obs_noise': pf_obs_noise
        }
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.vwap_window = vwap_window
        self.delta_threshold = delta_threshold

        # Track filters per asset
        self.filters: Dict[str, PredictionMarketParticleFilter] = {}

    def _get_or_create_filter(self, asset: str) -> PredictionMarketParticleFilter:
        """Get or create particle filter for asset."""
        if asset not in self.filters:
            self.filters[asset] = PredictionMarketParticleFilter(
                n_particles=self.pf_params['n_particles'],
                process_vol=self.pf_params['process_vol'],
                obs_noise=self.pf_params['obs_noise'],
                prior_prob=0.50
            )
        return self.filters[asset]

    def calculate_rsi(self, prices: pd.Series) -> float:
        """
        Calculate Relative Strength Index.

        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss
        """
        if len(prices) < self.rsi_period:
            return 50.0

        deltas = prices.diff()
        gains = deltas.clip(lower=0)
        losses = -deltas.clip(upper=0)

        avg_gain = gains.rolling(window=self.rsi_period).mean().iloc[-1]
        avg_loss = losses.rolling(window=self.rsi_period).mean().iloc[-1]

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_macd(self, prices: pd.Series) -> Tuple[float, float, float]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        MACD = Fast EMA - Slow EMA
        Signal = EMA of MACD
        Histogram = MACD - Signal
        """
        if len(prices) < self.macd_slow:
            return 0.0, 0.0, 0.0

        ema_fast = prices.ewm(span=self.macd_fast).mean()
        ema_slow = prices.ewm(span=self.macd_slow).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal).mean()
        histogram = macd_line - signal_line

        return (
            macd_line.iloc[-1],
            signal_line.iloc[-1],
            histogram.iloc[-1]
        )

    def calculate_vwap_deviation(self, 
                                 df: pd.DataFrame,
                                 current_price: float) -> float:
        """
        Calculate deviation from VWAP.

        Returns: (current_price - vwap) / vwap
        """
        if len(df) < self.vwap_window:
            return 0.0

        recent = df.iloc[-self.vwap_window:]
        vwap = (recent['close'] * recent['volume']).sum() / recent['volume'].sum()

        return (current_price - vwap) / vwap if vwap != 0 else 0.0

    def calculate_order_flow_delta(self, df: pd.DataFrame) -> float:
        """
        Calculate order flow delta.

        Delta = (BidVol - AskVol) / TotalVol
        """
        if 'bid_volume' not in df.columns or 'ask_volume' not in df.columns:
            return 0.0

        recent = df.iloc[-self.vwap_window:]
        total_bid = recent['bid_volume'].sum()
        total_ask = recent['ask_volume'].sum()
        total = total_bid + total_ask

        if total == 0:
            return 0.0

        return (total_bid - total_ask) / total

    def generate_signal(self, 
                       asset: str,
                       df: pd.DataFrame,
                       market_price: float) -> Signal:
        """
        Generate trading signal for asset.

        Args:
            asset: Asset symbol
            df: DataFrame with OHLCV data
            market_price: Current market price (0-1 for prediction markets)

        Returns:
            Signal with strength (-1 to +1) and confidence
        """
        # 1. Particle filter for probability estimation
        pf = self._get_or_create_filter(asset)

        # Normalize market price for prediction market if needed
        if market_price > 1:
            normalized_price = market_price / 10000  # Scale down if needed
        else:
            normalized_price = market_price

        filtered_prob, ci_low, ci_high = pf.update(normalized_price)
        confidence = 1 - (ci_high - ci_low)  # Narrower CI = higher confidence

        # 2. Technical indicators
        prices = df['close']

        rsi = self.calculate_rsi(prices)
        macd, macd_signal, macd_hist = self.calculate_macd(prices)
        vwap_dev = self.calculate_vwap_deviation(df, market_price)

        # 3. Order flow
        delta = self.calculate_order_flow_delta(df)

        # 4. Combine into signal strength
        signal_components = []

        # Particle filter signal (probability vs market price)
        pf_signal = (filtered_prob - normalized_price) * 4  # Scale to -1 to 1
        signal_components.append(('particle_filter', pf_signal, 0.35))

        # RSI signal (mean reversion)
        if rsi > self.rsi_overbought:
            rsi_signal = -1.0  # Overbought, sell
        elif rsi < self.rsi_oversold:
            rsi_signal = 1.0   # Oversold, buy
        else:
            rsi_signal = 0.0
        signal_components.append(('rsi', rsi_signal, 0.20))

        # MACD signal (momentum)
        macd_signal_str = np.clip(macd_hist / abs(macd_hist).max() if macd_hist != 0 else 0, -1, 1)
        signal_components.append(('macd', macd_signal_str, 0.20))

        # VWAP deviation (trend/mean reversion)
        vwap_signal = -np.clip(vwap_dev * 10, -1, 1)  # Mean reversion
        signal_components.append(('vwap', vwap_signal, 0.15))

        # Order flow delta
        if abs(delta) > self.delta_threshold:
            delta_signal = np.sign(delta)
        else:
            delta_signal = 0.0
        signal_components.append(('delta', delta_signal, 0.10))

        # Weighted combination
        total_strength = sum(comp[1] * comp[2] for comp in signal_components)
        total_strength = np.clip(total_strength, -1, 1)

        # Determine direction
        if total_strength > 0.3:
            direction = 'buy'
        elif total_strength < -0.3:
            direction = 'sell'
        else:
            direction = 'hold'

        signal = Signal(
            asset=asset,
            timestamp=datetime.now(),
            strength=total_strength,
            confidence=confidence,
            raw_prob=normalized_price,
            filtered_prob=filtered_prob,
            vwap_deviation=vwap_dev,
            rsi=rsi,
            macd=macd_hist,
            order_flow_delta=delta,
            direction=direction
        )

        logger.info(f"Signal generated: {asset} | "
                   f"Strength: {total_strength:.3f} | "
                   f"Direction: {direction} | "
                   f"Confidence: {confidence:.3f}")

        return signal

    def get_signal_summary(self, signal: Signal) -> Dict:
        """Get human-readable signal summary."""
        return {
            'asset': signal.asset,
            'strength': f"{signal.strength:.3f} ({'Strong' if abs(signal.strength) > 0.7 else 'Moderate' if abs(signal.strength) > 0.3 else 'Weak'})",
            'direction': signal.direction.upper(),
            'confidence': f"{signal.confidence*100:.1f}%",
            'technical': {
                'RSI': f"{signal.rsi:.1f} ({'Overbought' if signal.rsi > 70 else 'Oversold' if signal.rsi < 30 else 'Neutral'})",
                'MACD': f"{signal.macd:.4f}",
                'VWAP Dev': f"{signal.vwap_deviation*100:.2f}%",
                'Order Flow Delta': f"{signal.order_flow_delta:.3f}"
            }
        }


if __name__ == "__main__":
    # Demo
    engine = SignalEngine()

    # Generate synthetic data
    dates = pd.date_range('2025-01-01', periods=100, freq='1h')
    df = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'high': 102 + np.cumsum(np.random.randn(100) * 0.5),
        'low': 98 + np.cumsum(np.random.randn(100) * 0.5),
        'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'volume': np.random.randint(1000, 10000, 100),
        'bid_volume': np.random.randint(500, 5000, 100),
        'ask_volume': np.random.randint(500, 5000, 100)
    }, index=dates)

    signal = engine.generate_signal("ETH", df, 0.62)
    summary = engine.get_signal_summary(signal)

    print(f"\nSignal Summary:")
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
