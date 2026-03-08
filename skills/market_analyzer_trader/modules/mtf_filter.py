"""
APEX Multi-Timeframe Filter
===========================
Require signal alignment across multiple timeframes.

AND logic: signal must appear on ≥3 timeframes to be valid.
Timeframes: 1m, 5m, 15m, 1h
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MTFSignal:
    """Multi-timeframe signal result."""
    timeframes: Dict[str, float]  # timeframe -> signal value
    aligned: bool
    alignment_count: int
    direction: str
    confidence: float


class MTFFilter:
    """
    Multi-timeframe alignment filter.

    Requires signal agreement across timeframes
    to reduce noise and false signals.
    """

    def __init__(self, 
                 min_alignment: int = 3,
                 timeframes: List[str] = None,
                 signal_threshold: float = 0.3):
        self.min_alignment = min_alignment
        self.timeframes = timeframes or ['1m', '5m', '15m', '1h']
        self.signal_threshold = signal_threshold

        # Cache for timeframe data
        self.data_cache: Dict[str, pd.DataFrame] = {}

    def _resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample data to specified timeframe."""
        if timeframe == '1m':
            return df

        rules = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }

        rule = rules.get(timeframe, '1h')
        return df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

    def _calculate_signal(self, df: pd.DataFrame) -> float:
        """
        Calculate signal for a timeframe.

        Returns: -1 to +1 signal strength
        """
        if len(df) < 20:
            return 0.0

        # Simple momentum + trend signal
        returns = df['close'].pct_change().dropna()

        # Trend (EMA slope)
        ema_fast = df['close'].ewm(span=12).mean()
        ema_slow = df['close'].ewm(span=26).mean()
        trend = np.sign(ema_fast.iloc[-1] - ema_slow.iloc[-1])

        # Momentum (recent returns)
        momentum = np.sign(returns.iloc[-5:].mean())

        # Volume confirmation
        vol_avg = df['volume'].iloc[-20:].mean()
        vol_recent = df['volume'].iloc[-5:].mean()
        vol_conf = 1 if vol_recent > vol_avg else -1

        # Combined signal
        signal = (trend + momentum + vol_conf) / 3
        return np.clip(signal, -1, 1)

    def check_alignment(self, 
                      df_1m: pd.DataFrame,
                      signal_generator = None) -> MTFSignal:
        """
        Check signal alignment across timeframes.

        Args:
            df_1m: 1-minute OHLCV data
            signal_generator: Optional custom signal function

        Returns:
            MTFSignal with alignment info
        """
        signals = {}
        directions = []

        for tf in self.timeframes:
            # Resample to timeframe
            if tf == '1m':
                tf_df = df_1m
            else:
                tf_df = self._resample_data(df_1m, tf)

            if len(tf_df) < 5:
                signals[tf] = 0.0
                continue

            # Calculate signal
            if signal_generator:
                sig = signal_generator(tf_df)
            else:
                sig = self._calculate_signal(tf_df)

            signals[tf] = sig

            # Count direction
            if abs(sig) >= self.signal_threshold:
                directions.append('buy' if sig > 0 else 'sell')

        # Check alignment
        if not directions:
            alignment_count = 0
            aligned = False
            direction = 'hold'
        else:
            buy_count = directions.count('buy')
            sell_count = directions.count('sell')

            if buy_count >= self.min_alignment:
                aligned = True
                alignment_count = buy_count
                direction = 'buy'
            elif sell_count >= self.min_alignment:
                aligned = True
                alignment_count = sell_count
                direction = 'sell'
            else:
                aligned = False
                alignment_count = max(buy_count, sell_count)
                direction = 'hold'

        # Confidence = alignment / total timeframes
        confidence = alignment_count / len(self.timeframes) if self.timeframes else 0

        return MTFSignal(
            timeframes=signals,
            aligned=aligned,
            alignment_count=alignment_count,
            direction=direction,
            confidence=confidence
        )

    def should_trade(self, mtf_signal: MTFSignal) -> Tuple[bool, str]:
        """Check if we should trade based on MTF alignment."""
        if mtf_signal.aligned:
            return True, f"Aligned on {mtf_signal.alignment_count} timeframes"
        else:
            return False, f"Only {mtf_signal.alignment_count}/{len(self.timeframes)} timeframes aligned"

    def get_summary(self, mtf_signal: MTFSignal) -> Dict:
        """Get MTF summary."""
        return {
            'aligned': mtf_signal.aligned,
            'direction': mtf_signal.direction,
            'alignment_count': mtf_signal.alignment_count,
            'confidence': mtf_signal.confidence,
            'timeframe_signals': mtf_signal.timeframes
        }


if __name__ == "__main__":
    print("APEX Multi-Timeframe Filter Demo")
    print("=" * 60)

    mtf = MTFFilter(min_alignment=3)

    # Generate synthetic 1m data
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=1000, freq='1min')
    prices = 3000 + np.cumsum(np.random.randn(1000) * 0.05)

    df = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + abs(np.random.randn(1000) * 0.001)),
        'low': prices * (1 - abs(np.random.randn(1000) * 0.001)),
        'close': prices,
        'volume': np.random.randint(100, 1000, 1000)
    }, index=dates)

    result = mtf.check_alignment(df)

    print(f"\nMTF Analysis:")
    print(f"  Aligned: {result.aligned}")
    print(f"  Direction: {result.direction.upper()}")
    print(f"  Alignment Count: {result.alignment_count}/{len(mtf.timeframes)}")
    print(f"  Confidence: {result.confidence:.1%}")

    print(f"\nTimeframe Signals:")
    for tf, sig in result.timeframes.items():
        direction = 'BUY' if sig > 0.3 else 'SELL' if sig < -0.3 else 'HOLD'
        print(f"  {tf}: {sig:+.3f} ({direction})")
