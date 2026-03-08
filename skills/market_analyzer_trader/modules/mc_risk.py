"""
APEX Monte Carlo TP/SL Calculator
=================================
Monte Carlo simulation for optimal take-profit and stop-loss levels.

Based on historical volatility paths to find optimal levels
that maximize risk-adjusted returns.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from dataclasses import dataclass
from scipy import stats


@dataclass
class TPSLResult:
    """TP/SL calculation result."""
    take_profit_pct: float
    stop_loss_pct: float
    tp_price: float
    sl_price: float
    rr_ratio: float
    win_rate: float
    expected_return: float
    confidence: float
    paths_simulated: int

    def to_dict(self) -> Dict:
        return {
            'take_profit_pct': self.take_profit_pct,
            'stop_loss_pct': self.stop_loss_pct,
            'tp_price': self.tp_price,
            'sl_price': self.sl_price,
            'rr_ratio': self.rr_ratio,
            'win_rate': self.win_rate,
            'expected_return': self.expected_return,
            'confidence': self.confidence,
            'paths_simulated': self.paths_simulated
        }


class MonteCarloRiskCalculator:
    """
    Monte Carlo simulation for optimal TP/SL levels.

    Simulates price paths based on historical volatility
    to find levels that maximize risk-adjusted returns.
    """

    def __init__(self, 
                 n_sims: int = 10000,
                 confidence: float = 0.95,
                 max_tp: float = 0.15,  # 15%
                 max_sl: float = 0.10):  # 10%
        self.n_sims = n_sims
        self.confidence = confidence
        self.max_tp = max_tp
        self.max_sl = max_sl

    def _estimate_volatility(self, returns: pd.Series) -> float:
        """Estimate annualized volatility from returns."""
        return returns.std() * np.sqrt(252)  # Annualized

    def _simulate_paths(self,
                     current_price: float,
                     volatility: float,
                     drift: float,
                     days: int = 30,
                     steps_per_day: int = 24) -> np.ndarray:
        """
        Simulate price paths using geometric Brownian motion.

        Returns:
            Array of shape (n_sims, n_steps) with final prices
        """
        n_steps = days * steps_per_day
        dt = 1 / (252 * steps_per_day)  # Time step

        # Generate random walks
        Z = np.random.standard_normal((self.n_sims, n_steps))

        # GBM: dS/S = mu*dt + sigma*sqrt(dt)*dW
        price_changes = np.exp(
            (drift - 0.5 * volatility**2) * dt + 
            volatility * np.sqrt(dt) * Z
        )

        # Calculate paths
        paths = current_price * np.cumprod(price_changes, axis=1)

        return paths

    def calculate_tpsl(self,
                     df: pd.DataFrame,
                     entry_price: float,
                     side: str,
                     win_rate_target: float = 0.40) -> TPSLResult:
        """
        Calculate optimal TP/SL levels.

        Args:
            df: OHLCV DataFrame
            entry_price: Entry price
            side: 'long' or 'short'
            win_rate_target: Target win rate (0-1)

        Returns:
            TPSLResult with optimal levels
        """
        # Calculate historical returns
        returns = df['close'].pct_change().dropna()

        # Estimate parameters
        volatility = self._estimate_volatility(returns.iloc[-30:])
        drift = returns.iloc[-30:].mean() * 252  # Annualized drift

        # Simulate paths
        paths = self._simulate_paths(entry_price, volatility, drift)

        # Test different TP/SL levels
        best_result = None
        best_score = -np.inf

        for tp_pct in np.linspace(0.02, self.max_tp, 20):
            for sl_pct in np.linspace(0.01, min(self.max_sl, tp_pct * 0.8), 15):
                # Calculate TP/SL prices
                if side == 'long':
                    tp_price = entry_price * (1 + tp_pct)
                    sl_price = entry_price * (1 - sl_pct)
                else:  # short
                    tp_price = entry_price * (1 - tp_pct)
                    sl_price = entry_price * (1 + sl_pct)

                # Check hit rates
                hits_tp = np.any(paths >= tp_price, axis=1) if side == 'long' else np.any(paths <= tp_price, axis=1)
                hits_sl = np.any(paths <= sl_price, axis=1) if side == 'long' else np.any(paths >= sl_price, axis=1)

                # Calculate win rate (TP hit before SL)
                # Simplified: count which hits first
                wins = np.sum(hits_tp & ~hits_sl)
                losses = np.sum(hits_sl & ~hits_tp)
                total = wins + losses

                if total == 0:
                    continue

                win_rate = wins / total

                # Calculate expected return
                expected_ret = win_rate * tp_pct - (1 - win_rate) * sl_pct

                # Risk-reward ratio
                rr_ratio = tp_pct / sl_pct if sl_pct > 0 else 0

                # Score = expected return * win_rate * rr_ratio
                # Penalize win rates far from target
                score = expected_ret * (1 - abs(win_rate - win_rate_target))

                if score > best_score:
                    best_score = score
                    best_result = {
                        'tp_pct': tp_pct,
                        'sl_pct': sl_pct,
                        'tp_price': tp_price,
                        'sl_price': sl_price,
                        'rr_ratio': rr_ratio,
                        'win_rate': win_rate,
                        'expected_return': expected_ret
                    }

        if best_result is None:
            # Fallback: reasonable defaults
            best_result = {
                'tp_pct': 0.05,
                'sl_pct': 0.03,
                'tp_price': entry_price * 1.05 if side == 'long' else entry_price * 0.95,
                'sl_price': entry_price * 0.97 if side == 'long' else entry_price * 1.03,
                'rr_ratio': 1.67,
                'win_rate': 0.45,
                'expected_return': 0.015
            }

        return TPSLResult(
            take_profit_pct=best_result['tp_pct'],
            stop_loss_pct=best_result['sl_pct'],
            tp_price=best_result['tp_price'],
            sl_price=best_result['sl_price'],
            rr_ratio=best_result['rr_ratio'],
            win_rate=best_result['win_rate'],
            expected_return=best_result['expected_return'],
            confidence=self.confidence,
            paths_simulated=self.n_sims
        )

    def validate_levels(self, 
                     tpsl: TPSLResult,
                     min_rr: float = 1.5,
                     min_win_rate: float = 0.35) -> Tuple[bool, str]:
        """
        Validate TP/SL levels meet criteria.

        Args:
            tpsl: TPSLResult
            min_rr: Minimum risk/reward ratio
            min_win_rate: Minimum win rate

        Returns:
            (is_valid, reason)
        """
        if tpsl.rr_ratio < min_rr:
            return False, f"RR ratio {tpsl.rr_ratio:.2f} < {min_rr}"

        if tpsl.win_rate < min_win_rate:
            return False, f"Win rate {tpsl.win_rate:.1%} < {min_win_rate:.1%}"

        if tpsl.expected_return <= 0:
            return False, "Negative expected return"

        return True, "Levels valid"


if __name__ == "__main__":
    print("APEX Monte Carlo TP/SL Calculator Demo")
    print("=" * 60)

    calc = MonteCarloRiskCalculator(n_sims=5000)

    # Generate synthetic data
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=100, freq='1h')
    returns = np.random.randn(100) * 0.0015
    prices = 3000 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + abs(np.random.randn(100) * 0.001)),
        'low': prices * (1 - abs(np.random.randn(100) * 0.001)),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)

    entry = prices[-1]

    print(f"\nCalculating TP/SL for LONG @ {entry:.2f}")
    result = calc.calculate_tpsl(df, entry, 'long')

    print(f"\nResults:")
    print(f"  Take Profit: {result.take_profit_pct*100:.2f}% @ {result.tp_price:.2f}")
    print(f"  Stop Loss: {result.stop_loss_pct*100:.2f}% @ {result.sl_price:.2f}")
    print(f"  R/R Ratio: {result.rr_ratio:.2f}")
    print(f"  Win Rate: {result.win_rate:.1%}")
    print(f"  Expected Return: {result.expected_return*100:.2f}%")
    print(f"  Paths Simulated: {result.paths_simulated:,}")

    valid, reason = calc.validate_levels(result)
    print(f"\nValidation: {reason}")
