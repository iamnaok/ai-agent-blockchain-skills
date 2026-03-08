"""
APEX Smart Position Sizer
=========================
Enhanced Kelly Criterion with correlation matrix.

Features:
- Student-t copula for tail dependence (nu=4)
- Importance sampling for extreme contracts
- Position constraints and VaR limits
"""

import sys
sys.path.insert(0, '/a0/usr/workdir/ai-agent-blockchain-skills/skills/professional_quant_trader/modules')

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.special import gammaln

from kelly_portfolio_optimizer import KellyPortfolioOptimizer
from decimal import Decimal, ROUND_HALF_UP


@dataclass
class SmartSizeResult:
    """Smart sizer result with full risk breakdown."""
    position_size: float  # % of bankroll
    leverage: float
    notional: float
    var_95: float
    var_99: float
    tail_risk: float
    correlation_adjustment: float
    confidence: float

    def to_dict(self) -> Dict:
        return {
            'position_size': self.position_size,
            'leverage': self.leverage,
            'notional': self.notional,
            'var_95': self.var_95,
            'var_99': self.var_99,
            'tail_risk': self.tail_risk,
            'correlation_adjustment': self.correlation_adjustment,
            'confidence': self.confidence
        }


class StudentTCopula:
    """Student-t copula for tail dependence modeling."""

    def __init__(self, nu: float = 4.0):
        """
        Args:
            nu: Degrees of freedom (default 4 for moderate tails)
        """
        self.nu = nu

    def sample(self, correlation_matrix: np.ndarray, n: int = 1000) -> np.ndarray:
        """
        Sample from Student-t copula.

        Args:
            correlation_matrix: Correlation matrix
            n: Number of samples

        Returns:
            Samples in uniform space (n, dim)
        """
        d = correlation_matrix.shape[0]

        # Sample from multivariate t
        # X ~ t_nu(0, Sigma)
        Z = np.random.multivariate_normal(np.zeros(d), correlation_matrix, n)
        S = np.random.chisquare(self.nu, n) / self.nu

        X = Z / np.sqrt(S)[:, None]

        # Transform to uniform via CDF
        U = stats.t.cdf(X, self.nu)

        return U

    def tail_dependence(self, correlation: float) -> Tuple[float, float]:
        """
        Calculate tail dependence coefficients.

        Returns:
            (lower_tail, upper_tail)
        """
        # For Student-t copula, symmetric tail dependence
        # lambda = 2 * t_{nu+1}(-sqrt((nu+1)*(1-rho)/(1+rho)))
        if correlation >= 1:
            return 1.0, 1.0

        t_stat = -np.sqrt((self.nu + 1) * (1 - correlation) / (1 + correlation))
        tail = 2 * stats.t.cdf(t_stat, self.nu + 1)

        return tail, tail


class ImportanceSampler:
    """Importance sampling for extreme contract scenarios."""

    def __init__(self, n_samples: int = 10000):
        self.n_samples = n_samples

    def sample_extremes(self, 
                       mean: float,
                       std: float,
                       tail_prob: float = 0.05) -> np.ndarray:
        """
        Sample from tail regions.

        Args:
            mean: Distribution mean
            std: Distribution std
            tail_prob: Tail probability to sample from

        Returns:
            Extreme samples
        """
        # Sample from both tails
        n_tail = int(self.n_samples * tail_prob / 2)

        # Lower tail: z < -2
        lower_tail = stats.norm.ppf(
            np.random.uniform(0, tail_prob/2, n_tail)
        ) * std + mean

        # Upper tail: z > 2
        upper_tail = stats.norm.ppf(
            np.random.uniform(1 - tail_prob/2, 1, n_tail)
        ) * std + mean

        # Middle (standard sampling)
        middle = np.random.normal(mean, std, self.n_samples - 2 * n_tail)

        return np.concatenate([lower_tail, middle, upper_tail])


class SmartPositionSizer:
    """
    Smart position sizing with copula and importance sampling.

    Extends Kelly criterion with:
    1. Student-t copula for tail dependence
    2. Importance sampling for extremes
    3. Correlation adjustment
    """

    def __init__(self,
                 bankroll: float = 10000,
                 kelly_fraction: float = 0.25,
                 max_position: float = 0.10,
                 max_drawdown: float = 0.20,
                 max_leverage: float = 10.0,
                 copula_nu: float = 4.0):
        """
        Args:
            bankroll: Total capital
            kelly_fraction: Kelly multiplier (default 0.25 for quarter Kelly)
            max_position: Max position size as % of bankroll
            max_drawdown: Circuit breaker drawdown
            max_leverage: Max leverage allowed
            copula_nu: Student-t degrees of freedom
        """
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.max_position = max_position
        self.max_drawdown = max_drawdown
        self.max_leverage = max_leverage
        self.copula_nu = copula_nu

        self.copula = StudentTCopula(nu=copula_nu)
        self.sampler = ImportanceSampler()
        self.base_optimizer = KellyPortfolioOptimizer(
            bankroll=bankroll,
            max_position_pct=max_position,
            max_drawdown=max_drawdown
        )

    def calculate_size(self,
                     asset: str,
                     edge: float,
                     win_prob: float,
                     odds: float,
                     correlation_matrix: Optional[np.ndarray] = None,
                     market_data: Optional[pd.DataFrame] = None) -> SmartSizeResult:
        """
        Calculate smart position size.

        Args:
            asset: Asset symbol
            edge: Edge score (-1 to 1)
            win_prob: Estimated win probability
            odds: Average win/loss ratio
            correlation_matrix: Optional correlation matrix for multi-asset
            market_data: Market data for volatility estimation

        Returns:
            SmartSizeResult with full sizing info
        """
        # Base Kelly size
        base_size = self.base_optimizer.calculate_position_size(
            market_price=1.0,
            estimated_prob=win_prob,
            odds=odds,
            asset=asset
        )

        # Apply Kelly fraction
        kelly_size = base_size.position_pct * self.kelly_fraction

        # Calculate tail risk adjustment using copula
        tail_adjustment = 1.0
        if market_data is not None and len(market_data) > 20:
            # Calculate VaR with copula
            var_95, var_99 = self._calculate_var_copula(market_data, kelly_size)

            # Adjust size if VaR exceeds limit
            max_var = self.bankroll * 0.02  # 2% max VaR
            if var_95 > max_var:
                tail_adjustment = max_var / var_95

        # Correlation adjustment
        corr_adjustment = 1.0
        if correlation_matrix is not None:
            # Reduce size for correlated assets
            avg_corr = np.mean(np.abs(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]))
            corr_adjustment = 1.0 - (avg_corr * 0.3)  # Up to 30% reduction for high correlation

        # Final size
        final_size = kelly_size * tail_adjustment * corr_adjustment
        final_size = min(final_size, self.max_position)
        final_size = max(final_size, 0)  # No short via sizing

        # Calculate leverage
        if final_size > 0:
            leverage = min(final_size / 0.01, self.max_leverage)  # Base on 1% base position
            leverage = max(leverage, 1.0)
        else:
            leverage = 1.0

        # Calculate notional
        notional = self.bankroll * final_size * leverage

        # Recalculate VaR with final size
        var_95, var_99 = self._calculate_var_copula(
            market_data if market_data is not None else pd.DataFrame(),
            final_size
        )

        # Confidence based on adjustments
        confidence = min(tail_adjustment, corr_adjustment, 1.0)

        return SmartSizeResult(
            position_size=final_size,
            leverage=leverage,
            notional=notional,
            var_95=var_95,
            var_99=var_99,
            tail_risk=1 - tail_adjustment,
            correlation_adjustment=corr_adjustment,
            confidence=confidence
        )

    def _calculate_var_copula(self, 
                             df: pd.DataFrame, 
                             position_size: float) -> Tuple[float, float]:
        """Calculate VaR using copula."""
        if len(df) < 20:
            return position_size * 0.02, position_size * 0.05  # Default 2%, 5%

        returns = df['close'].pct_change().dropna()

        if len(returns) < 20:
            return position_size * 0.02, position_size * 0.05

        # Sample from copula
        corr = np.array([[1.0]])
        samples = self.copula.sample(corr, 1000)

        # Transform to returns
        mean = returns.mean()
        std = returns.std()
        sim_returns = stats.norm.ppf(samples[:, 0]) * std + mean

        # Calculate portfolio P&L
        portfolio_returns = position_size * sim_returns

        # VaR at 95% and 99%
        var_95 = np.percentile(portfolio_returns, 5)
        var_99 = np.percentile(portfolio_returns, 1)

        # Convert to positive VaR (loss)
        var_95 = -var_95 * self.bankroll if var_95 < 0 else 0
        var_99 = -var_99 * self.bankroll if var_99 < 0 else 0

        return var_95, var_99

    def validate_position(self, result: SmartSizeResult) -> Tuple[bool, str]:
        """Validate position meets risk criteria."""
        if result.position_size > self.max_position:
            return False, f"Position {result.position_size:.2%} > max {self.max_position:.2%}"

        if result.leverage > self.max_leverage:
            return False, f"Leverage {result.leverage:.1f}x > max {self.max_leverage:.1f}x"

        if result.var_95 > self.bankroll * 0.03:  # 3% VaR limit
            return False, f"VaR 95% {result.var_95:.2f} exceeds 3% limit"

        return True, "Position valid"


if __name__ == "__main__":
    print("APEX Smart Position Sizer Demo")
    print("=" * 60)

    sizer = SmartPositionSizer(
        bankroll=10000,
        kelly_fraction=0.25,
        max_position=0.10
    )

    # Generate synthetic market data
    np.random.seed(42)
    returns = np.random.randn(100) * 0.02  # 2% vol
    prices = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    })

    print(f"\nBankroll: ${sizer.bankroll:,.0f}")
    print(f"Kelly Fraction: {sizer.kelly_fraction}")

    result = sizer.calculate_size(
        asset="ETH",
        edge=0.6,
        win_prob=0.55,
        odds=2.0,
        market_data=df
    )

    print(f"\nSmart Size Result:")
    print(f"  Position Size: {result.position_size*100:.2f}% of bankroll")
    print(f"  Leverage: {result.leverage:.1f}x")
    print(f"  Notional: ${result.notional:,.2f}")
    print(f"  VaR 95%: ${result.var_95:,.2f}")
    print(f"  VaR 99%: ${result.var_99:,.2f}")
    print(f"  Confidence: {result.confidence:.1%}")

    valid, reason = sizer.validate_position(result)
    print(f"\nValidation: {reason}")
