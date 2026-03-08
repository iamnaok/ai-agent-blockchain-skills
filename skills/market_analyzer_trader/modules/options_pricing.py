"""
Options Pricing Module
======================
Black-Scholes, Monte Carlo, and Greeks calculation for crypto options.
Jane Street-style quant pricing for perpetual options and expiring contracts.
"""

import numpy as np
from scipy import stats
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger('OptionsPricing')


@dataclass
class OptionParams:
    """Standardized option parameters."""
    S: float          # Spot price
    K: float          # Strike price
    T: float          # Time to maturity (years)
    r: float          # Risk-free rate
    sigma: float      # Volatility
    option_type: str  # 'call' or 'put'


@dataclass
class Greeks:
    """Option Greeks."""
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


@dataclass
class OptionPrice:
    """Complete option pricing result."""
    price: float
    intrinsic: float
    time_value: float
    greeks: Greeks
    implied_vol: Optional[float] = None


class BlackScholesPricer:
    """
    Black-Scholes-Merton option pricing model.

    The foundation of modern quantitative finance.
    d1 = (ln(S/K) + (r + σ²/2)T) / (σ√T)
    d2 = d1 - σ√T

    Call = S*N(d1) - K*exp(-rT)*N(d2)
    Put = K*exp(-rT)*N(-d2) - S*N(-d1)
    """

    @staticmethod
    def calculate_d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
        """Calculate d1 and d2 (the Jane Street whiteboard test)."""
        if T <= 0 or sigma <= 0:
            return 0.0, 0.0

        # d1 = (ln(S/K) + (r + σ²/2)T) / (σ√T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

        # d2 = d1 - σ√T
        d2 = d1 - sigma * np.sqrt(T)

        return d1, d2

    @classmethod
    def price(cls, params: OptionParams) -> float:
        """Calculate option price using Black-Scholes."""
        d1, d2 = cls.calculate_d1_d2(params.S, params.K, params.T, params.r, params.sigma)

        if params.option_type == 'call':
            # Call = S*N(d1) - K*exp(-rT)*N(d2)
            price = params.S * stats.norm.cdf(d1) - params.K * np.exp(-params.r * params.T) * stats.norm.cdf(d2)
        else:
            # Put = K*exp(-rT)*N(-d2) - S*N(-d1)
            price = params.K * np.exp(-params.r * params.T) * stats.norm.cdf(-d2) - params.S * stats.norm.cdf(-d1)

        return max(price, 0.0)

    @classmethod
    def greeks(cls, params: OptionParams) -> Greeks:
        """Calculate all Greeks analytically."""
        d1, d2 = cls.calculate_d1_d2(params.S, params.K, params.T, params.r, params.sigma)

        # Delta
        if params.option_type == 'call':
            delta = stats.norm.cdf(d1)
        else:
            delta = stats.norm.cdf(d1) - 1

        # Gamma (same for calls and puts)
        gamma = stats.norm.pdf(d1) / (params.S * params.sigma * np.sqrt(params.T))

        # Vega (same for calls and puts)
        vega = params.S * stats.norm.pdf(d1) * np.sqrt(params.T)

        # Theta
        if params.option_type == 'call':
            theta = (-params.S * stats.norm.pdf(d1) * params.sigma / (2 * np.sqrt(params.T)) 
                     - params.r * params.K * np.exp(-params.r * params.T) * stats.norm.cdf(d2))
        else:
            theta = (-params.S * stats.norm.pdf(d1) * params.sigma / (2 * np.sqrt(params.T)) 
                     + params.r * params.K * np.exp(-params.r * params.T) * stats.norm.cdf(-d2))

        # Rho
        if params.option_type == 'call':
            rho = params.K * params.T * np.exp(-params.r * params.T) * stats.norm.cdf(d2)
        else:
            rho = -params.K * params.T * np.exp(-params.r * params.T) * stats.norm.cdf(-d2)

        return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)

    @classmethod
    def implied_volatility(cls, market_price: float, params: OptionParams, 
                          tol: float = 1e-6, max_iter: int = 100) -> Optional[float]:
        """Calculate implied volatility from market price."""
        if market_price <= 0:
            return None

        def objective(sigma):
            test_params = OptionParams(
                S=params.S, K=params.K, T=params.T, r=params.r, sigma=sigma, option_type=params.option_type
            )
            return cls.price(test_params) - market_price

        try:
            # Use Brent's method for robust root finding
            implied_vol = brentq(objective, 0.001, 5.0, xtol=tol, maxiter=max_iter)
            return implied_vol
        except ValueError:
            logger.warning("Implied volatility calculation failed: no root in range")
            return None


class MonteCarloPricer:
    """
    Monte Carlo option pricing using risk-neutral valuation.

    S_T = S_0 * exp((r - σ²/2)T + σ√T * Z)

    Price = exp(-rT) * E[Payoff(S_T)]

    For path-dependent options (Asian, Barrier), we simulate full paths.
    """

    def __init__(self, n_paths: int = 100000, n_steps: int = 252, seed: Optional[int] = None):
        self.n_paths = n_paths
        self.n_steps = n_steps
        if seed:
            np.random.seed(seed)

    def price_european(self, params: OptionParams) -> Tuple[float, float]:
        """
        Price European option using Monte Carlo.
        Returns (price, standard_error).
        """
        dt = params.T / self.n_steps

        # Generate random paths
        Z = np.random.standard_normal((self.n_paths, self.n_steps))

        # GBM: S_T = S_0 * exp((r - σ²/2)T + σ√T * Z)
        # Drift = (r - σ²/2), Diffusion = σ√T
        drift = (params.r - 0.5 * params.sigma ** 2) * params.T
        diffusion = params.sigma * np.sqrt(params.T)

        # Terminal prices
        log_returns = drift + diffusion * Z.mean(axis=1)  # Simplified
        S_T = params.S * np.exp(log_returns)

        # Payoffs
        if params.option_type == 'call':
            payoffs = np.maximum(S_T - params.K, 0)
        else:
            payoffs = np.maximum(params.K - S_T, 0)

        # Discount to present
        discounted_payoffs = np.exp(-params.r * params.T) * payoffs

        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(self.n_paths)

        return price, std_error

    def price_asian(self, params: OptionParams, averaging_type: str = 'arithmetic') -> float:
        """
        Price Asian option (average price option).
        Path-dependent: payoff depends on average price over time.
        """
        dt = params.T / self.n_steps
        paths = np.zeros((self.n_paths, self.n_steps + 1))
        paths[:, 0] = params.S

        for t in range(1, self.n_steps + 1):
            Z = np.random.standard_normal(self.n_paths)
            paths[:, t] = paths[:, t-1] * np.exp(
                (params.r - 0.5 * params.sigma ** 2) * dt + 
                params.sigma * np.sqrt(dt) * Z
            )

        # Calculate average price
        if averaging_type == 'arithmetic':
            avg_prices = np.mean(paths[:, 1:], axis=1)
        else:  # geometric
            avg_prices = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))

        # Payoff
        if params.option_type == 'call':
            payoffs = np.maximum(avg_prices - params.K, 0)
        else:
            payoffs = np.maximum(params.K - avg_prices, 0)

        return np.mean(np.exp(-params.r * params.T) * payoffs)

    def price_barrier(self, params: OptionParams, barrier: float, 
                     barrier_type: str = 'up-and-out') -> float:
        """
        Price barrier option (knock-out or knock-in).
        Path-dependent: option ceases to exist if barrier is hit.
        """
        dt = params.T / self.n_steps
        paths = np.zeros((self.n_paths, self.n_steps + 1))
        paths[:, 0] = params.S

        for t in range(1, self.n_steps + 1):
            Z = np.random.standard_normal(self.n_paths)
            paths[:, t] = paths[:, t-1] * np.exp(
                (params.r - 0.5 * params.sigma ** 2) * dt + 
                params.sigma * np.sqrt(dt) * Z
            )

        # Check barrier conditions
        if barrier_type == 'up-and-out':
            valid_paths = ~np.any(paths > barrier, axis=1)
        elif barrier_type == 'down-and-out':
            valid_paths = ~np.any(paths < barrier, axis=1)
        elif barrier_type == 'up-and-in':
            valid_paths = np.any(paths > barrier, axis=1)
        else:  # down-and-in
            valid_paths = np.any(paths < barrier, axis=1)

        # Calculate payoff for valid paths
        S_T = paths[:, -1]
        if params.option_type == 'call':
            payoffs = np.maximum(S_T - params.K, 0)
        else:
            payoffs = np.maximum(params.K - S_T, 0)

        payoffs[~valid_paths] = 0

        return np.mean(np.exp(-params.r * params.T) * payoffs)


class OptionsPositionSizer:
    """
    Kelly criterion position sizing for options trades.

    Combines options pricing with Kelly optimal bet sizing.
    """

    def __init__(self, capital: float, max_position_pct: float = 0.1):
        self.capital = capital
        self.max_position_pct = max_position_pct

    def calculate_edge(self, model_price: float, market_price: float) -> float:
        """
        Calculate edge as expected return.
        Edge = (ModelPrice - MarketPrice) / MarketPrice
        """
        if market_price <= 0:
            return 0.0
        return (model_price - market_price) / market_price

    def kelly_size(self, option_params: OptionParams, market_price: float,
                   win_prob: float = 0.5, payout_ratio: float = 2.0) -> dict:
        """
        Calculate Kelly-optimal position size for options.

        For options: Kelly = (p*b - q) / b
        where p = win probability, b = payout ratio, q = 1-p
        """
        model_price = BlackScholesPricer.price(option_params)
        edge = self.calculate_edge(model_price, market_price)

        # Kelly fraction
        q = 1 - win_prob
        if payout_ratio <= 0:
            kelly_fraction = 0
        else:
            kelly_fraction = (win_prob * payout_ratio - q) / payout_ratio

        # Quarter Kelly for options (higher variance)
        fractional_kelly = max(0, kelly_fraction * 0.25)

        # Position sizing
        max_position = self.capital * self.max_position_pct
        position_size = min(self.capital * fractional_kelly, max_position)

        # Calculate Greeks for risk assessment
        greeks = BlackScholesPricer.greeks(option_params)

        return {
            'model_price': model_price,
            'market_price': market_price,
            'edge': edge,
            'kelly_fraction': kelly_fraction,
            'fractional_kelly': fractional_kelly,
            'position_size': position_size,
            'max_position': max_position,
            'contracts': int(position_size / market_price) if market_price > 0 else 0,
            'greeks': greeks,
            'notional': int(position_size / market_price) * option_params.K if market_price > 0 else 0
        }


# Test
def test_options_pricing():
    """Test all pricing methods."""
    print("=" * 60)
    print("OPTIONS PRICING MODULE TEST")
    print("=" * 60)

    # Standard parameters (BTC-like)
    params = OptionParams(
        S=67000,      # Spot
        K=70000,      # Strike (OTM call)
        T=0.25,       # 3 months
        r=0.05,       # 5% risk-free
        sigma=0.6,    # 60% volatility (crypto)
        option_type='call'
    )

    # Black-Scholes
    print("\n--- Black-Scholes Pricing ---")
    bs_price = BlackScholesPricer.price(params)
    print(f"Call Price: ${bs_price:,.2f}")

    d1, d2 = BlackScholesPricer.calculate_d1_d2(
        params.S, params.K, params.T, params.r, params.sigma
    )
    print(f"d1: {d1:.4f}, d2: {d2:.4f}")

    # Greeks
    print("\n--- Greeks ---")
    greeks = BlackScholesPricer.greeks(params)
    print(f"Delta: {greeks.delta:.4f}")
    print(f"Gamma: {greeks.gamma:.6f}")
    print(f"Vega: ${greeks.vega:.2f}")
    print(f"Theta: ${greeks.theta:.2f} per day")
    print(f"Rho: ${greeks.rho:.2f}")

    # Implied volatility
    print("\n--- Implied Volatility ---")
    market_price = bs_price * 0.95  # Assume market underprices by 5%
    implied_vol = BlackScholesPricer.implied_volatility(market_price, params)
    print(f"Market Price: ${market_price:,.2f}")
    print(f"Implied Vol: {implied_vol:.2%}" if implied_vol else "Failed to converge")

    # Monte Carlo
    print("\n--- Monte Carlo (100,000 paths) ---")
    mc_pricer = MonteCarloPricer(n_paths=100000, seed=42)
    mc_price, mc_error = mc_pricer.price_european(params)
    print(f"MC Price: ${mc_price:,.2f} ± ${mc_error:,.2f}")
    print(f"Difference from BS: ${abs(mc_price - bs_price):,.2f}")

    # Asian option
    print("\n--- Asian Option ---")
    asian_price = mc_pricer.price_asian(params, 'arithmetic')
    print(f"Asian Call Price: ${asian_price:,.2f}")

    # Position sizing
    print("\n--- Kelly Position Sizing ---")
    sizer = OptionsPositionSizer(capital=10000, max_position_pct=0.1)
    sizing = sizer.kelly_size(params, market_price=bs_price * 0.98, 
                              win_prob=0.45, payout_ratio=3.0)
    print(f"Model Price: ${sizing['model_price']:,.2f}")
    print(f"Market Price: ${sizing['market_price']:,.2f}")
    print(f"Edge: {sizing['edge']:.2%}")
    print(f"Kelly Fraction: {sizing['kelly_fraction']:.2%}")
    print(f"Quarter Kelly: {sizing['fractional_kelly']:.2%}")
    print(f"Position Size: ${sizing['position_size']:,.2f}")
    print(f"Contracts: {sizing['contracts']}")
    print(f"Notional: ${sizing['notional']:,.2f}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_options_pricing()
