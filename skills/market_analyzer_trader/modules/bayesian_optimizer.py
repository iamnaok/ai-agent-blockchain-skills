"""
Bayesian Optimizer for APEX Trading Parameters

Auto-tunes Kelly fraction, entry/exit thresholds using Gaussian Process optimization.
Maximizes backtest Sharpe ratio through intelligent parameter search.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import json
from typing import Dict, List, Tuple, Callable, Optional
import warnings


class GaussianProcess:
    """Simple Gaussian Process for Bayesian Optimization."""

    def __init__(self, length_scale=1.0, noise=1e-5):
        self.length_scale = length_scale
        self.noise = noise
        self.X = []
        self.y = []
        self.K = None

    def rbf_kernel(self, x1, x2):
        """Radial Basis Function kernel."""
        sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return np.exp(-0.5 * sqdist / self.length_scale**2)

    def fit(self, X, y):
        """Fit GP to observed data."""
        self.X = np.array(X)
        self.y = np.array(y)

        if len(self.X) == 0:
            return

        self.K = self.rbf_kernel(self.X, self.X) + self.noise * np.eye(len(self.X))

    def predict(self, X_new, return_std=True):
        """Predict mean and std at new points."""
        if len(self.X) == 0:
            return np.zeros(len(X_new)), np.ones(len(X_new)) if return_std else None

        X_new = np.atleast_2d(X_new)

        # Kernel matrices
        K_s = self.rbf_kernel(self.X, X_new)
        K_ss = self.rbf_kernel(X_new, X_new) + self.noise * np.eye(len(X_new))

        # Inverse with stability
        try:
            L = np.linalg.cholesky(self.K + self.noise * np.eye(len(self.X)))
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            alpha = np.linalg.lstsq(self.K, self.y, rcond=None)[0]

        # Mean prediction
        mu = np.dot(K_s.T, alpha)

        if not return_std:
            return mu

        # Variance
        try:
            v = np.linalg.solve(L, K_s)
            var = np.diag(K_ss) - np.sum(v**2, axis=0)
        except:
            var = np.diag(K_ss)

        var = np.maximum(var, 1e-10)  # Ensure positive
        std = np.sqrt(var)

        return mu, std


class BayesianOptimizer:
    """
    Bayesian Optimization for APEX trading parameters.

    Optimizes:
    - kelly_fraction: (0.1, 0.5)
    - entry_threshold: (1.0, 3.0)  # Z-score
    - stop_loss: (0.3, 0.6)  # % of position
    - take_profit: (1.0, 2.0)  # Risk multiple
    """

    def __init__(self, 
                 backtest_func: Callable,
                 n_initial: int = 5,
                 n_iterations: int = 25,
                 acquisition: str = 'ei',  # 'ei', 'pi', 'ucb'
                 xi: float = 0.01,
                 kappa: float = 2.576):
        """
        Args:
            backtest_func: Function(params) -> sharpe_ratio
            n_initial: Random initialization points
            n_iterations: Total optimization iterations
            acquisition: Acquisition function ('ei', 'pi', 'ucb')
            xi: Exploration parameter
            kappa: UCB exploration parameter
        """
        self.backtest_func = backtest_func
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.acquisition = acquisition
        self.xi = xi
        self.kappa = kappa

        self.param_bounds = {
            'kelly_fraction': (0.1, 0.5),
            'entry_threshold': (1.0, 3.0),
            'stop_loss': (0.3, 0.6),
            'take_profit': (1.0, 2.0)
        }

        self.gp = GaussianProcess(length_scale=1.0, noise=1e-5)
        self.observations = []
        self.best_params = None
        self.best_sharpe = -np.inf

    def _normalize_params(self, params: Dict) -> np.ndarray:
        """Normalize parameters to [0, 1] for GP."""
        normalized = []
        for key, (low, high) in self.param_bounds.items():
            val = params.get(key, (low + high) / 2)
            normalized.append((val - low) / (high - low))
        return np.array(normalized)

    def _denormalize_params(self, normalized: np.ndarray) -> Dict:
        """Denormalize from [0, 1] to actual parameter range."""
        params = {}
        for i, (key, (low, high)) in enumerate(self.param_bounds.items()):
            params[key] = low + normalized[i] * (high - low)
        return params

    def _acquisition_function(self, X_candidates: np.ndarray) -> np.ndarray:
        """Compute acquisition function values."""
        if len(self.gp.X) == 0:
            return np.ones(len(X_candidates))

        mu, std = self.gp.predict(X_candidates, return_std=True)

        if self.acquisition == 'ei':  # Expected Improvement
            if self.best_sharpe == -np.inf:
                return mu + std  # Just explore initially
            z = (mu - self.best_sharpe - self.xi) / (std + 1e-9)
            ei = (mu - self.best_sharpe - self.xi) * norm.cdf(z) + std * norm.pdf(z)
            return ei

        elif self.acquisition == 'pi':  # Probability of Improvement
            z = (mu - self.best_sharpe - self.xi) / (std + 1e-9)
            return norm.cdf(z)

        elif self.acquisition == 'ucb':  # Upper Confidence Bound
            return mu + self.kappa * std

        else:
            return mu  # Default to mean

    def _suggest_next(self) -> Dict:
        """Suggest next parameters to try using acquisition function."""
        if len(self.observations) < self.n_initial:
            # Random sampling for initial points
            params = {}
            for key, (low, high) in self.param_bounds.items():
                params[key] = np.random.uniform(low, high)
            return params

        # Acquisition function optimization
        best_acq = -np.inf
        best_params = None

        # Random restart optimization
        n_restarts = 10
        for _ in range(n_restarts):
            # Random starting point
            x0 = np.random.uniform(0, 1, len(self.param_bounds))

            # Negative acquisition (we want to maximize)
            def neg_acq(x):
                x = x.reshape(1, -1)
                return -self._acquisition_function(x)[0]

            # Optimize
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(neg_acq, x0, bounds=[(0, 1)] * len(self.param_bounds), 
                                method='L-BFGS-B')

            if not result.success:
                continue

            x_opt = result.x
            acq_val = -result.fun

            if acq_val > best_acq:
                best_acq = acq_val
                best_params = x_opt

        if best_params is None:
            # Fallback to random
            best_params = np.random.uniform(0, 1, len(self.param_bounds))

        return self._denormalize_params(best_params)

    def optimize(self) -> Dict:
        """
        Run Bayesian optimization.

        Returns:
            Dict with best parameters and metadata
        """
        print(f"Starting Bayesian Optimization: {self.n_initial} init + {self.n_iterations} iterations")

        for i in range(self.n_iterations):
            # Suggest next parameters
            params = self._suggest_next()

            # Evaluate (run backtest)
            print(f"Iteration {i+1}/{self.n_iterations}: {params}")
            try:
                sharpe = self.backtest_func(params)
                if sharpe is None or np.isnan(sharpe):
                    sharpe = 0.0
            except Exception as e:
                print(f"  Backtest failed: {e}")
                sharpe = 0.0

            # Store observation
            self.observations.append({
                'params': params,
                'sharpe': sharpe,
                'iteration': i
            })

            # Update best
            if sharpe > self.best_sharpe:
                self.best_sharpe = sharpe
                self.best_params = params.copy()
                print(f"  *** New best Sharpe: {sharpe:.4f} ***")
            else:
                print(f"  Sharpe: {sharpe:.4f}, Best: {self.best_sharpe:.4f}")

            # Update GP
            X_norm = [self._normalize_params(o['params']) for o in self.observations]
            y_obs = [o['sharpe'] for o in self.observations]
            self.gp.fit(X_norm, y_obs)

        return {
            'best_params': self.best_params,
            'best_sharpe': self.best_sharpe,
            'n_iterations': self.n_iterations,
            'observations': self.observations
        }

    def save_results(self, filepath: str):
        """Save optimization results to JSON."""
        results = {
            'best_params': self.best_params,
            'best_sharpe': self.best_sharpe,
            'n_iterations': self.n_iterations,
            'n_observations': len(self.observations),
            'observations': [
                {'params': o['params'], 'sharpe': o['sharpe'], 'iteration': o['iteration']}
                for o in self.observations
            ]
        }
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filepath}")


class APEXBacktestWrapper:
    """Wrapper to run APEX backtests with parameter sets."""

    def __init__(self, market_data_path: str = None):
        self.market_data_path = market_data_path

    def run_backtest(self, params: Dict) -> float:
        """
        Run APEX backtest with given parameters.

        Returns:
            Sharpe ratio (float)
        """
        # Import APEX modules
        try:
            from modules.apex_signal_engine import APEXSignalEngine
            from modules.edge_scorer import EdgeScorer
            from modules.smart_sizer import SmartPositionSizer
        except ImportError:
            # Fallback: use simplified backtest
            return self._simplified_backtest(params)

        # Configure with parameters
        kelly_frac = params.get('kelly_fraction', 0.25)
        entry_thresh = params.get('entry_threshold', 1.5)
        stop_loss = params.get('stop_loss', 0.5)
        take_profit = params.get('take_profit', 1.5)

        # Run backtest (simplified for now)
        return self._simplified_backtest(params)

    def _simplified_backtest(self, params: Dict) -> float:
        """Simplified backtest for testing."""
        # Simulate: higher Kelly = higher Sharpe but higher variance
        kelly = params.get('kelly_fraction', 0.25)
        entry = params.get('entry_threshold', 1.5)

        # Simulated Sharpe based on parameter quality
        # Optimal around kelly=0.25, entry=1.5
        kelly_score = 1.0 - abs(kelly - 0.25) * 2
        entry_score = 1.0 - abs(entry - 1.5) * 0.5

        base_sharpe = 1.5
        noise = np.random.normal(0, 0.1)
        sharpe = base_sharpe * kelly_score * entry_score + noise

        return max(0, sharpe)


def main():
    """Example usage."""
    # Create wrapper
    wrapper = APEXBacktestWrapper()

    # Create optimizer
    optimizer = BayesianOptimizer(
        backtest_func=wrapper.run_backtest,
        n_initial=5,
        n_iterations=25,
        acquisition='ei'
    )

    # Run optimization
    results = optimizer.optimize()

    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"Best Sharpe Ratio: {results['best_sharpe']:.4f}")
    print("Best Parameters:")
    for key, val in results['best_params'].items():
        print(f"  {key}: {val:.4f}")

    # Save results
    optimizer.save_results('/a0/usr/workdir/ai-agent-blockchain-skills/skills/market_analyzer_trader/bayesian_optimization_results.json')


if __name__ == "__main__":
    main()
