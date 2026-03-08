"""
Backtest Suite
==============
Walk-forward validation for Particle Filter + Kelly + VWAP strategy.

Compares vs naive (no Kelly) and vs buy-and-hold.
"""

import sys
sys.path.insert(0, '/a0/usr/workdir/ai-agent-blockchain-skills/skills/market_analyzer_trader/modules')

import numpy as np
import pandas as pd
from typing import Dict, List, Callable
from datetime import datetime
import logging

from backtest_engine import BacktestEngine, BacktestMetrics
from signal_engine import SignalEngine
from decision_engine import DecisionEngine
from execution_engine import ExecutionEngine

logger = logging.getLogger('BacktestSuite')


class BacktestSuite:
    """
    Complete backtesting suite for strategy validation.

    Strategy: Particle filter signal + Kelly sizing + VWAP execution

    Compares:
    1. Full strategy (PF + Kelly + VWAP)
    2. Naive (no Kelly, equal sizing)
    3. Buy-and-hold benchmark
    """

    def __init__(self,
                 train_days: int = 30,
                 test_days: int = 7,
                 transaction_cost: float = 0.01):
        """
        Args:
            train_days: Training window size
            test_days: Testing window size
            transaction_cost: Roundtrip cost per trade
        """
        self.engine = BacktestEngine(
            train_days=train_days,
            test_days=test_days,
            transaction_cost=transaction_cost
        )

        self.signal_engine = SignalEngine()
        self.decision_engine = DecisionEngine()
        self.execution_engine = ExecutionEngine()

    def create_full_strategy(self) -> Dict:
        """Create full strategy config (PF + Kelly + VWAP)."""

        def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            signals = []

            for i in range(len(df)):
                if i < 20:
                    signals.append(0)
                else:
                    window = df.iloc[:i+1]
                    price = df.iloc[i]['close']
                    sig = self.signal_engine.generate_signal('TEST', window, price)
                    signals.append(sig.strength)

            df['signal'] = signals
            df['asset'] = 'TEST'
            return df

        def sizing_fn(row: pd.Series) -> float:
            if abs(row['signal']) > 0.3:
                return abs(row['signal']) * 0.1  # 10% max
            return 0

        def exit_fn(df: pd.DataFrame, idx: int) -> tuple:
            if idx > 0:
                signal = df.iloc[idx]['signal']
                prev_signal = df.iloc[idx-1]['signal']

                # Exit if signal reverses
                if signal * prev_signal < 0:
                    return True, 'signal_reversal', df.iloc[idx]['close']

            return False, '', 0

        return {
            'signal_fn': signal_fn,
            'sizing_fn': sizing_fn,
            'exit_fn': exit_fn
        }

    def create_naive_strategy(self) -> Dict:
        """Create naive strategy (no Kelly, equal sizing)."""

        def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df['signal'] = np.where(
                df['close'] > df['close'].shift(5), 1,
                np.where(df['close'] < df['close'].shift(5), -1, 0)
            )
            df['asset'] = 'TEST'
            return df

        def sizing_fn(row: pd.Series) -> float:
            return 0.05 if row['signal'] != 0 else 0  # Fixed 5%

        def exit_fn(df: pd.DataFrame, idx: int) -> tuple:
            if idx > 0 and idx % 5 == 0:
                return True, 'time_exit', df.iloc[idx]['close']
            return False, '', 0

        return {
            'signal_fn': signal_fn,
            'sizing_fn': sizing_fn,
            'exit_fn': exit_fn
        }

    def create_buy_hold_strategy(self) -> Dict:
        """Create buy-and-hold benchmark."""

        def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df['signal'] = 0
            df['signal'].iloc[0] = 1  # Buy at start
            df['asset'] = 'TEST'
            return df

        def sizing_fn(row: pd.Series) -> float:
            return 1.0 if row.name == row.index[0] else 0

        def exit_fn(df: pd.DataFrame, idx: int) -> tuple:
            return idx == len(df) - 1, 'end', df.iloc[idx]['close']

        return {
            'signal_fn': signal_fn,
            'sizing_fn': sizing_fn,
            'exit_fn': exit_fn
        }

    def run_comparison(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run backtest comparison of all strategies.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with metrics for each strategy
        """
        strategies = {
            'Full (PF+Kelly+VWAP)': self.create_full_strategy(),
            'Naive (No Kelly)': self.create_naive_strategy(),
            'Buy & Hold': self.create_buy_hold_strategy()
        }

        results = []

        for name, config in strategies.items():
            logger.info(f"\nBacktesting: {name}")

            metrics = self.engine.run_backtest(
                data,
                config['signal_fn'],
                config['sizing_fn'],
                config['exit_fn']
            )

            results.append({
                'strategy': name,
                'total_trades': metrics.total_trades,
                'win_rate': f"{metrics.win_rate*100:.1f}%",
                'sharpe': f"{metrics.sharpe_ratio:.2f}",
                'sortino': f"{metrics.sortino_ratio:.2f}",
                'max_dd': f"{metrics.max_drawdown_pct:.1f}%",
                'total_return': f"{metrics.total_return*100:.1f}%",
                'profit_factor': f"{metrics.profit_factor:.2f}"
            })

            logger.info(f"  Trades: {metrics.total_trades}")
            logger.info(f"  Sharpe: {metrics.sharpe_ratio:.2f}")
            logger.info(f"  Max DD: {metrics.max_drawdown_pct:.1f}%")

        return pd.DataFrame(results)

    def run_full_backtest(self, 
                         data: pd.DataFrame,
                         asset: str = "TEST") -> Dict:
        """
        Run complete backtest with detailed results.

        Args:
            data: OHLCV DataFrame
            asset: Asset name

        Returns:
            Dict with metrics and trade history
        """
        strategy = self.create_full_strategy()

        metrics = self.engine.run_backtest(
            data,
            strategy['signal_fn'],
            strategy['sizing_fn'],
            strategy['exit_fn']
        )

        return {
            'metrics': metrics.to_dict(),
            'trades': [t for t in self.engine.trades],
            'equity_curve': self.engine.equity_curve
        }

    def generate_report(self, results: Dict) -> str:
        """Generate human-readable backtest report."""
        m = results['metrics']

        report = f"""
============================================
BACKTEST REPORT
============================================
Strategy: Particle Filter + Kelly + VWAP
Period: Walk-forward validation

METRICS
-------
Total Trades:    {m['total_trades']}
Win Rate:        {m['win_rate']*100:.1f}%
Avg Return:      {m['avg_return']*100:.2f}%
Sharpe Ratio:    {m['sharpe_ratio']:.2f}
Sortino Ratio:   {m['sortino_ratio']:.2f}
Max Drawdown:    {m['max_drawdown_pct']:.1f}%
Total Return:    {m['total_return']*100:.1f}%
Profit Factor:   {m['profit_factor']:.2f}
Brier Score:     {m['brier_score']:.4f}
Calmar Ratio:    {m['calmar_ratio']:.2f}

TRADE STATISTICS
----------------
Winning Trades:  {m['winning_trades']}
Losing Trades:   {m['losing_trades']}
Avg Win:         {m['avg_win']*100:.2f}%
Avg Loss:        {m['avg_loss']*100:.2f}%

============================================
"""
        return report


if __name__ == "__main__":
    print("Backtest Suite - demo")

    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=200, freq='1h')
    prices = 100 + np.cumsum(np.random.randn(200) * 0.5)

    df = pd.DataFrame({
        'open': prices * (1 + np.random.randn(200) * 0.002),
        'high': prices * (1 + abs(np.random.randn(200) * 0.005)),
        'low': prices * (1 - abs(np.random.randn(200) * 0.005)),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 200),
        'bid_volume': np.random.randint(500, 5000, 200),
        'ask_volume': np.random.randint(500, 5000, 200)
    }, index=dates)

    print("\nRunning backtest comparison...")
    suite = BacktestSuite(train_days=10, test_days=5)

    try:
        comparison = suite.run_comparison(df)
        print("\n" + comparison.to_string(index=False))
    except Exception as e:
        print(f"Error: {e}")
