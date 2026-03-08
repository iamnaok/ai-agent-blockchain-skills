"""
Backtest Engine v1.1
Walk-forward validation for trading strategies.

Based on: Professional Quant Trader research
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import logging
from decimal import Decimal, ROUND_HALF_UP
from decimal import Decimal, ROUND_HALF_UP

logger = logging.getLogger('BacktestEngine')


@dataclass
class TradeRecord:
    """Single trade record."""
    entry_time: datetime
    exit_time: Optional[datetime]
    asset: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    fees: float
    net_pnl: float
    exit_reason: str


@dataclass
class BacktestMetrics:
    """Performance metrics from backtest."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_return: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    total_return: float
    brier_score: float
    calmar_ratio: float

    def to_dict(self) -> Dict:
        return asdict(self)


class BacktestEngine:
    """
    Walk-forward backtesting with proper time-series validation.

    Key insight: Simple train/test split leads to overfitting.
    Walk-forward: Train [0:30], Test [30:37], Slide, Repeat

    Prevents overfitting by never peeking at future data.
    """

    def __init__(self,
                 train_days: int = 30,
                 test_days: int = 7,
                 transaction_cost: float = 0.01,  # 1% per trade (0.5% each way)
                 initial_capital: float = 10000.0):
        """
        Args:
            train_days: Days for training period
            test_days: Days for testing period
            transaction_cost: Roundtrip cost per trade
            initial_capital: Starting capital
        """
        self.train_days = train_days
        self.test_days = test_days
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital

        self.trades: List[TradeRecord] = []
        self.equity_curve: List[Tuple[datetime, float]] = []

    def walk_forward_split(self, 
                          data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate walk-forward train/test splits.

        Returns list of (train_data, test_data) tuples.
        """
        splits = []
        data = data.sort_index()

        # Calculate number of complete windows
        total_days = (data.index[-1] - data.index[0]).days
        n_windows = (total_days - self.train_days) // self.test_days

        for i in range(n_windows):
            train_start = data.index[0] + timedelta(days=i * self.test_days)
            train_end = train_start + timedelta(days=self.train_days)
            test_end = train_end + timedelta(days=self.test_days)

            train_mask = (data.index >= train_start) & (data.index < train_end)
            test_mask = (data.index >= train_end) & (data.index < test_end)

            if train_mask.sum() > 0 and test_mask.sum() > 0:
                train_data = data[train_mask].copy()
                test_data = data[test_mask].copy()
                splits.append((train_data, test_data))

        logger.info(f"Generated {len(splits)} walk-forward windows")
        return splits

    def simulate_trade(self,
                       entry_time: datetime,
                       asset: str,
                       side: str,
                       entry_price: float,
                       exit_price: float,
                       size: float,
                       exit_reason: str) -> TradeRecord:
        """
        Simulate a single trade with costs.
        """
        # Calculate gross PnL
        if side == 'long':
            # Decimal conversion for precise PnL
            entry_dec = Decimal(str(entry_price))
            exit_dec = Decimal(str(exit_price))
            size_dec = Decimal(str(size))
            pnl = (exit_dec - entry_dec) * size_dec
            pnl = pnl.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            pnl_pct = float((exit_dec - entry_dec) / entry_dec) if entry_dec > 0 else 0.0
        else:
            # Decimal conversion (variables already defined above)
            pnl = (entry_dec - exit_dec) * size_dec
            pnl = pnl.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            pnl_pct = float((entry_dec - exit_dec) / entry_dec) if entry_dec > 0 else 0.0

        # Apply transaction costs
        fees = entry_price * size * self.transaction_cost
        net_pnl = pnl - fees

        return TradeRecord(
            entry_time=entry_time,
            exit_time=datetime.now(),
            asset=asset,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            size=size,
            pnl=pnl,
            pnl_pct=pnl_pct,
            fees=fees,
            net_pnl=net_pnl,
            exit_reason=exit_reason
        )

    def run_backtest(self,
                     data: pd.DataFrame,
                     signal_fn: Callable[[pd.DataFrame], pd.DataFrame],
                     sizing_fn: Callable[[pd.Series], float],
                     exit_fn: Callable[[pd.DataFrame, int], Tuple[bool, str, float]]) -> BacktestMetrics:
        """
        Run full walk-forward backtest.

        Args:
            data: OHLCV DataFrame with datetime index
            signal_fn: Function that takes data and returns signals
            sizing_fn: Function that takes signal and returns position size
            exit_fn: Function that takes data and position index, returns (should_exit, reason, price)

        Returns:
            BacktestMetrics with performance statistics
        """
        splits = self.walk_forward_split(data)
        self.trades = []

        capital = self.initial_capital

        for i, (train_data, test_data) in enumerate(splits):
            logger.info(f"Window {i+1}/{len(splits)}: Training on {len(train_data)} rows, testing on {len(test_data)} rows")

            # Generate signals on test data
            signals = signal_fn(test_data)

            # Simulate trades
            position = None
            for idx, row in signals.iterrows():
                if position is None:
                    # Check for entry signal
                    if row.get('signal', 0) != 0:
                        size = sizing_fn(row) * capital
                        if size > 0:
                            position = {
                                'entry_time': idx,
                                'asset': row.get('asset', 'UNKNOWN'),
                                'side': 'long' if row['signal'] > 0 else 'short',
                                'entry_price': row['close'],
                                'size': size / row['close'],  # Convert to units
                                'max_price': row['close'],
                                'min_price': row['close']
                            }
                else:
                    # Update position tracking
                    position['max_price'] = max(position['max_price'], row['high'])
                    position['min_price'] = min(position['min_price'], row['low'])

                    # Check for exit
                    should_exit, reason, exit_price = exit_fn(test_data, test_data.index.get_loc(idx))

                    if should_exit or idx == test_data.index[-1]:  # Force exit at end of window
                        trade = self.simulate_trade(
                            entry_time=position['entry_time'],
                            asset=position['asset'],
                            side=position['side'],
                            entry_price=position['entry_price'],
                            exit_price=exit_price if should_exit else row['close'],
                            size=position['size'],
                            exit_reason=reason if should_exit else 'end_of_window'
                        )
                        self.trades.append(trade)
                        capital += trade.net_pnl
                        position = None

            self.equity_curve.append((test_data.index[-1], capital))

        return self.calculate_metrics()

    def calculate_metrics(self) -> BacktestMetrics:
        """Calculate performance metrics from trades."""
        if not self.trades:
            return BacktestMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.net_pnl > 0)
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        returns = [t.pnl_pct for t in self.trades]
        avg_return = np.mean(returns)

        wins = [t.pnl_pct for t in self.trades if t.pnl_pct > 0]
        losses = [t.pnl_pct for t in self.trades if t.pnl_pct <= 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        gross_profit = sum(t.net_pnl for t in self.trades if t.net_pnl > 0)
        gross_loss = abs(sum(t.net_pnl for t in self.trades if t.net_pnl <= 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Sharpe ratio (annualized)
        if len(returns) > 1:
            returns_arr = np.array(returns)
            sharpe = np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(252) if np.std(returns_arr) > 0 else 0
        else:
            sharpe = 0

        # Sortino ratio (downside deviation only)
        if len(returns) > 1:
            downside_returns = [r for r in returns if r < 0]
            downside_std = np.std(downside_returns) if downside_returns else 0
            sortino = np.mean(returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0
        else:
            sortino = 0

        # Max drawdown
        equity = [e[1] for e in self.equity_curve]
        max_dd = 0
        peak = self.initial_capital
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak
            max_dd = max(max_dd, dd)

        # Total return
        final_equity = equity[-1] if equity else self.initial_capital
        total_return = (final_equity - self.initial_capital) / self.initial_capital

        # Brier score (probability calibration)
        # Assume we track predicted vs actual outcomes
        brier_scores = []
        for trade in self.trades:
            if hasattr(trade, 'predicted_prob') and trade.predicted_prob is not None:
                outcome = 1 if trade.net_pnl > 0 else 0
                brier_scores.append((trade.predicted_prob - outcome) ** 2)
        brier = np.mean(brier_scores) if brier_scores else 0.25

        # Calmar ratio (return / max drawdown)
        calmar = total_return / max_dd if max_dd > 0 else 0

        return BacktestMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_return=avg_return,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd * 100,
            total_return=total_return,
            brier_score=brier,
            calmar_ratio=calmar
        )

    def compare_strategies(self,
                          data: pd.DataFrame,
                          strategies: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple strategies.

        Args:
            data: Market data
            strategies: Dict of {name: {signal_fn, sizing_fn, exit_fn}}

        Returns:
            DataFrame with metrics for each strategy
        """
        results = []

        for name, config in strategies.items():
            logger.info(f"\nBacktesting strategy: {name}")

            # Run backtest
            metrics = self.run_backtest(
                data,
                config['signal_fn'],
                config['sizing_fn'],
                config['exit_fn']
            )

            results.append({
                'strategy': name,
                **metrics.to_dict()
            })

        return pd.DataFrame(results)

    def save_results(self, filepath: str):
        """Save trades and metrics to JSON."""
        results = {
            'trades': [asdict(t) for t in self.trades],
            'metrics': self.calculate_metrics().to_dict() if self.trades else {},
            'equity_curve': [(e[0].isoformat(), e[1]) for e in self.equity_curve]
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Saved results to {filepath}")


if __name__ == "__main__":
    # Demo
    engine = BacktestEngine(train_days=10, test_days=5)

    # Generate sample data
    dates = pd.date_range(start='2025-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)

    data = pd.DataFrame({
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)

    # Simple strategy
    def signal_fn(df):
        df = df.copy()
        df['signal'] = np.where(df['close'] > df['close'].shift(5), 1, 
                               np.where(df['close'] < df['close'].shift(5), -1, 0))
        df['asset'] = 'SAMPLE'
        return df

    def sizing_fn(row):
        return 0.1 if row['signal'] != 0 else 0

    def exit_fn(df, idx):
        if idx > 0 and idx % 3 == 0:
            return True, 'time_exit', df.iloc[idx]['close']
        return False, '', 0

    metrics = engine.run_backtest(data, signal_fn, sizing_fn, exit_fn)

    print(f"\nBacktest Results:")
    print(f"Total Trades: {metrics.total_trades}")
    print(f"Win Rate: {metrics.win_rate*100:.1f}%")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown_pct:.1f}%")
    print(f"Total Return: {metrics.total_return*100:.1f}%")
