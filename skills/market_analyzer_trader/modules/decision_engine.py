"""
Decision Engine
===============
Uses Kelly criterion for position sizing with entry/exit logic.

Output: Decision dict with action, size, price, confidence.
"""

import sys
sys.path.insert(0, '/a0/usr/workdir/ai-agent-blockchain-skills/skills/professional_quant_trader/modules')
sys.path.insert(0, '/a0/usr/workdir')

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

# Import Kelly optimizer
try:
    from kelly_portfolio_optimizer import KellyPortfolioOptimizer
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "kelly_portfolio_optimizer",
        "/a0/usr/workdir/ai-agent-blockchain-skills/skills/professional_quant_trader/modules/kelly_portfolio_optimizer.py"
    )
    ko_module = importlib.util.module_from_spec(spec)
    sys.modules['kelly_portfolio_optimizer'] = ko_module
    spec.loader.exec_module(ko_module)
    KellyPortfolioOptimizer = ko_module.KellyPortfolioOptimizer

logger = logging.getLogger('DecisionEngine')


@dataclass
class TradingDecision:
    """Trading decision with all parameters."""
    asset: str
    action: str  # 'enter', 'exit', 'hold'
    side: str  # 'long', 'short', 'none'
    size: float  # Position size as % of capital
    leverage: float
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    signal_strength: float
    kelly_fraction: float
    timestamp: datetime
    reason: str


class DecisionEngine:
    """
    Decision engine integrating Kelly criterion with signal validation.

    Entry logic:
    - Signal strength > threshold (e.g., 0.3)
    - Liquidity score > 0.7
    - Kelly edge > 0 (positive expected value)

    Exit logic:
    - Stop loss: Kelly-derived (50% of entry signal)
    - Take profit: Kelly-derived (100% of entry signal)
    - Signal decay: Exit when signal flips
    """

    def __init__(self,
                 kelly_fraction: float = 0.25,
                 max_position: float = 0.10,
                 max_drawdown: float = 0.20,
                 var_limit: float = 0.05,
                 signal_threshold: float = 0.3,
                 liquidity_threshold: float = 0.7,
                 stop_loss_pct: float = 0.50,
                 take_profit_pct: float = 1.0):
        """
        Args:
            kelly_fraction: Kelly fraction (0.25 = quarter Kelly)
            max_position: Max position size (10%)
            max_drawdown: Circuit breaker level (20%)
            var_limit: VaR limit (5%)
            signal_threshold: Minimum signal strength to enter
            liquidity_threshold: Minimum liquidity score
            stop_loss_pct: Stop loss as % of expected edge
            take_profit_pct: Take profit as % of expected edge
        """
        self.kelly = KellyPortfolioOptimizer(
            kelly_fraction=kelly_fraction,
            max_position=max_position,
            max_drawdown=max_drawdown,
            var_limit=var_limit
        )
        self.signal_threshold = signal_threshold
        self.liquidity_threshold = liquidity_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        # Track open positions
        self.positions: Dict[str, Dict] = {}

    def should_enter(self, 
                    signal: 'Signal',  # type: ignore
                    liquidity_score: float) -> Tuple[bool, str]:
        """
        Determine if should enter position.

        Args:
            signal: Signal from signal engine
            liquidity_score: Current liquidity score

        Returns:
            (should_enter, reason)
        """
        checks = []

        # 1. Check signal strength
        if abs(signal.strength) >= self.signal_threshold:
            checks.append(f"PASS: Signal strength {abs(signal.strength):.3f} >= {self.signal_threshold}")
        else:
            checks.append(f"FAIL: Signal strength {abs(signal.strength):.3f} < {self.signal_threshold}")

        # 2. Check liquidity
        if liquidity_score >= self.liquidity_threshold:
            checks.append(f"PASS: Liquidity {liquidity_score:.3f} >= {self.liquidity_threshold}")
        else:
            checks.append(f"FAIL: Liquidity {liquidity_score:.3f} < {self.liquidity_threshold}")

        # 3. Check Kelly edge
        gross_edge, net_edge = self.kelly.calculate_edge(
            signal.raw_prob, 
            signal.filtered_prob
        )

        if net_edge > 0:
            checks.append(f"PASS: Net edge {net_edge:.4f} > 0")
        else:
            checks.append(f"FAIL: Net edge {net_edge:.4f} <= 0")

        # 4. Check if already have position
        if signal.asset in self.positions:
            checks.append(f"FAIL: Already have position in {signal.asset}")
        else:
            checks.append(f"PASS: No existing position")

        failures = [c for c in checks if c.startswith('FAIL')]
        should_enter = len(failures) == 0

        reason = "; ".join(checks)
        return should_enter, reason

    def should_exit(self,
                   asset: str,
                   signal: 'Signal',  # type: ignore
                   current_price: float) -> Tuple[bool, str, float]:
        """
        Determine if should exit position.

        Returns:
            (should_exit, reason, exit_price)
        """
        if asset not in self.positions:
            return False, "No position", 0.0

        position = self.positions[asset]
        entry_price = position['entry_price']
        side = position['side']

        # Check stop loss
        if side == 'long':
            stop_price = entry_price * (1 - self.stop_loss_pct)
            take_profit_price = entry_price * (1 + self.take_profit_pct)

            if current_price <= stop_price:
                return True, f"Stop loss triggered at ${current_price:.4f} (limit: ${stop_price:.4f})", current_price

            if current_price >= take_profit_price:
                return True, f"Take profit triggered at ${current_price:.4f} (target: ${take_profit_price:.4f})", current_price

        else:  # short
            stop_price = entry_price * (1 + self.stop_loss_pct)
            take_profit_price = entry_price * (1 - self.take_profit_pct)

            if current_price >= stop_price:
                return True, f"Stop loss triggered at ${current_price:.4f} (limit: ${stop_price:.4f})", current_price

            if current_price <= take_profit_price:
                return True, f"Take profit triggered at ${current_price:.4f} (target: ${take_profit_price:.4f})", current_price

        # Check signal decay (opposite signal)
        if signal.strength * (-1 if side == 'short' else 1) < -0.3:
            return True, f"Signal decay: {side} position with opposite signal {signal.strength:.3f}", current_price

        return False, "Hold", 0.0

    def make_decision(self,
                     signal: 'Signal',  # type: ignore
                     current_price: float,
                     liquidity_score: float,
                     bankroll: float = 10000.0) -> TradingDecision:
        """
        Generate trading decision.

        Args:
            signal: Signal from signal engine
            current_price: Current market price
            liquidity_score: Current liquidity score
            bankroll: Current bankroll

        Returns:
            TradingDecision with all parameters
        """
        asset = signal.asset

        # Check if we have position
        if asset in self.positions:
            # Check exit
            should_exit, reason, exit_price = self.should_exit(asset, signal, current_price)

            if should_exit:
                del self.positions[asset]
                return TradingDecision(
                    asset=asset,
                    action='exit',
                    side='none',
                    size=0.0,
                    leverage=0.0,
                    entry_price=0.0,
                    stop_loss=0.0,
                    take_profit=0.0,
                    confidence=signal.confidence,
                    signal_strength=signal.strength,
                    kelly_fraction=0.0,
                    timestamp=datetime.now(),
                    reason=reason
                )

            # Hold position
            pos = self.positions[asset]
            return TradingDecision(
                asset=asset,
                action='hold',
                side=pos['side'],
                size=pos['size'],
                leverage=pos['leverage'],
                entry_price=pos['entry_price'],
                stop_loss=pos['stop_loss'],
                take_profit=pos['take_profit'],
                confidence=signal.confidence,
                signal_strength=signal.strength,
                kelly_fraction=pos['kelly_fraction'],
                timestamp=datetime.now(),
                reason="Holding position"
            )

        # Check entry
        should_enter, reason = self.should_enter(signal, liquidity_score)

        if not should_enter:
            return TradingDecision(
                asset=asset,
                action='hold',
                side='none',
                size=0.0,
                leverage=0.0,
                entry_price=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                confidence=signal.confidence,
                signal_strength=signal.strength,
                kelly_fraction=0.0,
                timestamp=datetime.now(),
                reason=reason
            )

        # Calculate Kelly sizing
        gross_edge, net_edge = self.kelly.calculate_edge(
            signal.raw_prob,
            signal.filtered_prob
        )

        kelly_f = self.kelly.kelly_single(signal.raw_prob, signal.filtered_prob)
        position_size = kelly_f * self.kelly.kelly_fraction
        position_size = np.clip(position_size, -self.kelly.max_position, self.kelly.max_position)

        # Calculate leverage based on edge
        leverage = min(abs(net_edge) * 100, 50)
        leverage = max(leverage, 1.0)

        # Determine side
        side = 'long' if signal.strength > 0 else 'short'

        # Calculate stops
        if side == 'long':
            stop_loss = current_price * (1 - self.stop_loss_pct)
            take_profit = current_price * (1 + self.take_profit_pct)
        else:
            stop_loss = current_price * (1 + self.stop_loss_pct)
            take_profit = current_price * (1 - self.take_profit_pct)

        # Record position
        self.positions[asset] = {
            'entry_price': current_price,
            'side': side,
            'size': abs(position_size),
            'leverage': leverage,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'kelly_fraction': kelly_f
        }

        decision = TradingDecision(
            asset=asset,
            action='enter',
            side=side,
            size=abs(position_size),
            leverage=leverage,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=signal.confidence,
            signal_strength=signal.strength,
            kelly_fraction=kelly_f,
            timestamp=datetime.now(),
            reason=reason
        )

        logger.info(f"Decision: {decision.action.upper()} {side} on {asset} | "
                   f"Size: {abs(position_size)*100:.2f}% | "
                   f"Leverage: {leverage:.1f}x")

        return decision

    def get_open_positions(self) -> Dict:
        """Get all open positions."""
        return self.positions.copy()


if __name__ == "__main__":
    from signal_engine import SignalEngine, Signal

    # Demo
    engine = DecisionEngine()

    # Create sample signal
    signal = Signal(
        asset="ETH",
        timestamp=datetime.now(),
        strength=0.65,
        confidence=0.8,
        raw_prob=0.62,
        filtered_prob=0.70,
        vwap_deviation=0.02,
        rsi=45,
        macd=0.05,
        order_flow_delta=0.25,
        direction='buy'
    )

    # Make decision
    decision = engine.make_decision(signal, 3000, 0.85, bankroll=10000)

    print(f"\nTrading Decision:")
    print(f"  Action: {decision.action.upper()}")
    print(f"  Side: {decision.side}")
    print(f"  Size: {decision.size*100:.2f}% of capital")
    print(f"  Leverage: {decision.leverage:.1f}x")
    print(f"  Entry: ${decision.entry_price:.2f}")
    print(f"  Stop Loss: ${decision.stop_loss:.2f}")
    print(f"  Take Profit: ${decision.take_profit:.2f}")
    print(f"  Confidence: {decision.confidence*100:.1f}%")
    print(f"  Reason: {decision.reason}")
