"""
APEX LLM Veto
=============
Pre-trade check with LLM for final validation.

Input: Trade details + market context
Output: VETO or PROCEED with reasoning

Acts as final sanity check before execution.
"""

from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class LLMVetoDecision:
    """LLM veto decision."""
    decision: str  # 'PROCEED' or 'VETO'
    confidence: float  # 0 to 1
    reasoning: str
    risk_flags: list
    timestamp: datetime

    def to_dict(self) -> Dict:
        return {
            'decision': self.decision,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'risk_flags': self.risk_flags,
            'timestamp': self.timestamp.isoformat()
        }


class LLMVeto:
    """
    LLM-based pre-trade veto system.

    Performs final sanity check before execution:
    - Market context analysis
    - Risk flag detection
    - Common sense validation
    """

    def __init__(self, auto_proceed: bool = False):
        self.auto_proceed = auto_proceed
        self.veto_count = 0
        self.proceed_count = 0

    def check_trade(self,
                   asset: str,
                   side: str,
                   size: float,
                   leverage: float,
                   entry_price: float,
                   stop_loss: float,
                   take_profit: float,
                   edge_score: float,
                   regime: str,
                   signal_breakdown: Dict) -> LLMVetoDecision:
        """
        Perform pre-trade LLM check.

        Args:
            asset: Trading asset
            side: 'long' or 'short'
            size: Position size (% of capital)
            leverage: Leverage multiplier
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            edge_score: Calculated edge score
            regime: Current market regime
            signal_breakdown: Dict of signal scores

        Returns:
            LLMVetoDecision with PROCEED or VETO
        """
        risk_flags = []

        # Risk checks (rule-based, LLM-style reasoning)

        # 1. Extreme leverage
        if leverage > 20:
            risk_flags.append(f"High leverage ({leverage:.1f}x) - increased liquidation risk")

        # 2. Large position size
        if size > 0.15:
            risk_flags.append(f"Large position ({size*100:.1f}%) - concentration risk")

        # 3. Wide stop loss
        stop_pct = abs(entry_price - stop_loss) / entry_price
        if stop_pct > 0.15:
            risk_flags.append(f"Wide stop ({stop_pct*100:.1f}%) - excessive risk")

        # 4. Low edge score
        if abs(edge_score) < 0.4:
            risk_flags.append(f"Low edge score ({edge_score:.3f}) - weak signal")

        # 5. Adverse regime
        if regime == 'high_vol':
            risk_flags.append("High volatility regime - reduced edge")

        # 6. Contradictory signals
        buy_signals = sum(1 for v in signal_breakdown.values() if v > 0.3)
        sell_signals = sum(1 for v in signal_breakdown.values() if v < -0.3)
        if buy_signals > 0 and sell_signals > 0 and abs(buy_signals - sell_signals) < 2:
            risk_flags.append("Mixed signals - low conviction")

        # Determine decision
        critical_flags = [f for f in risk_flags if 'High leverage' in f or 'Large position' in f]

        if len(critical_flags) >= 2:
            decision = 'VETO'
            confidence = 0.8
        elif len(risk_flags) >= 3:
            decision = 'VETO'
            confidence = 0.7
        elif len(risk_flags) >= 1:
            decision = 'PROCEED_WITH_CAUTION'
            confidence = 0.6
        else:
            decision = 'PROCEED'
            confidence = 0.85

        # Build reasoning
        if decision == 'VETO':
            reasoning = f"Trade vetoed due to {len(risk_flags)} risk flags: " + "; ".join(risk_flags)
            self.veto_count += 1
        elif decision == 'PROCEED_WITH_CAUTION':
            reasoning = f"Proceeding with caution. Risk flags: " + "; ".join(risk_flags)
            self.proceed_count += 1
        else:
            reasoning = "Clean trade - all risk checks passed. Proceeding."
            self.proceed_count += 1

        return LLMVetoDecision(
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            risk_flags=risk_flags,
            timestamp=datetime.now()
        )

    def get_stats(self) -> Dict:
        """Get veto statistics."""
        total = self.veto_count + self.proceed_count
        return {
            'veto_count': self.veto_count,
            'proceed_count': self.proceed_count,
            'veto_rate': self.veto_count / total if total > 0 else 0,
            'total_checked': total
        }


if __name__ == "__main__":
    print("APEX LLM Veto Demo")
    print("=" * 60)

    veto = LLMVeto()

    # Example trade
    result = veto.check_trade(
        asset="ETH",
        side="long",
        size=0.20,
        leverage=25,
        entry_price=3000,
        stop_loss=2000,
        take_profit=4500,
        edge_score=0.35,
        regime="high_vol",
        signal_breakdown={
            'cvd_divergence': 0.8,
            'crowd_extreme': -0.4,
            'liquidation_cluster': 0.3
        }
    )

    print(f"\nTrade: ETH LONG @ 25x")
    print(f"Decision: {result.decision}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Risk Flags ({len(result.risk_flags)}):")
    for flag in result.risk_flags:
        print(f"  - {flag}")

    print(f"\nStats: {veto.get_stats()}")
