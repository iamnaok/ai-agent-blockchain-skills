"""
Live Trader Analysis
====================
Uses Hyperliquid live data to generate LONG/SHORT/NEUTRAL recommendations.

Analysis Framework:
- Funding Rate (momentum + carry)
- Open Interest (crowdedness)
- Price Momentum (trend)
- Liquidity (execution quality)
- Order Book Imbalance (directional pressure)

Output: Entry, Stop, Target, Size + Reasoning
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

from modules.hyperliquid_connector import HyperliquidConnector

logger = logging.getLogger('TraderAnalysis')


@dataclass
class TradeRecommendation:
    """Structured trade recommendation."""
    asset: str
    timestamp: datetime

    # Decision
    action: str  # 'LONG', 'SHORT', 'NEUTRAL'
    confidence: float  # 0.0 to 1.0

    # Levels
    current_price: float
    entry_price: float
    stop_loss: float
    take_profit: float

    # Sizing
    position_size_pct: float  # % of capital
    leverage: float

    # Risk
    risk_reward: float
    max_loss_pct: float  # % of capital at risk

    # Reasoning
    thesis: str
    key_signals: list
    risks: list

    # Data freshness
    data_source: str
    spread_pct: float


class LiveTraderAnalyzer:
    """
    Analyzes Hyperliquid markets and generates trade recommendations.

    Uses 5-factor model:
    1. Funding Rate (carry + momentum)
    2. Open Interest (crowdedness)
    3. Price Momentum (24h trend)
    4. Order Book Imbalance (directional pressure)
    5. Liquidity Quality (execution feasibility)
    """

    def __init__(self, capital: float = 1000.0):
        self.connector = HyperliquidConnector()
        self.capital = capital

    def _calculate_funding_score(self, funding_rate: float) -> float:
        """
        Score funding rate (-1 to +1).

        Negative funding = shorts pay longs = LONG bias
        Positive funding = longs pay shorts = SHORT bias
        """
        # Normalize: typical range ±0.01% (annualized ~36%)
        score = -np.clip(funding_rate * 10000, -1, 1)  # Flip: negative = good for longs
        return score

    def _calculate_momentum_score(self, price_change_24h: float) -> float:
        """
        Score 24h momentum (-1 to +1).

        Mild positive = trend continuation
        Extreme positive = mean reversion risk
        """
        change_pct = price_change_24h * 100

        if change_pct > 20:
            return 0.5  # Overextended, but strong
        elif change_pct > 10:
            return 0.8  # Strong trend
        elif change_pct > 2:
            return 0.5  # Mild trend
        elif change_pct > -2:
            return 0.0  # Neutral
        elif change_pct > -10:
            return -0.5  # Mild downtrend
        elif change_pct > -20:
            return -0.8  # Strong downtrend
        else:
            return -0.5  # Oversold

    def _calculate_liquidity_score(self, spread_pct: float) -> Tuple[float, str]:
        """
        Score liquidity quality.

        Returns: (score, quality_label)
        """
        if spread_pct < 0.01:
            return 1.0, "excellent"
        elif spread_pct < 0.05:
            return 0.8, "good"
        elif spread_pct < 0.1:
            return 0.5, "moderate"
        elif spread_pct < 0.5:
            return 0.2, "poor"
        else:
            return -1.0, "avoid"

    def _calculate_imbalance_score(self, bid_depth: float, ask_depth: float) -> float:
        """
        Score order book imbalance (-1 to +1).

        Positive = more bids than asks = buying pressure
        """
        total = bid_depth + ask_depth
        if total == 0:
            return 0.0

        imbalance = (bid_depth - ask_depth) / total
        return np.clip(imbalance, -1, 1)

    def analyze_token(self, asset: str, risk_profile: str = "moderate") -> TradeRecommendation:
        """
        Generate complete trade recommendation for a token.

        Args:
            asset: Token symbol (e.g., 'BTC', 'ETH', 'SOL')
            risk_profile: 'conservative', 'moderate', 'aggressive'

        Returns:
            TradeRecommendation with full analysis
        """
        # Fetch data
        market = self.connector.get_market_data(asset)
        if not market:
            raise ValueError(f"Could not fetch data for {asset}")

        liq = self.connector.analyze_liquidity(asset)

        # Calculate signal scores
        funding_score = self._calculate_funding_score(market.funding_rate)
        momentum_score = self._calculate_momentum_score(market.price_change_24h)
        liq_score, liq_quality = self._calculate_liquidity_score(liq.get('spread_pct', 0.5))
        imbalance_score = self._calculate_imbalance_score(
            liq.get('bid_depth', 0),
            liq.get('ask_depth', 0)
        )

        # Composite signal
        weights = {
            'funding': 0.25,
            'momentum': 0.30,
            'imbalance': 0.25,
            'liquidity': 0.20
        }

        composite_score = (
            weights['funding'] * funding_score +
            weights['momentum'] * momentum_score +
            weights['imbalance'] * imbalance_score +
            weights['liquidity'] * liq_score
        )

        # Determine action
        if composite_score > 0.3:
            action = "LONG"
            confidence = min(abs(composite_score), 1.0)
        elif composite_score < -0.3:
            action = "SHORT"
            confidence = min(abs(composite_score), 1.0)
        else:
            action = "NEUTRAL"
            confidence = 0.5

        # Calculate levels
        current = market.mark_price
        spread = market.ask - market.bid

        if action == "LONG":
            entry = current + spread * 0.5  # Mid
            stop = entry * 0.95  # 5% stop
            target = entry * 1.08  # 8% target
            leverage = 2.0 if risk_profile == "conservative" else 3.0 if risk_profile == "moderate" else 5.0
        elif action == "SHORT":
            entry = current - spread * 0.5
            stop = entry * 1.05  # 5% stop (above)
            target = entry * 0.92  # 8% target (below)
            leverage = 2.0 if risk_profile == "conservative" else 3.0 if risk_profile == "moderate" else 5.0
        else:
            entry = current
            stop = current * 0.90
            target = current * 1.10
            leverage = 0  # No position

        # Position sizing (conservative: 2% risk per trade)
        risk_per_trade = 0.02 if risk_profile == "conservative" else 0.03 if risk_profile == "moderate" else 0.05
        position_size = (self.capital * risk_per_trade) / abs(entry - stop) if action != "NEUTRAL" else 0
        position_size_pct = min(position_size / self.capital * 100, 50)  # Cap at 50%

        # Risk metrics
        rr = abs(target - entry) / abs(stop - entry) if action != "NEUTRAL" else 0
        max_loss = position_size_pct * abs(entry - stop) / entry if action != "NEUTRAL" else 0

        # Build thesis
        thesis_parts = []
        if funding_score > 0.3:
            thesis_parts.append(f"Negative funding ({market.funding_rate*100:.4f}%): longs get paid to hold")
        elif funding_score < -0.3:
            thesis_parts.append(f"Positive funding ({market.funding_rate*100:.4f}%): shorts get paid to hold")

        if momentum_score > 0.5:
            thesis_parts.append(f"Strong 24h momentum (+{market.price_change_24h*100:.2f}%)")
        elif momentum_score < -0.5:
            thesis_parts.append(f"Strong 24h downtrend ({market.price_change_24h*100:.2f}%)")

        if abs(imbalance_score) > 0.3:
            direction = "buying" if imbalance_score > 0 else "selling"
            thesis_parts.append(f"Order book shows {direction} pressure")

        thesis = "; ".join(thesis_parts) if thesis_parts else "No strong directional signals"

        # Key signals
        key_signals = [
            f"Funding: {market.funding_rate*100:.4f}% ({'LONG bias' if funding_score > 0 else 'SHORT bias' if funding_score < 0 else 'neutral'})",
            f"24h Change: {market.price_change_24h*100:+.2f}%",
            f"OI: ${market.open_interest:,.0f}",
            f"Spread: {liq.get('spread_pct', 0)*100:.4f}% ({liq_quality})",
            f"OB Imbalance: {imbalance_score:+.2f}"
        ]

        # Risks
        risks = []
        if liq_score < 0.5:
            risks.append(f"Poor liquidity ({liq_quality}): wide spreads, slippage risk")
        if abs(market.price_change_24h) > 0.15:
            risks.append("High 24h volatility: mean reversion risk")
        if market.funding_rate > 0.001:
            risks.append(f"High funding cost: {market.funding_rate*100:.4f}% paid by longs")

        return TradeRecommendation(
            asset=asset,
            timestamp=datetime.utcnow(),
            action=action,
            confidence=round(confidence, 2),
            current_price=current,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            position_size_pct=round(position_size_pct, 2),
            leverage=leverage,
            risk_reward=round(rr, 2),
            max_loss_pct=round(max_loss, 2),
            thesis=thesis,
            key_signals=key_signals,
            risks=risks if risks else ["No significant risks identified"],
            data_source="Hyperliquid",
            spread_pct=liq.get('spread_pct', 0)
        )

    def format_recommendation(self, rec: TradeRecommendation) -> str:
        """Format recommendation as readable report."""
        lines = [
            f"# {rec.asset} Trade Analysis",
            f"",
            f"## Recommendation: {rec.action}",
            f"**Confidence:** {rec.confidence*100:.0f}%",
            f"**Timestamp:** {rec.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC",
            f"",
            f"## Price Levels",
            f"| Level | Price |",
            f"|-------|-------|",
            f"| Current | ${rec.current_price:,.2f} |",
            f"| Entry | ${rec.entry_price:,.2f} |",
            f"| Stop Loss | ${rec.stop_loss:,.2f} ({abs((rec.stop_loss-rec.entry_price)/rec.entry_price)*100:.2f}%) |",
            f"| Take Profit | ${rec.take_profit:,.2f} (+{abs((rec.take_profit-rec.entry_price)/rec.entry_price)*100:.2f}%) |",
            f"",
            f"## Position",
            f"- **Size:** {rec.position_size_pct:.1f}% of capital",
            f"- **Leverage:** {rec.leverage}x" if rec.action != "NEUTRAL" else "- **Leverage:** N/A",
            f"- **Risk/Reward:** {rec.risk_reward:.2f}",
            f"- **Max Loss:** {rec.max_loss_pct:.2f}% of capital",
            f"",
            f"## Thesis",
            f"{rec.thesis}",
            f"",
            f"## Key Signals",
        ]

        for sig in rec.key_signals:
            lines.append(f"- {sig}")

        lines.extend([
            f"",
            f"## Risks",
        ])

        for risk in rec.risks:
            lines.append(f"- ⚠️ {risk}")

        lines.extend([
            f"",
            f"## Execution",
            f"- **Source:** {rec.data_source}",
            f"- **Spread:** {rec.spread_pct*100:.4f}%",
            f"",
            f"---",
            f"*This is an automated analysis. Not financial advice. Always DYOR.*"
        ])

        return "\n".join(lines)


def main():
    """Example analysis."""
    analyzer = LiveTraderAnalyzer(capital=1000)

    print("="*60)
    print("LIVE TRADER ANALYSIS")
    print("="*60)

    # Analyze SOL
    print("\n" + "="*60)
    print("SOL/USDC Analysis")
    print("="*60)

    try:
        rec = analyzer.analyze_token("SOL", risk_profile="moderate")
        print(analyzer.format_recommendation(rec))
    except Exception as e:
        print(f"Error: {e}")

    # Analyze BTC
    print("\n" + "="*60)
    print("BTC/USDC Analysis")
    print("="*60)

    try:
        rec = analyzer.analyze_token("BTC", risk_profile="moderate")
        print(analyzer.format_recommendation(rec))
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
