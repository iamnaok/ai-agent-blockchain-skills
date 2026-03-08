# Market Analyzer Trader Skill

## Overview

Real-time cryptocurrency trading analysis system using DEX data (Hyperliquid, ApeX) to generate LONG/SHORT/NEUTRAL trade recommendations with entry levels, stop losses, take profits, and position sizing.

## Capabilities

- **Live DEX Data**: Real-time price, order book, funding rates from Hyperliquid
- **Liquidation Oracle**: Liquidation cluster detection and cascade risk analysis
- **Trader Analysis**: 5-signal framework for directional recommendations
- **Risk Management**: Position sizing, stop/target levels, leverage recommendations

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  MARKET ANALYZER TRADER                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐    ┌──────────────────┐               │
│  │ Hyperliquid     │    │ Liquidation      │               │
│  │ Connector       │───→│ Oracle           │               │
│  │ (REST API)      │    │ (WebSocket)      │               │
│  └─────────────────┘    └──────────────────┘               │
│           │                        │                       │
│           └──────────┬─────────────┘                       │
│                      ▼                                     │
│           ┌──────────────────┐                            │
│           │ Live Trader      │                            │
│           │ Analysis         │                            │
│           │                  │                            │
│           │ • Funding Score  │                            │
│           │ • Momentum Score │                            │
│           │ • Liquidity      │                            │
│           │ • OB Imbalance   │                            │
│           │ • Liquidation    │                            │
│           │   Clusters       │                            │
│           └──────────────────┘                            │
│                      │                                     │
│                      ▼                                     │
│           ┌──────────────────┐                            │
│           │ Trade            │                            │
│           │ Recommendation   │                            │
│           │                  │                            │
│           │ • Action         │                            │
│           │ • Entry/Stop/TP  │                            │
│           │ • Position Size    │                            │
│           │ • Reasoning      │                            │
│           └──────────────────┘                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Modules

### 1. Hyperliquid Connector

**File:** `modules/hyperliquid_connector.py`

Real-time market data from Hyperliquid DEX.

**Features:**
- Mark price, index price, oracle price
- Order book depth (L2)
- Funding rates (annualized)
- 24h volume and price change
- Open interest
- Bid/ask spread analysis

**Usage:**
```python
from modules.hyperliquid_connector import HyperliquidConnector

connector = HyperliquidConnector()

# Get market data
btc = connector.get_market_data("BTC")
print(f"BTC: ${btc.mark_price:,.2f}")
print(f"Funding: {btc.funding_rate*100:.4f}%")
print(f"24h Change: {btc.price_change_24h*100:+.2f}%")

# Analyze liquidity
liq = connector.analyze_liquidity("BTC")
print(f"Spread: {liq['spread_pct']*100:.4f}%")
print(f"Quality: {liq['liquidity_quality']}")
```

### 2. Liquidation Oracle

**File:** `modules/liquidation_oracle.py`

Real-time liquidation monitoring and cluster analysis.

**Features:**
- WebSocket connection to Hyperliquid trade stream
- Liquidation event detection
- Cluster heatmap by price level
- Risk zone identification
- Cascade risk assessment
- Support/resistance from liquidation levels

**Usage:**
```python
from modules.liquidation_oracle import LiquidationOracle

oracle = LiquidationOracle()

# Start live monitoring
oracle.start(['BTC', 'ETH', 'SOL'])

# Get liquidation heatmap
heatmap = oracle.get_liquidation_heatmap('BTC', n_levels=10)

# Get risk zones
risk = oracle.get_risk_zones('BTC', current_price=67300)

# Cascade risk
cascade_risk = oracle.estimate_liquidation_cascade_risk(
    'BTC', 'down', 0.05  # 5% drop
)
print(f"Cascade risk: {cascade_risk*100:.0f}%")
```

### 3. Live Trader Analysis

**File:** `modules/live_trader_analysis.py`

Complete trade analysis with recommendation.

**5-Signal Framework:**
1. **Funding Rate** — Carry and momentum signal
2. **24h Momentum** — Trend direction
3. **Order Book Imbalance** — Buying/selling pressure
4. **Liquidity Quality** — Execution feasibility
5. **Liquidation Clusters** — Support/resistance levels

**Output:**
- Action: LONG / SHORT / NEUTRAL
- Confidence: 0-100%
- Entry / Stop / Target prices
- Position size (% of capital)
- Leverage recommendation
- Full thesis and key signals

**Usage:**
```python
from modules.live_trader_analysis import LiveTraderAnalyzer

analyzer = LiveTraderAnalyzer(capital=1000)

# Analyze token
rec = analyzer.analyze_token('BTC', risk_profile='moderate')

print(f"Action: {rec.action}")
print(f"Confidence: {rec.confidence*100:.0f}%")
print(f"Entry: ${rec.entry_price:,.2f}")
print(f"Stop: ${rec.stop_loss:,.2f}")
print(f"Target: ${rec.take_profit:,.2f}")
print(f"Size: {rec.position_size_pct:.1f}%")
print(f"Leverage: {rec.leverage}x")

# Format as report
report = analyzer.format_recommendation(rec)
print(report)
```

## Quick Start

### Get Live Trade Recommendation

```python
from modules.live_trader_analysis import LiveTraderAnalyzer

# Create analyzer
analyzer = LiveTraderAnalyzer(capital=1000)

# Analyze any token on Hyperliquid
rec = analyzer.analyze_token('SOL', risk_profile='moderate')

# Print recommendation
print(f"""
{'='*60}
{rec.asset} Trade Recommendation
{'='*60}

Action: {rec.action}
Confidence: {rec.confidence*100:.0f}%

Entry: ${rec.entry_price:,.2f}
Stop: ${rec.stop_loss:,.2f} ({abs((rec.stop_loss-rec.entry_price)/rec.entry_price)*100:.2f}%)
Target: ${rec.take_profit:,.2f} ({abs((rec.take_profit-rec.entry_price)/rec.entry_price)*100:.2f}%)

Position: {rec.position_size_pct:.1f}% of capital
Leverage: {rec.leverage}x
Risk/Reward: {rec.risk_reward:.2f}

Thesis: {rec.thesis}

Key Signals:
{chr(10).join('  • ' + s for s in rec.key_signals)}

Risks:
{chr(10).join('  • ' + r for r in rec.risks)}
""")
```

### Enhanced Analysis with Liquidation Data

```python
from modules.liquidation_oracle import LiquidationEnhancedTrader

# Create enhanced trader with liquidation data
trader = LiquidationEnhancedTrader(capital=1000)

# Start monitoring
trader.start_monitoring(['BTC', 'ETH', 'SOL'])

# Get analysis with liquidation insights
analysis = trader.analyze_with_liquidations('BTC', current_price=67300)

print(f"""
Liquidation-Enhanced Analysis:

Heatmap:
{chr(10).join(f"  ${h['price_level']:,.0f}: ${h['total_liquidated']:,.0f}" 
              for h in analysis['liquidation_heatmap'][:5])}

Cascade Risk:
  Long cascade if -5%: {analysis['cascade_risk']['long_cascade_if_down_5pct']*100:.0f}%
  Short cascade if +5%: {analysis['cascade_risk']['short_cascade_if_up_5pct']*100:.0f}%

Implications:
{chr(10).join('  • ' + i for i in analysis['trading_implications'])}
""")
```

## Risk Profiles

| Profile | Position Size | Leverage | Risk/Trade |
|---------|--------------|----------|------------|
| **Conservative** | 2% | 2x | Low |
| **Moderate** | 3% | 3x | Medium |
| **Aggressive** | 5% | 5x | High |

## Data Sources

**Hyperliquid DEX:**
- REST API: `https://api.hyperliquid.xyz`
- WebSocket: `wss://api.hyperliquid.xyz/ws`
- No authentication required for public data

**Data Points:**
- ✅ Real-time prices
- ✅ Order book depth
- ✅ Funding rates
- ✅ 24h volume
- ✅ Open interest
- ✅ Recent trades (with liquidation detection)

## Testing

### Test Hyperliquid Connection

```python
from modules.hyperliquid_connector import HyperliquidConnector

connector = HyperliquidConnector()

# Test multiple assets
for asset in ['BTC', 'ETH', 'SOL']:
    data = connector.get_market_data(asset)
    print(f"{asset}: ${data.mark_price:,.2f} "
          f"(spread: {((data.ask-data.bid)/data.mark_price)*100:.4f}%)")
```

### Test Liquidation Oracle

```python
from modules.liquidation_oracle import LiquidationOracle

oracle = LiquidationOracle()
oracle.start(['BTC', 'ETH'])

# Wait for data...
import time
time.sleep(5)

# Get liquidation heatmap
heatmap = oracle.get_liquidation_heatmap('BTC')
print(f"Top cluster: ${heatmap[0]['price_level']:,.0f} "
      f"(${heatmap[0]['total_liquidated']:,.0f} liquidated)")
```

## Files

| File | Purpose | Size |
|------|---------|------|
| `modules/hyperliquid_connector.py` | DEX API connector | ~10KB |
| `modules/liquidation_oracle.py` | Liquidation monitoring | ~17KB |
| `modules/live_trader_analysis.py` | Trade recommendation engine | ~12KB |
| `modules/bayesian_optimizer.py` | Parameter optimization | ~12KB |

## Limitations

**Current:**
- Liquidation clusters estimated from trade data (not exact position data)
- Requires WebSocket for real-time liquidation stream
- No historical liquidation database (only recent events)

**Future:**
- Authenticated API for exact position data
- Historical liquidation analysis
- Multi-DEX aggregation (ApeX, dYdX)

## Dependencies

```bash
pip install websockets numpy pandas requests
```

## Notes

- All prices are real-time from Hyperliquid
- Analysis is data-driven, no fabricated signals
- Position sizing uses quarter-Kelly (25% of optimal)
- Liquidation data is simulated for testing; enable WebSocket for live data

---

*Last Updated: 2026-03-03*
