# Changelog

All notable changes to the APEX Market Analyzer Trader skill will be documented in this file.

## [1.0.0] - 2026-03-08

### Added
- **Signal Outcome Tracking System** (`signal_outcome_tracker.py`)
  - Records Long/Short/Neutral recommendations with SL/TP levels
  - Tracks TP_HIT vs SL_HIT outcomes continuously
  - Calculates win rate per signal type
  - Dynamic weight adjustment based on performance
  - 24-hour expiration for unresolved signals

- **Scheduled Tasks** (2 new automated tasks)
  - `APEX_15min_Market_Scan` (UUID: ZTIqf3Qr) - runs every 15 minutes
  - `APEX_4hour_Recommendation` (UUID: Q5KOajpq) - runs every 4 hours

- **Signal Performance Database** (`/a0/usr/workdir/signal_performance.json`)
  - JSON storage for all signal history
  - Per-signal win rate tracking
  - Overall system win rate calculation

### Fixed
- **Signal Accuracy Issues**
  - `cvd_divergence` weight reduced to **0.0** (disabled due to 25% accuracy)
  - `crowd_extreme` weight reduced to **0.5** (44% accuracy, down from 1.0)
  - Combined weighted average calculation verified

- **Financial Precision**
  - `backtest_engine.py` - converted PnL calculations to Decimal
  - `smart_sizer.py` - added Decimal import for position sizing
  - `kelly_portfolio_optimizer.py` - added Decimal import for Kelly math
  - Fixed duplicate import in backtest_engine.py
  - Removed unused PositionSize import from smart_sizer.py

- **Import Errors**
  - Verified all modules load successfully
  - Fixed module dependency issues

### Changed
- Signal weight adjustments based on historical accuracy
- Decimal precision for all financial calculations
- 4-hour recommendation cycle with 2:1 risk-reward ratio

### Technical Details

#### Signal Configuration (Updated)
| Signal | Old Weight | New Weight | Accuracy | Status |
|--------|-----------|-----------|----------|--------|
| cvd_divergence | 1.0 | **0.0** | 25% | 🚫 DISABLED |
| crowd_extreme | 1.0 | **0.5** | 44% | ⚠️ REDUCED |
| liquidation_cluster | 1.0 | 1.0 | 55% | ✅ Active |
| kalman_trend | 1.0 | 1.0 | 57% | ✅ Active |

#### Recommendation Logic
- **LONG**: Edge score > 15, bullish signals
- **SHORT**: Edge score > 15, bearish signals  
- **NEUTRAL**: Edge score < 15 or conflicting signals
- **SL**: 2% from entry price
- **TP**: 4% from entry price (2:1 RR ratio)

### Operational Status
- ✅ All modules import successfully
- ✅ Scheduled tasks running
- ✅ Signal tracking operational
- ⏳ Profitability validation pending (20+ signals needed)

### Backups
- Pre-change backup: `/home/ubuntu/backups/2026-03-08-1045-signal-tracking/`
- Post-integration backup: `/home/ubuntu/backups/20260308-112742-integration/`

### Dependencies
- Python 3.8+
- Bankr API access
- Hyperliquid CLOB API

