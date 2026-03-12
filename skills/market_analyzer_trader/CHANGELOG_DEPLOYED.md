# Apex Trading Bot Changelog

All notable changes to the Apex Trading Bot will be documented in this file.

## [1.2.0] - 2026-03-12

### Phase 1 Integration - Non-Stationary Distribution Training

#### Added
- **Phase 1 Modules Deployed**
  - `modules/adaptive_thresholds.py` (353 lines): Rolling window statistics, dynamic threshold adjustment based on volatility and skewness
  - `modules/concept_drift_detector.py` (456 lines): Multi-method drift detection (KS test, CUSUM, Page-Hinkley), 5 market regime classifications
  - `modules/non_stationary_integration.py` (334 lines): Unified signal processing interface, regime-aware normalization

- **Phase 1 Integration into Signal Processing**
  - Modified `main.py` to integrate Phase 1 adaptive processing at line 987
  - Edge scores now processed with Phase 1 confidence weighting based on market regime
  - Adaptive thresholds replace static thresholds (0.5) with dynamic calculations
  - Signal processing pipeline now uses `process_signal_adaptive()` for all signals

- **Phase 1 Monitoring**
  - `phase1_monitor.py`: Performance monitoring script
  - Daily automated monitoring via cron job (runs at 9 AM)
  - Reports saved to `phase1_monitor_report.json`

#### Changed
- **Signal Processing Pipeline**
  - Before: Static threshold check (`if abs(edge_score) >= MIN_EDGE_SCORE`)
  - After: Phase 1 adaptive processing with confidence weighting
  - Edge scores now adjusted based on volatility, regime, and drift detection
  - Thresholds dynamically adapt to market conditions

#### Technical Details
- Window Size: 100 samples (~4 days at hourly scans)
- Min Samples Before Adaptation: 20
- Drift Detection Methods: Kolmogorov-Smirnov, CUSUM, Page-Hinkley
- Regime Classifications: trending_volatile, trending, high_volatility, low_volatility, ranging
- Performance Impact: <5ms per signal
- Memory Usage: ~10KB per signal

#### Verification
- ✅ Phase 1 modules deployed to VPS
- ✅ Integrated into main.py signal processing
- ✅ Bot restarted and operational
- ✅ Phase 1 active: `[Phase 1] Adaptive signal processing enabled`
- ✅ Monitoring system active

---

## [1.1.0] - 2026-03-12

### File Structure Reorganization

#### Changed
- **Directory Structure**
  - Reorganized from flat structure to proper Python package
  - Main entry point: `main.py` (was `apex_trading_bot.py` at root)
  - All apex files moved to `/home/ubuntu/apex_bot/`
  - Service updated to point to new location

#### Files Moved
- `apex_trading_bot.py` → `apex_bot/main.py`
- `apex_signal_tracker.py` → `apex_bot/apex_signal_tracker.py`
- `apex_signal_tracker_wrapper.py` → `apex_bot/apex_signal_tracker_wrapper.py`
- `apex_market_data.py` → `apex_bot/apex_market_data.py`
- `apex_dex_skill.py` → `apex_bot/apex_dex_skill.py`
- `apex_signal_performance.json` → `apex_bot/apex_signal_performance.json`

#### Service Configuration
- **Entry Point**: `/usr/bin/python3 /home/ubuntu/apex_bot/main.py --daemon`
- **Working Directory**: `/home/ubuntu/apex_bot`

---

## [1.0.2] - 2026-03-08

### Signal Performance Tracking Fix

#### Fixed
- **Signal Outcome Tracking**
  - Fixed `check_apex_outcomes()` not being called in main loop
  - Added outcome checking to `run_analysis_cycle()` function
  - Signals now properly resolve when TP/SL levels are hit

- **Telegram Performance Display**
  - Replaced placeholder text with real performance stats
  - Now shows: "📊 Signal Performance: X/Y resolved | Win rate: Z%"
  - Uses `get_apex_performance()` for real-time data

#### Changed
- **Signal Weights** (based on performance data)
  - `cvd_divergence`: 0.0 (disabled, 25% accuracy)
  - `crowd_extreme`: 0.5 (44% accuracy, down from 1.0)
  - `liquidation_cluster`: 1.0 (58% accuracy)
  - `kalman_trend`: 1.0 (57-62% accuracy)

---

## [1.0.1] - 2026-03-08

### Decimal Precision Fix

#### Fixed
- **Financial Calculations**
  - `backtest_engine.py`: Converted PnL calculations to Decimal
  - `smart_sizer.py`: Added Decimal import for position sizing
  - `kelly_portfolio_optimizer.py`: Added Decimal import for Kelly math

---

## [1.0.0] - 2026-03-08

### Initial Release

#### Added
- **20-Factor Signal Engine**
  - 17 custom APEX signals (CVD_DIVERGENCE, LIQUIDATION_CLUSTER, FUNDING_ARB, KALMAN_TREND, etc.)
  - 3 traditional indicators (Particle Filter, RSI, MACD)
  - Weighted composite score calculation

- **Signal Outcome Tracking**
  - Records Long/Short/Neutral recommendations with SL/TP levels
  - Tracks TP_HIT vs SL_HIT outcomes
  - Calculates win rate per signal type
  - 24-hour expiration for unresolved signals

- **Risk Controls**
  - Edge scoring with MIN_EDGE_SCORE threshold
  - Regime analysis (HMM)
  - Multi-timeframe checks
  - Position sizing via Bankr

- **Scheduled Tasks**
  - APEX_15min_Market_Scan (every 15 minutes)
  - APEX_4hour_Recommendation (every 4 hours)
  - APEX_Health_Check (continuous monitoring)

#### Configuration
- **Mode**: DRY_RUN (no live trading)
- **Markets**: BTC, ETH, SOL, TON
- **Min Edge Score**: 20
- **R/R Ratio**: 3.0
- **Scan Interval**: 15 minutes

