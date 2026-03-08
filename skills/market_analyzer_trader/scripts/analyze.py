#!/usr/bin/env python3
"""
Market Analyzer Trader CLI
==========================

Analyze markets and execute trades using Kelly + Particle Filter + VWAP strategy.

Usage:
    python analyze.py --market "ETH-USDC" --analyze
    python analyze.py --market "ETH-USDC" --trade --dry-run
    python analyze.py --backtest --period 90d
"""

import sys
import os
# Add paths for imports
sys.path.insert(0, '/a0/usr/workdir')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from modules.market_data import MarketDataIngestor
from modules.signal_engine import SignalEngine
from modules.decision_engine import DecisionEngine
from modules.execution_manager import ExecutionManager
from modules.backtest_suite import BacktestSuite

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MarketAnalyzer')


def get_bankr_api_key() -> str:
    """Get Bankr API key from environment."""
    api_key = os.environ.get('BANKR_API_KEY') or "bk_KNKHH4Q3TMPXZN652FM7NM26EDSZU4SK"
    return api_key


def analyze_market(asset: str, ingestor: MarketDataIngestor) -> dict:
    """Analyze a market and return signal."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {asset}")
    print(f"{'='*60}")
    
    # Fetch data
    print(f"\n[1/4] Fetching market data...")
    df = ingestor.fetch_ohlcv(asset, timeframe="1h", source="synthetic")
    print(f"      Loaded {len(df)} data points")
    
    # Generate signal
    print(f"\n[2/4] Generating signals...")
    engine = SignalEngine()
    current_price = df['close'].iloc[-1]
    signal = engine.generate_signal(asset, df, current_price)
    summary = engine.get_signal_summary(signal)
    
    print(f"      Direction: {summary['direction']}")
    print(f"      Strength: {summary['strength']}")
    print(f"      Confidence: {summary['confidence']}")
    print(f"\n      Technical Indicators:")
    for k, v in summary['technical'].items():
        print(f"        {k}: {v}")
    
    # Generate decision
    print(f"\n[3/4] Kelly sizing...")
    decision_engine = DecisionEngine()
    liquidity = 0.85  # Simulated
    decision = decision_engine.make_decision(signal, current_price, liquidity)
    
    if decision.action == 'enter':
        print(f"      Action: {decision.action.upper()}")
        print(f"      Side: {decision.side}")
        print(f"      Size: {decision.size*100:.2f}% of capital")
        print(f"      Leverage: {decision.leverage:.1f}x")
        print(f"      Entry: ${decision.entry_price:.2f}")
        print(f"      Stop: ${decision.stop_loss:.2f}")
        print(f"      Take Profit: ${decision.take_profit:.2f}")
        print(f"      Kelly Fraction: {decision.kelly_fraction:.4f}")
    else:
        print(f"      Action: {decision.action.upper()}")
        print(f"      Reason: {decision.reason}")
    
    # Execution plan
    print(f"\n[4/4] Execution plan...")
    from modules.execution_engine import ExecutionEngine
    exec_engine = ExecutionEngine()
    vwap = exec_engine.calculate_vwap(df['close'].tolist(), df['volume'].tolist())
    deviation = exec_engine.calculate_vwap_deviation(current_price, df['close'].tolist(), df['volume'].tolist())
    
    print(f"      VWAP: ${vwap:.2f}")
    print(f"      VWAP Deviation: {deviation*100:.2f}%")
    print(f"      Strategy: {'Wait for price < VWAP' if deviation > 0 else 'Price below VWAP - favorable'}")
    
    return {
        'signal': signal,
        'decision': decision,
        'vwap': vwap,
        'deviation': deviation,
        'data': df
    }


def trade_market(asset: str, dry_run: bool = True) -> dict:
    """Execute trade on a market."""
    print(f"\n{'='*60}")
    print(f"TRADING: {asset} {'(DRY RUN)' if dry_run else '(LIVE)'}")
    print(f"{'='*60}")
    
    # Setup
    api_key = get_bankr_api_key()
    ingestor = MarketDataIngestor(bankr_api_key=api_key)
    
    # Analyze
    result = analyze_market(asset, ingestor)
    
    if result['decision'].action != 'enter':
        print(f"\nNo trade executed: {result['decision'].reason}")
        return result
    
    # Execute
    print(f"\n[EXECUTION]")
    
    if dry_run:
        print("      Mode: DRY RUN - no actual trade")
        print(f"      Would execute: {result['decision'].side} {asset}")
        print(f"      Size: {result['decision'].size*100:.2f}%")
        print(f"      Leverage: {result['decision'].leverage:.1f}x")
        print(f"      Entry: ${result['decision'].entry_price:.2f}")
    else:
        manager = ExecutionManager(api_key)
        # Simulate execution
        print(f"      Executing via Bankr API...")
        print(f"      (Actual execution would use kelly_bankr_adapter)")
    
    return result


def run_backtest(period: str = "90d"):
    """Run backtest."""
    print(f"\n{'='*60}")
    print(f"BACKTEST: {period}")
    print(f"{'='*60}")
    
    # Parse period
    days = int(period.replace('d', ''))
    
    # Generate data
    print(f"\n[1/3] Generating synthetic data ({days} days)...")
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days*24, freq='1h')
    returns = np.random.normal(0.0001, 0.015, len(dates))
    prices = 3000 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.randn(len(dates)) * 0.002),
        'high': prices * (1 + abs(np.random.randn(len(dates)) * 0.005)),
        'low': prices * (1 - abs(np.random.randn(len(dates)) * 0.005)),
        'close': prices,
        'volume': np.random.randint(1000, 50000, len(dates)),
        'bid_volume': np.random.randint(500, 25000, len(dates)),
        'ask_volume': np.random.randint(500, 25000, len(dates))
    }, index=dates)
    
    print(f"      Generated {len(df)} hourly data points")
    
    # Run backtest
    print(f"\n[2/3] Running strategy comparison...")
    suite = BacktestSuite(train_days=30, test_days=7)
    
    try:
        comparison = suite.run_comparison(df)
        print(f"\n{comparison.to_string(index=False)}")
        
        # Full backtest
        print(f"\n[3/3] Full backtest with trade log...")
        results = suite.run_full_backtest(df)
        
        # Report
        report = suite.generate_report(results)
        print(report)
        
        # Save results
        output_file = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        import json
        with open(output_file, 'w') as f:
            json.dump(results['metrics'], f, indent=2)
        print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        print(f"      Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='Market Analyzer Trader - Kelly + Particle Filter + VWAP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze.py --market "ETH-USDC" --analyze
  python analyze.py --market "ETH-USDC" --trade --dry-run
  python analyze.py --backtest --period 90d
  python analyze.py --market "BTC-USDC" --trade --dry-run
        """
    )
    
    parser.add_argument('--market', type=str, help='Market to analyze (e.g., ETH-USDC)')
    parser.add_argument('--analyze', action='store_true', help='Analyze market only')
    parser.add_argument('--trade', action='store_true', help='Execute trade')
    parser.add_argument('--dry-run', action='store_true', help='Simulate without execution')
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--period', type=str, default='90d', help='Backtest period (e.g., 90d)')
    parser.add_argument('--test', action='store_true', help='Run unit tests')
    
    args = parser.parse_args()
    
    if args.test:
        print("Running unit tests...")
        import subprocess
        subprocess.run([sys.executable, '-m', 'pytest', 'modules/', '-v'])
        return
    
    if args.backtest:
        run_backtest(args.period)
    elif args.trade:
        if not args.market:
            print("Error: --market required for trading")
            return
        trade_market(args.market, dry_run=args.dry_run)
    elif args.analyze:
        if not args.market:
            print("Error: --market required for analysis")
            return
        api_key = get_bankr_api_key()
        ingestor = MarketDataIngestor(bankr_api_key=api_key)
        analyze_market(args.market, ingestor)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
