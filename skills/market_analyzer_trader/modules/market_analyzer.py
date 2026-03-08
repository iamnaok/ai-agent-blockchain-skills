"""
APEX Market Analyzer
====================
Wrapper module to scan markets using APEX Signal Engine.
Runs signal calculations for configured assets and logs results.
"""

import sys
import os
import json
import logging
from datetime import datetime
from typing import List, Dict

# Add paths for imports
sys.path.insert(0, '/a0/usr/workdir/ai-agent-blockchain-skills/skills/market_analyzer_trader')
sys.path.insert(0, '/a0/usr/workdir/ai-agent-blockchain-skills/skills/professional_quant_trader/modules')

import numpy as np
import pandas as pd

from modules.market_data import MarketDataIngestor
from modules.apex_signal_engine import APEXSignalEngine

try:
    from modules.signal_outcome_tracker import SignalOutcomeTracker
    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False
    print("Warning: signal_outcome_tracker not available")


# Configure logging
log_dir = '/a0/usr/workdir/ai-agent-blockchain-skills/skills/market_analyzer_trader/logs'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'scan.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MarketAnalyzer')


class MarketAnalyzer:
    """
    Market analyzer wrapper for APEX trading bot.

    Runs signal calculations for selected assets and manages
    the signal outcome tracker for performance monitoring.
    """

    def __init__(self, assets: List[str] = None):
        """
        Initialize market analyzer.

        Args:
            assets: List of asset pairs to monitor (e.g., ['ETH-USDC', 'BTC-USDC'])
        """
        self.assets = assets or ['ETH-USDC', 'BTC-USDC']
        self.data_ingestor = MarketDataIngestor()
        self.signal_engine = APEXSignalEngine()
        self.tracker = SignalOutcomeTracker() if TRACKER_AVAILABLE else None

        logger.info("MarketAnalyzer initialized with assets: " + str(self.assets))

    def fetch_market_data(self, asset: str) -> pd.DataFrame:
        """Fetch OHLCV data for an asset."""
        try:
            df = self.data_ingestor.fetch_ohlcv(
                asset, 
                timeframe="1h", 
                source="synthetic"
            )
            logger.info("Fetched %d data points for %s" % (len(df), asset))
            return df
        except Exception as e:
            logger.error("Failed to fetch data for %s: %s" % (asset, str(e)))
            return None

    def calculate_signals(self, asset: str, df: pd.DataFrame) -> Dict:
        """Calculate all signals for an asset."""
        try:
            current_price = df['close'].iloc[-1]

            # Generate APEX signals
            signal = self.signal_engine.generate_signals(
                asset=asset,
                df=df,
                market_price=current_price,
                market_data={}
            )

            return signal.to_dict()
        except Exception as e:
            logger.error("Failed to calculate signals for %s: %s" % (asset, str(e)))
            import traceback
            traceback.print_exc()
            return None

    def check_pending_signals(self):
        """Check signal outcome tracker for pending signals."""
        if not self.tracker:
            logger.info("Signal tracker not available, skipping pending signal check")
            return

        try:
            # Get tracker summary to see pending signals
            summary = self.tracker.get_summary()
            total_signals = summary.get('total_signals', 0)
            logger.info("Found %d total tracked signals" % total_signals)

        except Exception as e:
            logger.error("Failed to check pending signals: %s" % str(e))

    def scan_markets(self) -> Dict:
        """
        Run full market scan for all configured assets.

        Returns:
            Dict with scan results for each asset
        """
        logger.info("=" * 60)
        logger.info("Starting market scan at " + str(datetime.now()))
        logger.info("=" * 60)

        results = {}

        # Check for pending signals first
        self.check_pending_signals()

        # Scan each asset
        for asset in self.assets:
            logger.info("--- Scanning " + asset + " ---")

            # Fetch data
            df = self.fetch_market_data(asset)
            if df is None:
                results[asset] = {"error": "Failed to fetch data"}
                continue

            # Calculate signals
            signals = self.calculate_signals(asset, df)
            if signals is None:
                results[asset] = {"error": "Failed to calculate signals"}
                continue

            results[asset] = signals

            # Log summary
            logger.info("Asset: " + asset)
            logger.info("  Direction: " + signals['direction'].upper())
            logger.info("  Combined Score: %.3f" % signals['combined_score'])
            logger.info("  Confidence: %.2f" % signals['confidence'])
            logger.info("  RSI: %.1f" % signals['rsi'])
            logger.info("  Kalman Trend: %+.2f" % signals['kalman_trend'])
            logger.info("  Crowd Extreme: %+.2f" % signals['crowd_extreme'])
            logger.info("  Liquidation Cluster: %+.2f" % signals['liquidation_cluster'])

        logger.info("=" * 60)
        logger.info("Market scan complete")
        logger.info("=" * 60)

        return results


if __name__ == "__main__":
    # Run standalone scan
    analyzer = MarketAnalyzer()
    results = analyzer.scan_markets()
    print(json.dumps(results, indent=2, default=str))
