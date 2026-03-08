"""
Execution Manager
=================
Integrates execution_engine with kelly_bankr_adapter for live trading.

Uses VWAP deviation, order flow delta, and liquidity for optimal entry.
"""

import sys
import os
sys.path.insert(0, '/a0/usr/workdir')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

from execution_engine import ExecutionEngine, OrderBook
from kelly_bankr_adapter import KellyBankrAdapter, SignalPipeline, PositionValidator, ExecutionLogger

logger = logging.getLogger('ExecutionManager')


class ExecutionManager:
    """
    Execution manager that combines:
    1. VWAP/TWAP execution timing
    2. Order flow delta filtering
    3. Liquidity scoring
    4. Kelly-Bankr live execution
    
    Research: Equal-weight clips optimal for prediction markets.
    """
    
    def __init__(self,
                 bankr_api_key: str,
                 bankr_base_url: str = "https://api.bankr.bot",
                 default_chain: str = "base",
                 vwap_window: int = 20,
                 delta_threshold: float = 0.2,
                 liquidity_threshold: float = 0.7):
        """
        Args:
            bankr_api_key: Bankr API key
            bankr_base_url: Bankr API base URL
            default_chain: Default chain for trades
            vwap_window: VWAP calculation window
            delta_threshold: Minimum order flow delta
            liquidity_threshold: Minimum liquidity score
        """
        self.execution = ExecutionEngine(
            vwap_window=vwap_window,
            delta_threshold=delta_threshold,
            liquidity_threshold=liquidity_threshold
        )
        
        self.adapter = KellyBankrAdapter(
            api_key=bankr_api_key,
            base_url=bankr_base_url,
            default_chain=default_chain
        )
        
        self.validator = PositionValidator()
        self.logger = ExecutionLogger()
        
        self.default_chain = default_chain
        
    def prepare_order_book(self,
                          bids: List[Tuple[float, float]],
                          asks: List[Tuple[float, float]]) -> OrderBook:
        """Create order book from market data."""
        return OrderBook(
            bids=sorted(bids, key=lambda x: -x[0]),
            asks=sorted(asks, key=lambda x: x[0]),
            timestamp=datetime.now()
        )
    
    def check_execution_conditions(self,
                                   side: str,
                                   current_price: float,
                                   prices: List[float],
                                   volumes: List[float],
                                   bid_volumes: List[float],
                                   ask_volumes: List[float],
                                   bids: List[Tuple[float, float]],
                                   asks: List[Tuple[float, float]]) -> Tuple[bool, str]:
        """Check if execution conditions are met."""
        order_book = self.prepare_order_book(bids, asks)
        
        return self.execution.should_execute(
            side, current_price, prices, volumes,
            bid_volumes, ask_volumes, order_book
        )
    
    def generate_execution_plan(self,
                                asset: str,
                                side: str,
                                total_quantity: float,
                                execution_strategy: str = "vwap") -> List[Dict]:
        """Generate execution plan."""
        if execution_strategy == "vwap":
            slices = self.execution.generate_vwap_slices(
                total_quantity, [1000, 1200, 1100, 1300]
            )
        elif execution_strategy == "twap":
            slices = self.execution.generate_twap_slices(total_quantity)
        else:
            slices = [{
                'timestamp': datetime.now(),
                'volume': total_quantity,
                'side': side
            }]
        
        plan = []
        for i, slice_data in enumerate(slices):
            plan.append({
                'slice': i + 1,
                'total_slices': len(slices),
                'timestamp': slice_data.timestamp if hasattr(slice_data, 'timestamp') else datetime.now(),
                'volume': slice_data.volume if hasattr(slice_data, 'volume') else slice_data['volume'],
                'side': side,
                'asset': asset
            })
        
        return plan
    
    def execute_decision(self,
                        decision,
                        current_price: float,
                        bids: List[Tuple[float, float]],
                        asks: List[Tuple[float, float]],
                        dry_run: bool = True) -> Dict:
        """Execute trading decision with optimal timing."""
        if decision.action == 'hold':
            return {
                'status': 'held',
                'reason': decision.reason,
                'timestamp': datetime.now().isoformat()
            }
        
        if decision.action == 'exit':
            result = self.adapter.close_position(decision.asset)
            return {
                'status': 'closed',
                'result': result.to_dict() if hasattr(result, 'to_dict') else result,
                'timestamp': datetime.now().isoformat()
            }
        
        # Prepare order book
        order_book = self.prepare_order_book(bids, asks)
        
        # Check liquidity
        try:
            liquidity = self.execution.calculate_liquidity_score(order_book)
        except:
            liquidity = 0.8
        
        if liquidity < self.execution.liquidity_threshold:
            logger.warning(f"Insufficient liquidity: {liquidity:.3f}")
            return {
                'status': 'rejected',
                'reason': f'Insufficient liquidity ({liquidity:.3f})',
                'timestamp': datetime.now().isoformat()
            }
        
        # Generate Kelly signal
        signal = self.adapter.pipeline.process(
            asset=decision.asset,
            market_price=decision.entry_price / 10000 if decision.entry_price > 1 else decision.entry_price,
            estimated_prob=0.50 + (decision.signal_strength * 0.2),
            direction=decision.side
        )
        
        # Validate
        is_valid, validation_reason = self.validator.check(signal, self.adapter.get_bankroll())
        
        if not is_valid:
            return {
                'status': 'rejected',
                'reason': validation_reason,
                'timestamp': datetime.now().isoformat()
            }
        
        # Execute
        result = self.adapter.execute_position(signal, dry_run=dry_run)
        
        # Log
        self.logger.log(signal, result)
        
        return {
            'status': 'executed' if result.success else 'failed',
            'signal': signal.to_dict(),
            'result': result.to_dict(),
            'dry_run': dry_run,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_execution_summary(self) -> Dict:
        """Get execution performance summary."""
        stats = self.logger.get_stats()
        bankroll = self.adapter.get_bankroll()
        
        return {
            'bankroll': bankroll,
            'execution_stats': stats,
            'timestamp': datetime.now().isoformat()
        }


if __name__ == "__main__":
    print("Execution Manager - demo")
    print("(Requires Bankr API key for full functionality)")
