"""
Kelly-Bankr Quant Execution Bridge
==================================
Connects Kelly portfolio optimizer to Bankr trading API.
Bridges quantitative signal generation to live execution.

Author: Agent Zero Master Developer
"""

import sys
import os
import json
import time
import logging
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict

import numpy as np

# Add path to existing quant modules
QUANT_MODULES_PATH = '/a0/usr/workdir/ai-agent-blockchain-skills/skills/professional_quant_trader/modules'
if QUANT_MODULES_PATH not in sys.path:
    sys.path.insert(0, QUANT_MODULES_PATH)

# Import existing quant modules (handle Dict import issue)
try:
    from kelly_portfolio_optimizer import KellyPortfolioOptimizer
    from particle_filter import PredictionMarketParticleFilter
except ImportError as e:
    # Fallback: import with Dict fix
    import importlib.util
    spec = importlib.util.spec_from_file_location("particle_filter", 
        f"{QUANT_MODULES_PATH}/particle_filter.py")
    pf_module = importlib.util.module_from_spec(spec)
    sys.modules['particle_filter'] = pf_module
    spec.loader.exec_module(pf_module)
    PredictionMarketParticleFilter = pf_module.PredictionMarketParticleFilter
    
    spec2 = importlib.util.spec_from_file_location("kelly_portfolio_optimizer",
        f"{QUANT_MODULES_PATH}/kelly_portfolio_optimizer.py")
    ko_module = importlib.util.module_from_spec(spec2)
    sys.modules['kelly_portfolio_optimizer'] = ko_module
    spec2.loader.exec_module(ko_module)
    KellyPortfolioOptimizer = ko_module.KellyPortfolioOptimizer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('KellyBankr')


@dataclass
class Signal:
    """Trading signal with Kelly-derived position sizing."""
    asset: str
    direction: str  # 'long' or 'short'
    market_price: float
    estimated_prob: float
    filtered_prob: float
    confidence: float  # confidence interval width
    kelly_fraction: float
    position_size: float
    leverage: float
    var_95: float
    expected_edge: float
    timestamp: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass 
class ExecutionResult:
    """Result of Bankr execution."""
    success: bool
    job_id: Optional[str]
    status: str
    response: Optional[str]
    error: Optional[str]
    bankroll_before: Optional[float]
    bankroll_after: Optional[float]
    execution_time_ms: int
    timestamp: str
    raw_data: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PositionRecord:
    """Record of a Kelly prediction → Bankr execution → outcome."""
    signal: Dict
    execution: Dict
    predicted_pnl: Optional[float] = None
    actual_pnl: Optional[float] = None
    outcome_timestamp: Optional[str] = None
    kelly_accuracy: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'signal': self.signal,
            'execution': self.execution,
            'predicted_pnl': self.predicted_pnl,
            'actual_pnl': self.actual_pnl,
            'outcome_timestamp': self.outcome_timestamp,
            'kelly_accuracy': self.kelly_accuracy
        }


class ExecutionLogger:
    """
    Logs every Kelly prediction → Bankr execution → outcome.
    Tracks: predicted vs actual, PnL, Kelly accuracy.
    Saves to JSONL for backtesting analysis.
    """
    
    def __init__(self, log_file: str = "/a0/usr/workdir/kelly_execution_log.jsonl"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.records: List[PositionRecord] = []
        
        # Load existing records
        self._load_existing()
        
    def _load_existing(self):
        """Load existing log entries."""
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            record = PositionRecord(**data)
                            self.records.append(record)
                        except Exception as e:
                            logger.warning(f"Failed to load log entry: {e}")
            logger.info(f"Loaded {len(self.records)} existing execution records")
    
    def log(self, signal: Signal, result: ExecutionResult) -> PositionRecord:
        """
        Log a signal and its execution result.
        
        Args:
            signal: The trading signal
            result: The execution result from Bankr
            
        Returns:
            PositionRecord that was logged
        """
        record = PositionRecord(
            signal=signal.to_dict(),
            execution=result.to_dict(),
            predicted_pnl=None,  # To be updated after position closes
            actual_pnl=None,
            outcome_timestamp=None,
            kelly_accuracy=None
        )
        
        self.records.append(record)
        
        # Append to JSONL
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(record.to_dict()) + '\n')
        
        logger.info(f"Logged execution: {signal.asset} {signal.direction} → {result.status}")
        return record
    
    def update_outcome(self, record_idx: int, actual_pnl: float):
        """
        Update a record with actual outcome PnL.
        
        Args:
            record_idx: Index in records list
            actual_pnl: Actual profit/loss realized
        """
        if 0 <= record_idx < len(self.records):
            record = self.records[record_idx]
            record.actual_pnl = actual_pnl
            record.outcome_timestamp = datetime.now(timezone.utc).isoformat()
            
            # Calculate Kelly accuracy
            if record.predicted_pnl is not None and record.predicted_pnl != 0:
                record.kelly_accuracy = actual_pnl / record.predicted_pnl
            
            # Rewrite log file
            with open(self.log_file, 'w') as f:
                for r in self.records:
                    f.write(json.dumps(r.to_dict()) + '\n')
            
            logger.info(f"Updated outcome for record {record_idx}: PnL={actual_pnl:.4f}")
    
    def get_stats(self) -> Dict:
        """Get execution statistics."""
        if not self.records:
            return {"total_trades": 0}
        
        successful = sum(1 for r in self.records if r.execution.get('success', False))
        with_outcomes = [r for r in self.records if r.actual_pnl is not None]
        
        stats = {
            "total_trades": len(self.records),
            "successful_executions": successful,
            "success_rate": successful / len(self.records),
            "trades_with_outcomes": len(with_outcomes)
        }
        
        if with_outcomes:
            total_pnl = sum(r.actual_pnl for r in with_outcomes)
            avg_pnl = total_pnl / len(with_outcomes)
            winning_trades = sum(1 for r in with_outcomes if r.actual_pnl and r.actual_pnl > 0)
            
            stats.update({
                "total_pnl": total_pnl,
                "avg_pnl_per_trade": avg_pnl,
                "win_rate": winning_trades / len(with_outcomes),
                "winning_trades": winning_trades,
                "losing_trades": len(with_outcomes) - winning_trades
            })
        
        return stats


class PositionValidator:
    """
    Pre-flight checks before execution.
    Verifies: sufficient margin, liquidation price safety, VaR limits.
    """
    
    def __init__(self, 
                 min_margin_pct: float = 0.10,
                 max_liquidation_proximity: float = 0.15,
                 max_leverage_crypto: float = 50.0,
                 max_leverage_fx: float = 100.0):
        """
        Args:
            min_margin_pct: Minimum margin as % of position (10%)
            max_liquidation_proximity: Max distance to liquidation (15%)
            max_leverage_crypto: Max leverage for crypto (50x)
            max_leverage_fx: Max leverage for forex/commodities (100x)
        """
        self.min_margin_pct = min_margin_pct
        self.max_liquidation_proximity = max_liquidation_proximity
        self.max_leverage_crypto = max_leverage_crypto
        self.max_leverage_fx = max_leverage_fx
    
    def calculate_liquidation_price(self, 
                                     entry_price: float, 
                                     leverage: float,
                                     direction: str) -> float:
        """
        Calculate liquidation price for a leveraged position.
        
        Long: Liquidation = Entry × (1 - 1/leverage)
        Short: Liquidation = Entry × (1 + 1/leverage)
        """
        if direction == 'long':
            return entry_price * (1 - 1/leverage)
        else:  # short
            return entry_price * (1 + 1/leverage)
    
    def check(self, signal: Signal, bankroll: float = 10000.0) -> Tuple[bool, str]:
        """
        Validate a signal before execution.
        
        Args:
            signal: The trading signal to validate
            bankroll: Current bankroll for margin calculation
            
        Returns:
            (is_valid, reason) tuple
        """
        checks = []
        
        # 1. Check leverage limits
        is_crypto = signal.asset in ['BTC', 'ETH', 'SOL', 'ARB', 'AVAX', 'BNB', 'DOGE', 'LINK', 'OP', 'MATIC']
        max_lev = self.max_leverage_crypto if is_crypto else self.max_leverage_fx
        
        if signal.leverage > max_lev:
            checks.append(f"FAIL: Leverage {signal.leverage}x exceeds max {max_lev}x for {signal.asset}")
        else:
            checks.append(f"PASS: Leverage {signal.leverage}x within limits")
        
        # 2. Check margin requirement
        position_value = bankroll * signal.position_size
        required_margin = position_value / signal.leverage
        available_margin = bankroll * self.min_margin_pct
        
        if required_margin > available_margin:
            checks.append(f"FAIL: Required margin ${required_margin:.2f} > available ${available_margin:.2f}")
        else:
            checks.append(f"PASS: Margin sufficient (${required_margin:.2f} required)")
        
        # 3. Check liquidation safety
        liq_price = self.calculate_liquidation_price(
            signal.market_price, signal.leverage, signal.direction
        )
        
        if signal.direction == 'long':
            distance = (signal.market_price - liq_price) / signal.market_price
        else:
            distance = (liq_price - signal.market_price) / signal.market_price
        
        if distance < self.max_liquidation_proximity:
            checks.append(f"WARN: Close to liquidation ({distance*100:.1f}% buffer)")
        else:
            checks.append(f"PASS: Liquidation buffer {distance*100:.1f}%")
        
        # 4. Check VaR limit
        if signal.var_95 > 0.05:  # 5% VaR limit
            checks.append(f"FAIL: VaR {signal.var_95*100:.2f}% exceeds 5% limit")
        else:
            checks.append(f"PASS: VaR {signal.var_95*100:.2f}% within limit")
        
        # 5. Check edge positivity
        if signal.expected_edge <= 0:
            checks.append(f"FAIL: Expected edge {signal.expected_edge:.4f} not positive")
        else:
            checks.append(f"PASS: Expected edge {signal.expected_edge:.4f}")
        
        # Determine overall validity
        failures = [c for c in checks if c.startswith('FAIL')]
        is_valid = len(failures) == 0
        
        reason = "; ".join(checks)
        
        if is_valid:
            logger.info(f"✓ Position validated: {signal.asset} {signal.direction}")
        else:
            logger.warning(f"✗ Position rejected: {'; '.join(failures)}")
        
        return is_valid, reason


class SignalPipeline:
    """
    Ingests price feeds, runs particle filter for signal extraction,
    feeds filtered signals to Kelly optimizer.
    """
    
    def __init__(self,
                 kelly_fraction: float = 0.25,
                 max_position: float = 0.10,
                 max_drawdown: float = 0.20,
                 var_limit: float = 0.05,
                 n_particles: int = 5000,
                 process_vol: float = 0.03,
                 obs_noise: float = 0.05):
        """
        Args:
            kelly_fraction: Fraction of full Kelly (0.25 = quarter Kelly)
            max_position: Hard cap per position (10%)
            max_drawdown: Circuit breaker level (20%)
            var_limit: 95% VaR limit (5%)
            n_particles: Number of particles for filter
            process_vol: Daily volatility in logit space
            obs_noise: Observation noise in probability space
        """
        self.kelly = KellyPortfolioOptimizer(
            kelly_fraction=kelly_fraction,
            max_position=max_position,
            max_drawdown=max_drawdown,
            var_limit=var_limit
        )
        self.n_particles = n_particles
        self.process_vol = process_vol
        self.obs_noise = obs_noise
        
        # Track filters per asset
        self.filters: Dict[str, PredictionMarketParticleFilter] = {}
        
    def get_or_create_filter(self, asset: str) -> PredictionMarketParticleFilter:
        """Get or create particle filter for asset."""
        if asset not in self.filters:
            self.filters[asset] = PredictionMarketParticleFilter(
                n_particles=self.n_particles,
                process_vol=self.process_vol,
                obs_noise=self.obs_noise,
                prior_prob=0.50
            )
        return self.filters[asset]
    
    def process(self, 
                asset: str,
                market_price: float,
                estimated_prob: float,
                direction: str = 'long') -> Signal:
        """
        Process market data through pipeline.
        
        Args:
            asset: Asset symbol (ETH, BTC, etc.)
            market_price: Current market price (0-1 for prediction markets, or actual price)
            estimated_prob: Raw probability estimate
            direction: 'long' or 'short'
            
        Returns:
            Signal with Kelly-derived position sizing
        """
        # 1. Run particle filter
        pf = self.get_or_create_filter(asset)
        filtered_prob, ci_low, ci_high = pf.update(market_price)
        confidence = ci_high - ci_low
        
        # 2. Calculate Kelly fraction
        kelly_f = self.kelly.kelly_single(market_price, estimated_prob)
        
        # 3. Apply constraints
        position_size = kelly_f * self.kelly.kelly_fraction
        position_size = np.clip(position_size, -self.kelly.max_position, self.kelly.max_position)
        
        # 4. Calculate leverage (Kelly-derived)
        # Higher edge = higher leverage, capped at limits
        gross_edge, net_edge = self.kelly.calculate_edge(market_price, estimated_prob)
        leverage = min(abs(net_edge) * 100, 50)  # Scale edge to leverage, max 50x
        leverage = max(leverage, 1.0)  # Minimum 1x
        
        # 5. Calculate VaR (simplified)
        var_95 = abs(position_size) * confidence * 0.5  # Simplified VaR estimate
        
        # 6. Build signal
        signal = Signal(
            asset=asset,
            direction=direction,
            market_price=market_price,
            estimated_prob=estimated_prob,
            filtered_prob=filtered_prob,
            confidence=confidence,
            kelly_fraction=kelly_f,
            position_size=abs(position_size),
            leverage=leverage,
            var_95=var_95,
            expected_edge=net_edge,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        logger.info(f"Signal generated: {asset} {direction} | "
                   f"Edge: {net_edge:.4f} | Size: {abs(position_size):.4f} | "
                   f"Leverage: {leverage:.1f}x")
        
        return signal
    
    def process_portfolio(self,
                         assets: List[str],
                         market_prices: List[float],
                         estimated_probs: List[float],
                         cov_matrix: Optional[np.ndarray] = None) -> List[Signal]:
        """
        Process multiple assets with correlation-aware Kelly.
        
        Args:
            assets: List of asset symbols
            market_prices: List of market prices
            estimated_probs: List of probability estimates
            cov_matrix: Covariance matrix (optional, uncorrelated if None)
            
        Returns:
            List of Signals
        """
        n = len(assets)
        if cov_matrix is None:
            cov_matrix = np.eye(n) * 0.01  # Uncorrelated, 1% variance
        
        # Get Kelly portfolio weights
        result = self.kelly.optimize(market_prices, estimated_probs, cov_matrix)
        weights = result['weights']
        
        signals = []
        for i, asset in enumerate(assets):
            # Get filtered probability
            pf = self.get_or_create_filter(asset)
            filtered_prob, ci_low, ci_high = pf.update(market_prices[i])
            confidence = ci_high - ci_low
            
            gross_edge, net_edge = self.kelly.calculate_edge(market_prices[i], estimated_probs[i])
            leverage = min(abs(net_edge) * 100, 50)
            leverage = max(leverage, 1.0)
            
            signal = Signal(
                asset=asset,
                direction='long' if weights[i] > 0 else 'short',
                market_price=market_prices[i],
                estimated_prob=estimated_probs[i],
                filtered_prob=filtered_prob,
                confidence=confidence,
                kelly_fraction=weights[i],
                position_size=abs(weights[i]),
                leverage=leverage,
                var_95=result['var_95'] / n,
                expected_edge=net_edge,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            signals.append(signal)
        
        return signals


class KellyBankrAdapter:
    """
    Takes Kelly output and translates to Bankr API calls.
    Handles both spot trades and Avantis leverage positions.
    """
    
    def __init__(self, 
                 api_key: str,
                 base_url: str = "https://api.bankr.bot",
                 default_chain: str = "base",
                 poll_interval: int = 2,
                 max_poll_attempts: int = 150):
        """
        Args:
            api_key: Bankr API key (bk_...)
            base_url: Bankr API base URL
            default_chain: Default chain for trades
            poll_interval: Seconds between status polls
            max_poll_attempts: Max polling attempts (5 min default)
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.default_chain = default_chain
        self.poll_interval = poll_interval
        self.max_poll_attempts = max_poll_attempts
        
        self.session = requests.Session()
        self.session.headers.update({
            'X-API-Key': api_key,
            'Content-Type': 'application/json'
        })
        
        # Track bankroll
        self._cached_bankroll: Optional[float] = None
        self._bankroll_timestamp: Optional[datetime] = None
        
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make authenticated request to Bankr API."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            return {"success": False, "error": str(e), "status_code": e.response.status_code}
        except Exception as e:
            logger.error(f"Request error: {e}")
            return {"success": False, "error": str(e)}
    
    def _poll_job(self, job_id: str) -> Dict:
        """Poll job until completion."""
        for attempt in range(self.max_poll_attempts):
            result = self._make_request('GET', f'/agent/job/{job_id}')
            status = result.get('status')
            
            if status in ['completed', 'failed', 'cancelled']:
                return result
            
            logger.debug(f"Job {job_id} status: {status}, polling... ({attempt+1}/{self.max_poll_attempts})")
            time.sleep(self.poll_interval)
        
        logger.warning(f"Job {job_id} polling timeout")
        return {"success": False, "error": "Polling timeout", "status": "timeout"}
    
    def get_balances(self, chains: Optional[List[str]] = None) -> Dict:
        """
        Get wallet balances across chains.
        
        Args:
            chains: List of chains to query (base, ethereum, polygon, solana, unichain)
            
        Returns:
            Balance data dict
        """
        params = {}
        if chains:
            params['chains'] = ','.join(chains)
        
        result = self._make_request('GET', '/agent/balances', params=params)
        
        if result.get('success'):
            # Extract total bankroll approximation
            balances = result.get('balances', [])
            total_usd = sum(
                float(b.get('usdValue', 0)) for b in balances if 'usdValue' in b
            )
            self._cached_bankroll = total_usd
            self._bankroll_timestamp = datetime.now(timezone.utc)
            logger.info(f"Total bankroll: ${total_usd:.2f}")
        
        return result
    
    def get_bankroll(self, force_refresh: bool = False) -> float:
        """Get current bankroll in USD."""
        if force_refresh or self._cached_bankroll is None:
            self.get_balances()
        return self._cached_bankroll or 0.0
    
    def execute_position(self, 
                       signal: Signal,
                       collateral_usd: Optional[float] = None,
                       stop_loss: Optional[float] = None,
                       take_profit: Optional[float] = None,
                       dry_run: bool = False) -> ExecutionResult:
        """
        Execute a position via Bankr API.
        
        Args:
            signal: Kelly-derived signal
            collateral_usd: Amount to use as collateral (auto-calculated if None)
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
            dry_run: If True, only simulate (don't actually trade)
            
        Returns:
            ExecutionResult with job status
        """
        start_time = time.time()
        
        # Calculate collateral
        bankroll = self.get_bankroll()
        if collateral_usd is None:
            collateral_usd = bankroll * signal.position_size
        
        collateral_usd = min(collateral_usd, bankroll * 0.5)  # Max 50% bankroll
        
        # Build prompt for Bankr
        direction_word = "long" if signal.direction == 'long' else "short"
        
        prompt = f"Open a {signal.leverage:.0f}x {direction_word} on {signal.asset} with ${collateral_usd:.2f}"
        
        if stop_loss:
            prompt += f" with stop loss at ${stop_loss:.2f}"
        if take_profit:
            prompt += f" and take profit at ${take_profit:.2f}"
        
        prompt += f" on {self.default_chain}"
        
        logger.info(f"Executing: {prompt}")
        
        if dry_run:
            logger.info("DRY RUN - not executing")
            return ExecutionResult(
                success=True,
                job_id=None,
                status="dry_run",
                response=prompt,
                error=None,
                bankroll_before=bankroll,
                bankroll_after=bankroll,
                execution_time_ms=int((time.time() - start_time) * 1000),
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        
        # Submit to Bankr
        submit_result = self._make_request(
            'POST', 
            '/agent/prompt',
            json={"prompt": prompt}
        )
        
        if not submit_result.get('success'):
            error = submit_result.get('error', 'Unknown error')
            logger.error(f"Failed to submit: {error}")
            return ExecutionResult(
                success=False,
                job_id=None,
                status="failed",
                response=None,
                error=error,
                bankroll_before=bankroll,
                bankroll_after=bankroll,
                execution_time_ms=int((time.time() - start_time) * 1000),
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        
        job_id = submit_result.get('jobId')
        logger.info(f"Job submitted: {job_id}")
        
        # Poll for completion
        job_result = self._poll_job(job_id)
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        status = job_result.get('status')
        success = status == 'completed'
        
        # Update bankroll estimate
        bankroll_after = bankroll
        if success:
            # Refresh balance after successful trade
            time.sleep(1)  # Brief delay for chain confirmation
            bankroll_after = self.get_bankroll(force_refresh=True)
        
        result = ExecutionResult(
            success=success,
            job_id=job_id,
            status=status,
            response=job_result.get('response'),
            error=job_result.get('error'),
            bankroll_before=bankroll,
            bankroll_after=bankroll_after,
            execution_time_ms=elapsed_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            raw_data=job_result
        )
        
        logger.info(f"Execution complete: {status} in {elapsed_ms}ms")
        return result
    
    def close_position(self, 
                       asset: str,
                       position_id: Optional[str] = None,
                       percentage: float = 100.0) -> ExecutionResult:
        """
        Close an existing position.
        
        Args:
            asset: Asset symbol
            position_id: Specific position ID (optional)
            percentage: Percentage to close (100 = full close)
            
        Returns:
            ExecutionResult
        """
        start_time = time.time()
        bankroll = self.get_bankroll()
        
        if position_id:
            prompt = f"Close position {position_id}"
        elif percentage == 100.0:
            prompt = f"Close my {asset} position"
        else:
            prompt = f"Close {percentage:.0f}% of my {asset} position"
        
        logger.info(f"Closing: {prompt}")
        
        submit_result = self._make_request(
            'POST',
            '/agent/prompt',
            json={"prompt": prompt}
        )
        
        if not submit_result.get('success'):
            error = submit_result.get('error', 'Unknown error')
            return ExecutionResult(
                success=False,
                job_id=None,
                status="failed",
                response=None,
                error=error,
                bankroll_before=bankroll,
                bankroll_after=bankroll,
                execution_time_ms=int((time.time() - start_time) * 1000),
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        
        job_id = submit_result.get('jobId')
        job_result = self._poll_job(job_id)
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        status = job_result.get('status')
        success = status == 'completed'
        
        # Refresh bankroll
        bankroll_after = bankroll
        if success:
            time.sleep(1)
            bankroll_after = self.get_bankroll(force_refresh=True)
        
        return ExecutionResult(
            success=success,
            job_id=job_id,
            status=status,
            response=job_result.get('response'),
            error=job_result.get('error'),
            bankroll_before=bankroll,
            bankroll_after=bankroll_after,
            execution_time_ms=elapsed_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            raw_data=job_result
        )
    
    def get_open_positions(self) -> Dict:
        """Get all open Avantis positions."""
        result = self._make_request(
            'POST',
            '/agent/prompt',
            json={"prompt": "Show my Avantis positions"}
        )
        
        if result.get('success'):
            job_result = self._poll_job(result.get('jobId'))
            return job_result
        
        return result


class KellyBankrSystem:
    """
    Integrated system combining all components.
    """
    
    def __init__(self, api_key: str, **kwargs):
        """
        Args:
            api_key: Bankr API key
            **kwargs: Configuration for pipeline and adapter
        """
        self.adapter = KellyBankrAdapter(api_key, **kwargs.get('adapter', {}))
        self.pipeline = SignalPipeline(**kwargs.get('pipeline', {}))
        self.validator = PositionValidator(**kwargs.get('validator', {}))
        self.logger = ExecutionLogger(**kwargs.get('logger', {}))
        
        self.drawdown_tracker: List[float] = []
        self.max_drawdown_limit = kwargs.get('max_drawdown', 0.20)
    
    def check_circuit_breakers(self) -> Tuple[bool, str]:
        """Check if any circuit breakers are triggered."""
        bankroll = self.adapter.get_bankroll()
        
        if len(self.drawdown_tracker) > 1:
            peak = max(self.drawdown_tracker)
            current_drawdown = (peak - bankroll) / peak if peak > 0 else 0
            
            if current_drawdown > self.max_drawdown_limit:
                return False, f"CIRCUIT BREAKER: Drawdown {current_drawdown*100:.1f}% exceeds {self.max_drawdown_limit*100:.0f}%"
        
        self.drawdown_tracker.append(bankroll)
        return True, "OK"
    
    def execute_signal(self, 
                       signal: Signal,
                       dry_run: bool = True) -> Tuple[Optional[ExecutionResult], PositionRecord]:
        """
        Full execution pipeline: validate → execute → log.
        
        Args:
            signal: Kelly-derived signal
            dry_run: If True, don't actually trade
            
        Returns:
            (ExecutionResult, PositionRecord) or (None, record) if rejected
        """
        # Check circuit breakers
        can_trade, reason = self.check_circuit_breakers()
        if not can_trade:
            logger.error(reason)
            return None, self.logger.log(signal, ExecutionResult(
                success=False,
                job_id=None,
                status="rejected",
                response=None,
                error=reason,
                bankroll_before=self.adapter.get_bankroll(),
                bankroll_after=self.adapter.get_bankroll(),
                execution_time_ms=0,
                timestamp=datetime.now(timezone.utc).isoformat()
            ))
        
        # Validate
        is_valid, validation_reason = self.validator.check(signal, self.adapter.get_bankroll())
        if not is_valid:
            logger.warning(f"Signal rejected: {validation_reason}")
            return None, self.logger.log(signal, ExecutionResult(
                success=False,
                job_id=None,
                status="rejected",
                response=None,
                error=validation_reason,
                bankroll_before=self.adapter.get_bankroll(),
                bankroll_after=self.adapter.get_bankroll(),
                execution_time_ms=0,
                timestamp=datetime.now(timezone.utc).isoformat()
            ))
        
        # Execute
        result = self.adapter.execute_position(signal, dry_run=dry_run)
        
        # Log
        record = self.logger.log(signal, result)
        
        return result, record


def demo():
    """
    Demo script showing ETH leverage position execution.
    """
    print("=" * 60)
    print("Kelly-Bankr Quant Execution Bridge - DEMO")
    print("=" * 60)
    
    # Use test API key from context
    API_KEY = "bk_HB93LB7QVW9LPDNGYFJLF6MHW9F6WHR"
    
    # Initialize system
    print("\n1. Initializing Kelly-Bankr System...")
    system = KellyBankrSystem(
        api_key=API_KEY,
        pipeline={
            'kelly_fraction': 0.25,
            'max_position': 0.10,
            'max_drawdown': 0.20,
        },
        validator={
            'min_margin_pct': 0.10,
            'max_liquidation_proximity': 0.15
        },
        adapter={
            'default_chain': 'base'
        }
    )
    
    # Get balances
    print("\n2. Checking Bankr balances...")
    try:
        balances = system.adapter.get_balances()
        if balances.get('success'):
            print(f"   ✓ Connected to Bankr API")
            print(f"   Total Bankroll: ${system.adapter.get_bankroll():.2f}")
        else:
            print(f"   ⚠ Balance check: {balances.get('error', 'Unknown')}")
    except Exception as e:
        print(f"   ⚠ API connection test: {e}")
    
    # Generate signal
    print("\n3. Generating Kelly signal for ETH...")
    # Simulate ETH market at $3000, we estimate 65% probability of rise
    signal = system.pipeline.process(
        asset="ETH",
        market_price=0.62,  # Normalized price (0-1 scale for Kelly calc)
        estimated_prob=0.70,  # Our estimate
        direction="long"
    )
    
    print(f"   Asset: {signal.asset}")
    print(f"   Direction: {signal.direction}")
    print(f"   Market Price: {signal.market_price}")
    print(f"   Filtered Prob: {signal.filtered_prob:.3f}")
    print(f"   Kelly Fraction: {signal.kelly_fraction:.4f}")
    print(f"   Position Size: {signal.position_size*100:.2f}% of bankroll")
    print(f"   Leverage: {signal.leverage:.1f}x")
    print(f"   Expected Edge: {signal.expected_edge:.4f}")
    print(f"   VaR 95%: {signal.var_95*100:.2f}%")
    
    # Validate
    print("\n4. Validating position...")
    is_valid, reason = system.validator.check(signal, bankroll=10000.0)
    print(f"   Valid: {is_valid}")
    print(f"   Details: {reason}")
    
    # Execute (dry run)
    print("\n5. Executing position (DRY RUN)...")
    result, record = system.execute_signal(signal, dry_run=True)
    
    if result:
        print(f"   Status: {result.status}")
        print(f"   Bankroll Before: ${result.bankroll_before:.2f}")
        print(f"   Bankroll After: ${result.bankroll_after:.2f}")
        print(f"   Execution Time: {result.execution_time_ms}ms")
    
    # Log stats
    print("\n6. Execution log statistics:")
    stats = system.logger.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    # Test portfolio optimization
    print("\n7. Testing portfolio optimization...")
    signals = system.pipeline.process_portfolio(
        assets=["ETH", "BTC", "SOL"],
        market_prices=[0.62, 0.55, 0.48],
        estimated_probs=[0.70, 0.65, 0.60]
    )
    
    print(f"   Generated {len(signals)} portfolio signals:")
    for sig in signals:
        print(f"   - {sig.asset}: {sig.direction} {sig.position_size*100:.2f}% @ {sig.leverage:.1f}x")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nTo execute live trades:")
    print("  1. Set dry_run=False in execute_signal()")
    print("  2. Ensure sufficient funds in Bankr wallet")
    print("  3. Start with small position sizes for testing")
    
    return system, signal, result


def run_tests():
    """Run unit tests for the module."""
    print("Running Kelly-Bankr adapter tests...")
    
    # Test 1: Signal pipeline
    print("\n[Test 1] Signal Pipeline...")
    pipeline = SignalPipeline(kelly_fraction=0.25, max_position=0.10)
    signal = pipeline.process("ETH", 0.62, 0.70, "long")
    assert signal.asset == "ETH"
    assert signal.direction == "long"
    assert 0 < signal.position_size <= 0.10
    print("   ✓ Signal generation works")
    
    # Test 2: Position validation
    print("\n[Test 2] Position Validator...")
    validator = PositionValidator()
    is_valid, reason = validator.check(signal, bankroll=10000.0)
    assert isinstance(is_valid, bool)
    assert isinstance(reason, str)
    print(f"   ✓ Validation works: {is_valid}")
    
    # Test 3: Liquidation calculation
    print("\n[Test 3] Liquidation Price...")
    liq_long = validator.calculate_liquidation_price(3000, 5, "long")
    liq_short = validator.calculate_liquidation_price(3000, 5, "short")
    assert liq_long < 3000  # Long liquidation below entry
    assert liq_short > 3000  # Short liquidation above entry
    print(f"   ✓ Long liq: ${liq_long:.2f}, Short liq: ${liq_short:.2f}")
    
    # Test 4: Execution logger
    print("\n[Test 4] Execution Logger...")
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        log_path = f.name
    
    exec_logger = ExecutionLogger(log_file=log_path)
    
    test_signal = Signal(
        asset="ETH", direction="long", market_price=0.62,
        estimated_prob=0.70, filtered_prob=0.65, confidence=0.10,
        kelly_fraction=0.15, position_size=0.05, leverage=5.0,
        var_95=0.03, expected_edge=0.08,
        timestamp=datetime.now(timezone.utc).isoformat()
    )
    
    test_result = ExecutionResult(
        success=True, job_id="job_test_123", status="completed",
        response="Test execution", error=None,
        bankroll_before=10000.0, bankroll_after=10200.0,
        execution_time_ms=1500,
        timestamp=datetime.now(timezone.utc).isoformat()
    )
    
    record = exec_logger.log(test_signal, test_result)
    assert len(exec_logger.records) == 1
    
    stats = exec_logger.get_stats()
    assert stats['total_trades'] == 1
    print("   ✓ Logging works")
    
    # Cleanup
    os.unlink(log_path)
    
    # Test 5: Kelly adapter initialization
    print("\n[Test 5] Bankr Adapter...")
    adapter = KellyBankrAdapter(api_key="bk_test_key")
    assert adapter.api_key == "bk_test_key"
    assert adapter.base_url == "https://api.bankr.bot"
    print("   ✓ Adapter initialization works")
    
    # Test 6: Portfolio optimization
    print("\n[Test 6] Portfolio Optimization...")
    signals = pipeline.process_portfolio(
        ["ETH", "BTC"],
        [0.62, 0.55],
        [0.70, 0.65]
    )
    assert len(signals) == 2
    print("   ✓ Portfolio optimization works")
    
    print("\n" + "=" * 40)
    print("ALL TESTS PASSED ✓")
    print("=" * 40)
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Kelly-Bankr Quant Execution Bridge")
    parser.add_argument("--test", action="store_true", help="Run unit tests")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    
    args = parser.parse_args()
    
    if args.test:
        run_tests()
    elif args.demo or len(sys.argv) == 1:
        demo()
    else:
        parser.print_help()
