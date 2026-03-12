"""
APEX Signal Engine
==================
20-factor signal generation combining advanced market microstructure analysis.

Signals (17 APEX + 3 existing):
1. CVD_DIVERGENCE - Price HH but CVD LH (buying exhaustion)
2. CVD_MOMENTUM - CVD slope matches price slope
3. LIQUIDATION_CLUSTER - Real liq data cluster within 2%
4. VOLUME_ABSORPTION - High vol + tight range at liq wall
5. FUNDING_ARB - Cross-exchange funding rate diff
6. LIQUIDITY_IMBALANCE - Bid/ask ratio (AND logic both venues)
7. PRICE_DISCREPANCY - Cross-exchange price diff
8. FUNDING_PERSISTENT - Funding sustained same direction
9. CROWD_EXTREME - L/S ratio contrarian
10. TAKER_FLOW - Recent vs avg taker ratio
11. ORDER_FLOW_IMBALANCE - Combined OFI both exchanges
12. GAMMA_WALL - Options gamma wall proximity
13. KALMAN_TREND - Kalman filter velocity
14. KALMAN_DEVIATION - Price deviation from fair value
15. LIQ_VEL_EXHAUSTION - Cascade exhaustion for re-entry
16. LIQ_VEL_ACCEL - Cascade acceleration (inhibitor)
17. WHALE_DELTA - Large trade vs retail delta
18. PARTICLE_FILTER - Existing (logit-space SMC)
19. RSI - Existing (mean reversion)
20. MACD - Existing (momentum)

Reference: APEX Omni DEX Trading System v1.2
"""

import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import deque
import logging
from scipy import stats

logger = logging.getLogger('APEXSignalEngine')

# Add paths
sys.path.insert(0, '/a0/usr/workdir/ai-agent-blockchain-skills/skills/professional_quant_trader/modules')
sys.path.insert(0, '/a0/usr/workdir/ai-agent-blockchain-skills/skills/market_analyzer_trader/modules')

from particle_filter import PredictionMarketParticleFilter


@dataclass
class APEXSignal:
    """Complete signal with all 20 components."""
    asset: str
    timestamp: datetime

    # 17 APEX Signals
    cvd_divergence: float = 0.0  # Range: -1 to +1
    cvd_momentum: float = 0.0
    liquidation_cluster: float = 0.0
    volume_absorption: float = 0.0
    funding_arb: float = 0.0
    liquidity_imbalance: float = 0.0
    price_discrepancy: float = 0.0
    funding_persistent: float = 0.0
    crowd_extreme: float = 0.0
    taker_flow: float = 0.0
    order_flow_imbalance: float = 0.0
    gamma_wall: float = 0.0
    kalman_trend: float = 0.0
    kalman_deviation: float = 0.0
    liq_vel_exhaustion: float = 0.0
    liq_vel_accel: float = 0.0
    whale_delta: float = 0.0

    # 3 Existing Signals
    particle_filter_signal: float = 0.0
    rsi: float = 50.0
    macd: float = 0.0

    # Composite
    combined_score: float = 0.0
    confidence: float = 0.0
    direction: str = 'hold'

    def to_dict(self) -> Dict:
        return {
            'asset': self.asset,
            'timestamp': self.timestamp.isoformat(),
            'cvd_divergence': self.cvd_divergence,
            'cvd_momentum': self.cvd_momentum,
            'liquidation_cluster': self.liquidation_cluster,
            'volume_absorption': self.volume_absorption,
            'funding_arb': self.funding_arb,
            'liquidity_imbalance': self.liquidity_imbalance,
            'price_discrepancy': self.price_discrepancy,
            'funding_persistent': self.funding_persistent,
            'crowd_extreme': self.crowd_extreme,
            'taker_flow': self.taker_flow,
            'order_flow_imbalance': self.order_flow_imbalance,
            'gamma_wall': self.gamma_wall,
            'kalman_trend': self.kalman_trend,
            'kalman_deviation': self.kalman_deviation,
            'liq_vel_exhaustion': self.liq_vel_exhaustion,
            'liq_vel_accel': self.liq_vel_accel,
            'whale_delta': self.whale_delta,
            'particle_filter': self.particle_filter_signal,
            'rsi': self.rsi,
            'macd': self.macd,
            'combined_score': self.combined_score,
            'confidence': self.confidence,
            'direction': self.direction
        }


class KalmanFilter:
    """Kalman filter for trend detection and fair value estimation."""

    def __init__(self, process_variance=1e-5, measurement_variance=1e-3):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0

    def update(self, measurement: float) -> Tuple[float, float]:
        """Update filter with new measurement."""
        # Prediction
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance

        # Update
        blending_factor = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

        # Velocity = change in estimate
        velocity = self.posteri_estimate - priori_estimate

        return self.posteri_estimate, velocity


class CVDAnalyzer:
    """Cumulative Volume Delta analyzer."""

    def __init__(self, window: int = 20):
        self.window = window
        self.cvd_history = deque(maxlen=window)
        self.price_history = deque(maxlen=window)

    def update(self, price: float, delta_volume: float):
        """Update CVD with new data."""
        if len(self.cvd_history) == 0:
            self.cvd_history.append(delta_volume)
        else:
            self.cvd_history.append(self.cvd_history[-1] + delta_volume)
        self.price_history.append(price)

    def calculate_divergence(self, span: int = 5) -> float:
        """
        Detect CVD divergence (buying/selling exhaustion).

        Price makes higher high but CVD makes lower high = bearish divergence
        Returns: -1 to +1 (negative = bearish, positive = bullish)
        """
        if len(self.cvd_history) < span * 2:
            return 0.0

        # Get recent windows
        prices = list(self.price_history)[-span*2:]
        cvd = list(self.cvd_history)[-span*2:]

        # First half vs second half
        price_first = np.mean(prices[:span])
        price_second = np.mean(prices[span:])
        cvd_first = np.mean(cvd[:span])
        cvd_second = np.mean(cvd[span:])

        price_trend = np.sign(price_second - price_first)
        cvd_trend = np.sign(cvd_second - cvd_first)

        # Divergence: price up, cvd down = bearish (-1)
        # Divergence: price down, cvd up = bullish (+1)
        if price_trend > 0 and cvd_trend < 0:
            return -1.0
        elif price_trend < 0 and cvd_trend > 0:
            return 1.0
        else:
            return 0.0

    def calculate_momentum(self) -> float:
        """Calculate CVD momentum alignment."""
        if len(self.cvd_history) < self.window:
            return 0.0

        cvd_slope = np.polyfit(range(self.window), list(self.cvd_history)[-self.window:], 1)[0]
        price_slope = np.polyfit(range(self.window), list(self.price_history)[-self.window:], 1)[0]

        # Alignment: same direction = positive signal
        if cvd_slope * price_slope > 0:
            return np.sign(cvd_slope) * min(abs(cvd_slope / price_slope), 1.0)
        return 0.0


class APEXSignalEngine:
    """
    Advanced 20-factor signal generation engine.

    Combines APEX Omni DEX signals with existing quant signals.
    """

    def __init__(self,
                 pf_particles: int = 5000,
                 pf_process_vol: float = 0.03,
                 rsi_period: int = 14,
                 rsi_overbought: float = 70,
                 rsi_oversold: float = 30,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 cvd_span: int = 5,
                 liq_cluster_threshold: float = 100000,  # $100K
                 liq_vel_exhaustion_threshold: float = 0.25,
                 liq_vel_accel_threshold: float = 0.80,
                 whale_threshold: float = 1000000,  # $1M
                 funding_threshold: float = 0.10,  # 10% annualized
                 spread_threshold: float = 0.003,  # 0.3%
                 gamma_proximity: float = 0.01):  # 1%

        self.pf_params = {'n_particles': pf_particles, 'process_vol': pf_process_vol}
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.cvd_span = cvd_span
        self.liq_cluster_threshold = liq_cluster_threshold
        self.liq_vel_exhaustion_threshold = liq_vel_exhaustion_threshold
        self.liq_vel_accel_threshold = liq_vel_accel_threshold
        self.whale_threshold = whale_threshold
        self.funding_threshold = funding_threshold
        self.spread_threshold = spread_threshold
        self.gamma_proximity = gamma_proximity

        # Track state
        self.filters: Dict[str, PredictionMarketParticleFilter] = {}
        self.cvd_analyzers: Dict[str, CVDAnalyzer] = {}
        self.kalman_filters: Dict[str, KalmanFilter] = {}
        self.liq_velocity_history: Dict[str, deque] = {}
        self.funding_history: Dict[str, deque] = {}

    def _get_or_create_filter(self, asset: str) -> PredictionMarketParticleFilter:
        """Get or create particle filter for asset."""
        if asset not in self.filters:
            self.filters[asset] = PredictionMarketParticleFilter(
                n_particles=self.pf_params['n_particles'],
                process_vol=self.pf_params['process_vol'],
                prior_prob=0.50
            )
        return self.filters[asset]

    def _get_or_create_cvd(self, asset: str) -> CVDAnalyzer:
        """Get or create CVD analyzer."""
        if asset not in self.cvd_analyzers:
            self.cvd_analyzers[asset] = CVDAnalyzer(window=20)
        return self.cvd_analyzers[asset]

    def _get_or_create_kalman(self, asset: str) -> KalmanFilter:
        """Get or create Kalman filter."""
        if asset not in self.kalman_filters:
            self.kalman_filters[asset] = KalmanFilter()
        return self.kalman_filters[asset]

    # ============== SIGNAL CALCULATIONS ==============

    def _calc_cvd_divergence(self, asset: str, price: float, delta_vol: float) -> float:
        """Signal 1: CVD Divergence."""
        cvd = self._get_or_create_cvd(asset)
        cvd.update(price, delta_vol)
        return cvd.calculate_divergence(span=self.cvd_span)

    def _calc_cvd_momentum(self, asset: str, price: float, delta_vol: float) -> float:
        """Signal 2: CVD Momentum Alignment."""
        cvd = self._get_or_create_cvd(asset)
        return cvd.calculate_momentum()

    def _calc_liquidation_cluster(self, liq_data: List[Dict]) -> float:
        """Signal 3: Liquidation Cluster."""
        if not liq_data:
            return 0.0

        # Check for cluster within 2% price range
        clusters = []
        for liq in liq_data:
            if liq.get('notional', 0) >= self.liq_cluster_threshold:
                clusters.append(liq)

        if len(clusters) >= 3:  # At least 3 large liquidations
            total_notional = sum(c.get('notional', 0) for c in clusters)
            if total_notional >= self.liq_cluster_threshold * 3:
                # Longs liquidated = bullish (shorts got squeezed)
                # Shorts liquidated = bearish (longs got squeezed)
                long_liq = sum(c.get('notional', 0) for c in clusters if c.get('side') == 'long')
                short_liq = sum(c.get('notional', 0) for c in clusters if c.get('side') == 'short')
                return 1.0 if long_liq > short_liq else -1.0
        return 0.0

    def _calc_volume_absorption(self, 
                                volume: float, 
                                price_range: float,
                                avg_volume: float,
                                liq_wall_proximity: float) -> float:
        """Signal 4: Volume Absorption at Liquidity Wall."""
        vol_ratio = volume / avg_volume if avg_volume > 0 else 0
        range_pct = price_range / 100 if price_range > 1 else price_range

        # High volume + tight range at liq wall
        if vol_ratio > 1.5 and range_pct < 0.005 and liq_wall_proximity < 0.02:
            return 1.0  # Buying absorbed at wall
        return 0.0

    def _calc_funding_arb(self, 
                         funding_rates: Dict[str, float]) -> float:
        """Signal 5: Cross-Exchange Funding Arbitrage."""
        if len(funding_rates) < 2:
            return 0.0

        rates = list(funding_rates.values())
        max_rate = max(rates)
        min_rate = min(rates)
        diff = max_rate - min_rate

        # Diff > 10% annualized
        if diff > self.funding_threshold:
            # Short highest funding, long lowest
            return 1.0 if max_rate > 0 else -1.0
        return 0.0

    def _calc_liquidity_imbalance(self,
                                 hyperliquid_ratio: float,
                                 apex_ratio: float) -> float:
        """Signal 6: Liquidity Imbalance (AND logic both venues)."""
        # HL > 2.5 AND ApeX > 1.8
        if hyperliquid_ratio > 2.5 and apex_ratio > 1.8:
            return 1.0 if hyperliquid_ratio > 1 else -1.0
        return 0.0

    def _calc_price_discrepancy(self,
                                prices: Dict[str, float]) -> float:
        """Signal 7: Cross-Exchange Price Discrepancy."""
        if len(prices) < 2:
            return 0.0

        price_list = list(prices.values())
        max_price = max(price_list)
        min_price = min(price_list)
        diff_pct = (max_price - min_price) / min_price

        # Diff > 0.1%
        if diff_pct > 0.001:
            return 1.0  # Arbitrage opportunity
        return 0.0

    def _calc_funding_persistent(self,
                               asset: str,
                               current_funding: float) -> float:
        """Signal 8: Persistent Funding Direction."""
        if asset not in self.funding_history:
            self.funding_history[asset] = deque(maxlen=10)

        self.funding_history[asset].append(current_funding)

        if len(self.funding_history[asset]) < 3:
            return 0.0

        # Check if same direction for 3+ consecutive
        history = list(self.funding_history[asset])[-3:]
        if all(f > 0 for f in history):
            return -1.0  # Positive funding = short expensive = bearish
        elif all(f < 0 for f in history):
            return 1.0   # Negative funding = long expensive = bullish
        return 0.0

    def _calc_crowd_extreme(self, long_short_ratio: float) -> float:
        """Signal 9: Crowd Extreme (Contrarian)."""
        # L/S > 1.8 = too many longs = SHORT
        # L/S < 0.55 = too many shorts = LONG
        if long_short_ratio > 1.8:
            return -1.0  # Short
        elif long_short_ratio < 0.55:
            return 1.0   # Long
        return 0.0

    def _calc_taker_flow(self,
                       recent_taker_vol: float,
                       avg_taker_vol: float) -> float:
        """Signal 10: Taker Flow."""
        if avg_taker_vol == 0:
            return 0.0

        ratio = recent_taker_vol / avg_taker_vol
        if ratio > 1.1:
            return 1.0 if recent_taker_vol > 0 else -1.0
        return 0.0

    def _calc_order_flow_imbalance(self,
                                  hyperliquid_ofi: float,
                                  apex_ofi: float) -> float:
        """Signal 11: Order Flow Imbalance (Combined)."""
        combined_ofi = hyperliquid_ofi + apex_ofi
        # OFI > 20% combined
        if combined_ofi > 0.20:
            return 1.0
        elif combined_ofi < -0.20:
            return -1.0
        return 0.0

    def _calc_gamma_wall(self,
                       current_price: float,
                       strikes: List[float]) -> float:
        """Signal 12: Gamma Wall Proximity."""
        for strike in strikes:
            if abs(current_price - strike) / current_price < self.gamma_proximity:
                # Near gamma wall
                return 1.0 if current_price < strike else -1.0
        return 0.0

    def _calc_kalman_trend(self,
                         asset: str,
                         price: float,
                         velocity_threshold: float = 0.001) -> float:
        """Signal 13: Kalman Filter Trend Velocity."""
        kf = self._get_or_create_kalman(asset)
        _, velocity = kf.update(price)

        if abs(velocity) > velocity_threshold:
            return np.sign(velocity)
        return 0.0

    def _calc_kalman_deviation(self,
                              asset: str,
                              price: float) -> float:
        """Signal 14: Kalman Fair Value Deviation."""
        kf = self._get_or_create_kalman(asset)
        fair_value, _ = kf.update(price)

        deviation = (price - fair_value) / fair_value
        if deviation > 0.015:
            return -1.0  # Overvalued
        elif deviation < -0.015:
            return 1.0   # Undervalued
        return 0.0

    def _calc_liq_vel_exhaustion(self,
                              asset: str,
                              current_velocity: float) -> float:
        """Signal 15: Liquidation Velocity Exhaustion."""
        if asset not in self.liq_velocity_history:
            self.liq_velocity_history[asset] = deque(maxlen=50)

        self.liq_velocity_history[asset].append(abs(current_velocity))

        if len(self.liq_velocity_history[asset]) < 10:
            return 0.0

        peak = max(self.liq_velocity_history[asset])
        if current_velocity / peak < self.liq_vel_exhaustion_threshold:
            return 1.0  # Cascade exhausted, re-entry opportunity
        return 0.0

    def _calc_liq_vel_accel(self,
                          asset: str,
                          current_velocity: float) -> float:
        """Signal 16: Liquidation Velocity Acceleration (Inhibitor)."""
        if asset not in self.liq_velocity_history:
            return 0.0

        if len(self.liq_velocity_history[asset]) < 10:
            return 0.0

        peak = max(self.liq_velocity_history[asset])
        if current_velocity / peak > self.liq_vel_accel_threshold:
            return -1.0  # Cascade accelerating, avoid entry
        return 0.0

    def _calc_whale_delta(self,
                        whale_trades: List[Dict],
                        retail_trades: List[Dict]) -> float:
        """Signal 17: Whale vs Retail Delta."""
        whale_net = sum(t.get('volume', 0) * (1 if t.get('side') == 'buy' else -1) 
                       for t in whale_trades if t.get('notional', 0) >= self.whale_threshold)

        retail_net = sum(t.get('volume', 0) * (1 if t.get('side') == 'buy' else -1)
                        for t in retail_trades)

        if abs(whale_net) >= self.whale_threshold:
            return np.sign(whale_net)
        return 0.0

    def _calc_particle_filter(self,
                            asset: str,
                            market_price: float,
                            estimated_prob: float) -> Tuple[float, float]:
        """Signal 18: Particle Filter Signal."""
        pf = self._get_or_create_filter(asset)
        filtered_prob, ci_low, ci_high = pf.update(market_price)
        confidence = 1 - (ci_high - ci_low)

        # Signal = (filtered - market) * 4 to scale to -1 to 1
        signal = (filtered_prob - market_price) * 4
        signal = np.clip(signal, -1, 1)

        return signal, confidence

    def _calc_rsi(self, prices: pd.Series) -> float:
        """Signal 19: RSI."""
        if len(prices) < self.rsi_period:
            return 50.0

        deltas = prices.diff()
        gains = deltas.clip(lower=0)
        losses = -deltas.clip(upper=0)

        avg_gain = gains.rolling(window=self.rsi_period).mean().iloc[-1]
        avg_loss = losses.rolling(window=self.rsi_period).mean().iloc[-1]

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calc_macd(self, prices: pd.Series) -> float:
        """Signal 20: MACD Histogram."""
        if len(prices) < self.macd_slow:
            return 0.0

        ema_fast = prices.ewm(span=self.macd_fast).mean()
        ema_slow = prices.ewm(span=self.macd_slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line

        return histogram.iloc[-1]

    # ============== MAIN INTERFACE ==============

    def generate_signals(self,
                      asset: str,
                      df: pd.DataFrame,
                      market_price: float,
                      market_data: Dict = None) -> APEXSignal:
        """
        Generate all 20 signals.

        Args:
            asset: Asset symbol
            df: OHLCV DataFrame
            market_price: Current market price
            market_data: Optional dict with additional data
                       {funding_rates, liq_data, whale_trades, etc}
        """
        market_data = market_data or {}

        # Calculate delta volume (bid - ask)
        delta_vol = df.get('bid_volume', pd.Series([0])).iloc[-1] - df.get('ask_volume', pd.Series([0])).iloc[-1]

        # 1-2: CVD signals
        cvd_div = self._calc_cvd_divergence(asset, market_price, delta_vol)
        cvd_mom = self._calc_cvd_momentum(asset, market_price, delta_vol)

        # 3: Liquidation cluster
        liq_cluster = self._calc_liquidation_cluster(
            market_data.get('liq_data', [])
        )

        # 4: Volume absorption
        vol_abs = self._calc_volume_absorption(
            df['volume'].iloc[-1],
            (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1],
            df['volume'].mean(),
            market_data.get('liq_wall_proximity', 0.5)
        )

        # 5: Funding arb
        fund_arb = self._calc_funding_arb(
            market_data.get('funding_rates', {})
        )

        # 6: Liquidity imbalance
        liq_imb = self._calc_liquidity_imbalance(
            market_data.get('hyperliquid_ratio', 1.0),
            market_data.get('apex_ratio', 1.0)
        )

        # 7: Price discrepancy
        price_disc = self._calc_price_discrepancy(
            market_data.get('exchange_prices', {})
        )

        # 8: Funding persistent
        fund_persist = self._calc_funding_persistent(
            asset,
            market_data.get('current_funding', 0)
        )

        # 9: Crowd extreme
        crowd = self._calc_crowd_extreme(
            market_data.get('long_short_ratio', 1.0)
        )

        # 10: Taker flow
        taker = self._calc_taker_flow(
            market_data.get('recent_taker_vol', 0),
            market_data.get('avg_taker_vol', 1)
        )

        # 11: Order flow imbalance
        ofi = self._calc_order_flow_imbalance(
            market_data.get('hyperliquid_ofi', 0),
            market_data.get('apex_ofi', 0)
        )

        # 12: Gamma wall
        gamma = self._calc_gamma_wall(
            market_price,
            market_data.get('gamma_strikes', [])
        )

        # 13-14: Kalman
        kal_trend = self._calc_kalman_trend(asset, market_price)
        kal_dev = self._calc_kalman_deviation(asset, market_price)

        # 15-16: Liquidation velocity
        liq_vel = market_data.get('liq_velocity', 0)
        liq_exhaust = self._calc_liq_vel_exhaustion(asset, liq_vel)
        liq_accel = self._calc_liq_vel_accel(asset, liq_vel)

        # 17: Whale delta
        whale = self._calc_whale_delta(
            market_data.get('whale_trades', []),
            market_data.get('retail_trades', [])
        )

        # 18: Particle filter
        pf_signal, pf_conf = self._calc_particle_filter(asset, market_price, 0.60)

        # 19-20: RSI and MACD
        rsi = self._calc_rsi(df['close'])
        macd = self._calc_macd(df['close'])

        # Create signal object
        signal = APEXSignal(
            asset=asset,
            timestamp=datetime.now(),
            cvd_divergence=cvd_div,
            cvd_momentum=cvd_mom,
            liquidation_cluster=liq_cluster,
            volume_absorption=vol_abs,
            funding_arb=fund_arb,
            liquidity_imbalance=liq_imb,
            price_discrepancy=price_disc,
            funding_persistent=fund_persist,
            crowd_extreme=crowd,
            taker_flow=taker,
            order_flow_imbalance=ofi,
            gamma_wall=gamma,
            kalman_trend=kal_trend,
            kalman_deviation=kal_dev,
            liq_vel_exhaustion=liq_exhaust,
            liq_vel_accel=liq_accel,
            whale_delta=whale,
            particle_filter_signal=pf_signal,
            rsi=rsi,
            macd=macd
        )

        # Calculate composite score
        # Weight APEX signals 70%, existing 30%
        # SIGNAL WEIGHTS ADJUSTMENT (2026-03-08):
        # - cvd_divergence: DISABLED (25% accuracy, worse than random)
        # - crowd_extreme: REDUCED to 50% weight (44% accuracy, below random)
        # - liquidation_cluster: FULL weight (58% accuracy, marginal but positive)
        # - kalman_trend: FULL weight (documented 57-62% accuracy)
        apex_signals_weighted = [
            (cvd_div * 0.0, 0.0),           # DISABLED - toxic signal
            (cvd_mom, 1.0),
            (liq_cluster, 1.0),
            (vol_abs, 1.0),
            (fund_arb, 1.0),
            (liq_imb, 1.0),
            (price_disc, 1.0),
            (fund_persist, 1.0),
            (crowd * 0.5, 0.5),             # REDUCED - weak signal
            (taker, 1.0),
            (ofi, 1.0),
            (gamma, 1.0),
            (kal_trend, 1.0),
            (kal_dev, 1.0),
            (liq_exhaust, 1.0),
            (liq_accel, 1.0),
            (whale, 1.0)
        ]

        # Calculate weighted average
        weighted_sum = sum(signal * weight for signal, weight in apex_signals_weighted)
        total_weight = sum(weight for _, weight in apex_signals_weighted)
        apex_score = weighted_sum / total_weight if total_weight > 0 else 0

        existing_signals = [
            pf_signal,
            (100 - rsi) / 50 - 1,  # Convert to -1 to 1
            np.sign(macd) * min(abs(macd) / 0.1, 1)
        ]

        existing_score = np.mean(existing_signals) if existing_signals else 0

        signal.combined_score = 0.7 * apex_score + 0.3 * existing_score
        signal.combined_score = np.clip(signal.combined_score, -1, 1)
        signal.confidence = pf_conf

        # Determine direction
        if signal.combined_score > 0.3:
            signal.direction = 'buy'
        elif signal.combined_score < -0.3:
            signal.direction = 'sell'
        else:
            signal.direction = 'hold'

        logger.info(f"APEX Signal: {asset} | "
                   f"Score: {signal.combined_score:.3f} | "
                   f"Dir: {signal.direction.upper()} | "
                   f"Conf: {signal.confidence:.2f}")

        return signal

    def get_signal_breakdown(self, signal: APEXSignal) -> Dict:
        """Get detailed signal breakdown."""
        return {
            'apex_signals': {
                'CVD Divergence': {'value': signal.cvd_divergence, 'strength': 'STRONG' if abs(signal.cvd_divergence) > 0.7 else 'MODERATE' if abs(signal.cvd_divergence) > 0.3 else 'WEAK'},
                'CVD Momentum': {'value': signal.cvd_momentum, 'strength': 'STRONG' if abs(signal.cvd_momentum) > 0.7 else 'MODERATE' if abs(signal.cvd_momentum) > 0.3 else 'WEAK'},
                'Liquidation Cluster': {'value': signal.liquidation_cluster, 'strength': 'STRONG' if abs(signal.liquidation_cluster) > 0.7 else 'MODERATE' if abs(signal.liquidation_cluster) > 0.3 else 'WEAK'},
                'Volume Absorption': {'value': signal.volume_absorption, 'strength': 'STRONG' if abs(signal.volume_absorption) > 0.7 else 'MODERATE' if abs(signal.volume_absorption) > 0.3 else 'WEAK'},
                'Funding Arb': {'value': signal.funding_arb, 'strength': 'STRONG' if abs(signal.funding_arb) > 0.7 else 'MODERATE' if abs(signal.funding_arb) > 0.3 else 'WEAK'},
                'Liquidity Imbalance': {'value': signal.liquidity_imbalance, 'strength': 'STRONG' if abs(signal.liquidity_imbalance) > 0.7 else 'MODERATE' if abs(signal.liquidity_imbalance) > 0.3 else 'WEAK'},
                'Price Discrepancy': {'value': signal.price_discrepancy, 'strength': 'STRONG' if abs(signal.price_discrepancy) > 0.7 else 'MODERATE' if abs(signal.price_discrepancy) > 0.3 else 'WEAK'},
                'Funding Persistent': {'value': signal.funding_persistent, 'strength': 'STRONG' if abs(signal.funding_persistent) > 0.7 else 'MODERATE' if abs(signal.funding_persistent) > 0.3 else 'WEAK'},
                'Crowd Extreme': {'value': signal.crowd_extreme, 'strength': 'STRONG' if abs(signal.crowd_extreme) > 0.7 else 'MODERATE' if abs(signal.crowd_extreme) > 0.3 else 'WEAK'},
                'Taker Flow': {'value': signal.taker_flow, 'strength': 'STRONG' if abs(signal.taker_flow) > 0.7 else 'MODERATE' if abs(signal.taker_flow) > 0.3 else 'WEAK'},
                'Order Flow Imbalance': {'value': signal.order_flow_imbalance, 'strength': 'STRONG' if abs(signal.order_flow_imbalance) > 0.7 else 'MODERATE' if abs(signal.order_flow_imbalance) > 0.3 else 'WEAK'},
                'Gamma Wall': {'value': signal.gamma_wall, 'strength': 'STRONG' if abs(signal.gamma_wall) > 0.7 else 'MODERATE' if abs(signal.gamma_wall) > 0.3 else 'WEAK'},
                'Kalman Trend': {'value': signal.kalman_trend, 'strength': 'STRONG' if abs(signal.kalman_trend) > 0.7 else 'MODERATE' if abs(signal.kalman_trend) > 0.3 else 'WEAK'},
                'Kalman Deviation': {'value': signal.kalman_deviation, 'strength': 'STRONG' if abs(signal.kalman_deviation) > 0.7 else 'MODERATE' if abs(signal.kalman_deviation) > 0.3 else 'WEAK'},
                'Liq Vel Exhaustion': {'value': signal.liq_vel_exhaustion, 'strength': 'STRONG' if abs(signal.liq_vel_exhaustion) > 0.7 else 'MODERATE' if abs(signal.liq_vel_exhaustion) > 0.3 else 'WEAK'},
                'Liq Vel Accel': {'value': signal.liq_vel_accel, 'strength': 'STRONG' if abs(signal.liq_vel_accel) > 0.7 else 'MODERATE' if abs(signal.liq_vel_accel) > 0.3 else 'WEAK'},
                'Whale Delta': {'value': signal.whale_delta, 'strength': 'STRONG' if abs(signal.whale_delta) > 0.7 else 'MODERATE' if abs(signal.whale_delta) > 0.3 else 'WEAK'}
            },
            'existing_signals': {
                'Particle Filter': {'value': signal.particle_filter_signal, 'strength': 'STRONG' if abs(signal.particle_filter_signal) > 0.7 else 'MODERATE' if abs(signal.particle_filter_signal) > 0.3 else 'WEAK'},
                'RSI': {'value': signal.rsi, 'strength': 'STRONG' if signal.rsi < 30 or signal.rsi > 70 else 'MODERATE' if signal.rsi < 40 or signal.rsi > 60 else 'WEAK'},
                'MACD': {'value': signal.macd, 'strength': 'STRONG' if abs(signal.macd) > 0.1 else 'MODERATE' if abs(signal.macd) > 0.05 else 'WEAK'}
            },
            'combined': {
                'score': signal.combined_score,
                'direction': signal.direction,
                'confidence': signal.confidence
            }
        }


if __name__ == "__main__":
    # Demo
    print("APEX Signal Engine - 20 Factor Demo")
    print("=" * 60)

    engine = APEXSignalEngine()

    # Generate synthetic data
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=100, freq='1h')
    prices = 3000 + np.cumsum(np.random.randn(100) * 0.5)

    df = pd.DataFrame({
        'open': prices * (1 + np.random.randn(100) * 0.002),
        'high': prices * (1 + abs(np.random.randn(100) * 0.005)),
        'low': prices * (1 - abs(np.random.randn(100) * 0.005)),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100),
        'bid_volume': np.random.randint(500, 5000, 100),
        'ask_volume': np.random.randint(500, 5000, 100)
    }, index=dates)

    # Generate signals
    signal = engine.generate_signals("ETH", df, prices[-1])

    print(f"\nSignal Summary:")
    print(f"  Asset: {signal.asset}")
    print(f"  Direction: {signal.direction.upper()}")
    print(f"  Combined Score: {signal.combined_score:.3f}")
    print(f"  Confidence: {signal.confidence:.2f}")

    print(f"\nTop APEX Signals:")
    breakdown = engine.get_signal_breakdown(signal)
    apex_sorted = sorted(breakdown['apex_signals'].items(), 
                        key=lambda x: abs(x[1]['value']), reverse=True)
    for name, data in apex_sorted[:5]:
        print(f"  {name}: {data['value']:+.2f} ({data['strength']})")
