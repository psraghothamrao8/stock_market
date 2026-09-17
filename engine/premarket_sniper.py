"""
Pre-Market Call Auction (9:00-9:08 AM) Dislocation Sniping Engine

Identifies 3-sigma statistical opening mispricings between 9:08 and 9:14 AM,
validates microstructure order book imbalance, and generates precision limit
order routing for the 9:15:00 AM regular session open under strict Rs 3 Lakh
capital and risk constraints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import requests

from .config import (
    CAPITAL_CONFIG,
    PRE_MARKET_CONFIG,
    CapitalConfig,
    PreMarketConfig,
)
from .risk import RiskEngine, TradeSizingResult


@dataclass(frozen=True)
class AuctionQuote:
    """
    Standardized NSE Pre-Market Call Auction Clearing Quote snapshot at 9:08 AM.
    """
    symbol: str
    timestamp: datetime
    previous_close: float
    equilibrium_price: float      # Discovered open price / Indicative Equilibrium Price (IEP)
    matched_volume: int           # Total matched volume during call auction cross
    total_buy_qty: int            # Total cumulative buy orders remaining in auction book
    total_sell_qty: int           # Total cumulative sell orders remaining in auction book
    lower_circuit: float          # Lower daily price band limit
    upper_circuit: float          # Upper daily price band limit
    is_fno: bool = True           # Eligible for intraday two-way trading


@dataclass(frozen=True)
class HistoricalOvernightStats:
    """
    Historical overnight statistical profile for an asset against a benchmark index.
    Computed over a rolling lookback window (default: 60 trading days).
    """
    symbol: str
    beta: float                   # Sensitivity of stock overnight gap to index overnight gap
    alpha: float                  # Unexplained mean overnight drift
    residual_sigma: float         # Sample standard deviation of idiosyncratic gap residuals (sigma_epsilon)
    residual_mean: float          # Mean of idiosyncratic gap residuals
    adv_20: float                 # 20-day Average Daily Volume (shares)
    sample_size: int              # Number of historical trading days used in calculation


@dataclass
class DislocationSignal:
    """
    Evaluated signal for a pre-market call auction snapshot.
    """
    symbol: str
    timestamp: datetime
    direction: str                # "LONG" (undervalued gap) or "SHORT" (overvalued gap)
    previous_close: float
    equilibrium_price: float
    stock_gap_pct: float          # (P_eq - P_prev) / P_prev
    index_gap_pct: float          # (P_index_eq - P_index_prev) / P_index_prev
    beta: float
    expected_gap_pct: float       # beta * index_gap_pct
    residual_gap: float           # stock_gap_pct - expected_gap_pct
    residual_sigma: float         # Historical standard deviation of residual gap
    z_score: float                # (residual_gap - mean_residual) / residual_sigma
    matched_volume: int
    adv_20: float
    volume_ratio: float           # matched_volume / adv_20 (AVR)
    imbalance_ratio: float        # (BuyQty - SellQty) / (BuyQty + SellQty + 1) (IBR)
    fair_price: float             # P_prev * (1 + expected_gap_pct)
    target_price: float           # Mean-reversion target price (61.8% reversion)
    stop_loss: float              # Protective stop loss giving >= 2.0 R:R
    is_actionable: bool           # Passed 3-sigma and all microstructure filters
    rejection_reasons: List[str] = field(default_factory=list)


@dataclass
class SniperOrder:
    """
    Precision execution order prepared for 9:15:00 AM dispatch.
    """
    symbol: str
    action: str                   # "BUY" or "SELL"
    order_type: str = "LIMIT"     # Always LIMIT to prevent opening spread slippage
    quantity: int = 0
    limit_price: float = 0.0      # Limit price with 20 bps slippage buffer
    stop_loss: float = 0.0        # Bracket stop loss price
    target_price: float = 0.0     # Bracket profit target price
    expected_risk: float = 0.0    # Exact Rupee risk (<= Rs 3,000 ceiling)
    capital_allocated: float = 0.0# Capital required (<= Rs 75,000 ceiling)
    reward_to_risk: float = 0.0   # Reward / Risk ratio (>= 1.5:1)
    timeout_seconds: float = 3.0  # Cancel entry if unfilled after 3.0 seconds
    time_stop: str = "09:45:00"   # Hard time exit if neither TP nor SL triggered
    status: str = "STAGED"        # STAGED, TRANSMITTED, FILLED, CANCELLED, REJECTED
    fill_price: Optional[float] = None
    execution_notes: List[str] = field(default_factory=list)


# =====================================================================
# Quantitative Mathematical Computations & Helpers
# =====================================================================

def round_to_tick(price: float, tick_size: float = 0.05) -> float:
    """
    Rounds a price to the nearest valid exchange tick (0.05 INR on NSE).
    Enforces exchange microstructure compliance to prevent OMS reject errors.
    """
    if price <= 0:
        return 0.0
    return round(round(price / tick_size) * tick_size, 2)


def normalize_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes DataFrame index by stripping timezone info and resetting time of day
    to midnight, ensuring robust calendar alignment between stock and index series.
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = df.index.normalize()
    else:
        df.index = pd.to_datetime(df.index).normalize()
    return df


def compute_overnight_gap_stats(
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame,
    lookback_days: int = 60
) -> HistoricalOvernightStats:
    """
    Computes overnight gap beta, idiosyncratic residual volatility (sigma),
    and 20-day ADV from historical Daily OHLCV data.

    Expects DataFrame with columns: ['Open', 'High', 'Low', 'Close', 'Volume']
    indexed by DatetimeIndex or Date.
    """
    symbol = getattr(stock_df, "symbol", "UNKNOWN")

    # Clean, normalize timestamps, and align dates
    s_clean = normalize_datetime_index(stock_df.dropna(subset=['Open', 'Close', 'Volume']))
    i_clean = normalize_datetime_index(index_df.dropna(subset=['Open', 'Close']))

    # Drop duplicate calendar dates if present
    s_clean = s_clean[~s_clean.index.duplicated(keep='last')].sort_index()
    i_clean = i_clean[~i_clean.index.duplicated(keep='last')].sort_index()

    # Calculate overnight gap returns: (Open_t - Close_{t-1}) / Close_{t-1}
    s_prev_close = s_clean['Close'].shift(1)
    i_prev_close = i_clean['Close'].shift(1)

    s_clean['gap_ret'] = (s_clean['Open'] - s_prev_close) / s_prev_close
    i_clean['gap_ret'] = (i_clean['Open'] - i_prev_close) / i_prev_close

    # Join on matching trading dates
    merged = pd.DataFrame({
        'stock_gap': s_clean['gap_ret'],
        'stock_vol': s_clean['Volume'],
        'index_gap': i_clean['gap_ret']
    }).dropna()

    if len(merged) < 20:
        # Fallback for short histories
        return HistoricalOvernightStats(
            symbol=symbol,
            beta=1.0,
            alpha=0.0,
            residual_sigma=0.015, # conservative 1.5% default sigma
            residual_mean=0.0,
            adv_20=float(s_clean['Volume'].tail(20).mean() if len(s_clean) >= 5 else 100_000.0),
            sample_size=len(merged)
        )

    # Use rolling lookback window
    window_data = merged.tail(lookback_days)
    n = len(window_data)

    stock_gaps = window_data['stock_gap'].to_numpy()
    index_gaps = window_data['index_gap'].to_numpy()

    # Beta estimation via Cov(stock_gap, index_gap) / Var(index_gap)
    var_index = float(np.var(index_gaps, ddof=1))
    if var_index > 1e-10:
        cov_matrix = np.cov(stock_gaps, index_gaps, ddof=1)
        beta = float(cov_matrix[0, 1] / var_index)
        alpha = float(np.mean(stock_gaps) - beta * np.mean(index_gaps))
    else:
        beta = 1.0
        alpha = 0.0

    # Calculate idiosyncratic residuals: epsilon_t = stock_gap_t - (alpha + beta * index_gap_t)
    residuals = stock_gaps - (alpha + beta * index_gaps)
    residual_mean = float(np.mean(residuals))
    residual_sigma = float(np.std(residuals, ddof=1))

    # Guard against zero or NaN sigma
    if residual_sigma < 1e-5 or math.isnan(residual_sigma):
        residual_sigma = 0.015

    # 20-day Average Daily Volume
    adv_20 = float(window_data['stock_vol'].tail(20).mean())
    if adv_20 <= 0 or math.isnan(adv_20):
        adv_20 = 100_000.0

    return HistoricalOvernightStats(
        symbol=symbol,
        beta=beta,
        alpha=alpha,
        residual_sigma=residual_sigma,
        residual_mean=residual_mean,
        adv_20=adv_20,
        sample_size=n
    )


def calculate_imbalance_ratio(unmatched_buy_qty: int, unmatched_sell_qty: int) -> float:
    """
    Computes normalized Order Book Imbalance Ratio (IBR):
    IBR in [-1.0, +1.0]
    +1.0 = Pure buyer demand dominance (no sellers)
    -1.0 = Pure seller supply dominance (no buyers)
    """
    buy = max(0, unmatched_buy_qty)
    sell = max(0, unmatched_sell_qty)
    total = buy + sell + 1  # Laplace smoothing against zero division
    return (buy - sell) / total


# =====================================================================
# Pre-Market Dislocation Sniping Engine
# =====================================================================

class PreMarketSniperEngine:
    """
    Engine that analyzes NSE 9:08 AM Call Auction results, executes the
    3-sigma mathematical filter, verifies microstructure safety, and routes
    sized limit orders for the 9:15:00 AM market open.
    """

    def __init__(
        self,
        capital_config: CapitalConfig = CAPITAL_CONFIG,
        sniper_config: PreMarketConfig = PRE_MARKET_CONFIG,
        risk_engine: Optional[RiskEngine] = None
    ):
        self.capital_cfg = capital_config
        self.sniper_cfg = sniper_config
        self.risk_engine = risk_engine or RiskEngine(capital_config)

    def evaluate_auction_dislocation(
        self,
        quote: AuctionQuote,
        index_quote: AuctionQuote,
        hist_stats: HistoricalOvernightStats
    ) -> DislocationSignal:
        """
        Applies the exact mathematical 3-sigma filter and microstructure rules
        to evaluate whether a pre-market auction quote presents an exploitable dislocation.
        """
        rejection_reasons = []

        # 1. Calculate raw overnight gap percentages
        if quote.previous_close <= 0 or quote.equilibrium_price <= 0:
            return self._build_invalid_signal(quote, hist_stats, "Invalid previous close or equilibrium price <= 0")

        stock_gap_pct = (quote.equilibrium_price - quote.previous_close) / quote.previous_close

        if index_quote.previous_close > 0 and index_quote.equilibrium_price > 0:
            index_gap_pct = (index_quote.equilibrium_price - index_quote.previous_close) / index_quote.previous_close
        else:
            index_gap_pct = 0.0

        # 2. Beta decomposition & Expected Gap
        expected_gap_pct = hist_stats.alpha + (hist_stats.beta * index_gap_pct)
        residual_gap = stock_gap_pct - expected_gap_pct

        # 3. Standardized Z-score calculation
        sigma = hist_stats.residual_sigma if hist_stats.residual_sigma > 1e-5 else 0.015
        z_score = (residual_gap - hist_stats.residual_mean) / sigma

        # 4. Imbalance Ratio (IBR) and Auction Volume Ratio (AVR)
        imbalance_ratio = calculate_imbalance_ratio(quote.total_buy_qty, quote.total_sell_qty)
        volume_ratio = quote.matched_volume / hist_stats.adv_20 if hist_stats.adv_20 > 0 else 0.0

        # 5. Determine Trade Direction & Fair Price (aligned to exchange tick size)
        fair_price = round_to_tick(quote.previous_close * (1.0 + expected_gap_pct))

        if z_score <= -self.sniper_cfg.z_score_threshold:
            # Undervalued dislocation: stock opened way below beta expectation -> BUY / LONG
            direction = "LONG"
            reversion_distance = max(0.0, fair_price - quote.equilibrium_price)
            raw_target = quote.equilibrium_price + (self.sniper_cfg.reversion_target_pct * reversion_distance)
            target_price = round_to_tick(raw_target)
            
            # Stop loss calculation enforcing >= 2.0 R:R
            reward = target_price - quote.equilibrium_price
            stop_distance = reward / 2.0
            raw_stop = quote.equilibrium_price - stop_distance
            stop_loss = round_to_tick(max(quote.lower_circuit, raw_stop))
            
        elif z_score >= self.sniper_cfg.z_score_threshold:
            # Overvalued dislocation: stock opened way above beta expectation -> SELL / SHORT
            direction = "SHORT"
            reversion_distance = max(0.0, quote.equilibrium_price - fair_price)
            raw_target = quote.equilibrium_price - (self.sniper_cfg.reversion_target_pct * reversion_distance)
            target_price = round_to_tick(raw_target)
            
            reward = quote.equilibrium_price - target_price
            stop_distance = reward / 2.0
            raw_stop = quote.equilibrium_price + stop_distance
            stop_loss = round_to_tick(min(quote.upper_circuit, raw_stop))
        else:
            direction = "NEUTRAL"
            target_price = quote.equilibrium_price
            stop_loss = quote.equilibrium_price
            rejection_reasons.append(
                f"Z-score ({z_score:+.2f}) does not meet 3-sigma threshold (+/-{self.sniper_cfg.z_score_threshold:.1f})"
            )

        # Microstructure Validation Filter 1: Circuit Proximity Headroom
        if quote.lower_circuit > 0 and quote.upper_circuit > 0:
            dist_to_lc = (quote.equilibrium_price - quote.lower_circuit) / quote.previous_close
            dist_to_uc = (quote.upper_circuit - quote.equilibrium_price) / quote.previous_close
            if dist_to_lc < self.sniper_cfg.circuit_buffer_pct:
                rejection_reasons.append(
                    f"Too close to Lower Circuit: margin {dist_to_lc:.2%} < {self.sniper_cfg.circuit_buffer_pct:.2%}"
                )
            if dist_to_uc < self.sniper_cfg.circuit_buffer_pct:
                rejection_reasons.append(
                    f"Too close to Upper Circuit: margin {dist_to_uc:.2%} < {self.sniper_cfg.circuit_buffer_pct:.2%}"
                )

        # Microstructure Validation Filter 2: Auction Volume Significance Ratio (AVR)
        if volume_ratio < self.sniper_cfg.min_auction_volume_ratio:
            rejection_reasons.append(
                f"Auction volume too low: {volume_ratio:.3%} of ADV < minimum {self.sniper_cfg.min_auction_volume_ratio:.2%} (Phantom quote)"
            )
        elif volume_ratio > self.sniper_cfg.max_auction_volume_ratio:
            rejection_reasons.append(
                f"Auction volume excessive: {volume_ratio:.2%} of ADV > maximum {self.sniper_cfg.max_auction_volume_ratio:.1%} (Block dump / News event)"
            )

        # Microstructure Validation Filter 3: Order Book Imbalance (IBR)
        if direction == "LONG" and imbalance_ratio < -self.sniper_cfg.max_imbalance_ratio:
            rejection_reasons.append(
                f"Severe sell book imbalance (IBR {imbalance_ratio:+.2f} < -{self.sniper_cfg.max_imbalance_ratio:.2f}) - Falling knife risk"
            )
        elif direction == "SHORT" and imbalance_ratio > self.sniper_cfg.max_imbalance_ratio:
            rejection_reasons.append(
                f"Severe buy book imbalance (IBR {imbalance_ratio:+.2f} > +{self.sniper_cfg.max_imbalance_ratio:.2f}) - Squeeze risk"
            )

        # Non-F&O intraday shorting restriction safety check
        if direction == "SHORT" and not quote.is_fno:
            rejection_reasons.append("Shorting not permitted on non-F&O cash equity under exchange rules")

        is_actionable = (len(rejection_reasons) == 0 and direction in ("LONG", "SHORT"))

        return DislocationSignal(
            symbol=quote.symbol,
            timestamp=quote.timestamp,
            direction=direction,
            previous_close=quote.previous_close,
            equilibrium_price=quote.equilibrium_price,
            stock_gap_pct=stock_gap_pct,
            index_gap_pct=index_gap_pct,
            beta=hist_stats.beta,
            expected_gap_pct=expected_gap_pct,
            residual_gap=residual_gap,
            residual_sigma=hist_stats.residual_sigma,
            z_score=z_score,
            matched_volume=quote.matched_volume,
            adv_20=hist_stats.adv_20,
            volume_ratio=volume_ratio,
            imbalance_ratio=imbalance_ratio,
            fair_price=fair_price,
            target_price=target_price,
            stop_loss=stop_loss,
            is_actionable=is_actionable,
            rejection_reasons=rejection_reasons
        )

    def route_sniping_orders(
        self,
        signals: List[DislocationSignal]
    ) -> List[SniperOrder]:
        """
        Executes order routing logic for the 9:15:00 AM market open.
        Enforces:
        - Daily kill switch status.
        - Prioritization by statistical significance (|Z-score| descending).
        - Fixed fractional risk ceiling (Rs 3,000 max loss per trade on Rs 3 Lakh capital).
        - Concentration ceiling (Rs 75,000 max capital per trade).
        - Cumulative portfolio capital bounds across the execution batch.
        - Precision limit order pricing with slippage buffer and NSE 0.05 tick size conformance.
        """
        # Circuit Breaker Check
        if self.risk_engine.is_halted:
            return []

        # Filter actionable candidates
        actionable_signals = [s for s in signals if s.is_actionable]
        if not actionable_signals:
            return []

        # Rank by magnitude of statistical mispricing (|Z-score| descending)
        ranked_signals = sorted(actionable_signals, key=lambda s: abs(s.z_score), reverse=True)

        routed_orders: List[SniperOrder] = []
        max_trades = self.sniper_cfg.max_concurrent_trades
        cumulative_allocated_capital = 0.0

        for sig in ranked_signals:
            if len(routed_orders) >= max_trades:
                break

            is_long = (sig.direction == "LONG")

            # Limit Price with 20 bps buffer to ensure immediate execution at 9:15:00 AM, rounded to tick
            buffer = self.sniper_cfg.limit_buffer_pct
            raw_limit = sig.equilibrium_price * (1.0 + buffer if is_long else 1.0 - buffer)
            limit_price = round_to_tick(raw_limit)

            # Precision Bracket Stop Loss & Target Price aligned to tick size
            target_price = round_to_tick(sig.target_price)
            if is_long:
                if target_price <= limit_price:
                    continue
                reward = target_price - limit_price
                stop_distance = reward / 2.0
                order_stop = round_to_tick(max(sig.stop_loss, limit_price - stop_distance))
            else:
                if target_price >= limit_price:
                    continue
                reward = limit_price - target_price
                stop_distance = reward / 2.0
                order_stop = round_to_tick(min(sig.stop_loss, limit_price + stop_distance))

            # Portfolio capital allocation boundary check
            remaining_cap = self.risk_engine.current_capital - cumulative_allocated_capital
            if remaining_cap < limit_price:
                break

            # Mathematical sizing from RiskEngine using limit price and remaining available capital
            sizing: TradeSizingResult = self.risk_engine.calculate_position_size(
                symbol=sig.symbol,
                entry_price=limit_price,
                stop_loss=order_stop,
                target_price=target_price,
                is_long=is_long,
                max_available_capital=remaining_cap
            )

            if not sizing.is_valid or sizing.shares <= 0:
                continue

            order = SniperOrder(
                symbol=sig.symbol,
                action="BUY" if is_long else "SELL",
                order_type="LIMIT",
                quantity=sizing.shares,
                limit_price=limit_price,
                stop_loss=order_stop,
                target_price=target_price,
                expected_risk=round(sizing.total_risk_amount, 2),
                capital_allocated=round(sizing.total_trade_capital, 2),
                reward_to_risk=round(sizing.reward_to_risk_ratio, 2),
                timeout_seconds=self.sniper_cfg.order_timeout_seconds,
                time_stop="09:45:00",
                status="STAGED",
                execution_notes=[
                    f"Z-Score: {sig.z_score:+.2f}",
                    f"Discovered Open (IEP): Rs {sig.equilibrium_price:.2f}",
                    f"Fair Value: Rs {sig.fair_price:.2f}",
                    f"Slippage Buffer: {self.sniper_cfg.limit_buffer_pct:.2%}",
                    f"Tick Compliance: Valid Rs 0.05 multiple",
                    f"Kill Switch Safety: OK"
                ]
            )
            cumulative_allocated_capital += sizing.total_trade_capital
            routed_orders.append(order)

        return routed_orders

    def _build_invalid_signal(
        self,
        quote: AuctionQuote,
        hist_stats: HistoricalOvernightStats,
        reason: str
    ) -> DislocationSignal:
        return DislocationSignal(
            symbol=quote.symbol,
            timestamp=quote.timestamp,
            direction="NEUTRAL",
            previous_close=quote.previous_close,
            equilibrium_price=quote.equilibrium_price,
            stock_gap_pct=0.0,
            index_gap_pct=0.0,
            beta=hist_stats.beta,
            expected_gap_pct=0.0,
            residual_gap=0.0,
            residual_sigma=hist_stats.residual_sigma,
            z_score=0.0,
            matched_volume=quote.matched_volume,
            adv_20=hist_stats.adv_20,
            volume_ratio=0.0,
            imbalance_ratio=0.0,
            fair_price=quote.equilibrium_price,
            target_price=quote.equilibrium_price,
            stop_loss=quote.equilibrium_price,
            is_actionable=False,
            rejection_reasons=[reason]
        )


# =====================================================================
# NSE Live Data Feed Adapter
# =====================================================================

class NSEPreMarketFeedAdapter:
    """
    Ingests and parses official NSE Pre-Market Call Auction live data feeds
    or simulated quote packets.
    """

    NSE_PREOPEN_URL = "https://www.nseindia.com/api/market-data-pre-open?key=FO"
    NSE_INDEX_URL = "https://www.nseindia.com/api/allIndices"
    DEFAULT_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/market-data/pre-open-market-cm-and-emerge-market"
    }

    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or requests.Session()
        self.session.headers.update(self.DEFAULT_HEADERS)
        self._cookies_initialized = False

    @staticmethod
    def _safe_float(val: Any, default: float = 0.0) -> float:
        if val is None:
            return default
        try:
            return float(str(val).replace(",", "").strip())
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_int(val: Any, default: int = 0) -> int:
        if val is None:
            return default
        try:
            return int(float(str(val).replace(",", "").strip()))
        except (ValueError, TypeError):
            return default

    def _init_session(self):
        """Initializes NSE session cookies by touching base domain."""
        if not self._cookies_initialized:
            try:
                self.session.get("https://www.nseindia.com", timeout=5)
                self._cookies_initialized = True
            except Exception:
                pass

    def fetch_live_auction_quotes(self) -> Tuple[List[AuctionQuote], AuctionQuote]:
        """
        Fetches live pre-market call auction snapshot from NSE API.
        Returns (list of stock AuctionQuotes, index AuctionQuote).
        """
        self._init_session()
        response = self.session.get(self.NSE_PREOPEN_URL, timeout=8)
        response.raise_for_status()
        data = response.json().get("data", [])

        stock_quotes: List[AuctionQuote] = []
        now = datetime.now()

        for item in data:
            meta = item.get("metadata", {})
            detail = item.get("detail", {}).get("preOpenMarket", {})

            symbol = meta.get("symbol")
            if not symbol:
                continue

            prev_close = self._safe_float(meta.get("previousClose"), 0.0)
            iep = self._safe_float(meta.get("iep"), self._safe_float(meta.get("lastPrice"), 0.0))
            if prev_close <= 0 or iep <= 0:
                continue

            matched_vol = self._safe_int(detail.get("finalQuantity", meta.get("finalQuantity")), 0)
            total_buy = self._safe_int(detail.get("totalBuyQuantity"), 0)
            total_sell = self._safe_int(detail.get("totalSellQuantity"), 0)

            # Circuit bands: defaults to +/- 10% if not explicitly in feed, aligned to tick size
            lower_circuit = round_to_tick(prev_close * 0.90)
            upper_circuit = round_to_tick(prev_close * 1.10)

            stock_quotes.append(
                AuctionQuote(
                    symbol=f"{symbol}.NS",
                    timestamp=now,
                    previous_close=prev_close,
                    equilibrium_price=round_to_tick(iep),
                    matched_volume=matched_vol,
                    total_buy_qty=total_buy,
                    total_sell_qty=total_sell,
                    lower_circuit=lower_circuit,
                    upper_circuit=upper_circuit,
                    is_fno=True
                )
            )

        # Index snapshot (NIFTY 50)
        index_quote = self._fetch_index_quote(stock_quotes)
        return stock_quotes, index_quote

    def _fetch_index_quote(self, stock_quotes: List[AuctionQuote]) -> AuctionQuote:
        """
        Attempts to fetch live NIFTY 50 from index endpoint; falls back to
        median gap of constituent quotes if index endpoint is unavailable.
        """
        now = datetime.now()
        try:
            resp = self.session.get(self.NSE_INDEX_URL, timeout=5)
            if resp.status_code == 200:
                indices = resp.json().get("data", [])
                for idx in indices:
                    if idx.get("index") == "NIFTY 50":
                        p_close = self._safe_float(idx.get("previousClose"), 0.0)
                        last_p = self._safe_float(idx.get("last"), p_close)
                        if p_close > 0:
                            return AuctionQuote(
                                symbol="^NSEI",
                                timestamp=now,
                                previous_close=p_close,
                                equilibrium_price=round_to_tick(last_p),
                                matched_volume=0,
                                total_buy_qty=0,
                                total_sell_qty=0,
                                lower_circuit=round_to_tick(p_close * 0.85),
                                upper_circuit=round_to_tick(p_close * 1.15),
                                is_fno=False
                            )
        except Exception:
            pass

        # Robust Fallback: Median gap of top liquid equities
        if stock_quotes:
            valid_gaps = [
                (q.equilibrium_price - q.previous_close) / q.previous_close
                for q in stock_quotes if q.previous_close > 0
            ]
            median_gap = float(np.median(valid_gaps)) if valid_gaps else 0.0
        else:
            median_gap = 0.0

        base_index_price = 24_000.0
        return AuctionQuote(
            symbol="^NSEI",
            timestamp=now,
            previous_close=base_index_price,
            equilibrium_price=round_to_tick(base_index_price * (1.0 + median_gap)),
            matched_volume=0,
            total_buy_qty=0,
            total_sell_qty=0,
            lower_circuit=round_to_tick(base_index_price * 0.85),
            upper_circuit=round_to_tick(base_index_price * 1.15),
            is_fno=False
        )


class SyntheticPreMarketGenerator:
    """
    Generates deterministic, mathematically verifiable pre-market scenarios
    for backtesting, selftest verification, and simulation.
    """

    @staticmethod
    def create_scenario(
        symbol: str = "RELIANCE.NS",
        prev_close: float = 3000.0,
        beta: float = 1.2,
        residual_sigma: float = 0.008,     # 0.8% overnight gap standard deviation
        index_gap_pct: float = -0.005,     # Index down -0.50%
        sigma_multiple: float = -3.5,      # 3.5-sigma dislocation downwards
        adv_20: float = 5_000_000.0,
        volume_pct_adv: float = 0.02,      # 2% of ADV matched
        imbalance: float = 0.10            # Moderate positive buy demand
    ) -> Tuple[AuctionQuote, AuctionQuote, HistoricalOvernightStats]:
        """
        Constructs an exact synthetic test scenario conforming to NSE tick rules.
        """
        now = datetime.now()

        # Expected stock gap = beta * index_gap = 1.2 * (-0.005) = -0.006 (-0.6%)
        # Dislocation = sigma_multiple * sigma = -3.5 * 0.008 = -0.028 (-2.8%)
        # Total gap = -0.006 + (-0.028) = -0.034 (-3.4%)
        expected_gap = beta * index_gap_pct
        residual = sigma_multiple * residual_sigma
        total_gap = expected_gap + residual

        equilibrium_price = round_to_tick(prev_close * (1.0 + total_gap))
        matched_volume = int(adv_20 * volume_pct_adv)

        # Imbalance to buy/sell quantities
        base_depth = 100_000
        buy_qty = int(base_depth * (1.0 + imbalance))
        sell_qty = int(base_depth * (1.0 - imbalance))

        quote = AuctionQuote(
            symbol=symbol,
            timestamp=now,
            previous_close=prev_close,
            equilibrium_price=equilibrium_price,
            matched_volume=matched_volume,
            total_buy_qty=buy_qty,
            total_sell_qty=sell_qty,
            lower_circuit=round_to_tick(prev_close * 0.90),
            upper_circuit=round_to_tick(prev_close * 1.10),
            is_fno=True
        )

        index_prev = 24_000.0
        index_quote = AuctionQuote(
            symbol="^NSEI",
            timestamp=now,
            previous_close=index_prev,
            equilibrium_price=round_to_tick(index_prev * (1.0 + index_gap_pct)),
            matched_volume=0,
            total_buy_qty=0,
            total_sell_qty=0,
            lower_circuit=round_to_tick(index_prev * 0.85),
            upper_circuit=round_to_tick(index_prev * 1.15),
            is_fno=False
        )

        hist_stats = HistoricalOvernightStats(
            symbol=symbol,
            beta=beta,
            alpha=0.0,
            residual_sigma=residual_sigma,
            residual_mean=0.0,
            adv_20=adv_20,
            sample_size=60
        )

        return quote, index_quote, hist_stats
