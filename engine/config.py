"""
Trading Configuration & Single Source of Truth
"""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class CapitalConfig:
    total_capital: float = 300_000.0        # Total starting trading capital (INR)
    risk_per_trade_pct: float = 0.010       # 1.0% max loss per trade (Rs 3,000)
    max_portfolio_allocation_pct: float = 0.25 # Max 25% of capital per single position
    daily_drawdown_limit_pct: float = 0.025 # 2.5% max daily loss kill-switch (Rs 7,500)
    weekly_drawdown_limit_pct: float = 0.050# 5.0% max weekly loss circuit breaker (Rs 15,000)


@dataclass(frozen=True)
class BrokerConfig:
    # Transaction cost estimates for NSE equity delivery/swing
    brokerage_per_order: float = 20.0       # Standard discount broker fee (INR)
    stt_delivery_rate: float = 0.001        # 0.1% on buy & sell (Securities Transaction Tax)
    exchange_txn_charge_rate: float = 0.0000345 # NSE turnover charge
    gst_rate: float = 0.18                  # 18% on (brokerage + txn charges)
    stamp_duty_rate: float = 0.00015        # 0.015% on buy side
    estimated_slippage_pct: float = 0.001   # 0.1% slippage assumption


@dataclass(frozen=True)
class UniverseConfig:
    # Core high-liquidity large & midcap NSE stocks for swing trading
    symbols: List[str] = field(default_factory=lambda: [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
        "BHARTIARTL.NS", "ITC.NS", "SBIN.NS", "LICI.NS", "HINDUNILVR.NS",
        "LT.NS", "BAJFINANCE.NS", "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS",
        "TATAMOTORS.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "TITAN.NS"
    ])


CAPITAL_CONFIG = CapitalConfig()
BROKER_CONFIG = BrokerConfig()
UNIVERSE_CONFIG = UniverseConfig()


@dataclass(frozen=True)
class PreMarketConfig:
    z_score_threshold: float = 3.0          # Absolute Z-score threshold for 3-sigma dislocation
    lookback_days: int = 60                 # Lookback window for overnight beta and sigma calculation
    min_auction_volume_ratio: float = 0.002 # Min 0.2% of 20-day ADV (filters out phantom/illiquid quotes)
    max_auction_volume_ratio: float = 0.150 # Max 15% of 20-day ADV (filters out institutional block dumps/leaks)
    max_imbalance_ratio: float = 0.80       # Max order book imbalance to avoid circuit freezes
    circuit_buffer_pct: float = 0.005       # Min 0.5% distance from Upper/Lower Circuit limits
    limit_buffer_pct: float = 0.002         # 0.20% price buffer for 9:15:00 AM limit orders
    order_timeout_seconds: float = 3.0      # Cancel entry order if unfilled after 3.0 seconds
    time_stop_minutes: int = 30             # Hard time stop: exit at 9:45:00 AM if trade still open
    reversion_target_pct: float = 0.618     # Target 61.8% mean reversion towards fair value
    max_concurrent_trades: int = 2          # Max concurrent trades for ?3L capital budget
    min_reward_to_risk: float = 1.5         # Minimum Reward-to-Risk ratio required


PRE_MARKET_CONFIG = PreMarketConfig()
