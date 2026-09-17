"""
Risk Management Engine
Implements strict mathematical position sizing, capital protection, and drawdown circuit breakers.
"""

import math
from dataclasses import dataclass
from typing import Optional
from .config import CAPITAL_CONFIG, BROKER_CONFIG, CapitalConfig, BrokerConfig


@dataclass
class TradeSizingResult:
    shares: int
    entry_price: float
    stop_loss: float
    target_price: float
    risk_per_share: float
    total_risk_amount: float
    total_trade_capital: float
    reward_to_risk_ratio: float
    is_valid: bool
    rejection_reason: Optional[str] = None


class RiskEngine:
    def __init__(self, capital_config: CapitalConfig = CAPITAL_CONFIG, broker_config: BrokerConfig = BROKER_CONFIG):
        self.capital_cfg = capital_config
        self.broker_cfg = broker_config
        self.current_capital = capital_config.total_capital
        self.daily_pnl = 0.0
        self.is_halted = False

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        target_price: float,
        is_long: bool = True
    ) -> TradeSizingResult:
        """
        Computes the exact number of shares to buy/sell based on strict risk ceiling
        and portfolio concentration rules.
        """
        if self.is_halted:
            return TradeSizingResult(
                shares=0, entry_price=entry_price, stop_loss=stop_loss,
                target_price=target_price, risk_per_share=0, total_risk_amount=0,
                total_trade_capital=0, reward_to_risk_ratio=0, is_valid=False,
                rejection_reason="Risk engine is halted due to daily drawdown limit."
            )

        if is_long and stop_loss >= entry_price:
            return TradeSizingResult(
                shares=0, entry_price=entry_price, stop_loss=stop_loss,
                target_price=target_price, risk_per_share=0, total_risk_amount=0,
                total_trade_capital=0, reward_to_risk_ratio=0, is_valid=False,
                rejection_reason="Stop loss must be strictly below entry price for long positions."
            )

        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return TradeSizingResult(
                shares=0, entry_price=entry_price, stop_loss=stop_loss,
                target_price=target_price, risk_per_share=0, total_risk_amount=0,
                total_trade_capital=0, reward_to_risk_ratio=0, is_valid=False,
                rejection_reason="Invalid risk per share (zero or negative)."
            )

        reward_per_share = abs(target_price - entry_price)
        reward_risk_ratio = reward_per_share / risk_per_share

        if reward_risk_ratio < 1.5:
            return TradeSizingResult(
                shares=0, entry_price=entry_price, stop_loss=stop_loss,
                target_price=target_price, risk_per_share=risk_per_share, total_risk_amount=0,
                total_trade_capital=0, reward_to_risk_ratio=reward_risk_ratio, is_valid=False,
                rejection_reason=f"Reward-to-Risk ratio ({reward_risk_ratio:.2f}) is below minimum 1.5:1 requirement."
            )

        # 1. Max allowed loss for this trade (e.g. 1.0% of capital = Rs 3,000)
        max_trade_risk = self.current_capital * self.capital_cfg.risk_per_trade_pct

        # 2. Shares bounded by risk ceiling
        qty_by_risk = math.floor(max_trade_risk / risk_per_share)

        # 3. Shares bounded by portfolio allocation ceiling (e.g. 25% max per position = Rs 75,000)
        max_position_capital = self.current_capital * self.capital_cfg.max_portfolio_allocation_pct
        qty_by_capital = math.floor(max_position_capital / entry_price)

        final_shares = min(qty_by_risk, qty_by_capital)

        if final_shares <= 0:
            return TradeSizingResult(
                shares=0, entry_price=entry_price, stop_loss=stop_loss,
                target_price=target_price, risk_per_share=risk_per_share, total_risk_amount=0,
                total_trade_capital=0, reward_to_risk_ratio=reward_risk_ratio, is_valid=False,
                rejection_reason="Calculated position size is 0 shares due to capital/risk limits."
            )

        actual_risk_amount = final_shares * risk_per_share
        actual_trade_capital = final_shares * entry_price

        return TradeSizingResult(
            shares=final_shares,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            risk_per_share=risk_per_share,
            total_risk_amount=actual_risk_amount,
            total_trade_capital=actual_trade_capital,
            reward_to_risk_ratio=reward_risk_ratio,
            is_valid=True
        )

    def record_pnl(self, pnl: float) -> bool:
        """
        Records realized profit/loss and checks circuit breakers.
        Returns True if trading can continue, False if halted.
        """
        self.daily_pnl += pnl
        self.current_capital += pnl

        daily_halt_threshold = -(self.capital_cfg.total_capital * self.capital_cfg.daily_drawdown_limit_pct)
        if self.daily_pnl <= daily_halt_threshold:
            self.is_halted = True
            return False
        return True
