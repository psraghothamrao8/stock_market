"""
Unit and Regression Test Suite for Pre-Market Call Auction Dislocation Sniping Engine
"""

from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from engine.config import CapitalConfig, PreMarketConfig
from engine.risk import RiskEngine
from engine.premarket_sniper import (
    AuctionQuote,
    HistoricalOvernightStats,
    PreMarketSniperEngine,
    SyntheticPreMarketGenerator,
    calculate_imbalance_ratio,
    compute_overnight_gap_stats,
)


@pytest.fixture
def base_setup():
    capital_cfg = CapitalConfig(
        total_capital=300_000.0,
        risk_per_trade_pct=0.01,         # 1% = Rs 3,000 max risk
        max_portfolio_allocation_pct=0.25 # 25% = Rs 75,000 max capital per trade
    )
    sniper_cfg = PreMarketConfig(
        z_score_threshold=3.0,
        min_auction_volume_ratio=0.002,
        max_auction_volume_ratio=0.150,
        max_imbalance_ratio=0.80,
        circuit_buffer_pct=0.005,
        limit_buffer_pct=0.002,
        max_concurrent_trades=2
    )
    risk_engine = RiskEngine(capital_cfg)
    engine = PreMarketSniperEngine(capital_cfg, sniper_cfg, risk_engine)
    return engine, risk_engine, sniper_cfg, capital_cfg


def test_imbalance_ratio_calculation():
    # Extreme buy dominance
    assert calculate_imbalance_ratio(100_000, 0) > 0.99
    # Extreme sell dominance
    assert calculate_imbalance_ratio(0, 100_000) < -0.99
    # Equal balance
    assert abs(calculate_imbalance_ratio(50_000, 50_000)) < 0.001
    # Zero orders
    assert calculate_imbalance_ratio(0, 0) == 0.0


def test_overnight_gap_stats_computation():
    # Create 60 days of synthetic data where Stock overnight gap = 1.5 * Index gap + noise
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=65, freq="B")

    index_gaps = np.random.normal(0.001, 0.005, 65)
    stock_gaps = 1.5 * index_gaps + np.random.normal(0.0, 0.003, 65)

    index_prices = [24000.0]
    stock_prices = [3000.0]
    for i in range(1, 65):
        index_prices.append(index_prices[-1] * (1.0 + index_gaps[i]))
        stock_prices.append(stock_prices[-1] * (1.0 + stock_gaps[i]))

    index_df = pd.DataFrame({
        "Open": index_prices,
        "High": [p * 1.01 for p in index_prices],
        "Low": [p * 0.99 for p in index_prices],
        "Close": [p * 0.999 for p in index_prices],
        "Volume": [10_000_000] * 65
    }, index=dates)

    stock_df = pd.DataFrame({
        "Open": stock_prices,
        "High": [p * 1.01 for p in stock_prices],
        "Low": [p * 0.99 for p in stock_prices],
        "Close": [p * 0.999 for p in stock_prices],
        "Volume": [2_000_000] * 65
    }, index=dates)

    stats = compute_overnight_gap_stats(stock_df, index_df, lookback_days=60)
    assert 1.2 <= stats.beta <= 1.8, f"Expected beta near 1.5, got {stats.beta}"
    assert stats.residual_sigma > 0.001
    assert stats.adv_20 == 2_000_000.0


def test_actionable_long_dislocation(base_setup):
    engine, _, _, _ = base_setup
    quote, idx_quote, h_stats = SyntheticPreMarketGenerator.create_scenario(
        symbol="RELIANCE.NS",
        prev_close=3000.0,
        beta=1.2,
        residual_sigma=0.008,
        index_gap_pct=-0.005,
        sigma_multiple=-3.5,     # 3.5-sigma gap down
        adv_20=5_000_000.0,
        volume_pct_adv=0.02,     # 2% volume
        imbalance=0.10           # Positive buy demand
    )

    sig = engine.evaluate_auction_dislocation(quote, idx_quote, h_stats)
    assert sig.is_actionable
    assert sig.direction == "LONG"
    assert sig.z_score < -3.0
    assert sig.target_price > sig.equilibrium_price
    assert sig.stop_loss < sig.equilibrium_price

    # Order routing test
    orders = engine.route_sniping_orders([sig])
    assert len(orders) == 1
    order = orders[0]
    assert order.action == "BUY"
    assert order.order_type == "LIMIT"
    assert order.quantity > 0
    # Strict capital limit: capital allocated <= Rs 75,000
    assert order.capital_allocated <= 75_000.0
    # Strict risk ceiling: expected risk <= Rs 3,000
    assert order.expected_risk <= 3_000.0
    # Strict Reward-to-Risk ratio >= 1.5
    assert order.reward_to_risk >= 1.5
    # Limit price includes buffer
    assert order.limit_price == round(sig.equilibrium_price * 1.002, 2)


def test_actionable_short_dislocation(base_setup):
    engine, _, _, _ = base_setup
    quote, idx_quote, h_stats = SyntheticPreMarketGenerator.create_scenario(
        symbol="TCS.NS",
        prev_close=4000.0,
        beta=0.9,
        residual_sigma=0.007,
        index_gap_pct=+0.004,
        sigma_multiple=+3.2,     # 3.2-sigma gap up
        adv_20=3_000_000.0,
        volume_pct_adv=0.02,
        imbalance=-0.10          # Sellers stepping in
    )

    sig = engine.evaluate_auction_dislocation(quote, idx_quote, h_stats)
    assert sig.is_actionable
    assert sig.direction == "SHORT"
    assert sig.z_score > 3.0
    assert sig.target_price < sig.equilibrium_price
    assert sig.stop_loss > sig.equilibrium_price

    orders = engine.route_sniping_orders([sig])
    assert len(orders) == 1
    order = orders[0]
    assert order.action == "SELL"
    assert order.capital_allocated <= 75_000.0
    assert order.expected_risk <= 3_000.0
    assert order.reward_to_risk >= 1.5


def test_microstructure_filter_circuit_proximity(base_setup):
    engine, _, _, _ = base_setup
    quote, idx_quote, h_stats = SyntheticPreMarketGenerator.create_scenario(
        symbol="HDFCBANK.NS",
        prev_close=1600.0,
        sigma_multiple=-3.5
    )
    # Move price dangerously close to lower circuit (within 0.2%)
    lc = 1440.0
    quote = AuctionQuote(
        symbol=quote.symbol,
        timestamp=quote.timestamp,
        previous_close=1600.0,
        equilibrium_price=lc * 1.002,
        matched_volume=quote.matched_volume,
        total_buy_qty=quote.total_buy_qty,
        total_sell_qty=quote.total_sell_qty,
        lower_circuit=lc,
        upper_circuit=1760.0,
        is_fno=True
    )

    sig = engine.evaluate_auction_dislocation(quote, idx_quote, h_stats)
    assert not sig.is_actionable
    assert any("Lower Circuit" in r for r in sig.rejection_reasons)


def test_microstructure_filter_low_volume_phantom(base_setup):
    engine, _, _, _ = base_setup
    quote, idx_quote, h_stats = SyntheticPreMarketGenerator.create_scenario(
        symbol="INFY.NS",
        prev_close=1800.0,
        sigma_multiple=-3.5,
        adv_20=5_000_000.0,
        volume_pct_adv=0.0005 # 0.05% < 0.2% minimum
    )

    sig = engine.evaluate_auction_dislocation(quote, idx_quote, h_stats)
    assert not sig.is_actionable
    assert any("Auction volume too low" in r for r in sig.rejection_reasons)


def test_microstructure_filter_extreme_sell_imbalance(base_setup):
    engine, _, _, _ = base_setup
    quote, idx_quote, h_stats = SyntheticPreMarketGenerator.create_scenario(
        symbol="TATAMOTORS.NS",
        prev_close=1000.0,
        sigma_multiple=-4.0,
        imbalance=-0.95 # Severe sell imbalance
    )

    sig = engine.evaluate_auction_dislocation(quote, idx_quote, h_stats)
    assert not sig.is_actionable
    assert any("Severe sell book imbalance" in r for r in sig.rejection_reasons)


def test_below_3sigma_threshold_rejected(base_setup):
    engine, _, _, _ = base_setup
    quote, idx_quote, h_stats = SyntheticPreMarketGenerator.create_scenario(
        symbol="ICICIBANK.NS",
        prev_close=1200.0,
        sigma_multiple=-1.5 # Only 1.5-sigma
    )

    sig = engine.evaluate_auction_dislocation(quote, idx_quote, h_stats)
    assert not sig.is_actionable
    assert sig.direction == "NEUTRAL"
    assert any("does not meet 3-sigma threshold" in r for r in sig.rejection_reasons)


def test_kill_switch_blocks_order_routing(base_setup):
    engine, risk_engine, _, _ = base_setup
    quote, idx_quote, h_stats = SyntheticPreMarketGenerator.create_scenario(
        symbol="RELIANCE.NS",
        prev_close=3000.0,
        sigma_multiple=-3.5
    )

    sig = engine.evaluate_auction_dislocation(quote, idx_quote, h_stats)
    assert sig.is_actionable

    # Trigger daily drawdown kill switch (-Rs 7,500)
    risk_engine.record_pnl(-7600.0)
    assert risk_engine.is_halted

    # Engine must refuse to route any orders
    orders = engine.route_sniping_orders([sig])
    assert len(orders) == 0


def test_max_concurrent_trades_constraint(base_setup):
    engine, _, sniper_cfg, _ = base_setup
    # Create 3 actionable signals
    signals = []
    for sym, mult in [("RELIANCE.NS", -4.0), ("TCS.NS", -3.8), ("INFY.NS", -3.2)]:
        q, idx_q, hs = SyntheticPreMarketGenerator.create_scenario(
            symbol=sym,
            prev_close=3000.0,
            sigma_multiple=mult
        )
        sig = engine.evaluate_auction_dislocation(q, idx_q, hs)
        assert sig.is_actionable
        signals.append(sig)

    # Max concurrent trades is 2
    orders = engine.route_sniping_orders(signals)
    assert len(orders) == 2
    # Should select top 2 by |Z-score|: RELIANCE and TCS
    routed_symbols = [o.symbol for o in orders]
    assert "RELIANCE.NS" in routed_symbols
    assert "TCS.NS" in routed_symbols
    assert "INFY.NS" not in routed_symbols
