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
    round_to_tick,
    normalize_datetime_index,
    NSEPreMarketFeedAdapter,
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


def test_timezone_and_timestamp_normalization():
    # Test 1: Tz-aware stock DataFrame (Asia/Kolkata) with tz-naive index DataFrame
    dates_stock = pd.date_range("2026-01-01 09:15", periods=30, freq="B", tz="Asia/Kolkata")
    dates_index = pd.date_range("2026-01-01 00:00", periods=30, freq="B")

    s_df = pd.DataFrame({
        "Open": [100.0 + i for i in range(30)],
        "High": [102.0 + i for i in range(30)],
        "Low": [98.0 + i for i in range(30)],
        "Close": [101.0 + i for i in range(30)],
        "Volume": [500_000] * 30
    }, index=dates_stock)

    i_df = pd.DataFrame({
        "Open": [20000.0 + i * 10 for i in range(30)],
        "High": [20100.0 + i * 10 for i in range(30)],
        "Low": [19900.0 + i * 10 for i in range(30)],
        "Close": [20050.0 + i * 10 for i in range(30)],
        "Volume": [10_000_000] * 30
    }, index=dates_index)

    # Must execute cleanly without TypeError and correctly match calendar dates
    stats = compute_overnight_gap_stats(s_df, i_df, lookback_days=25)
    assert stats.sample_size >= 25, f"Expected >= 25 matched dates, got {stats.sample_size}"
    assert stats.adv_20 == 500_000.0


def test_tick_size_compliance(base_setup):
    engine, _, _, _ = base_setup
    # Verify helper rounding
    assert round_to_tick(2908.12) == 2908.10
    assert round_to_tick(2908.13) == 2908.15
    assert round_to_tick(2908.17) == 2908.15
    assert round_to_tick(2908.18) == 2908.20

    quote, idx_quote, h_stats = SyntheticPreMarketGenerator.create_scenario(
        symbol="RELIANCE.NS",
        prev_close=3000.0,
        sigma_multiple=-3.5
    )
    sig = engine.evaluate_auction_dislocation(quote, idx_quote, h_stats)
    assert sig.is_actionable

    # Verify that fair price, target, and stop loss are strictly on the 0.05 tick
    for price in [sig.fair_price, sig.target_price, sig.stop_loss]:
        remainder = round(price * 20) - (price * 20)
        assert abs(remainder) < 1e-5, f"Price {price} is not a valid 0.05 multiple"

    orders = engine.route_sniping_orders([sig])
    assert len(orders) == 1
    o = orders[0]
    for p in [o.limit_price, o.stop_loss, o.target_price]:
        remainder = round(p * 20) - (p * 20)
        assert abs(remainder) < 1e-5, f"Order price {p} is not a valid 0.05 multiple"


def test_directional_safety_in_risk_engine():
    risk_engine = RiskEngine()

    # Long with target below entry -> MUST be rejected
    res_long_bad = risk_engine.calculate_position_size("LONG_BAD", 100.0, 90.0, 80.0, is_long=True)
    assert not res_long_bad.is_valid
    assert "Target price must be strictly above entry price" in res_long_bad.rejection_reason

    # Short with stop below entry -> MUST be rejected
    res_short_stop_bad = risk_engine.calculate_position_size("SHORT_BAD", 100.0, 90.0, 70.0, is_long=False)
    assert not res_short_stop_bad.is_valid
    assert "Stop loss must be strictly above entry price for short positions." in res_short_stop_bad.rejection_reason

    # Short with target above entry -> MUST be rejected
    res_short_tp_bad = risk_engine.calculate_position_size("SHORT_TP_BAD", 100.0, 110.0, 120.0, is_long=False)
    assert not res_short_tp_bad.is_valid
    assert "Target price must be strictly below entry price for short positions." in res_short_tp_bad.rejection_reason


def test_zero_and_negative_entry_price_rejected():
    risk_engine = RiskEngine()
    # Zero entry price
    res_zero = risk_engine.calculate_position_size("ZERO", 0.0, -10.0, 20.0, is_long=True)
    assert not res_zero.is_valid
    assert "Entry price must be strictly positive" in res_zero.rejection_reason

    # Negative entry price
    res_neg = risk_engine.calculate_position_size("NEG", -10.0, -20.0, 20.0, is_long=True)
    assert not res_neg.is_valid
    assert "Entry price must be strictly positive" in res_neg.rejection_reason


def test_limit_price_risk_and_capital_bounds(base_setup):
    engine, _, _, capital_cfg = base_setup
    quote, idx_quote, h_stats = SyntheticPreMarketGenerator.create_scenario(
        symbol="RELIANCE.NS",
        prev_close=3000.0,
        sigma_multiple=-3.5
    )
    sig = engine.evaluate_auction_dislocation(quote, idx_quote, h_stats)
    orders = engine.route_sniping_orders([sig])
    assert len(orders) == 1
    order = orders[0]

    # At the worst-case limit fill:
    # 1. Capital allocated must be <= Rs 75,000 (25% capital ceiling)
    assert order.capital_allocated <= capital_cfg.total_capital * capital_cfg.max_portfolio_allocation_pct
    assert order.quantity * order.limit_price <= 75_000.0

    # 2. Risk must be <= Rs 3,000 (1% risk ceiling)
    worst_case_risk = order.quantity * (order.limit_price - order.stop_loss)
    assert worst_case_risk <= 3_000.0
    assert order.expected_risk <= 3_000.0

    # 3. Reward-to-Risk must be >= 1.5:1
    assert order.reward_to_risk >= 1.5


def test_batch_capital_allocation_ceiling():
    cap_cfg = CapitalConfig(total_capital=100_000.0) # Rs 1 Lakh capital
    sniper_cfg = PreMarketConfig(max_concurrent_trades=4)
    risk_engine = RiskEngine(cap_cfg)
    engine = PreMarketSniperEngine(cap_cfg, sniper_cfg, risk_engine)

    signals = []
    for sym in ["SYM1.NS", "SYM2.NS", "SYM3.NS", "SYM4.NS"]:
        q, idx_q, hs = SyntheticPreMarketGenerator.create_scenario(
            symbol=sym,
            prev_close=1000.0,
            sigma_multiple=-3.5
        )
        sig = engine.evaluate_auction_dislocation(q, idx_q, hs)
        signals.append(sig)

    orders = engine.route_sniping_orders(signals)
    total_alloc = sum(o.capital_allocated for o in orders)
    # Total allocated capital across all staged orders must not exceed total capital
    assert total_alloc <= 100_000.0


def test_safe_parsing_of_malformed_live_feed():
    adapter = NSEPreMarketFeedAdapter()
    assert adapter._safe_float(None, 10.0) == 10.0
    assert adapter._safe_float("1,250.75", 0.0) == 1250.75
    assert adapter._safe_float("invalid", 5.0) == 5.0
    assert adapter._safe_int(None, 0) == 0
    assert adapter._safe_int("5,000", 0) == 5000
    assert adapter._safe_int("3.14", 0) == 3
