"""
Offline Self-Test Script
Validates mathematical risk rules and position sizing logic without requiring external network access.
"""

from engine.risk import RiskEngine
from engine.config import CapitalConfig

def run_tests():
    print("========================================")
    print("Running Risk Engine Self-Tests")
    print("========================================")

    config = CapitalConfig(
        total_capital=300_000.0,
        risk_per_trade_pct=0.01,         # 1% = Rs 3,000 max risk
        max_portfolio_allocation_pct=0.25 # 25% = Rs 75,000 max position capital
    )
    engine = RiskEngine(config)

    # Test Case 1: Standard swing trade setup (e.g. Reliance buy at 3000, SL at 2950, TP at 3125)
    # Risk per share = 50. Max risk = 3000 -> 60 shares.
    # Capital required for 60 shares = 60 * 3000 = 180,000.
    # BUT max capital allocation is 75,000 -> 75,000 / 3000 = 25 shares.
    # Therefore, allocation ceiling should constrain it to 25 shares.
    result = engine.calculate_position_size(
        symbol="RELIANCE.NS",
        entry_price=3000.0,
        stop_loss=2950.0,
        target_price=3125.0,
        is_long=True
    )

    print(f"Test 1 - Allocation Constrained Trade:")
    print(f"  Valid: {result.is_valid}")
    print(f"  Shares: {result.shares} (Expected: 25)")
    print(f"  Actual Risk: Rs {result.total_risk_amount:.2f}")
    print(f"  Actual Capital: Rs {result.total_trade_capital:.2f}")
    print(f"  R:R Ratio: {result.reward_to_risk_ratio:.2f}")
    assert result.is_valid
    assert result.shares == 25, f"Expected 25 shares, got {result.shares}"

    # Test Case 2: Risk Constrained Trade (e.g. Smallcap / High-volatility stock)
    # Entry: 500, SL: 450 (Risk = 50 per share). Target: 650 (Reward = 150).
    # Risk limit: 3,000 / 50 = 60 shares.
    # Capital for 60 shares: 60 * 500 = 30,000 <= 75,000 allocation limit.
    # Expected shares = 60.
    result2 = engine.calculate_position_size(
        symbol="MIDCAP.NS",
        entry_price=500.0,
        stop_loss=450.0,
        target_price=650.0,
        is_long=True
    )
    print(f"\nTest 2 - Risk Constrained Trade:")
    print(f"  Valid: {result2.is_valid}")
    print(f"  Shares: {result2.shares} (Expected: 60)")
    print(f"  Actual Risk: Rs {result2.total_risk_amount:.2f}")
    assert result2.is_valid
    assert result2.shares == 60, f"Expected 60 shares, got {result2.shares}"

    # Test Case 3: Poor Reward:Risk ratio (< 1.5)
    result3 = engine.calculate_position_size(
        symbol="BAD_RR.NS",
        entry_price=100.0,
        stop_loss=90.0,
        target_price=110.0 # R:R = 1.0 < 1.5
    )
    print(f"\nTest 3 - Poor R:R Rejection:")
    print(f"  Valid: {result3.is_valid} (Expected: False)")
    print(f"  Reason: {result3.rejection_reason}")
    assert not result3.is_valid

    # Test Case 4: Daily Loss Kill-Switch (Circuit breaker)
    # 2.5% of 300,000 = Rs 7,500
    engine.record_pnl(-4000.0) # Down 4000
    assert not engine.is_halted
    can_continue = engine.record_pnl(-3600.0) # Cumulative -7600
    assert not can_continue
    assert engine.is_halted
    print(f"\nTest 4 - Kill-Switch Circuit Breaker:")
    print(f"  Halted: {engine.is_halted} (Expected: True)")
    print(f"  Daily PnL: Rs {engine.daily_pnl:.2f}")

    print("\nALL RISK ENGINE SELF-TESTS PASSED!")

    print("\n========================================")
    print("Running Pre-Market Sniping Engine Self-Tests")
    print("========================================")

    from engine.config import PRE_MARKET_CONFIG
    from engine.premarket_sniper import (
        PreMarketSniperEngine,
        SyntheticPreMarketGenerator,
        AuctionQuote
    )

    sniper_engine = PreMarketSniperEngine(config, PRE_MARKET_CONFIG, RiskEngine(config))

    # Test Case 5: 3.5-Sigma Long Dislocation (Reliance panic dip absorbed by institutional bids)
    quote_long, idx_q, h_stats = SyntheticPreMarketGenerator.create_scenario(
        symbol="RELIANCE.NS",
        prev_close=3000.0,
        beta=1.15,
        residual_sigma=0.008,
        index_gap_pct=-0.004,
        sigma_multiple=-3.5,
        adv_20=5_000_000.0,
        volume_pct_adv=0.02,
        imbalance=0.15
    )
    sig_long = sniper_engine.evaluate_auction_dislocation(quote_long, idx_q, h_stats)
    print("Test 5 - 3.5-Sigma Long Dislocation Detection:")
    print(f"  Z-score: {sig_long.z_score:+.2f} (Expected: <= -3.0)")
    print(f"  Actionable: {sig_long.is_actionable} (Expected: True)")
    assert sig_long.is_actionable
    assert sig_long.z_score <= -3.0
    assert sig_long.direction == "LONG"

    orders = sniper_engine.route_sniping_orders([sig_long])
    print(f"  Routed Orders Count: {len(orders)} (Expected: 1)")
    print(f"  Limit Price: Rs {orders[0].limit_price:.2f} (Buffer: 0.2%)")
    print(f"  Allocated Capital: Rs {orders[0].capital_allocated:.2f} <= Rs 75,000")
    print(f"  Max Risk: Rs {orders[0].expected_risk:.2f} <= Rs 3,000")
    assert len(orders) == 1
    assert orders[0].capital_allocated <= 75_000.0
    assert orders[0].expected_risk <= 3_000.0
    assert orders[0].reward_to_risk >= 1.5

    # Test Case 6: Microstructure Circuit Guard Rejection
    quote_lc = AuctionQuote(
        symbol="HDFCBANK.NS",
        timestamp=quote_long.timestamp,
        previous_close=1600.0,
        equilibrium_price=1442.0, # Within 0.15% of LC 1440
        matched_volume=50_000,
        total_buy_qty=10_000,
        total_sell_qty=8_000,
        lower_circuit=1440.0,
        upper_circuit=1760.0,
        is_fno=True
    )
    sig_circuit = sniper_engine.evaluate_auction_dislocation(quote_lc, idx_q, h_stats)
    print("\nTest 6 - Circuit Proximity Guard Rejection:")
    print(f"  Actionable: {sig_circuit.is_actionable} (Expected: False)")
    print(f"  Rejection Reason: {sig_circuit.rejection_reasons[0]}")
    assert not sig_circuit.is_actionable
    assert any("Lower Circuit" in r for r in sig_circuit.rejection_reasons)

    # Test Case 7: Severe Falling Knife Imbalance Rejection (IBR < -0.80)
    quote_fk, _, _ = SyntheticPreMarketGenerator.create_scenario(
        symbol="TATAMOTORS.NS",
        prev_close=1000.0,
        sigma_multiple=-4.0,
        imbalance=-0.95
    )
    sig_fk = sniper_engine.evaluate_auction_dislocation(quote_fk, idx_q, h_stats)
    print("\nTest 7 - Severe Order Book Imbalance Rejection:")
    print(f"  Actionable: {sig_fk.is_actionable} (Expected: False)")
    print(f"  Rejection Reason: {sig_fk.rejection_reasons[0]}")
    assert not sig_fk.is_actionable
    assert any("Severe sell book imbalance" in r for r in sig_fk.rejection_reasons)

    # Test Case 8: Daily Drawdown Kill Switch Block on Order Router
    halted_engine = PreMarketSniperEngine(config, PRE_MARKET_CONFIG, RiskEngine(config))
    halted_engine.risk_engine.record_pnl(-7600.0) # Trip 2.5% daily drawdown kill switch
    halted_orders = halted_engine.route_sniping_orders([sig_long])
    print("\nTest 8 - Kill-Switch Order Routing Interceptor:")
    print(f"  Halted: {halted_engine.risk_engine.is_halted} (Expected: True)")
    print(f"  Orders Routed: {len(halted_orders)} (Expected: 0)")
    assert halted_engine.risk_engine.is_halted
    assert len(halted_orders) == 0

    print("\nALL PRE-MARKET SNIPING ENGINE SELF-TESTS PASSED!")


if __name__ == "__main__":
    run_tests()

