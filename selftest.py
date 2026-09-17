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

if __name__ == "__main__":
    run_tests()
