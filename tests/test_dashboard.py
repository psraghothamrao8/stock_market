"""
Unit and Integration Tests for Pre-Market Dislocation Dashboard Generator
Validates automated HTML rendering, 3-sigma candidate aggregation,
and robust cloud-runner geo-blocking fallback behavior.
"""

import os
import tempfile
from unittest.mock import patch
import pytest
import requests

from generate_dashboard import (
    build_simulation_scenarios,
    get_current_timestamps,
    render_html_dashboard,
    run_live_pipeline,
    run_scanner,
    run_simulation_pipeline,
    write_dashboard_file,
)


def test_get_current_timestamps():
    ist_str, utc_str = get_current_timestamps()
    assert "IST" in ist_str
    assert "UTC" in utc_str


def test_build_simulation_scenarios():
    scenarios = build_simulation_scenarios()
    assert len(scenarios) >= 6
    symbols = [s["symbol"] for s in scenarios]
    assert "RELIANCE.NS" in symbols
    assert "TCS.NS" in symbols
    assert "HDFCBANK.NS" in symbols
    assert "INFY.NS" in symbols
    assert "TATAMOTORS.NS" in symbols
    assert "ICICIBANK.NS" in symbols


def test_run_simulation_pipeline():
    signals, orders, engine, risk_engine = run_simulation_pipeline(
        capital=300_000.0, z_threshold=3.0, max_trades=2
    )
    assert len(signals) >= 6
    assert len(orders) == 2  # RELIANCE Long and TCS Short

    reliance_order = next((o for o in orders if "RELIANCE" in o.symbol), None)
    tcs_order = next((o for o in orders if "TCS" in o.symbol), None)

    assert reliance_order is not None
    assert reliance_order.action == "BUY"
    assert reliance_order.capital_allocated <= 75_000.0
    assert reliance_order.expected_risk <= 3_000.0
    assert reliance_order.reward_to_risk >= 1.5

    assert tcs_order is not None
    assert tcs_order.action == "SELL"
    assert tcs_order.capital_allocated <= 75_000.0
    assert tcs_order.expected_risk <= 3_000.0
    assert tcs_order.reward_to_risk >= 1.5

    # Check tick compliance on all orders
    for o in orders:
        for p in [o.limit_price, o.stop_loss, o.target_price]:
            rem = round(p * 20) - (p * 20)
            assert abs(rem) < 1e-4, f"Price {p} violates NSE 0.05 tick size"

    # Check rejection cases
    hdfc_sig = next((s for s in signals if "HDFCBANK" in s.symbol), None)
    assert hdfc_sig is not None
    assert not hdfc_sig.is_actionable
    assert any("Circuit" in r for r in hdfc_sig.rejection_reasons)

    infy_sig = next((s for s in signals if "INFY" in s.symbol), None)
    assert infy_sig is not None
    assert not infy_sig.is_actionable
    assert any("Auction volume" in r or "Phantom quote" in r for r in infy_sig.rejection_reasons)

    tata_sig = next((s for s in signals if "TATAMOTORS" in s.symbol), None)
    assert tata_sig is not None
    assert not tata_sig.is_actionable
    assert any("imbalance" in r for r in tata_sig.rejection_reasons)


def test_run_scanner_simulate_mode():
    data = run_scanner(mode="simulate", capital=300_000.0, z_threshold=3.0, max_trades=2)
    assert data["is_live"] is False
    assert "Simulation" in data["data_source"]
    assert len(data["signals"]) >= 6
    assert len(data["orders"]) == 2
    assert data["total_capital"] == 300_000.0
    assert data["allocated_capital"] > 0
    assert data["total_risk"] <= 6_000.0
    assert data["daily_loss_limit"] == 7_500.0


def test_run_scanner_auto_fallback_on_network_error():
    """Simulates foreign cloud runner encountering geo-blocking or network failure."""
    with patch(
        "engine.premarket_sniper.NSEPreMarketFeedAdapter.fetch_live_auction_quotes",
        side_effect=requests.exceptions.ConnectionError("Foreign IP blocked by Akamai WAF")
    ):
        data = run_scanner(mode="auto", capital=300_000.0, z_threshold=3.0, max_trades=2)
        assert data["is_live"] is False
        assert data["fallback_reason"] is not None
        assert "Foreign IP blocked" in data["fallback_reason"]
        assert len(data["signals"]) >= 6
        assert len(data["orders"]) == 2


def test_run_scanner_auto_fallback_on_empty_quotes():
    """Simulates market closed or empty feed returned by NSE API."""
    with patch(
        "engine.premarket_sniper.NSEPreMarketFeedAdapter.fetch_live_auction_quotes",
        return_value=([], None)
    ):
        data = run_scanner(mode="auto", capital=300_000.0, z_threshold=3.0, max_trades=2)
        assert data["is_live"] is False
        assert data["fallback_reason"] is not None
        assert "zero auction quotes" in data["fallback_reason"]
        assert len(data["signals"]) >= 6


def test_render_html_dashboard_with_orders():
    data = run_scanner(mode="simulate", capital=300_000.0, z_threshold=3.0, max_trades=2)
    html = render_html_dashboard(data)

    assert "<!DOCTYPE html>" in html
    assert "NSE Pre-Market 3-Sigma Sniper" in html
    assert "RELIANCE" in html
    assert "TCS" in html
    assert "HDFCBANK" in html
    assert "STAGED (09:15)" in html
    assert "09:15:00 AM Precision Execution Schedule" in html
    assert "09:08 AM Call Auction Complete Dislocation Scan" in html
    assert "Quantitative Edge & Microstructure Safety Architecture" in html
    assert "filterTable" in html
    assert "setFilter" in html


def test_render_html_dashboard_empty_orders():
    # Construct data with zero orders
    data = run_scanner(mode="simulate", capital=300_000.0, z_threshold=10.0, max_trades=2)
    # Threshold 10.0 guarantees zero actionable orders
    data["orders"] = []
    data["actionable_signals"] = []
    data["allocated_capital"] = 0.0
    data["total_risk"] = 0.0

    html = render_html_dashboard(data)
    assert "<!DOCTYPE html>" in html
    assert "Capital Preserved" in html
    assert "No Actionable Sniping Orders Staged" in html


def test_write_dashboard_file(tmp_path):
    out_file = tmp_path / "subdir" / "index.html"
    content = "<html><body><h1>Test Dashboard</h1></body></html>"
    write_dashboard_file(content, str(out_file))

    assert out_file.exists()
    assert out_file.read_text(encoding="utf-8") == content
