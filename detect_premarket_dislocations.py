#!/usr/bin/env python
"""
NSE Pre-Market Call Auction (9:00-9:08 AM) Dislocation Sniping Engine CLI
Detects 3-sigma statistical opening mispricings, confirms microstructure safety,
and routes sized limit orders for the 9:15:00 AM market open.

Usage:
    python detect_premarket_dislocations.py --simulate
    python detect_premarket_dislocations.py --live
    python detect_premarket_dislocations.py --live --threshold 2.5
    python detect_premarket_dislocations.py --capital 300000
"""

import argparse
import sys
from datetime import datetime
from typing import Dict, List, Optional

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
from tabulate import tabulate

from engine.config import (
    CAPITAL_CONFIG,
    PRE_MARKET_CONFIG,
    UNIVERSE_CONFIG,
    CapitalConfig,
    PreMarketConfig,
)
from engine.risk import RiskEngine
from engine.premarket_sniper import (
    AuctionQuote,
    DislocationSignal,
    HistoricalOvernightStats,
    NSEPreMarketFeedAdapter,
    PreMarketSniperEngine,
    SniperOrder,
    SyntheticPreMarketGenerator,
    compute_overnight_gap_stats,
)


def fetch_historical_ohlcv(symbol: str, period: str = "6mo") -> pd.DataFrame:
    """Fetches historical daily OHLCV from Yahoo Finance with fallback."""
    import yfinance as yf
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval="1d")
        if df is not None and not df.empty and len(df) >= 20:
            df.symbol = symbol
            return df
    except Exception as e:
        pass
    return pd.DataFrame()


def run_live_detection(
    capital: float = 300_000.0,
    z_threshold: float = 3.0,
    max_trades: int = 2
):
    """
    Connects to the official live NSE Pre-Market Call Auction feed,
    evaluates candidates against historical beta/sigma, and generates
    executable 9:15:00 AM orders.
    """
    print("=" * 85)
    print("  NSE LIVE PRE-MARKET CALL AUCTION DISLOCATION SCANNER")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
    print(f"  Capital Budget: Rs {capital:,.2f} | 3-Sigma Threshold: +/-{z_threshold:.1f}")
    print("=" * 85)

    capital_cfg = CapitalConfig(total_capital=capital)
    sniper_cfg = PreMarketConfig(
        z_score_threshold=z_threshold,
        max_concurrent_trades=max_trades
    )
    risk_engine = RiskEngine(capital_cfg)
    engine = PreMarketSniperEngine(capital_cfg, sniper_cfg, risk_engine)

    adapter = NSEPreMarketFeedAdapter()
    print("\n[1/3] Connecting to live NSE Pre-Market clearing feed...")
    try:
        stock_quotes, index_quote = adapter.fetch_live_auction_quotes()
        print(f"      Successfully ingested {len(stock_quotes)} auction quotes from NSE.")
    except Exception as exc:
        print(f"[ERROR] Failed to fetch live NSE feed: {exc}")
        print("Falling back to deterministic simulation mode...")
        run_simulation(capital=capital, z_threshold=z_threshold, max_trades=max_trades)
        return

    # Focus on target universe or top liquid quotes
    universe_symbols = set(UNIVERSE_CONFIG.symbols)
    target_quotes = [q for q in stock_quotes if q.symbol in universe_symbols]
    if not target_quotes:
        # Fallback to top liquid quotes from feed
        target_quotes = sorted(stock_quotes, key=lambda q: q.matched_volume, reverse=True)[:25]

    print(f"\n[2/3] Computing historical overnight gap betas & sigma for {len(target_quotes)} symbols...")
    index_df = fetch_historical_ohlcv("^NSEI", period="6mo")
    if index_df.empty:
        # Fallback synthetic index series if network restricted
        dates = pd.date_range(end=datetime.now(), periods=90, freq='B')
        index_df = pd.DataFrame({
            'Open': np.linspace(23000, 24000, 90),
            'High': np.linspace(23100, 24100, 90),
            'Low': np.linspace(22900, 23900, 90),
            'Close': np.linspace(23050, 24050, 90),
            'Volume': 10_000_000
        }, index=dates)

    signals: List[DislocationSignal] = []
    for q in target_quotes:
        s_df = fetch_historical_ohlcv(q.symbol, period="6mo")
        if s_df.empty:
            # Fallback historical stats based on typical large cap NSE behavior
            hist_stats = HistoricalOvernightStats(
                symbol=q.symbol,
                beta=1.1,
                alpha=0.0,
                residual_sigma=0.012, # 1.2% typical overnight idiosyncratic std
                residual_mean=0.0,
                adv_20=max(float(q.matched_volume * 25), 500_000.0),
                sample_size=60
            )
        else:
            hist_stats = compute_overnight_gap_stats(s_df, index_df, lookback_days=sniper_cfg.lookback_days)

        sig = engine.evaluate_auction_dislocation(q, index_quote, hist_stats)
        signals.append(sig)

    print("\n[3/3] Evaluating statistical dislocations and routing 9:15:00 AM orders...")
    _display_results(signals, engine, risk_engine)


def run_simulation(
    capital: float = 300_000.0,
    z_threshold: float = 3.0,
    max_trades: int = 2
):
    """
    Runs a deterministic simulation of realistic pre-market auction scenarios,
    demonstrating the 3-sigma mathematical filter, circuit guards, order book
    imbalance rejection, and 9:15:00 AM order routing.
    """
    print("=" * 85)
    print("  NSE PRE-MARKET DISLOCATION SNIPING ENGINE: DETERMINISTIC SIMULATION")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
    print(f"  Capital Budget: Rs {capital:,.2f} | 3-Sigma Threshold: +/-{z_threshold:.1f}")
    print("=" * 85)

    capital_cfg = CapitalConfig(total_capital=capital)
    sniper_cfg = PreMarketConfig(
        z_score_threshold=z_threshold,
        max_concurrent_trades=max_trades
    )
    risk_engine = RiskEngine(capital_cfg)
    engine = PreMarketSniperEngine(capital_cfg, sniper_cfg, risk_engine)

    # Scenarios designed to test all branches of the mathematical engine
    scenarios = [
        # Scenario 1: Clean 3.5-sigma Long Dislocation on RELIANCE (undervalued panic, buyer depth confirms)
        {
            "symbol": "RELIANCE.NS",
            "prev_close": 3000.0,
            "beta": 1.15,
            "residual_sigma": 0.008,
            "index_gap_pct": -0.004,     # Index down -0.4%
            "sigma_multiple": -3.5,      # 3.5-sigma gap down
            "adv_20": 4_500_000.0,
            "volume_pct_adv": 0.025,     # 2.5% of ADV matched
            "imbalance": 0.15            # Buyers absorbing
        },
        # Scenario 2: Clean 3.2-sigma Short Dislocation on TCS (overbought euphoria, supply emerges)
        {
            "symbol": "TCS.NS",
            "prev_close": 4200.0,
            "beta": 0.90,
            "residual_sigma": 0.007,
            "index_gap_pct": +0.003,     # Index up +0.3%
            "sigma_multiple": +3.2,      # 3.2-sigma gap up
            "adv_20": 2_200_000.0,
            "volume_pct_adv": 0.018,     # 1.8% of ADV matched
            "imbalance": -0.10           # Sellers offering
        },
        # Scenario 3: 3.6-sigma Gap Down on HDFCBANK BUT within 0.3% of Lower Circuit -> REJECTED (Circuit lock)
        {
            "symbol": "HDFCBANK.NS",
            "prev_close": 1600.0,
            "beta": 1.20,
            "residual_sigma": 0.009,
            "index_gap_pct": -0.005,
            "sigma_multiple": -3.6,
            "adv_20": 8_000_000.0,
            "volume_pct_adv": 0.030,
            "imbalance": 0.05,
            "circuit_override": (1450.0, 1760.0) # Lower circuit near 1450
        },
        # Scenario 4: 3.8-sigma Gap Down on INFY BUT Illiquid (AVR = 0.05% of ADV) -> REJECTED (Phantom quote)
        {
            "symbol": "INFY.NS",
            "prev_close": 1800.0,
            "beta": 1.05,
            "residual_sigma": 0.008,
            "index_gap_pct": -0.002,
            "sigma_multiple": -3.8,
            "adv_20": 5_000_000.0,
            "volume_pct_adv": 0.0005,    # Only 0.05% of ADV (very illiquid auction)
            "imbalance": 0.20
        },
        # Scenario 5: 4.0-sigma Gap Down on TATAMOTORS BUT Extreme Sell Imbalance (IBR = -0.92) -> REJECTED (Falling knife)
        {
            "symbol": "TATAMOTORS.NS",
            "prev_close": 1000.0,
            "beta": 1.40,
            "residual_sigma": 0.012,
            "index_gap_pct": -0.006,
            "sigma_multiple": -4.0,
            "adv_20": 6_000_000.0,
            "volume_pct_adv": 0.035,
            "imbalance": -0.92           # Massive sell wall / circuit dump
        },
        # Scenario 6: Normal Market Fluctuation on ICICIBANK (1.2-sigma gap) -> REJECTED (Below 3-sigma)
        {
            "symbol": "ICICIBANK.NS",
            "prev_close": 1250.0,
            "beta": 1.10,
            "residual_sigma": 0.009,
            "index_gap_pct": -0.003,
            "sigma_multiple": -1.2,      # Only 1.2-sigma
            "adv_20": 7_000_000.0,
            "volume_pct_adv": 0.015,
            "imbalance": 0.02
        }
    ]

    signals: List[DislocationSignal] = []

    for sc in scenarios:
        q, idx_q, h_stats = SyntheticPreMarketGenerator.create_scenario(
            symbol=sc["symbol"],
            prev_close=sc["prev_close"],
            beta=sc["beta"],
            residual_sigma=sc["residual_sigma"],
            index_gap_pct=sc["index_gap_pct"],
            sigma_multiple=sc["sigma_multiple"],
            adv_20=sc["adv_20"],
            volume_pct_adv=sc["volume_pct_adv"],
            imbalance=sc["imbalance"]
        )
        if "circuit_override" in sc:
            # Force circuit proximity to test filter
            q = AuctionQuote(
                symbol=q.symbol,
                timestamp=q.timestamp,
                previous_close=q.previous_close,
                equilibrium_price=sc["circuit_override"][0] * 1.002, # 0.2% above LC
                matched_volume=q.matched_volume,
                total_buy_qty=q.total_buy_qty,
                total_sell_qty=q.total_sell_qty,
                lower_circuit=sc["circuit_override"][0],
                upper_circuit=sc["circuit_override"][1],
                is_fno=True
            )

        sig = engine.evaluate_auction_dislocation(q, idx_q, h_stats)
        signals.append(sig)

    _display_results(signals, engine, risk_engine)


def _display_results(
    signals: List[DislocationSignal],
    engine: PreMarketSniperEngine,
    risk_engine: RiskEngine
):
    """Prints beautiful quantitative tables of signals and routed orders."""
    # 1. Auction Scan Summary Table
    scan_headers = [
        "Symbol", "Prev Close", "9:08 IEP", "Stock Gap", "Index Gap",
        "Beta", "Exp Gap", "Res Gap", "Sigma", "Z-Score", "AVR %", "IBR", "Status"
    ]
    scan_rows = []
    for s in signals:
        status = "[*] ACTIONABLE" if s.is_actionable else f"[X] REJECTED ({s.rejection_reasons[0][:28]}...)"
        scan_rows.append([
            s.symbol.replace(".NS", ""),
            f"{s.previous_close:,.1f}",
            f"{s.equilibrium_price:,.1f}",
            f"{s.stock_gap_pct:+.2%}",
            f"{s.index_gap_pct:+.2%}",
            f"{s.beta:.2f}",
            f"{s.expected_gap_pct:+.2%}",
            f"{s.residual_gap:+.2%}",
            f"{s.residual_sigma:.2%}",
            f"{s.z_score:+.2f}",
            f"{s.volume_ratio:.2%}",
            f"{s.imbalance_ratio:+.2f}",
            status
        ])

    print("\n" + "=" * 85)
    print("  TABLE 1: 9:08 AM CALL AUCTION DISLOCATION SCANNER RESULTS")
    print("=" * 85)
    print(tabulate(scan_rows, headers=scan_headers, tablefmt="grid"))

    # 2. Route Orders for 9:15:00 AM
    orders = engine.route_sniping_orders(signals)

    print("\n" + "=" * 85)
    print("  TABLE 2: 9:15:00 AM PRECISION LIMIT ORDER ROUTING SCHEDULE")
    print("=" * 85)

    if not orders:
        print("  [!] No actionable orders qualified for 9:15:00 AM routing.")
        return

    order_headers = [
        "Symbol", "Action", "Type", "Qty", "Limit Price",
        "Stop Loss", "Target", "R:R", "Allocated Capital", "Max Risk (Rs)", "Timeout"
    ]
    order_rows = []
    total_alloc = 0.0
    total_risk = 0.0

    for o in orders:
        total_alloc += o.capital_allocated
        total_risk += o.expected_risk
        order_rows.append([
            o.symbol.replace(".NS", ""),
            o.action,
            o.order_type,
            o.quantity,
            f"Rs {o.limit_price:,.2f}",
            f"Rs {o.stop_loss:,.2f}",
            f"Rs {o.target_price:,.2f}",
            f"{o.reward_to_risk:.2f}:1",
            f"Rs {o.capital_allocated:,.2f}",
            f"Rs {o.expected_risk:,.2f}",
            f"{o.timeout_seconds:.1f}s"
        ])

    print(tabulate(order_rows, headers=order_headers, tablefmt="grid"))

    # 3. Capital & Risk Allocation Metrics
    total_cap = engine.capital_cfg.total_capital
    remaining_cap = total_cap - total_alloc
    heat_pct = total_risk / total_cap

    print("\n" + "-" * 85)
    print("  PORTFOLIO CAPITAL & RISK BUDGET AUDIT")
    print("-" * 85)
    print(f"  * Total Portfolio Capital     : Rs {total_cap:,.2f}")
    print(f"  * Total Capital Allocated     : Rs {total_alloc:,.2f} ({total_alloc / total_cap:.1%}) [Limit: <= 50% for 2 trades]")
    print(f"  * Remaining Unencumbered Cash : Rs {remaining_cap:,.2f}")
    print(f"  * Total Committed Risk (Heat) : Rs {total_risk:,.2f} ({heat_pct:.2%}) [Limit: <= 4.0%]")
    print(f"  * Daily Loss Kill-Switch Level: Rs {total_cap * engine.capital_cfg.daily_drawdown_limit_pct:,.2f} (2.5% max daily drawdown)")
    print(f"  * Execution Window            : Exactly 09:15:00.050 AM IST")
    print(f"  * Protective Hard Time-Stop   : 09:45:00 AM IST (Cancel & Close all open positions)")
    print("-" * 85)


def main():
    parser = argparse.ArgumentParser(
        description="NSE Pre-Market Call Auction Dislocation Sniping Engine (9:00-9:08 AM)"
    )
    parser.add_argument(
        "--live", action="store_true", help="Connect to live NSE Pre-Open feed"
    )
    parser.add_argument(
        "--simulate", action="store_true", help="Run deterministic test scenarios"
    )
    parser.add_argument(
        "--capital", type=float, default=300_000.0, help="Trading capital in INR (default: 300,000)"
    )
    parser.add_argument(
        "--threshold", type=float, default=3.0, help="Z-score threshold for dislocation (default: 3.0)"
    )
    parser.add_argument(
        "--max-trades", type=int, default=2, help="Max simultaneous trades (default: 2)"
    )

    args = parser.parse_args()

    if args.live:
        run_live_detection(
            capital=args.capital,
            z_threshold=args.threshold,
            max_trades=args.max_trades
        )
    else:
        # Default to simulation if no flag or --simulate passed
        run_simulation(
            capital=args.capital,
            z_threshold=args.threshold,
            max_trades=args.max_trades
        )


if __name__ == "__main__":
    main()
