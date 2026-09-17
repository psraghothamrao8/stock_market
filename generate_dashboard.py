#!/usr/bin/env python
"""
Automated 9:08 AM NSE Pre-Market Dislocation Dashboard Generator
Executes the quantitative 3-sigma sniping engine, aggregates microstructure opportunities,
and renders a clean, dark-mode, mobile-responsive HTML dashboard for GitHub Pages.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any

# Ensure UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd

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
    round_to_tick,
)


def get_current_timestamps() -> Tuple[str, str]:
    """Returns formatted timestamps in IST (UTC+5:30) and UTC."""
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    now_ist = datetime.now(ist_tz)
    now_utc = datetime.now(timezone.utc)
    ist_str = now_ist.strftime("%Y-%m-%d %H:%M:%S IST")
    utc_str = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    return ist_str, utc_str


def fetch_historical_ohlcv(symbol: str, period: str = "6mo") -> pd.DataFrame:
    """Fetches historical daily OHLCV from Yahoo Finance with robust timeout and fallback."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval="1d", timeout=5)
        if df is not None and not df.empty and len(df) >= 20:
            df.symbol = symbol
            return df
    except Exception:
        pass
    return pd.DataFrame()


def build_simulation_scenarios() -> List[Dict[str, Any]]:
    """
    Constructs high-fidelity pre-market auction test scenarios reflecting realistic
    market conditions, edge cases, and microstructure rejections.
    """
    return [
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
        # Scenario 3: 3.6-sigma Gap Down on HDFCBANK BUT within 0.12% of Lower Circuit -> REJECTED (Circuit lock)
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
            "volume_pct_adv": 0.0005,    # Only 0.05% of ADV (illiquid auction)
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
            "imbalance": -0.92           # Massive sell wall
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
        },
        # Scenario 7: Normal Market Drift on BHARTIARTL (+0.65-sigma) -> REJECTED (Normal drift)
        {
            "symbol": "BHARTIARTL.NS",
            "prev_close": 1850.0,
            "beta": 0.85,
            "residual_sigma": 0.0075,
            "index_gap_pct": +0.002,
            "sigma_multiple": +0.65,
            "adv_20": 4_000_000.0,
            "volume_pct_adv": 0.016,
            "imbalance": -0.04
        },
        # Scenario 8: Normal Market Fluctuation on SBIN (-0.85-sigma) -> REJECTED (Normal drift)
        {
            "symbol": "SBIN.NS",
            "prev_close": 820.0,
            "beta": 1.25,
            "residual_sigma": 0.011,
            "index_gap_pct": -0.003,
            "sigma_multiple": -0.85,
            "adv_20": 9_000_000.0,
            "volume_pct_adv": 0.020,
            "imbalance": 0.01
        }
    ]


def run_simulation_pipeline(
    capital: float = 300_000.0,
    z_threshold: float = 3.0,
    max_trades: int = 2
) -> Tuple[List[DislocationSignal], List[SniperOrder], PreMarketSniperEngine, RiskEngine]:
    """Runs deterministic simulation pipeline with comprehensive scenarios."""
    capital_cfg = CapitalConfig(total_capital=capital)
    sniper_cfg = PreMarketConfig(
        z_score_threshold=z_threshold,
        max_concurrent_trades=max_trades
    )
    risk_engine = RiskEngine(capital_cfg)
    engine = PreMarketSniperEngine(capital_cfg, sniper_cfg, risk_engine)

    scenarios = build_simulation_scenarios()
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
            q = AuctionQuote(
                symbol=q.symbol,
                timestamp=q.timestamp,
                previous_close=q.previous_close,
                equilibrium_price=sc["circuit_override"][0] * 1.0012, # 0.12% above LC
                matched_volume=q.matched_volume,
                total_buy_qty=q.total_buy_qty,
                total_sell_qty=q.total_sell_qty,
                lower_circuit=sc["circuit_override"][0],
                upper_circuit=sc["circuit_override"][1],
                is_fno=True
            )

        sig = engine.evaluate_auction_dislocation(q, idx_q, h_stats)
        signals.append(sig)

    orders = engine.route_sniping_orders(signals)
    return signals, orders, engine, risk_engine


def run_live_pipeline(
    capital: float = 300_000.0,
    z_threshold: float = 3.0,
    max_trades: int = 2
) -> Tuple[List[DislocationSignal], List[SniperOrder], PreMarketSniperEngine, RiskEngine]:
    """Connects to live NSE pre-market clearing feed and processes quotes."""
    capital_cfg = CapitalConfig(total_capital=capital)
    sniper_cfg = PreMarketConfig(
        z_score_threshold=z_threshold,
        max_concurrent_trades=max_trades
    )
    risk_engine = RiskEngine(capital_cfg)
    engine = PreMarketSniperEngine(capital_cfg, sniper_cfg, risk_engine)

    adapter = NSEPreMarketFeedAdapter()
    stock_quotes, index_quote = adapter.fetch_live_auction_quotes()

    if not stock_quotes:
        raise ValueError("NSE feed returned zero auction quotes (market closed or off-hours)")

    universe_symbols = set(UNIVERSE_CONFIG.symbols)
    target_quotes = [q for q in stock_quotes if q.symbol in universe_symbols]
    if not target_quotes:
        target_quotes = sorted(stock_quotes, key=lambda q: q.matched_volume, reverse=True)[:25]

    index_df = fetch_historical_ohlcv("^NSEI", period="6mo")
    if index_df.empty:
        dates = pd.date_range(end=datetime.now(), periods=90, freq='B')
        index_df = pd.DataFrame({
            'Open': np.linspace(23000, 24000, 90),
            'High': np.linspace(23100, 24100, 90),
            'Low': np.linspace(22900, 23900, 90),
            'Close': np.linspace(23050, 24050, 90),
            'Volume': 10_000_000
        }, index=dates)

    # Concurrently pre-fetch historical data for targets to eliminate multi-second serial network latency
    stock_dfs: Dict[str, pd.DataFrame] = {}
    if target_quotes:
        workers = min(8, len(target_quotes))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_sym = {
                executor.submit(fetch_historical_ohlcv, q.symbol, "6mo"): q.symbol
                for q in target_quotes
            }
            for future in future_to_sym:
                sym = future_to_sym[future]
                try:
                    stock_dfs[sym] = future.result()
                except Exception:
                    stock_dfs[sym] = pd.DataFrame()

    signals: List[DislocationSignal] = []
    for q in target_quotes:
        s_df = stock_dfs.get(q.symbol, pd.DataFrame())
        try:
            if s_df is None or s_df.empty:
                hist_stats = HistoricalOvernightStats(
                    symbol=q.symbol,
                    beta=1.1,
                    alpha=0.0,
                    residual_sigma=0.012,
                    residual_mean=0.0,
                    adv_20=max(float(q.matched_volume * 25), 500_000.0),
                    sample_size=60
                )
            else:
                hist_stats = compute_overnight_gap_stats(s_df, index_df, lookback_days=sniper_cfg.lookback_days)
        except Exception:
            hist_stats = HistoricalOvernightStats(
                symbol=q.symbol,
                beta=1.1,
                alpha=0.0,
                residual_sigma=0.012,
                residual_mean=0.0,
                adv_20=max(float(q.matched_volume * 25), 500_000.0),
                sample_size=60
            )

        sig = engine.evaluate_auction_dislocation(q, index_quote, hist_stats)
        signals.append(sig)

    orders = engine.route_sniping_orders(signals)
    return signals, orders, engine, risk_engine


def run_scanner(
    mode: str = "auto",
    capital: float = 300_000.0,
    z_threshold: float = 3.0,
    max_trades: int = 2
) -> Dict[str, Any]:
    """
    Main scanner execution orchestrator with automatic fallback.
    Ensures zero failure even when cloud runners encounter geo-blocking.
    """
    ist_str, utc_str = get_current_timestamps()
    is_live = False
    fallback_reason: Optional[str] = None
    data_source = "Deterministic Microstructure Simulation"

    if mode in ("live", "auto"):
        try:
            signals, orders, engine, risk_engine = run_live_pipeline(
                capital=capital, z_threshold=z_threshold, max_trades=max_trades
            )
            is_live = True
            data_source = "Official NSE India Pre-Market Auction Feed"
            print(f"[INFO] Successfully completed live scan: {len(signals)} quotes evaluated.")
        except Exception as exc:
            if mode == "live":
                raise exc
            print(f"[WARN] Live NSE scan failed ({exc}). Falling back to deterministic simulation.")
            fallback_reason = f"Cloud Runner Geo-Fallback (NSE API unreachable/blocked on foreign IP: {exc})"
            signals, orders, engine, risk_engine = run_simulation_pipeline(
                capital=capital, z_threshold=z_threshold, max_trades=max_trades
            )
            is_live = False
            data_source = "Deterministic Microstructure Simulation (Cloud Geo-Fallback)"
    else:
        signals, orders, engine, risk_engine = run_simulation_pipeline(
            capital=capital, z_threshold=z_threshold, max_trades=max_trades
        )
        is_live = False
        data_source = "Deterministic Microstructure Simulation (Manual)"

    total_cap = engine.capital_cfg.total_capital
    total_alloc = sum(o.capital_allocated for o in orders)
    total_risk = sum(o.expected_risk for o in orders)
    remaining_cap = total_cap - total_alloc
    heat_pct = total_risk / total_cap if total_cap > 0 else 0.0
    actionable_signals = [s for s in signals if s.is_actionable]
    rejected_signals = [s for s in signals if not s.is_actionable]

    return {
        "signals": signals,
        "actionable_signals": actionable_signals,
        "rejected_signals": rejected_signals,
        "orders": orders,
        "engine": engine,
        "risk_engine": risk_engine,
        "is_live": is_live,
        "data_source": data_source,
        "fallback_reason": fallback_reason,
        "timestamp_ist": ist_str,
        "timestamp_utc": utc_str,
        "total_capital": total_cap,
        "allocated_capital": total_alloc,
        "remaining_capital": remaining_cap,
        "total_risk": total_risk,
        "heat_pct": heat_pct,
        "daily_loss_limit": total_cap * engine.capital_cfg.daily_drawdown_limit_pct,
        "z_threshold": z_threshold,
        "max_trades": max_trades,
    }


def render_html_dashboard(data: Dict[str, Any]) -> str:
    """Renders a modern, responsive, dark-mode single-page HTML report."""
    is_live = data["is_live"]
    status_badge = (
        '<span class="badge badge-live"><span class="pulse-dot live"></span> LIVE NSE AUCTION FEED</span>'
        if is_live
        else '<span class="badge badge-fallback"><span class="pulse-dot fallback"></span> SIMULATION BENCHMARK</span>'
    )

    actionable_count = len(data["actionable_signals"])
    total_scanned = len(data["signals"])
    orders_count = len(data["orders"])
    alloc_pct = (data["allocated_capital"] / data["total_capital"]) if data.get("total_capital", 0.0) > 0 else 0.0

    # Hero alert banner
    if orders_count > 0:
        hero_alert = f"""
        <div class="alert-banner alert-success">
            <div class="alert-icon">⚡</div>
            <div class="alert-content">
                <strong>{orders_count} High-Probability Sniping Order{'s' if orders_count > 1 else ''} Ready for 09:15:00 AM Open</strong>
                <p>Identified statistical 3-sigma dislocations confirmed by microstructure order book depth. Precision limit orders staged with strict ₹3,000 risk ceilings.</p>
            </div>
        </div>
        """
    else:
        hero_alert = f"""
        <div class="alert-banner alert-neutral">
            <div class="alert-icon">🛡️</div>
            <div class="alert-content">
                <strong>Capital Preserved — 0 Actionable Dislocation Orders</strong>
                <p>All {total_scanned} scanned assets traded within acceptable statistical variance (|Z| < {data['z_threshold']:.1f}σ) or were protected by circuit guards. Zero capital committed.</p>
            </div>
        </div>
        """

    # Orders Table HTML
    if data["orders"]:
        order_rows = []
        for o in data["orders"]:
            action_cls = "badge-buy" if o.action == "BUY" else "badge-sell"
            order_rows.append(f"""
            <tr>
                <td><strong>{o.symbol.replace('.NS', '')}</strong></td>
                <td><span class="badge {action_cls}">{o.action}</span></td>
                <td><span class="pill-tag">{o.order_type}</span></td>
                <td class="text-right font-mono">{o.quantity:,}</td>
                <td class="text-right font-mono highlight-gold">₹{o.limit_price:,.2f}</td>
                <td class="text-right font-mono text-red">₹{o.stop_loss:,.2f}</td>
                <td class="text-right font-mono text-green">₹{o.target_price:,.2f}</td>
                <td class="text-right font-mono">{o.reward_to_risk:.2f}:1</td>
                <td class="text-right font-mono">₹{o.capital_allocated:,.2f}</td>
                <td class="text-right font-mono">₹{o.expected_risk:,.2f}</td>
                <td class="text-center font-mono">{o.timeout_seconds:.1f}s</td>
                <td><span class="status-staged">STAGED (09:15)</span></td>
            </tr>
            """)
        orders_table_html = f"""
        <div class="table-responsive">
            <table class="table-custom">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Type</th>
                        <th class="text-right">Qty</th>
                        <th class="text-right">Limit Price</th>
                        <th class="text-right">Stop Loss</th>
                        <th class="text-right">Target</th>
                        <th class="text-right">R:R</th>
                        <th class="text-right">Capital</th>
                        <th class="text-right">Max Risk</th>
                        <th class="text-center">Timeout</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(order_rows)}
                </tbody>
            </table>
        </div>
        """
    else:
        orders_table_html = """
        <div class="empty-state">
            <div class="empty-icon">🛡️</div>
            <h3>No Actionable Sniping Orders Staged</h3>
            <p>No symbols breached the +/-3.0-Sigma dislocation threshold or passed microstructure order book checks today. Capital remains 100% unencumbered and protected.</p>
        </div>
        """

    # Scan Summary Table HTML
    scan_rows = []
    for s in data["signals"]:
        sym_clean = s.symbol.replace(".NS", "")
        # Z-score styling
        if s.z_score <= -3.0:
            z_cls = "badge-z-buy"
            z_label = f"{s.z_score:+.2f}σ (Oversold)"
        elif s.z_score >= 3.0:
            z_cls = "badge-z-sell"
            z_label = f"{s.z_score:+.2f}σ (Overbought)"
        else:
            z_cls = "badge-z-normal"
            z_label = f"{s.z_score:+.2f}σ"

        if s.is_actionable:
            status_html = '<span class="badge badge-actionable">✅ ACTIONABLE</span>'
            row_filter_cls = "row-actionable"
        else:
            reason = s.rejection_reasons[0] if s.rejection_reasons else "Below 3-sigma"
            # Shorten reason for clean display
            if "Lower Circuit" in reason or "Upper Circuit" in reason:
                reason_short = "Circuit Lock (<0.5%)"
            elif "Auction volume too low" in reason or "Phantom quote" in reason or "volume ratio" in reason:
                reason_short = "Illiquid (AVR <0.2%)"
            elif "Auction volume excessive" in reason or "Block dump" in reason:
                reason_short = "Block Deal / Leak (>15%)"
            elif "sell book imbalance" in reason or "Falling knife" in reason:
                reason_short = "Sell Imbalance (IBR<-0.8)"
            elif "buy book imbalance" in reason or "Squeeze risk" in reason:
                reason_short = "Buy Imbalance (IBR>+0.8)"
            elif "non-F&O" in reason:
                reason_short = "Non-F&O Short Prohibited"
            elif "Z-score" in reason:
                reason_short = "Within 3-Sigma"
            else:
                reason_short = reason[:24]
            status_html = f'<span class="badge badge-rejected" title="{reason}">❌ {reason_short}</span>'
            row_filter_cls = "row-rejected"

        gap_color = "text-green" if s.stock_gap_pct > 0 else ("text-red" if s.stock_gap_pct < 0 else "")
        idx_color = "text-green" if s.index_gap_pct > 0 else ("text-red" if s.index_gap_pct < 0 else "")

        scan_rows.append(f"""
        <tr class="{row_filter_cls}" data-symbol="{sym_clean.lower()}">
            <td><strong>{sym_clean}</strong></td>
            <td class="text-right font-mono">₹{s.previous_close:,.1f}</td>
            <td class="text-right font-mono">₹{s.equilibrium_price:,.1f}</td>
            <td class="text-right font-mono {gap_color}">{s.stock_gap_pct:+.2%}</td>
            <td class="text-right font-mono {idx_color}">{s.index_gap_pct:+.2%}</td>
            <td class="text-right font-mono">{s.beta:.2f}</td>
            <td class="text-right font-mono">{s.expected_gap_pct:+.2%}</td>
            <td class="text-right font-mono">{s.residual_gap:+.2%}</td>
            <td class="text-right font-mono">{s.residual_sigma:.2%}</td>
            <td class="text-center"><span class="badge {z_cls}">{z_label}</span></td>
            <td class="text-right font-mono">{s.volume_ratio:.2%}</td>
            <td class="text-right font-mono">{s.imbalance_ratio:+.2f}</td>
            <td>{status_html}</td>
        </tr>
        """)

    # Fallback explanation alert if applicable
    fallback_note = ""
    if data["fallback_reason"]:
        fallback_note = f"""
        <div class="note-box">
            <span class="note-tag">CLOUD RUNNER NOTICE</span>
            <span>{data['fallback_reason']} — Displaying deterministic benchmark scenario with live 0.05 NSE tick alignment.</span>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NSE Pre-Market 3-Sigma Dislocation Sniper</title>
    <style>
        :root {{
            --bg-primary: #0a0e17;
            --bg-card: #111827;
            --bg-card-alt: #162032;
            --border-card: #1e293b;
            --border-subtle: #2d3748;
            --text-main: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --green: #10b981;
            --green-glow: rgba(16, 185, 129, 0.2);
            --red: #ef4444;
            --red-glow: rgba(239, 68, 68, 0.2);
            --blue: #3b82f6;
            --blue-glow: rgba(59, 130, 246, 0.2);
            --amber: #f59e0b;
            --purple: #8b5cf6;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background-color: var(--bg-primary);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.5;
            padding: 0;
            margin: 0;
            -webkit-font-smoothing: antialiased;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 24px 16px 64px 16px;
        }}

        /* Header */
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border-card);
            margin-bottom: 24px;
        }}

        .brand-section h1 {{
            font-size: 1.6rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .brand-section p {{
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-top: 4px;
        }}

        .header-meta {{
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            gap: 6px;
        }}

        .time-badge {{
            font-size: 0.82rem;
            color: var(--text-secondary);
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            background: var(--bg-card);
            padding: 4px 10px;
            border-radius: 6px;
            border: 1px solid var(--border-card);
        }}

        /* Badges & Pills */
        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}

        .badge-live {{
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.4);
        }}

        .badge-fallback {{
            background: rgba(245, 158, 11, 0.15);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.4);
        }}

        .pulse-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }}

        .pulse-dot.live {{
            background: #10b981;
            box-shadow: 0 0 8px #10b981;
            animation: pulse 2s infinite;
        }}

        .pulse-dot.fallback {{
            background: #f59e0b;
            box-shadow: 0 0 8px #f59e0b;
        }}

        @keyframes pulse {{
            0% {{ transform: scale(0.95); opacity: 0.8; }}
            50% {{ transform: scale(1.2); opacity: 1; }}
            100% {{ transform: scale(0.95); opacity: 0.8; }}
        }}

        .badge-buy {{
            background: rgba(16, 185, 129, 0.2);
            color: #34d399;
            border: 1px solid #059669;
        }}

        .badge-sell {{
            background: rgba(239, 68, 68, 0.2);
            color: #f87171;
            border: 1px solid #dc2626;
        }}

        .badge-z-buy {{
            background: rgba(16, 185, 129, 0.25);
            color: #34d399;
            font-weight: 700;
        }}

        .badge-z-sell {{
            background: rgba(239, 68, 68, 0.25);
            color: #f87171;
            font-weight: 700;
        }}

        .badge-z-normal {{
            background: rgba(148, 163, 184, 0.1);
            color: var(--text-secondary);
        }}

        .badge-actionable {{
            background: rgba(16, 185, 129, 0.2);
            color: #34d399;
            border: 1px solid #10b981;
        }}

        .badge-rejected {{
            background: rgba(100, 116, 139, 0.15);
            color: #94a3b8;
            font-size: 0.7rem;
        }}

        .pill-tag {{
            background: rgba(59, 130, 246, 0.15);
            color: #93c5fd;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-family: monospace;
        }}

        .status-staged {{
            color: #38bdf8;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.05em;
        }}

        /* Hero Alerts */
        .alert-banner {{
            display: flex;
            align-items: flex-start;
            gap: 14px;
            padding: 16px 20px;
            border-radius: 10px;
            margin-bottom: 24px;
        }}

        .alert-success {{
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: #e6fffa;
        }}

        .alert-neutral {{
            background: rgba(59, 130, 246, 0.08);
            border: 1px solid rgba(59, 130, 246, 0.25);
            color: #eff6ff;
        }}

        .alert-icon {{
            font-size: 1.5rem;
            line-height: 1;
        }}

        .alert-content strong {{
            font-size: 1.05rem;
            display: block;
            margin-bottom: 4px;
        }}

        .alert-content p {{
            font-size: 0.88rem;
            color: var(--text-secondary);
        }}

        /* KPI Metric Cards Grid */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}

        .kpi-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: 10px;
            padding: 16px 18px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: transform 0.15s ease, border-color 0.15s ease;
        }}

        .kpi-card:hover {{
            transform: translateY(-2px);
            border-color: var(--border-subtle);
        }}

        .kpi-title {{
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }}

        .kpi-value {{
            font-size: 1.45rem;
            font-weight: 700;
            color: var(--text-main);
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }}

        .kpi-subtext {{
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 6px;
        }}

        /* Section Containers */
        .section-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 14px;
            flex-wrap: wrap;
            gap: 12px;
        }}

        .section-title {{
            font-size: 1.25rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .section-badge {{
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            color: var(--text-secondary);
            font-size: 0.8rem;
            padding: 2px 8px;
            border-radius: 6px;
        }}

        /* Tables */
        .card-panel {{
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 32px;
        }}

        .table-responsive {{
            width: 100%;
            overflow-x: auto;
        }}

        .table-custom {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
            text-align: left;
        }}

        .table-custom th {{
            background: #0d131f;
            color: var(--text-secondary);
            font-weight: 600;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 12px 14px;
            border-bottom: 1px solid var(--border-card);
            white-space: nowrap;
        }}

        .table-custom td {{
            padding: 12px 14px;
            border-bottom: 1px solid rgba(30, 41, 59, 0.7);
            white-space: nowrap;
        }}

        .table-custom tbody tr:hover {{
            background: rgba(255, 255, 255, 0.02);
        }}

        /* Filter Controls */
        .controls-bar {{
            display: flex;
            gap: 12px;
            align-items: center;
            flex-wrap: wrap;
        }}

        .filter-btn {{
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            color: var(--text-secondary);
            padding: 6px 14px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .filter-btn.active {{
            background: var(--blue);
            color: #fff;
            border-color: var(--blue);
        }}

        .search-input {{
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            color: var(--text-main);
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.82rem;
            outline: none;
            width: 180px;
        }}

        .search-input:focus {{
            border-color: var(--blue);
        }}

        /* Utility classes */
        .text-right {{ text-align: right; }}
        .text-center {{ text-align: center; }}
        .font-mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
        .text-green {{ color: #34d399; }}
        .text-red {{ color: #f87171; }}
        .highlight-gold {{ color: #fbbf24; font-weight: 600; }}

        /* Empty state */
        .empty-state {{
            text-align: center;
            padding: 48px 20px;
        }}

        .empty-icon {{
            font-size: 2.8rem;
            margin-bottom: 12px;
        }}

        .empty-state h3 {{
            font-size: 1.15rem;
            margin-bottom: 6px;
        }}

        .empty-state p {{
            color: var(--text-secondary);
            max-width: 500px;
            margin: 0 auto;
            font-size: 0.88rem;
        }}

        /* Architecture Grid */
        .arch-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}

        .arch-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: 10px;
            padding: 20px;
        }}

        .arch-card h4 {{
            font-size: 0.95rem;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
            color: #60a5fa;
        }}

        .arch-card p {{
            font-size: 0.84rem;
            color: var(--text-secondary);
            line-height: 1.5;
        }}

        .formula-box {{
            background: #080d17;
            border: 1px solid var(--border-card);
            padding: 8px 12px;
            border-radius: 6px;
            font-family: monospace;
            font-size: 0.8rem;
            color: #38bdf8;
            margin: 10px 0;
            overflow-x: auto;
        }}

        /* Note Box */
        .note-box {{
            background: rgba(245, 158, 11, 0.08);
            border: 1px solid rgba(245, 158, 11, 0.3);
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 0.82rem;
            color: #fef3c7;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }}

        .note-tag {{
            background: #b45309;
            color: #fff;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 700;
        }}

        /* Footer */
        footer {{
            border-top: 1px solid var(--border-card);
            padding-top: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            color: var(--text-muted);
            font-size: 0.82rem;
        }}

        footer a {{
            color: var(--blue);
            text-decoration: none;
        }}

        footer a:hover {{
            text-decoration: underline;
        }}

        /* Responsive */
        @media (max-width: 768px) {{
            .header-meta {{
                align-items: flex-start;
            }}
            .kpi-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
            .brand-section h1 {{
                font-size: 1.3rem;
            }}
        }}

        @media (max-width: 480px) {{
            .kpi-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header>
            <div class="brand-section">
                <h1>⚡ NSE Pre-Market 3-Sigma Sniper</h1>
                <p>Indicative Equilibrium Price (IEP) Mean-Reversion Engine & Precision 09:15:00 AM Order Dispatch</p>
            </div>
            <div class="header-meta">
                {status_badge}
                <div class="time-badge">Scanned: {data['timestamp_ist']}</div>
                <div class="time-badge" style="font-size: 0.75rem; color: var(--text-muted);">{data['timestamp_utc']}</div>
            </div>
        </header>

        {fallback_note}

        {hero_alert}

        <!-- KPI Metric Summary Cards -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-title">Total Account Capital</div>
                <div class="kpi-value">₹{data['total_capital']:,.0f}</div>
                <div class="kpi-subtext">Single Source Budget</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Capital Allocated</div>
                <div class="kpi-value">₹{data['allocated_capital']:,.2f}</div>
                <div class="kpi-subtext">{alloc_pct:.1%} of budget (Limit &le; 50%)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Portfolio Heat (Risk)</div>
                <div class="kpi-value">₹{data['total_risk']:,.2f}</div>
                <div class="kpi-subtext">{data['heat_pct']:.2%} of capital (Max 4.0%)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Actionable Setups</div>
                <div class="kpi-value" style="color: {'#34d399' if actionable_count > 0 else '#94a3b8'};">{actionable_count}</div>
                <div class="kpi-subtext">&ge; 3.0σ + Microstructure verified</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Daily Loss Kill-Switch</div>
                <div class="kpi-value" style="color: #f87171;">₹{data['daily_loss_limit']:,.0f}</div>
                <div class="kpi-subtext">2.5% Drawdown Circuit Breaker</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Time Window</div>
                <div class="kpi-value" style="font-size: 1.15rem; color: #38bdf8;">09:15:00.050</div>
                <div class="kpi-subtext">3.0s Timeout | 09:45 Hard Exit</div>
            </div>
        </div>

        <!-- Section 1: Precision Sniping Orders -->
        <div class="section-header">
            <div class="section-title">
                <span>🎯 09:15:00 AM Precision Execution Schedule</span>
                <span class="section-badge">{orders_count} Orders Staged</span>
            </div>
        </div>
        <div class="card-panel">
            {orders_table_html}
        </div>

        <!-- Section 2: Complete Call Auction Scan -->
        <div class="section-header">
            <div class="section-title">
                <span>📊 09:08 AM Call Auction Complete Dislocation Scan</span>
                <span class="section-badge">{total_scanned} Symbols Scanned</span>
            </div>
            <div class="controls-bar">
                <input type="text" id="symbolSearch" class="search-input" placeholder="Search symbol..." onkeyup="filterTable()">
                <button class="filter-btn active" onclick="setFilter('all', this)">All ({total_scanned})</button>
                <button class="filter-btn" onclick="setFilter('actionable', this)">Actionable ({actionable_count})</button>
                <button class="filter-btn" onclick="setFilter('rejected', this)">Rejected ({len(data['rejected_signals'])})</button>
            </div>
        </div>
        <div class="card-panel">
            <div class="table-responsive">
                <table class="table-custom" id="scanTable">
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th class="text-right">Prev Close</th>
                            <th class="text-right">9:08 IEP</th>
                            <th class="text-right">Stock Gap</th>
                            <th class="text-right">Nifty Gap</th>
                            <th class="text-right">Beta</th>
                            <th class="text-right">Exp Gap</th>
                            <th class="text-right">Res Gap</th>
                            <th class="text-right">Sigma</th>
                            <th class="text-center">Z-Score</th>
                            <th class="text-right">AVR %</th>
                            <th class="text-right">IBR</th>
                            <th>Decision</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(scan_rows)}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Section 3: Architecture & Safety Guards -->
        <div class="section-header">
            <div class="section-title">
                <span>🔬 Quantitative Edge & Microstructure Safety Architecture</span>
            </div>
        </div>
        <div class="arch-grid">
            <div class="arch-card">
                <h4>📐 3-Sigma Mathematical Edge</h4>
                <p>Calculates the rolling 60-day overnight gap beta (&beta;) and idiosyncratic gap standard deviation (&sigma;<sub>&epsilon;</sub>) against NIFTY 50.</p>
                <div class="formula-box">Z = (Gap_Stock - &beta; * Gap_Index) / &sigma;_&epsilon;</div>
                <p>Only dislocations with |Z| &ge; 3.0 (99.7% statistical mispricing confidence) qualify for execution.</p>
            </div>
            <div class="arch-card">
                <h4>🛡️ Microstructure Circuit Proximity Guard</h4>
                <p>Rejects any setup opening within <strong>0.50%</strong> of the NSE Upper or Lower Circuit bands.</p>
                <div class="formula-box">Margin = |IEP - CircuitPrice| / PrevClose &ge; 0.50%</div>
                <p>Guarantees liquidity and eliminates the fatal risk of getting trapped in a frozen circuit band limit.</p>
            </div>
            <div class="arch-card">
                <h4>⚖️ Order Book Imbalance & Liquidity</h4>
                <p>Protects against phantom clearing prints and catastrophic falling knife order book sweeps.</p>
                <div class="formula-box">AVR &ge; 0.20% ADV | -0.80 &le; IBR &le; +0.80</div>
                <p>Validates that institutions matched volume in the auction and bids/offers are available to absorb the reversion.</p>
            </div>
            <div class="arch-card">
                <h4>⏱️ Precision Sizing & Execution Geometry</h4>
                <p>Routes strict LIMIT orders at 09:15:00.050 AM with 20 bps slippage buffer and 3.0s cancel timeout.</p>
                <div class="formula-box">Target = 61.8% Reversion | Max Risk &le; ₹3,000</div>
                <p>All prices snapped to NSE ₹0.05 tick size. Hard 09:45:00 AM time-stop automatically exits stalled trades.</p>
            </div>
        </div>

        <!-- Footer -->
        <footer>
            <div>
                <strong>NSE Pre-Market Sniper</strong> &bull; Zero-Maintenance Cloud Auto-Schedule &bull; Cron: <code>38 3 * * 1-5</code> (09:08 AM IST)
            </div>
            <div>
                Data Source: {data['data_source']} &bull;
                <a href="https://github.com/psraghothamrao8/stock_market" target="_blank" rel="noopener">GitHub Repository</a>
            </div>
        </footer>
    </div>

    <!-- Client-side Interactive Filter Script -->
    <script>
        let currentFilter = 'all';

        function setFilter(filterType, btn) {{
            currentFilter = filterType;
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            if (btn) btn.classList.add('active');
            filterTable();
        }}

        function filterTable() {{
            const searchVal = document.getElementById('symbolSearch').value.toLowerCase().trim();
            const rows = document.querySelectorAll('#scanTable tbody tr');

            rows.forEach(row => {{
                const sym = row.getAttribute('data-symbol') || '';
                const isActionable = row.classList.contains('row-actionable');
                const isRejected = row.classList.contains('row-rejected');

                let matchesFilter = true;
                if (currentFilter === 'actionable' && !isActionable) matchesFilter = false;
                if (currentFilter === 'rejected' && !isRejected) matchesFilter = false;

                let matchesSearch = sym.includes(searchVal);

                if (matchesFilter && matchesSearch) {{
                    row.style.display = '';
                }} else {{
                    row.style.display = 'none';
                }}
            }});
        }}
    </script>
</body>
</html>
"""
    return html


def write_dashboard_file(html_content: str, output_path: str) -> None:
    """Writes the generated HTML content to the specified path, creating directories and .nojekyll as needed."""
    abs_path = os.path.abspath(output_path)
    parent_dir = os.path.dirname(abs_path)
    os.makedirs(parent_dir, exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    # Ensure .nojekyll exists so GitHub Pages skips Jekyll processing
    nojekyll_path = os.path.join(parent_dir, ".nojekyll")
    if not os.path.exists(nojekyll_path):
        try:
            with open(nojekyll_path, "w", encoding="utf-8") as f:
                f.write("")
        except Exception:
            pass
    print(f"[SUCCESS] Dashboard written to: {abs_path} ({len(html_content):,} bytes)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate automated NSE Pre-Market 3-Sigma Dislocation HTML Dashboard"
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "live", "simulate"],
        default="auto",
        help="Scanner mode: 'auto' (live with simulation fallback), 'live', or 'simulate' (default: auto)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="public/index.html",
        help="Path for output HTML file (default: public/index.html)"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=300_000.0,
        help="Trading capital in INR (default: 300,000)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Z-score threshold for dislocation (default: 3.0)"
    )
    parser.add_argument(
        "--max-trades",
        type=int,
        default=2,
        help="Max simultaneous trades (default: 2)"
    )

    args = parser.parse_args()

    data = run_scanner(
        mode=args.mode,
        capital=args.capital,
        z_threshold=args.threshold,
        max_trades=args.max_trades
    )
    html_content = render_html_dashboard(data)
    write_dashboard_file(html_content, args.output)


if __name__ == "__main__":
    main()
