# Quantitative Trading Roadmap: From ?3 Lakhs to Consistent Profitability

## Executive Summary
* **Account Size**: ?3,00,000 (INR)
* **Target Style**: Systematic Swing Trading (3?10 day holding period) + EOD Quantitative Scanning
* **Primary Objective**: Capital preservation, positive mathematical expectancy, and compounding without risking ruin.

---

## Phase 1: Infrastructure & Market Data (Weeks 1?3)
- [x] Create centralized Git repository with clean architecture.
- [ ] Implement historical data downloader for Nifty 50 / Nifty Midcap 150 using `yfinance`.
- [ ] Build pure indicator library (ATR, Exponential Moving Averages, Session VWAP, Relative Volume).
- [ ] Establish trade journaling process in `journal/entries/`.

---

## Phase 2: Systematic Strategy Development & Backtesting (Weeks 4?8)
- [ ] Implement Baseline Strategies:
  1. **Momentum Pullback (Trend Following)**: High relative strength stocks pulling back to 20-day EMA in an uptrending regime.
  2. **Volume Breakout (Volatility Expansion)**: 50-day high breakouts supported by 2x average volume.
- [ ] Code event-driven backtesting engine accounting for:
  - Slippage (0.10% per transaction)
  - Brokerage & STT (Securities Transaction Tax)
  - Exchange turnover and GST
- [ ] Verification criteria:
  - Minimum 100 historical trades across 3?5 years.
  - Profit Factor > 1.6
  - Max Historical Drawdown < 12%
  - Win Rate * Avg Win / (Loss Rate * Avg Loss) > 1.3

---

## Phase 3: Forward Testing & Paper Trading (Weeks 9?16)
- [ ] Run daily EOD screener at 3:45 PM IST.
- [ ] Log potential trade setups into `journal/entries/YYYY-MM-DD.md`.
- [ ] Paper-trade entries and exits with realistic limit orders.
- [ ] Measure slippage between signal trigger and hypothetical fill.

---

## Phase 4: Live Execution with Micro-Capital (Month 4 onwards)
- [ ] Fund broker with initial test capital (?50,000 of the ?3,00,000).
- [ ] Maximum risk per trade capped at ?500.
- [ ] Scale up to full ?3L capital only after 50 live trades with positive expectancy.
