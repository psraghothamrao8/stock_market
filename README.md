# Systematic Trading & Quantitative Research Engine

A code-driven algorithmic and quantitative trading repository designed for systematic swing and intraday trading on Indian equities (NSE/BSE).

---

## 1. Capital Allocation & Risk Geometry (?3,00,000 Budget)

This system is built around strict capital preservation and risk sizing:

* **Total Account Capital**: ?3,00,000 (INR)
* **Risk Per Trade (Max)**: 1.0% (?3,000 ceiling)
* **Daily Drawdown Limit (Kill-Switch)**: 2.5% (?7,500)
* **Position Sizing Engine**:
  $$\text{Shares} = \min\left(\left\lfloor \frac{\text{Risk Amount}}{|\text{Entry Price} - \text{Stop Loss}|} \right\rfloor, \left\lfloor \frac{\text{Max Capital Per Position}}{\text{Entry Price}} \right\rfloor\right)$$

---

## 2. Directory Structure

```
stock_market/
?-- docs/
?   ?-- ROADMAP.md              # Milestones from ?3L capital to scaling
?   ?-- RISK_RULES.md           # Mathematical rules, drawdown limits, kill switches
?-- journal/
?   ?-- TEMPLATE.md             # Daily market review & trade post-mortem template
?   ?-- entries/                # Daily trading logs and notes
?-- engine/
?   ?-- __init__.py
?   ?-- config.py               # Tunable constants & parameters (Single Source of Truth)
?   ?-- risk.py                 # Risk budget, position sizing, kill-switch logic
?   ?-- indicators.py           # Technical indicators (EMA, VWAP, ATR, RVOL)
?-- backtest/
?   ?-- __init__.py
?   ?-- runner.py               # Historical backtester & metric evaluator
?-- data/                       # Historical price cache & symbol universe (git-ignored)
?-- requirements.txt            # Python dependencies
?-- .gitignore
```

---

## 3. Workflow Philosophy

1. **Hypothesis First**: Formulate mathematical edge before writing code.
2. **Backtesting & Verification**: Rigorous historical verification accounting for slippage, STT, and transaction friction.
3. **Walk-Forward Validation**: Out-of-sample testing to prevent curve-fitting and data snooping.
4. **Paper Execution**: Real-time forward testing before committing hard capital.
5. **Systematic Journaling**: Recording every setup, emotional deviation, and execution metric in `journal/`.
