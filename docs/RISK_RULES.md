# Risk Management Rules (?3,00,000 Budget)

> "The first rule of compounding is to never interrupt it unnecessarily." ? Charlie Munger

## 1. Capital Preservation Rules
1. **Total Capital**: ?3,00,000
2. **Fixed Fractional Risk**: Never risk more than **1.0% (?3,000)** on any single trade idea.
3. **Portfolio Heat (Total Open Risk)**: Maximum open risk across all simultaneous positions is capped at **4.0% (?12,000)**.
   - That means at most 4 positions risking 1.0% each, or 8 positions risking 0.5% each.
4. **Maximum Allocation Per Position**:
   - In cash swing trading, allocate maximum **25% (?75,000)** of total portfolio to a single stock.
   - Max 4?5 open positions concurrently.

---

## 2. The Golden Position Sizing Formula

For every trade with entry price $P_{\text{entry}}$ and stop loss $P_{\text{stop}}$:

$$\Delta P = |P_{\text{entry}} - P_{\text{stop}}|$$

$$\text{Risk Ceiling} = \text{Capital} \times 1\% = ?3,000$$

$$\text{Shares by Risk} = \left\lfloor \frac{\text{Risk Ceiling}}{\Delta P} \right\rfloor$$

$$\text{Shares by Allocation} = \left\lfloor \frac{?75,000}{P_{\text{entry}}} \right\rfloor$$

$$\text{Final Order Qty} = \min(\text{Shares by Risk}, \text{Shares by Allocation})$$

---

## 3. Daily & Weekly Drawdown Circuit Breakers

* **Daily Loss Limit**: If cumulative closed and open loss in a day reaches **?7,500 (2.5%)**, stop trading immediately for that day.
* **Weekly Loss Limit**: If cumulative loss across a calendar week reaches **?15,000 (5.0%)**, take 3 trading days completely off to review trade logs.
