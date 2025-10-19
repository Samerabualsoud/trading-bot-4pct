# Debug Version Instructions

## What's New

I've created **`ai_bot_maximum_8pct_debug.py`** with enhanced logging to show:

1. **Which symbols are being analyzed**
2. **Why signals are rejected** (low confidence, no data, outside hours, etc.)
3. **Confidence levels** for BUY and SELL for each symbol
4. **Market data status** (if data is available)

---

## How to Use

### Step 1: Stop Current Bot

Press `Ctrl+C` to stop the running bot.

### Step 2: Run Debug Version

```bash
python ai_bot_maximum_8pct_debug.py
```

### Step 3: Watch the Output

You'll now see detailed information like:

```
CYCLE - 2025-10-19 19:15:00 - Session: OVERLAP
Balance: $990,739.49 | Equity: $990,739.49
Daily: 0 trades | P/L: $0.00 (+0.00%)

Analyzing 25 symbols...

[ANALYZING] EURUSD...
  [EURUSD] No data or insufficient candles (got 0)

[ANALYZING] GBPUSD...
  [GBPUSD] No data or insufficient candles (got 0)

[ANALYZING] BTCUSD...
  [BTCUSD] No signal - BUY: 62.3%, SELL: 58.1% (need 65.0%)

[ANALYZING] ETHUSD...
  [ETHUSD] No signal - BUY: 59.8%, SELL: 61.2% (need 65.0%)

Cycle complete. Signals: 0, Trades: 0
```

---

## What to Look For

### **If you see "No data or insufficient candles (got 0)":**
- **Forex pairs:** Markets are closed (weekend)
- **Crypto pairs:** MT5 connection issue or symbol not available

### **If you see "No signal - BUY: XX%, SELL: YY% (need ZZ%)":**
- Bot is analyzing the symbol
- Confidence is below threshold
- **This is normal** - bot is being selective

### **If you see "Outside active hours":**
- Market hours filter is blocking trades
- Check if it's the right time

### **If you see ">>> SIGNAL <<<":**
- Bot found a high-confidence setup!
- Trade will be executed

---

## Expected Output on Weekend

**Saturday/Sunday:**
```
[ANALYZING] EURUSD... ← Forex
  [EURUSD] No data (got 0) ← Markets closed

[ANALYZING] BTCUSD... ← Crypto
  [BTCUSD] No signal - BUY: 62%, SELL: 58% (need 65%) ← Active but below threshold
```

**Monday-Friday during overlap (16:00-20:00 Riyadh time):**
```
[ANALYZING] EURUSD...
  [EURUSD] No signal - BUY: 68%, SELL: 55% (need 65%) ← Close!

[ANALYZING] GBPUSD...
  [GBPUSD] No signal - BUY: 71%, SELL: 52% (need 65%) ← Very close!

[ANALYZING] BTCUSD...
>>> SIGNAL #1 <<< ← FOUND ONE!
  Symbol: BTCUSD
  Action: BUY
  Confidence: 78.5%
```

---

## When to Switch Back

Once you understand what's happening:

1. Stop debug bot (Ctrl+C)
2. Run normal version:
   ```bash
   python ai_bot_maximum_8pct.py
   ```

The normal version has less logging (cleaner output).

---

## Troubleshooting

**Too much output?**
- This is expected with DEBUG mode
- Shows every detail
- Helps diagnose issues

**Still no signals on crypto?**
- Check if confidence is close (60-64%)
- Consider lowering thresholds temporarily
- Or wait for better market conditions

**Want even more detail?**
- Check the log file: `ai_trading_bot_maximum.log`
- Contains everything from console + more

---

## Summary

The debug version will show you **exactly** why the bot is or isn't generating signals.

**Run it now and send me the output!** I'll be able to tell you precisely what's happening.
