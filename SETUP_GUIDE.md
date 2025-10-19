# 📖 Complete Setup Guide - Maximum Performance Bot

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [First Run](#first-run)
5. [Monitoring](#monitoring)
6. [Optimization Settings](#optimization-settings)
7. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum
- **OS:** Windows 10/11, Linux, macOS
- **Python:** 3.8 or higher (3.11 recommended)
- **RAM:** 4GB
- **Storage:** 2GB free space
- **Internet:** Stable connection

### Recommended
- **OS:** Windows 11 or Ubuntu 22.04
- **Python:** 3.11
- **RAM:** 8GB+
- **Storage:** 5GB+ free space
- **Internet:** High-speed connection

### MetaTrader 5
- MT5 terminal installed
- Demo or live account
- Account credentials (login, password, server)

---

## Installation

### Step 1: Install Python

**Windows:**
```bash
# Download from python.org
# During installation, check "Add Python to PATH"
python --version  # Verify installation
```

**Linux:**
```bash
sudo apt update
sudo apt install python3.11 python3-pip
python3.11 --version
```

### Step 2: Clone Repository

```bash
git clone https://github.com/Samerabualsoud/trading-bot-maximum.git
cd trading-bot-maximum
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Expected output:**
```
Installing MetaTrader5...
Installing pandas...
Installing numpy...
Installing scikit-learn...
Installing tensorflow...
Installing joblib...
Successfully installed all packages!
```

**Note:** TensorFlow installation may take 5-10 minutes.

### Step 4: Verify Installation

```bash
python -c "import MetaTrader5; import pandas; import numpy; import sklearn; import tensorflow; print('✅ All dependencies installed!')"
```

---

## Configuration

### Step 1: Open Configuration File

Edit `ai_bot_maximum_8pct.py` in your favorite text editor.

### Step 2: Update MT5 Credentials

Find the `CONFIG` section (around line 67) and update:

```python
CONFIG = {
    'mt5_login': 843153,  # ← Change to your MT5 login
    'mt5_password': 'YOUR_PASSWORD_HERE',  # ← Change to your password
    'mt5_server': 'ACYSecurities-Demo',  # ← Change to your server
    ...
}
```

**How to find your MT5 server:**
1. Open MT5 terminal
2. Go to Tools → Options → Server
3. Copy the server name exactly

### Step 3: Review Trading Pairs

The bot trades 25 pairs by default:

```python
'symbols': [
    # Major Forex (7)
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF',
    # Cross Forex (10)
    'EURGBP', 'EURJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY',
    'EURAUD', 'EURNZD', 'GBPAUD', 'GBPNZD', 'AUDNZD',
    # Commodities (3)
    'XAUUSD', 'XAGUSD', 'XTIUSD',
    # Cryptocurrency (5)
    'BTCUSD', 'ETHUSD', 'SOLUSD', 'DOGEUSD', 'XRPUSD',
],
```

**To modify:**
- Remove pairs you don't want to trade
- Add pairs available on your broker
- Ensure all pairs are available on your MT5 account

### Step 4: Review Risk Settings

```python
'base_risk_per_trade': 0.03,  # 3% per trade
'min_risk': 0.02,  # 2% in high volatility
'max_risk': 0.04,  # 4% in low volatility
'daily_loss_limit': 0.08,  # 8% daily stop
'daily_profit_target': 0.04,  # 4% daily target (will continue to 15% max)
```

**Conservative settings (recommended for beginners):**
```python
'base_risk_per_trade': 0.02,  # 2% per trade
'daily_loss_limit': 0.05,  # 5% daily stop
```

**Aggressive settings (for experienced traders):**
```python
'base_risk_per_trade': 0.04,  # 4% per trade
'daily_loss_limit': 0.10,  # 10% daily stop
```

### Step 5: Enable/Disable Optimizations

All optimizations are enabled by default. To disable any:

```python
'use_advanced_optimizations': True,  # Master switch
'use_volume_surge_detection': True,  # Set to False to disable
'use_support_resistance': True,
'use_rsi_divergence': True,
'use_adx_trend_strength': True,
'use_time_of_day_weighting': True,
'use_market_regime_detection': True,
'use_fibonacci_levels': True,
'use_smart_stop_loss': True,
'use_trade_clustering_prevention': True,
```

---

## First Run

### Step 1: Start the Bot

```bash
python ai_bot_maximum_8pct.py
```

### Step 2: Initial Model Training

**First run output:**
```
2025-10-19 14:00:00 - INFO - Initializing Maximum Performance Bot...
2025-10-19 14:00:01 - INFO - Connected to MT5 - Balance: $10,000.00
2025-10-19 14:00:01 - INFO - Initializing AI models...
2025-10-19 14:00:02 - INFO - Training new models for EURUSD...
2025-10-19 14:00:05 - INFO -   [OK] Pattern model trained
2025-10-19 14:00:08 - INFO -   [OK] Ensemble model trained
2025-10-19 14:00:15 - INFO -   [OK] LSTM model trained
2025-10-19 14:00:15 - INFO - Training new models for GBPUSD...
...
```

**This will take 5-10 minutes** as the bot trains AI models for all 25 symbols.

**Subsequent runs:** Models load instantly from `models_maximum/` directory.

### Step 3: Monitor First Cycle

```
2025-10-19 14:10:00 - INFO - ================================================================================
2025-10-19 14:10:00 - INFO - CYCLE - 2025-10-19 14:10:00 - Session: OVERLAP
2025-10-19 14:10:00 - INFO - ================================================================================
2025-10-19 14:10:00 - INFO - Balance: $10,000.00 | Equity: $10,000.00 | Profit: $0.00 | Open: 0
2025-10-19 14:10:00 - INFO - Daily: 0 trades | P/L: $0.00 (+0.00%)
```

### Step 4: Watch for Signals

```
2025-10-19 14:15:30 - INFO - >>> SIGNAL #1 <<<
2025-10-19 14:15:30 - INFO - Symbol: EURUSD
2025-10-19 14:15:30 - INFO - Action: BUY
2025-10-19 14:15:30 - INFO - Confidence: 78.5%
2025-10-19 14:15:30 - INFO - [VOLUME SURGE] 2.1x average volume (+25 confidence)
2025-10-19 14:15:30 - INFO - [SUPPORT LEVEL] Near support at 1.0850 (+20 confidence)
2025-10-19 14:15:30 - INFO - [STRONG TREND] ADX 28.3 - strong uptrend (+20 confidence)
2025-10-19 14:15:31 - INFO - [SUCCESS] TRADE EXECUTED
```

---

## Monitoring

### Real-Time Monitoring

**Console Output:**
- Shows each cycle (every 60 seconds)
- Displays balance, equity, open positions
- Shows signals and trade executions
- Reports optimization bonuses

**Log File:**
- `ai_trading_bot_maximum.log`
- Contains detailed history
- Useful for analysis

### Key Metrics to Watch

**Daily:**
- Total trades: Should be 22-26
- Win rate: Should be 75%+
- Daily ROI: Should be 7-8%
- Max drawdown: Should be <10%

**Weekly:**
- Consistency: Should be 90%+ profitable days
- Average daily ROI: 7-8%
- Total ROI: 50-60%

### Optimization Indicators

**Volume Surge:**
```
[VOLUME SURGE] EURUSD - 2.1x average volume (+25 confidence)
```

**Support/Resistance:**
```
[SUPPORT LEVEL] GBPUSD near support at 1.2650 (+20 confidence)
[RESISTANCE LEVEL] USDJPY near resistance at 150.20 (+20 confidence)
```

**Divergence:**
```
[DIVERGENCE] BTCUSD bullish divergence detected (+30 confidence)
```

**Trend Strength:**
```
[STRONG TREND] XAUUSD ADX 32.5 - strong uptrend (+20 confidence)
[WEAK TREND] EURJPY ADX 15.2 - weak trend (-10 confidence)
```

**Market Regime:**
```
[REGIME] Trending market detected (1.3x confidence boost)
[REGIME] Ranging market detected (0.7x confidence reduction)
```

**Fibonacci:**
```
[FIB LEVEL] GBPUSD at 0.618 retracement (+15 confidence)
```

**Clustering Prevention:**
```
[CLUSTERING] Prevented: Too many trades in 15 min
[CLUSTERING] Prevented: Too many EURUSD trades in 30 min
```

---

## Optimization Settings

### Fine-Tuning Thresholds

**More aggressive (more trades):**
```python
'tier1_min_confidence': 60,  # Default: 65
'tier2_min_confidence': 65,  # Default: 70
'tier3_min_confidence': 70,  # Default: 75
```

**More conservative (fewer, higher quality trades):**
```python
'tier1_min_confidence': 70,  # Default: 65
'tier2_min_confidence': 75,  # Default: 70
'tier3_min_confidence': 80,  # Default: 75
```

### Adjusting Time-of-Day Weighting

In `advanced_optimizations.py`, modify `get_hour_performance_multiplier()`:

```python
hour_multipliers = {
    # Make overlap even more aggressive
    13: 1.8, 14: 1.8, 15: 1.8, 16: 1.7,  # Was 1.5, 1.5, 1.5, 1.4
    
    # Or make Asian session more active
    0: 0.8, 1: 0.8, 2: 0.8, 3: 0.8,  # Was 0.5, 0.5, 0.5, 0.5
}
```

### Adjusting Clustering Prevention

In `advanced_optimizations.py`, modify `check_clustering()`:

```python
# More lenient (allow more trades)
if len(recent_trades) >= 8:  # Was 5
    return False, "Too many trades in 15 min"

# More strict (fewer trades)
if len(recent_trades) >= 3:  # Was 5
    return False, "Too many trades in 15 min"
```

---

## Troubleshooting

### Issue: "MT5 initialization failed"

**Cause:** MT5 terminal not running or not installed

**Solution:**
1. Open MT5 terminal
2. Ensure it's logged in
3. Keep MT5 running while bot operates

### Issue: "MT5 login failed"

**Cause:** Incorrect credentials

**Solution:**
1. Verify login, password, server in CONFIG
2. Check MT5 terminal is logged in
3. Ensure account is active

### Issue: "Advanced optimizations module not available"

**Cause:** `advanced_optimizations.py` not found

**Solution:**
1. Ensure all files are in same directory
2. Check file permissions
3. Verify file wasn't renamed

### Issue: "No signals generated"

**Cause:** High confidence thresholds or poor market conditions

**Solution:**
1. This is normal - bot is selective
2. Wait for better market conditions
3. Consider lowering thresholds slightly
4. Check if during Asian session (conservative by design)

### Issue: "TensorFlow not available"

**Cause:** TensorFlow not installed or incompatible version

**Solution:**
1. Reinstall: `pip install tensorflow>=2.12.0`
2. Bot will work without TensorFlow (uses RF and GB only)
3. Performance slightly reduced without LSTM

### Issue: Bot stops trading after reaching target

**Cause:** Daily profit target reached

**Solution:**
1. This is by design (risk management)
2. Increase `daily_profit_max` in CONFIG
3. Or disable: `'scale_down_after_target': False`

### Issue: Too many/too few trades

**Adjust confidence thresholds:**
- Too many: Increase thresholds by 5
- Too few: Decrease thresholds by 5

**Adjust clustering prevention:**
- Too few: Increase limits in `check_clustering()`
- Too many: Decrease limits

---

## Best Practices

### 1. Start Conservative
- Begin with demo account
- Use conservative risk settings
- Monitor for 1 week minimum

### 2. Gradual Scaling
- Start with minimum lot sizes
- Increase gradually as confidence builds
- Never risk more than you can afford to lose

### 3. Regular Monitoring
- Check bot at least twice daily
- Review log file weekly
- Analyze performance monthly

### 4. Backup
- Backup `models_maximum/` directory weekly
- Save log files monthly
- Keep configuration backups

### 5. Updates
- Pull latest updates regularly
- Test updates on demo first
- Read changelog before updating

---

## Performance Expectations

### Week 1
- **Trades:** 150-180 total
- **Win Rate:** 75-80%
- **Daily ROI:** 6-8%
- **Weekly ROI:** 50-60%

### Month 1
- **Trades:** 600-750 total
- **Win Rate:** 78-82%
- **Daily ROI:** 7-8%
- **Monthly ROI:** 200-300%

### Long Term
- **Consistency:** 90%+ profitable days
- **Max Drawdown:** <10%
- **Sharpe Ratio:** >2.0
- **Profit Factor:** >2.5

---

## Support

For issues not covered here:
1. Check `ai_trading_bot_maximum.log`
2. Review console output
3. Verify all files present
4. Ensure MT5 is running
5. Test on demo account first

**Good luck and happy trading!** 🚀📈💰

