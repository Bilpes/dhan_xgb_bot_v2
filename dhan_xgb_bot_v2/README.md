# Dhan XGBoost Trading Bot — Setup Guide

## Project structure

```
dhan_xgb_bot/
├── config/
│   └── config.py          ← All your settings (capital, risk, API keys)
├── data/
│   ├── download_data.py   ← Downloads historical data via yfinance
│   ├── features.py        ← Builds XGBoost features from OHLCV
│   └── historical/        ← CSVs saved here after download
├── models/
│   ├── train.py           ← Train XGBoost locally, saves model
│   ├── xgb_model.pkl      ← Saved after training
│   └── scaler.pkl         ← Saved after training
├── bot/
│   ├── dhan_api.py        ← Dhan API wrapper (orders + data)
│   ├── signal_engine.py   ← XGBoost inference on live candles
│   ├── risk_manager.py    ← Position sizing, SL, trailing stop
│   ├── live_bot.py        ← Main trading loop
│   └── backtest.py        ← Simulate on historical data
├── logs/
│   ├── bot.log            ← Runtime logs
│   ├── trades.csv         ← Live trade journal
│   └── backtest_trades.csv
└── requirements.txt
```

---

## Step-by-step setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Dhan API credentials

- Go to https://login.dhan.co → My Profile → Data APIs
- Generate your `Client ID` and `Access Token`
- Paste them in `config/config.py`:

```python
DHAN_CLIENT_ID    = "your_id_here"
DHAN_ACCESS_TOKEN = "your_token_here"
```

Note: Access token expires. Regenerate daily or automate via Dhan login API.

### 3. Download historical data

```bash
python data/download_data.py
```

Downloads 60 days of 5-minute OHLCV for top Nifty 50 stocks via yfinance (free).

### 4. Train the model

```bash
python models/train.py
```

Trains XGBoost on your local machine (Ryzen 5 2400G handles this in ~10 minutes).
Saves `models/xgb_model.pkl` and `models/scaler.pkl`.

### 5. Backtest first — always

```bash
python bot/backtest.py
```

Simulates the bot on historical data. Check:
- Win rate > 50%
- Average win > Average loss
- Max drawdown < 20% of capital

If backtest looks bad, adjust `BUY_THRESHOLD` in config.py or retrain.

### 6. Paper trade (1–2 weeks)

Set in `config/config.py`:
```python
# Comment out real order placement in live_bot.py
# and just log signals without calling dhan_api
```

Watch the signals in the log. Would you have made money?

### 7. Go live

```bash
python bot/live_bot.py
```

Start this at 9:10 AM IST. The bot handles everything from here.

---

## How the bot trades

```
Every 5 minutes:
  ↓
  Fetch last 60 candles from Dhan API
  ↓
  Build 27 features (RSI, EMA, MACD, VWAP, ATR, etc.)
  ↓
  XGBoost outputs: prob_up (0.0 – 1.0)
  ↓
  prob_up >= 0.62 → BUY signal
  prob_up <= 0.38 → SELL signal (future: short via F&O)
  else           → HOLD
  ↓
  On BUY signal + no open position:
    → Calculate stop-loss (ATR-based, max 2.5% below entry)
    → Calculate qty (risk ₹1,500 max per trade on ₹50k)
    → Place bracket order on Dhan (entry + SL + target in 1 call)
  ↓
  While in position every candle:
    → Activate trailing stop after +1.5% profit
    → Exit immediately if signal flips (model turns bearish)
    → Dhan bracket order auto-exits at SL or target
  ↓
  3:10 PM → Force exit all intraday positions
```

---

## Key settings to tune

| Setting | Default | Meaning |
|---|---|---|
| `BUY_THRESHOLD` | 0.62 | Minimum model confidence to enter |
| `STOP_LOSS_PCT` | 2.5% | Max loss per trade |
| `MAX_RISK_PCT` | 3% | Capital risked per trade |
| `TRAIL_AFTER_PCT` | 1.5% | When trailing stop activates |
| `TRAIL_DISTANCE` | 1% | How tight the trail follows price |
| `DAILY_LOSS_LIMIT` | 6% | Shuts bot down if daily loss hits this |

---

## Retrain schedule

Retrain every week to keep the model fresh:

```bash
# Sunday evening
python data/download_data.py   # refresh data
python models/train.py         # retrain
python bot/backtest.py         # validate before Monday
```

---

## What the bot does NOT do (yet)

- Short selling / F&O (only long equity for now)
- Multiple simultaneous positions (1 at a time during trial)
- Telegram/email alerts (add later once stable)
- Auto token refresh for Dhan API
