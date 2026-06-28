# dhan_xgb_bot_v2

XGBoost-based intraday algo trading bot for NSE via Dhan broker API.

## Project Structure

```
dhan_xgb_bot_v2/          ← GitHub repo root
  README.md               ← this file
  dhan_xgb_bot_v2/        ← Python package root (cd here to run)
    bot/
      auto_retrain.py     ← nightly walk-forward retraining
      live_bot.py         ← main trading loop
      signal_engine.py    ← XGBoost signal generation
      dhan_api.py         ← Dhan broker API wrapper
      risk_manager.py     ← position sizing & risk controls
      trade_policy.py     ← BUY_THRESHOLD, ATR, HORIZON params
      backtest.py         ← offline backtesting
      telegram_alert.py   ← Telegram notifications
      health_check.py     ← system health monitor
      token_refresh.py    ← access token refresh
    config/
      config.py           ← paths, timing, filters, credentials
      watchlist.json      ← 21-stock curated universe
      .env.example        ← copy to .env and fill credentials
    data/
      features.py         ← build_features() pipeline
      historical/         ← CSV fallback data (SYMBOL_5min.csv)
    models/
      xgb_model.pkl       ← deployed model (git-ignored)
      scaler.pkl          ← deployed scaler (git-ignored)
    logs/                 ← runtime logs (git-ignored)
    requirements.txt
    scheduler.bat         ← Windows Task Scheduler launcher
```

## Setup

```cmd
cd dhan_xgb_bot_v2\dhan_xgb_bot_v2
pip install -r requirements.txt
copy config\.env.example config\.env
# Fill in DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN, TELEGRAM_BOT_TOKEN etc.
```

## Running

**All commands must be run from inside `dhan_xgb_bot_v2\dhan_xgb_bot_v2\`**

```cmd
# Retrain model
python -m bot.auto_retrain

# Start live/paper trading bot
python -m bot.live_bot

# Run backtest
python -m bot.backtest

# Health check
python -m bot.health_check
```

## Model Performance (2026-06-28)

| Metric | OOS Value |
|--------|-----------|
| Accuracy | 0.772 |
| AUC | 0.868 |
| Precision | 0.681 |
| Recall | 0.776 |
| Training rows | 90,320 |
| BUY% | 39.0% |

Walk-forward validated across 5 time-series folds, 21 stocks, 90-day window.

## Universe

21 curated NSE stocks — 12 Tier-A (always scanned) + 9 Tier-B (scanned after 10:00).
All stocks: daily vol > 300Cr, MCap > 15,000Cr, no news-event driven names.
See `config/watchlist.json` for the full list.

## Key Parameters

All trading parameters (thresholds, ATR multipliers, position limits) live in `bot/trade_policy.py`.  
All infrastructure (paths, timing, credentials) lives in `config/config.py`.
