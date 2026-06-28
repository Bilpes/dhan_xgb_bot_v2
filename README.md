# dhan_xgb_bot_v3 — NSE Intraday XGBoost Algo Bot

## Critical Fixes vs v2

| Issue | v2 | v3 Fix |
|---|---|---|
| Label entry price | `close[t]` — **LEAKAGE** | `open[t+1]` — correct |
| Prob cap | Hard cap at 0.90/0.95 | Removed — raw model output |
| Retrain embargo | None | 14-day gap enforced |
| Prob calibration | 90% pred → 27% actual win | Aligns after retrain |
| Watchlist | 150 stocks incl. penny/PSU | 38 quality stocks ≥₹200 |
| ATR_SL_MULT | 1.5 (noise SL hits) | 2.2 (breathing room) |
| VWAP gate | Hard filter ON (kills signals) | Feature only — not gated |
| Trade mode | CNC (overnight risk) | INTRADAY / MIS |
| Trailing SL | None | Activates at 1× ATR gain |
| Signal scan log | None | `logs/signal_scan.csv` |
| Sector limit | Not enforced | MAX_PER_SECTOR = 2 |

## Repository Structure

```
config.py          — all parameters (thresholds, risk, timing, Redis)
watchlist.py       — 38-stock curated list with sector map
features.py        — leakage-free features + label builder
train.py           — walk-forward with 14-day embargo
signal_engine.py   — signal generation with scan logger
trade_manager.py   — risk sizing, trailing SL, CSV trade log
bot.py             — main event loop (paper + live)
auto_retrain.py    — weekly retrain scheduler
diagnostics.py     — calibration + reject reason + P&L reports
```

## Quick Start

```bash
pip install -r requirements.txt

# 1. Set environment variables
export DHAN_CLIENT_ID=your_id
export DHAN_ACCESS_TOKEN=your_token
export TELEGRAM_BOT_TOKEN=your_bot_token   # optional
export TELEGRAM_CHAT_ID=your_chat_id       # optional

# 2. Download 1yr of 5-min OHLCV data for each symbol to:
#    data/SYMBOL_5min.csv  (columns: datetime, open, high, low, close, volume)

# 3. First train
python train.py

# 4. Paper mode run
python bot.py

# 5. After 1 week — check calibration
python diagnostics.py
```

## Leakage Verification

After first retrain, run `python diagnostics.py`.

**Healthy**: `prob=0.65` bucket → actual win rate ~50–65%  
**Leakage still present**: `prob=0.90` → actual win ~25%

If still leaking, verify:
- `features.py build_labels()`: `label_entry_shift=1` (not 0)
- `train.py`: `embargo_days=14` is set
- `signal_engine.py`: no prob capping before logging to `signal_scan.csv`

## Expected Performance (v3 Paper Mode)

| Day Type | Expected Trades | Notes |
|---|---|---|
| BULL (Nifty trending ↑) | 5–9 | Peak signal flow |
| SIDEWAYS | 3–6 | VWAP gate removed helps |
| WEAK (Nifty falling ↓) | 2–5 | Threshold bumps to 0.58 |
| Expiry Thursday | 7–12 | F&O unwinding = momentum |
| High Vol (Budget/Results) | 6–11 | Best day for this model |
