import os, time, json, csv, traceback, base64
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

import requests
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# =========================
# SINGLE INSTANCE LOCK
# =========================
LOCK_FILE = "/tmp/coin_sniper.lock"

def acquire_lock_or_exit():
    if os.path.exists(LOCK_FILE):
        try:
            raw = open(LOCK_FILE).read().strip()
            if raw:
                old_pid = int(raw.split("|")[0])
                try:
                    os.kill(old_pid, 0)
                    raise SystemExit(0)
                except OSError: pass
        except SystemExit: raise
        except Exception: pass
    open(LOCK_FILE, "w").write(f"{os.getpid()}|{int(time.time())}")

acquire_lock_or_exit()

# =========================
# CORE CONFIGURATION
# =========================
START_BALANCE = float(os.getenv("START_BALANCE", "1000"))
SCAN_INTERVAL = 6                 # Fast heartbeat
STATUS_INTERVAL = 300             # Profit report every 5 mins
ML_INTERVAL = 300                 

MAX_OPEN_TRADES = 12
MAX_NEW_BUYS_PER_SCAN = 2
MIN_TRADE_SIZE = 50               # Increased for profit impact

# --- Profit-Focused Exit Strategy ---
STOP_LOSS_PERCENT = 3.5           # Tight stop to protect equity
TRAILING_START_PERCENT = 1.0      # Lock in gains early
TRAILING_DISTANCE_PERCENT = 1.2   # Tight trail for micro-cap volatility
ATR_MULT = 2.5                    # Adaptive volatility multiplier

# --- High-Conviction Entry Gates ---
ENTRY_SCORE_MIN = 8               
MIN_VOLUME_RATIO = 3.5            # RVOL floor
EXTENSION_MAX = 0.04              # Anti-FOMO: Max distance from EMA20
COOLDOWN_SECONDS_AFTER_SELL = 600 

# --- Market Guard & AI ---
BTC_GUARD_DROP_PCT = 0.7          # Sensitive BTC dump protection
ML_MIN_PROB = 0.65                # High AI confidence required

# --- External Services ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
STATE_FILE = "coin_sniper_state.json"
ML_FILE = "coin_sniper_ml.json"
LEDGER_FILE = "coin_sniper_ledger.csv"

# =========================
# DATA MODELS & PERSISTENCE
# =========================
@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    entry_time: float
    high_water: float
    stop_price: float
    trailing_active: bool
    trail_dist_pct: float
    last_reason: str
    last_score: int
    entry_rvol: float
    entry_accel: float
    entry_ext: float
    entry_atrp: float

def notify(msg: str):
    print(msg, flush=True)
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                          json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}, 
                          timeout=10)
        except Exception: pass

# =========================
# TRADING ENGINE LOGIC
# =========================

def score_symbol(sym: str, candles: List[list]) -> Tuple[Optional[int], str, Dict[str, float]]:
    if not candles: return None, "no_data", {}
    recent = list(reversed(candles[:200]))
    closes = np.array([float(c[4]) for c in recent])
    vols = np.array([float(c[5]) for c in recent])
    
    p0 = closes[-1]
    avg_vol = np.mean(vols[-20:-1])
    rvol = vols[-1] / avg_vol if avg_vol > 0 else 1
    
    # Velocity calculation
    v_now = (closes[-1] - closes[-4]) / closes[-4] if len(closes) > 4 else 0
    v_prev = (closes[-4] - closes[-7]) / closes[-7] if len(closes) > 7 else 0
    accel = v_now - v_prev
    
    ema20 = _ema(closes, 20)[-1]
    ext = (p0 - ema20) / ema20
    atrp = _atr_percent(candles)

    score = 5
    if rvol >= MIN_VOLUME_RATIO: score += 3
    if rvol >= 5.0: score += 2
    if accel > 0: score += 1
    if ext > EXTENSION_MAX: score -= 7 # Heavily penalize overextension
    
    return int(max(1, min(10, score))), f"RVOL:{rvol:.1f} Accel:{accel:.3f}", {"rvol": rvol, "accel": accel, "ext": ext, "atrp": atrp}

def open_position(state, positions, sym, price, score, reason, extra):
    cash = state["cash"]
    if cash < MIN_TRADE_SIZE: return
    
    qty = MIN_TRADE_SIZE / price
    state["cash"] -= MIN_TRADE_SIZE
    
    positions[sym] = Position(
        symbol=sym, qty=qty, entry_price=price, entry_time=time.time(),
        high_water=price, stop_price=price * (1 - STOP_LOSS_PERCENT/100),
        trailing_active=False, trail_dist_pct=TRAILING_DISTANCE_PERCENT,
        last_reason=reason, last_score=score,
        entry_rvol=extra['rvol'], entry_accel=extra['accel'], 
        entry_ext=extra['ext'], entry_atrp=extra['atrp']
    )
    
    notify(f"🟢 *BUY {sym}*\nPrice: ${price:.6f}\nScore: {score}/10\nReason: {reason}")

def close_position(state, positions, sym, price, reason):
    p = positions.pop(sym)
    proceeds = p.qty * price
    pnl = proceeds - (p.qty * p.entry_price)
    pnl_pct = (pnl / (p.qty * p.entry_price)) * 100
    
    state["cash"] += proceeds
    state["realized_pnl"] += pnl
    if pnl > 0: state["wins"] += 1
    else: state["losses"] += 1
    
    state["cooldowns"][sym] = time.time() + COOLDOWN_SECONDS_AFTER_SELL
    
    icon = "📈" if pnl > 0 else "📉"
    notify(f"{icon} *SELL {sym}*\nExit: ${price:.6f}\nPnL: ${pnl:.2f} ({pnl_pct:.2f}%)\nReason: {reason}")

def status_report(state, positions, last_prices, guard_msg):
    cash = state["cash"]
    eq = cash + sum(p.qty * last_prices.get(s, p.entry_price) for s, p in positions.items())
    wins, losses = state["wins"], state["losses"]
    wr = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
    
    msg = (f"📊 *ELITE PROFIT REPORT*\n"
           f"Equity: ${eq:.2f} | Cash: ${cash:.2f}\n"
           f"PnL: ${state['realized_pnl']:.2f}\n"
           f"Win Rate: {wr:.1f}% ({wins}W/{losses}L)\n"
           f"Guard: {guard_msg}")
    notify(msg)

# =========================
# HELPER FUNCTIONS & LOOP
# =========================
def _ema(s, n):
    a = 2 / (n + 1)
    res = np.zeros_like(s)
    res[0] = s[0]
    for i in range(1, len(s)):
        res[i] = a * s[i] + (1 - a) * res[i-1]
    return res

def _atr_percent(candles):
    # Simplified ATR% calculation
    return 1.5 

def main():
    state = {"cash": START_BALANCE, "positions": {}, "wins": 0, "losses": 0, "realized_pnl": 0.0, "cooldowns": {}}
    positions = {}
    last_status = 0
    
    notify("🚀 *Savage ELITE Engine Initialized*")
    
    while True:
        try:
            # Placeholder for product universe and candle fetching
            # manage_positions(...)
            # score_symbol(...)
            # status_report(...)
            
            time.sleep(SCAN_INTERVAL)
        except Exception as e:
            notify(f"⚠️ Error: {str(e)}")
            time.sleep(10)

if __name__ == "__main__":
    main()
