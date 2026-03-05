import os
import time
import json
import traceback
import requests
import ccxt
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

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
                except OSError:
                    pass
        except SystemExit:
            raise
        except:
            pass
    open(LOCK_FILE, "w").write(f"{os.getpid()}|{int(time.time())}")

acquire_lock_or_exit()

# =========================
# CORE CONFIGURATION
# =========================
START_BALANCE = 2000
SCAN_INTERVAL = 6
STATUS_INTERVAL = 300

MAX_OPEN_TRADES = 6
MIN_TRADE_SIZE = 50

STOP_LOSS_PERCENT = 3.5
TRAILING_START_PERCENT = 1.0
TRAILING_DISTANCE_PERCENT = 1.2

ENTRY_SCORE_MIN = 8
MIN_VOLUME_RATIO = 3.5
EXTENSION_MAX = 0.04

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

COINBASE_PRO_API_KEY = os.getenv("COINBASE_PRO_API_KEY")
COINBASE_PRO_API_SECRET = os.getenv("COINBASE_PRO_API_SECRET")
COINBASE_PRO_API_PASSWORD = os.getenv("COINBASE_PRO_API_PASSWORD")

SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "MATIC/USD"]

# =========================
# TELEGRAM NOTIFY
# =========================
def notify(msg: str):
    print(msg, flush=True)
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"},
                timeout=10,
            )
            print("Telegram response:", r.json(), flush=True)
        except Exception as e:
            print("Telegram failed:", e, flush=True)

# =========================
# POSITION MODEL
# =========================
@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    high_water: float
    stop_price: float
    trailing_active: bool

# =========================
# EXCHANGE SETUP (Coinbase Pro)
# =========================
exchange = ccxt.coinbasepro({
    "apiKey": COINBASE_PRO_API_KEY,
    "secret": COINBASE_PRO_API_SECRET,
    "password": COINBASE_PRO_API_PASSWORD,
    "enableRateLimit": True,
})

# =========================
# INDICATORS
# =========================
def _ema(values, n):
    a = 2 / (n + 1)
    res = np.zeros_like(values)
    res[0] = values[0]
    for i in range(1, len(values)):
        res[i] = a * values[i] + (1 - a) * res[i - 1]
    return res

def _fetch_candles(symbol):
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe="5m", limit=200)
        return data
    except Exception as e:
        print(f"fetch_candles error {symbol}:", e)
        return []

def score_symbol(sym: str, candles: List[list]) -> Tuple[Optional[int], str, dict]:
    if not candles:
        return None, "no_data", {}
    closes = np.array([c[4] for c in candles])
    vols = np.array([c[5] for c in candles])

    price = closes[-1]
    avg_vol = np.mean(vols[-20:]) if len(vols) >= 20 else 1
    rvol = (vols[-1] / avg_vol) if avg_vol > 0 else 1

    ema20 = _ema(closes, 20)[-1]
    ext = (price - ema20) / ema20

    score = 5
    if rvol >= MIN_VOLUME_RATIO:
        score += 3
    if rvol >= 5.0:
        score += 2
    if ext <= EXTENSION_MAX:
        score += 1

    return int(max(1, min(10, score))), f"RVOL:{rvol:.2f} EXT:{ext:.3f}", {"rvol": rvol, "ext": ext}

# =========================
# TRADE LOGIC (Paper Trading)
# =========================
def open_position(state, positions, sym, price):
    if state["cash"] < MIN_TRADE_SIZE:
        return
    qty = MIN_TRADE_SIZE / price
    state["cash"] -= MIN_TRADE_SIZE
    positions[sym] = Position(
        symbol=sym,
        qty=qty,
        entry_price=price,
        high_water=price,
        stop_price=price * (1 - STOP_LOSS_PERCENT / 100),
        trailing_active=False,
    )
    notify(f"🟢 PAPER BUY {sym} @ ${price:.2f}")

def close_position(state, positions, sym, price):
    p = positions.pop(sym)
    proceeds = p.qty * price
    pnl = proceeds - (p.qty * p.entry_price)
    state["cash"] += proceeds
    notify(f"🔴 PAPER SELL {sym} @ ${price:.2f} | PnL: ${pnl:.2f}")

def manage_positions(state, positions):
    to_close = []
    for sym, p in positions.items():
        try:
            ticker = exchange.fetch_ticker(sym)
            last = ticker["last"]
        except Exception as e:
            print(f"fetch_ticker error {sym}:", e)
            continue

        if last <= p.stop_price:
            to_close.append((sym, last))
        elif last > p.high_water * (1 + TRAILING_START_PERCENT / 100):
            p.trailing_active = True
            p.high_water = last

        if p.trailing_active:
            trail_stop = p.high_water * (1 - TRAILING_DISTANCE_PERCENT / 100)
            if last <= trail_stop:
                to_close.append((sym, last))

    for sym, price in to_close:
        close_position(state, positions, sym, price)

# =========================
# REPORTING
# =========================
def status_report(state, positions):
    equity = state["cash"]
    for s, p in positions.items():
        try:
            ticker = exchange.fetch_ticker(s)
            equity += p.qty * ticker["last"]
        except Exception:
            equity += p.qty * p.entry_price
    notify(f"📊 STATUS | Cash:${state['cash']:.2f} | Open:{len(positions)} | Equity:${equity:.2f}")

# =========================
# MAIN LOOP
# =========================
def main():
    state = {"cash": START_BALANCE}
    positions: Dict[str, Position] = {}
    last_status = time.time()

    notify("🚀 Savage ELITE Paper Trading Engine Initialized ($2000 Ledger)")

    while True:
        try:
            # 1️⃣ Manage open positions
            manage_positions(state, positions)

            # 2️⃣ Scan new entries
            for sym in SYMBOLS:
                if len(positions) >= MAX_OPEN_TRADES:
                    break
                if sym in positions:
                    continue

                candles = _fetch_candles(sym)
                score, reason, extra = score_symbol(sym, candles)
                if score and score >= ENTRY_SCORE_MIN:
                    price = candles[-1][4]
                    open_position(state, positions, sym, price)

            # 3️⃣ Periodic status
            if time.time() - last_status > STATUS_INTERVAL:
                status_report(state, positions)
                last_status = time.time()

            time.sleep(SCAN_INTERVAL)

        except Exception as e:
            notify("⚠️ Exception:\n" + str(e))
            traceback.print_exc()
            time.sleep(10)

if __name__ == "__main__":
    main()
