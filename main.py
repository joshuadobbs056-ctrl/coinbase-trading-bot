import os
import time
import json
import csv
import math
import requests
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ==========================================================
# VARIABLES (Railway)
# ==========================================================

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 5))
STATUS_INTERVAL = int(os.getenv("STATUS_INTERVAL", 60))

COINS = os.getenv("COINS", "AUTO").replace(" ", "")
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", 200))
MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", 20))

START_BALANCE = float(os.getenv("START_BALANCE", 500))

# Position sizing
MIN_TRADE_SIZE_USD = float(os.getenv("MIN_TRADE_SIZE_USD", 25))
MIN_POSITION_SIZE_PERCENT = float(os.getenv("MIN_POSITION_SIZE_PERCENT", 5))
MAX_POSITION_SIZE_PERCENT = float(os.getenv("MAX_POSITION_SIZE_PERCENT", 10))
MIN_CASH_RESERVE_PERCENT = float(os.getenv("MIN_CASH_RESERVE_PERCENT", 15))

# Risk / exits
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", 2.5))
TRAILING_START_PERCENT = float(os.getenv("TRAILING_START_PERCENT", 0.9))
TRAILING_DISTANCE_PERCENT = float(os.getenv("TRAILING_DISTANCE_PERCENT", 0.5))
MAX_TRADE_DURATION_MINUTES = int(os.getenv("MAX_TRADE_DURATION_MINUTES", 60))

# Fee-aware exit floor (prevents tiny "wins" getting eaten)
MIN_PROFIT_TO_SELL_PERCENT = float(os.getenv("MIN_PROFIT_TO_SELL_PERCENT", 0.35))

# Trend continuation entry thresholds
RSI_MIN = float(os.getenv("RSI_MIN", 55))
RSI_MAX = float(os.getenv("RSI_MAX", 72))
MIN_VOL_RATIO = float(os.getenv("MIN_VOL_RATIO", 1.15))
MIN_MOMENTUM = float(os.getenv("MIN_MOMENTUM", 0.002))  # 0.2% over lookback
MIN_TREND_STRENGTH = float(os.getenv("MIN_TREND_STRENGTH", 0.0))

# Candles
CANDLE_GRANULARITY = int(os.getenv("CANDLE_GRANULARITY", 60))  # 60s candles
CANDLE_POINTS = int(os.getenv("CANDLE_POINTS", 60))            # last N candles

# ML
ML_MIN_TRADES_TO_ENABLE = int(os.getenv("ML_MIN_TRADES_TO_ENABLE", 50))
ML_MIN_PROB = float(os.getenv("ML_MIN_PROB", 0.60))
ML_RETRAIN_EVERY_TRADES = int(os.getenv("ML_RETRAIN_EVERY_TRADES", 25))

# Telegram (optional)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Paper trading toggle (this bot is PAPER by design)
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() == "true"

# Coinbase Exchange API base
BASE_URL = "https://api.exchange.coinbase.com"

# Files (local container). NOTE: if Railway restarts without persistent storage, these reset.
LEARNING_FILE = "learning.json"
HISTORY_FILE = "trade_history.csv"
STATE_FILE = "state.json"

# ==========================================================
# Telegram
# ==========================================================

def send_telegram(msg: str) -> None:
    print(msg, flush=True)
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
            timeout=10,
        )
    except Exception:
        pass

# ==========================================================
# Persistence (local files)
# ==========================================================

def load_json(path: str, default: dict) -> dict:
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def save_json(path: str, data: dict) -> None:
    try:
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception:
        pass

def ensure_history_file() -> None:
    if os.path.exists(HISTORY_FILE):
        return
    try:
        with open(HISTORY_FILE, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rsi", "volume_ratio", "trend_strength", "momentum", "profit"])
    except Exception:
        pass

ensure_history_file()

learning = load_json(LEARNING_FILE, {
    "trade_count": 0,
    "win_count": 0,
    "loss_count": 0,
    "total_profit_usd": 0.0,  # realized P/L in USD for paper
})

state = load_json(STATE_FILE, {
    "cash": START_BALANCE,
})

cash = float(state.get("cash", START_BALANCE))

# ==========================================================
# Coinbase data
# ==========================================================

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "coin-sniper-bot"})

def cb_get_products_usd() -> List[str]:
    try:
        r = SESSION.get(f"{BASE_URL}/products", timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
        symbols = []
        for p in data:
            try:
                if p.get("quote_currency") == "USD" and p.get("status") == "online":
                    symbols.append(p["id"])
            except Exception:
                continue
        return symbols
    except Exception:
        return []

def cb_get_price(symbol: str) -> Optional[float]:
    # Ticker sometimes doesn't include "price" for some products or when rate-limited.
    try:
        r = SESSION.get(f"{BASE_URL}/products/{symbol}/ticker", timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        p = data.get("price")
        if p is None:
            return None
        return float(p)
    except Exception:
        return None

def cb_get_candles(symbol: str, granularity: int, points: int) -> Optional[List[List[float]]]:
    try:
        r = SESSION.get(
            f"{BASE_URL}/products/{symbol}/candles",
            params={"granularity": str(granularity)},
            timeout=15,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        if not isinstance(data, list) or len(data) < 30:
            return None
        # newest-first -> oldest-first
        data = list(reversed(data))
        return data[-min(points, len(data)):]
    except Exception:
        return None

# ==========================================================
# Indicators
# ==========================================================

def ema(values: np.ndarray, period: int = 20) -> float:
    if len(values) == 0:
        return 0.0
    if len(values) < period:
        return float(np.mean(values))
    k = 2 / (period + 1)
    e = float(values[0])
    for v in values[1:]:
        e = float(v) * k + e * (1 - k)
    return float(e)

def rsi14(closes: np.ndarray) -> float:
    if len(closes) < 15:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains[-14:]))
    avg_loss = float(np.mean(losses[-14:]))
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))

def volume_ratio(vols: np.ndarray, period: int = 20) -> float:
    if len(vols) < 2:
        return 1.0
    n = min(period, len(vols))
    avg = float(np.mean(vols[-n:]))
    if avg <= 0:
        return 1.0
    return float(vols[-1] / avg)

def momentum(closes: np.ndarray, lookback: int = 5) -> float:
    if len(closes) < lookback + 1:
        return 0.0
    base = float(closes[-1 - lookback])
    if base <= 0:
        return 0.0
    return float((closes[-1] - base) / base)

def trend_strength(closes: np.ndarray) -> float:
    if len(closes) < 20:
        return 0.0
    e = ema(closes, 20)
    if e <= 0:
        return 0.0
    return float((closes[-1] - e) / e)

def build_features(symbol: str) -> Optional[Tuple[List[float], float]]:
    candles = cb_get_candles(symbol, CANDLE_GRANULARITY, CANDLE_POINTS)
    if candles is None or len(candles) < 30:
        return None

    closes = np.array([float(c[4]) for c in candles], dtype=float)
    vols = np.array([float(c[5]) for c in candles], dtype=float)

    r = rsi14(closes)
    vr = volume_ratio(vols, 20)
    ts = trend_strength(closes)
    mom = momentum(closes, 5)
    last = float(closes[-1])

    feats = [float(r), float(vr), float(ts), float(mom)]
    return feats, last

# ==========================================================
# ML (Real training on real features)
# ==========================================================

model: Optional[RandomForestClassifier] = None

def load_history_matrix() -> Optional[np.ndarray]:
    try:
        data = np.loadtxt(HISTORY_FILE, delimiter=",", skiprows=1)
        if data.size == 0:
            return None
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        return data
    except Exception:
        return None

def train_model_if_ready() -> None:
    global model
    data = load_history_matrix()
    if data is None or len(data) < ML_MIN_TRADES_TO_ENABLE:
        model = None
        return
    try:
        X = data[:, :-1]
        y = (data[:, -1] > 0).astype(int)
        m = RandomForestClassifier(n_estimators=200, random_state=42)
        m.fit(X, y)
        model = m
        send_telegram(f"ML MODEL ON ({len(data)} trades)")
    except Exception:
        model = None

def ml_allows(features: List[float]) -> bool:
    if model is None:
        return True
    try:
        prob = float(model.predict_proba([features])[0][1])
        return prob >= ML_MIN_PROB
    except Exception:
        return True

# ==========================================================
# Portfolio / Trading (PAPER)
# ==========================================================

open_trades: List[Dict] = []

def cash_reserve_amount() -> float:
    return cash * (MIN_CASH_RESERVE_PERCENT / 100.0)

def can_open_more() -> bool:
    return len(open_trades) < MAX_OPEN_TRADES

def already_open(symbol: str) -> bool:
    return any(t["symbol"] == symbol for t in open_trades)

def next_position_size_usd() -> float:
    # Random within percent range, but enforces MIN_TRADE_SIZE_USD and reserve cash.
    available = max(0.0, cash - cash_reserve_amount())
    if available <= 0:
        return 0.0

    pct = float(np.random.uniform(MIN_POSITION_SIZE_PERCENT, MAX_POSITION_SIZE_PERCENT))
    raw = cash * (pct / 100.0)
    size = max(raw, MIN_TRADE_SIZE_USD)
    return float(min(size, available))

def calculate_equity(latest_prices: Dict[str, float]) -> float:
    eq = cash
    for t in open_trades:
        p = latest_prices.get(t["symbol"])
        if p is None:
            continue
        eq += t["qty"] * p
    return float(eq)

def append_trade_history(features: List[float], profit_usd: float) -> None:
    try:
        with open(HISTORY_FILE, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([features[0], features[1], features[2], features[3], profit_usd])
    except Exception:
        pass

def save_state_now() -> None:
    save_json(LEARNING_FILE, learning)
    save_json(STATE_FILE, {"cash": cash})

def open_trade(symbol: str, price: float, features: List[float]) -> None:
    global cash
    if not PAPER_TRADING:
        # This bot is paper-only (no authenticated trading here).
        return

    if not can_open_more() or already_open(symbol):
        return

    size = next_position_size_usd()
    if size <= 0:
        return

    qty = size / price
    if qty <= 0:
        return

    cash -= size

    trade = {
        "symbol": symbol,
        "qty": qty,
        "entry": price,
        "time": time.time(),
        "peak": price,
        "trail": None,
        "stop": price * (1 - STOP_LOSS_PERCENT / 100.0),
        "features": features,  # REAL features saved for ML
    }
    open_trades.append(trade)

    send_telegram(
        f"BUY {symbol}\n"
        f"Price: {price:.6f}\n"
        f"Size: ${size:.2f}\n"
        f"Stop: {trade['stop']:.6f}\n"
        f"Cash: ${cash:.2f}\n"
        f"ML: {'ON' if model is not None else 'OFF'}"
    )
    save_state_now()

def close_trade(trade: Dict, exit_price: float, reason: str) -> None:
    global cash

    proceeds = trade["qty"] * exit_price
    cost = trade["qty"] * trade["entry"]
    profit_usd = proceeds - cost

    cash += proceeds

    learning["trade_count"] += 1
    if profit_usd > 0:
        learning["win_count"] += 1
    else:
        learning["loss_count"] += 1
    learning["total_profit_usd"] += float(profit_usd)

    append_trade_history(trade["features"], float(profit_usd))

    # Retrain periodically (keeps ML improving)
    if learning["trade_count"] % max(1, ML_RETRAIN_EVERY_TRADES) == 0:
        train_model_if_ready()

    wins = int(learning["win_count"])
    losses = int(learning["loss_count"])
    total = int(learning["trade_count"])
    winrate = (wins / total * 100.0) if total > 0 else 0.0

    send_telegram(
        f"SELL {trade['symbol']} ({reason})\n"
        f"Entry: {trade['entry']:.6f}\n"
        f"Exit:  {exit_price:.6f}\n"
        f"P/L:   ${profit_usd:.2f}\n"
        f"Cash:  ${cash:.2f}\n"
        f"Trades: {total} | W: {wins} | L: {losses} | WR: {winrate:.1f}%"
    )

    save_state_now()

def manage_open_trades(latest_prices: Dict[str, float]) -> None:
    global open_trades

    remaining = []

    for t in open_trades:
        price = latest_prices.get(t["symbol"])
        if price is None:
            remaining.append(t)
            continue

        # Update peak
        if price > t["peak"]:
            t["peak"] = price

        # Profit percent relative to entry
        profit_pct = (price - t["entry"]) / t["entry"] * 100.0

        # Activate/update trailing once profit >= start
        if profit_pct >= TRAILING_START_PERCENT:
            t["trail"] = t["peak"] * (1 - TRAILING_DISTANCE_PERCENT / 100.0)

        age_min = (time.time() - t["time"]) / 60.0

        # STOP
        if price <= t["stop"]:
            close_trade(t, price, "STOP")
            continue

        # TRAIL (fee-aware floor)
        if t["trail"] is not None and price <= t["trail"] and profit_pct >= MIN_PROFIT_TO_SELL_PERCENT:
            close_trade(t, price, "TRAIL")
            continue

        # STAGNATION (your exact requested logic)
        if age_min >= MAX_TRADE_DURATION_MINUTES and profit_pct <= 0:
            close_trade(t, price, "STAGNATION")
            continue

        remaining.append(t)

    open_trades = remaining

# ==========================================================
# Trend Continuation Entry Filter
# ==========================================================

def trend_continuation_ok(features: List[float]) -> bool:
    rsi_v, vol_r, ts, mom = features
    return (
        (RSI_MIN <= rsi_v <= RSI_MAX) and
        (vol_r >= MIN_VOL_RATIO) and
        (mom >= MIN_MOMENTUM) and
        (ts >= MIN_TREND_STRENGTH)
    )

# ==========================================================
# Status
# ==========================================================

def send_status(latest_prices: Dict[str, float]) -> None:
    eq = calculate_equity(latest_prices)
    unrealized = eq - cash
    net = eq - START_BALANCE

    total = int(learning.get("trade_count", 0))
    wins = int(learning.get("win_count", 0))
    losses = int(learning.get("loss_count", 0))
    win_rate = (wins / total * 100.0) if total > 0 else 0.0

    send_telegram(
        f"STATUS\n"
        f"Cash: ${cash:.2f}\n"
        f"Equity: ${eq:.2f}\n"
        f"Unrealized: ${unrealized:.2f}\n"
        f"Net P/L: ${net:.2f}\n"
        f"Open: {len(open_trades)}\n"
        f"Trades: {total} | W: {wins} | L: {losses} | WR: {win_rate:.1f}%\n"
        f"ML: {'ON' if model is not None else 'OFF'}"
    )

# ==========================================================
# Boot
# ==========================================================

if COINS == "AUTO":
    SYMBOLS = cb_get_products_usd()[:MAX_SYMBOLS]
else:
    SYMBOLS = [c for c in COINS.split(",") if c]

train_model_if_ready()

send_telegram(
    "BOT STARTED\n"
    f"Mode: {'PAPER' if PAPER_TRADING else 'LIVE(DISABLED)'}\n"
    f"Symbols: {len(SYMBOLS)}\n"
    f"Max Open: {MAX_OPEN_TRADES}\n"
    f"Min Trade: ${MIN_TRADE_SIZE_USD:.2f}\n"
    f"ML: {'ON' if model is not None else 'OFF'}"
)

last_status = time.time()

# ==========================================================
# Main loop
# ==========================================================

while True:
    try:
        # 1) Pull prices for symbols
        latest_prices: Dict[str, float] = {}
        for sym in SYMBOLS:
            p = cb_get_price(sym)
            if p is not None:
                latest_prices[sym] = p

        # 2) Manage open trades (stop/trail/stagnation)
        if latest_prices:
            manage_open_trades(latest_prices)

        # 3) Scan for new trend-continuation entries
        if can_open_more():
            for sym in SYMBOLS:
                if not can_open_more():
                    break
                if already_open(sym):
                    continue

                feats_last = build_features(sym)
                if feats_last is None:
                    continue

                feats, last_price = feats_last

                # Trend continuation filter
                if not trend_continuation_ok(feats):
                    continue

                # ML filter (if enabled)
                if not ml_allows(feats):
                    continue

                # Use latest ticker price when available; else candle close
                entry_price = latest_prices.get(sym, last_price)
                if entry_price is None:
                    continue

                open_trade(sym, float(entry_price), feats)

                # ultra-frequency safety: don't open 20 trades in one loop tick
                # open at most a few per scan
                if len(open_trades) >= MAX_OPEN_TRADES:
                    break

        # 4) Status
        if time.time() - last_status >= STATUS_INTERVAL:
            send_status(latest_prices)
            last_status = time.time()

        time.sleep(SCAN_INTERVAL)

    except Exception as e:
        send_telegram(f"ERROR: {str(e)}")
        time.sleep(5)
