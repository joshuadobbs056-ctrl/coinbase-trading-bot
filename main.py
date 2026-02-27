import os
import time
import json
import csv
import math
import requests
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier

#########################################
# RAILWAY VARIABLES
#########################################

# Core loop
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 30))          # seconds
STATUS_INTERVAL = int(os.getenv("STATUS_INTERVAL", 120))    # seconds

# Universe
COINS = os.getenv("COINS", "BTC-USD,ETH-USD,SOL-USD").replace(" ", "")
COIN_LIST = [c for c in COINS.split(",") if c]

# Risk / exposure
START_BALANCE = float(os.getenv("START_BALANCE", 1000))
MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", 6))
POSITION_SIZE_PERCENT = float(os.getenv("POSITION_SIZE_PERCENT", 15))  # % of balance per trade (compounds)
MIN_CASH_RESERVE_PERCENT = float(os.getenv("MIN_CASH_RESERVE_PERCENT", 5))  # keep some cash unused

# Stops
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", 2.0))          # hard stop
TRAILING_START_PERCENT = float(os.getenv("TRAILING_START_PERCENT", 1.2))  # start trailing after profit reaches X%
TRAILING_DISTANCE_PERCENT = float(os.getenv("TRAILING_DISTANCE_PERCENT", 0.7))  # trail distance from peak

# AI / Filters
MIN_SCORE = float(os.getenv("MIN_SCORE", 3.0))  # weighted points threshold
ML_MIN_PROB = float(os.getenv("ML_MIN_PROB", 0.60))
ML_MIN_TRADES_TO_ENABLE = int(os.getenv("ML_MIN_TRADES_TO_ENABLE", 25))

# Candle settings (Exchange API supports granularities: 60,300,900,3600,21600,86400)  [oai_citation:2‡Coinbase Developer Docs](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles?utm_source=chatgpt.com)
CANDLE_GRANULARITY = int(os.getenv("CANDLE_GRANULARITY", 60))
CANDLE_POINTS = int(os.getenv("CANDLE_POINTS", 60))  # use last N closes for indicators (<= 300)

# Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Optional: paper mode only (always True in this file)
PAPER_TRADING = True

#########################################
# FILES (BOT MEMORY)
#########################################

LEARNING_FILE = "learning.json"
HISTORY_FILE = "trade_history.csv"

#########################################
# TELEGRAM
#########################################

def send_telegram(message: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=10)
    except Exception:
        pass

#########################################
# LEARNING (LEVEL 1) + BACKWARD COMPAT
#########################################

def load_learning() -> Dict:
    default = {
        "weights": {"rsi": 1.0, "volume": 1.0, "trend": 1.0, "momentum": 1.0},
        "trade_count": 0,
        "win_count": 0,
        "total_profit": 0.0
    }

    if not os.path.exists(LEARNING_FILE):
        return default

    try:
        with open(LEARNING_FILE, "r") as f:
            data = json.load(f)

        # Patch missing fields (so upgrades don’t crash)
        if "weights" not in data:
            data["weights"] = default["weights"]
        else:
            for k, v in default["weights"].items():
                if k not in data["weights"]:
                    data["weights"][k] = v

        if "trade_count" not in data:
            data["trade_count"] = 0
        if "win_count" not in data:
            data["win_count"] = 0
        if "total_profit" not in data:
            data["total_profit"] = 0.0

        return data
    except Exception:
        return default

def save_learning(learning: Dict) -> None:
    try:
        with open(LEARNING_FILE, "w") as f:
            json.dump(learning, f)
    except Exception:
        pass

learning = load_learning()

#########################################
# HISTORY DATASET (LEVEL 2 TRAINING)
#########################################

def ensure_history_file() -> None:
    if os.path.exists(HISTORY_FILE):
        return
    try:
        with open(HISTORY_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["rsi", "volume_ratio", "trend_strength", "momentum", "profit"])
    except Exception:
        pass

ensure_history_file()

#########################################
# COINBASE MARKET DATA (REAL)
#########################################

BASE_URL = "https://api.exchange.coinbase.com"  #  [oai_citation:3‡Coinbase Developer Docs](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-ticker?utm_source=chatgpt.com)

def cb_get_ticker(product_id: str) -> Optional[Dict]:
    # GET /products/{product_id}/ticker  [oai_citation:4‡Coinbase Developer Docs](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-ticker?utm_source=chatgpt.com)
    try:
        r = requests.get(f"{BASE_URL}/products/{product_id}/ticker", timeout=10)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def cb_get_candles(product_id: str, granularity: int, points: int) -> Optional[List[List[float]]]:
    # GET /products/{product_id}/candles?granularity=...  [oai_citation:5‡Coinbase Developer Docs](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles?utm_source=chatgpt.com)
    # Response: [ time, low, high, open, close, volume ]
    try:
        params = {"granularity": str(granularity)}
        r = requests.get(f"{BASE_URL}/products/{product_id}/candles", params=params, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            return None
        # Coinbase returns newest-first; we want oldest-first
        data = list(reversed(data))
        return data[-min(points, len(data)):]
    except Exception:
        return None

#########################################
# INDICATORS
#########################################

def calc_rsi(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(np.array(closes, dtype=float))
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0

    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))

def ema(values: List[float], period: int = 20) -> float:
    if len(values) < period:
        return float(np.mean(values)) if values else 0.0
    arr = np.array(values, dtype=float)
    k = 2 / (period + 1)
    e = arr[0]
    for v in arr[1:]:
        e = v * k + e * (1 - k)
    return float(e)

def trend_strength(closes: List[float]) -> float:
    # Simple normalized trend: (close - EMA) / EMA clamped to [0..1]
    if len(closes) < 20:
        return 0.5
    e = ema(closes, 20)
    if e <= 0:
        return 0.5
    raw = (closes[-1] - e) / e
    # map roughly: -2%..+2% into 0..1
    scaled = (raw + 0.02) / 0.04
    return float(max(0.0, min(1.0, scaled)))

def momentum(closes: List[float], lookback: int = 5) -> float:
    if len(closes) < lookback + 1:
        return 0.0
    return float((closes[-1] - closes[-1 - lookback]) / closes[-1 - lookback])

def volume_ratio(volumes: List[float], period: int = 20) -> float:
    if not volumes:
        return 1.0
    if len(volumes) < period:
        avg = float(np.mean(volumes))
    else:
        avg = float(np.mean(volumes[-period:]))
    if avg <= 0:
        return 1.0
    return float(volumes[-1] / avg)

#########################################
# AI SCORING (LEVEL 1)
#########################################

def calculate_score(rsi_v: float, vol_ratio: float, trend_v: float, mom_v: float) -> float:
    w = learning["weights"]
    score = 0.0

    # RSI: prefer bullish but not extreme
    if 55 <= rsi_v <= 75:
        score += w["rsi"]
    elif rsi_v > 75:
        score += w["rsi"] * 0.5  # still bullish but more risky

    # Volume: must be above average
    if vol_ratio > 1.3:
        score += w["volume"]

    # Trend: above neutral
    if trend_v > 0.55:
        score += w["trend"]

    # Momentum: positive
    if mom_v > 0:
        score += w["momentum"]

    return float(score)

#########################################
# ML FILTER (LEVEL 2)
#########################################

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
        send_telegram(f"AI model trained ({len(data)} trades)")
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

#########################################
# PAPER PORTFOLIO + COMPOUNDING
#########################################

current_balance = START_BALANCE  # cash only (paper)
open_trades: List[Dict] = []     # positions

def cash_reserve_amount() -> float:
    return current_balance * (MIN_CASH_RESERVE_PERCENT / 100.0)

def next_position_cash() -> float:
    # Compounding: % of current balance
    raw = current_balance * (POSITION_SIZE_PERCENT / 100.0)
    # never use reserve cash
    available = max(0.0, current_balance - cash_reserve_amount())
    return float(min(raw, available))

#########################################
# TRADE RECORDING (feeds learning + ML)
#########################################

def append_trade_history(rsi_v: float, vol_ratio: float, trend_v: float, mom_v: float, profit: float) -> None:
    try:
        with open(HISTORY_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([rsi_v, vol_ratio, trend_v, mom_v, profit])
    except Exception:
        pass

def update_learning_on_close(profit: float) -> None:
    # Lightweight adaptive weights
    if profit > 0:
        learning["win_count"] += 1
        learning["weights"]["trend"] += 0.02
        learning["weights"]["volume"] += 0.02
    else:
        learning["weights"]["volume"] = max(0.2, learning["weights"]["volume"] - 0.01)

#########################################
# ENTRY / EXIT LOGIC (STOP + TRAIL)
#########################################

def can_open_more() -> bool:
    return len(open_trades) < MAX_OPEN_TRADES

def already_open(symbol: str) -> bool:
    return any(t["symbol"] == symbol for t in open_trades)

def open_trade(symbol: str, price: float, features: Dict[str, float], score: float) -> None:
    global current_balance

    if not can_open_more():
        return
    if already_open(symbol):
        return

    pos_cash = next_position_cash()
    if pos_cash <= 0:
        return

    qty = pos_cash / price
    if qty <= 0:
        return

    # Deduct cash (paper execution)
    current_balance -= pos_cash

    stop_price = price * (1 - STOP_LOSS_PERCENT / 100.0)

    trade = {
        "symbol": symbol,
        "qty": qty,
        "entry_price": price,
        "entry_time": time.time(),

        # features for learning
        "rsi": features["rsi"],
        "volume_ratio": features["volume_ratio"],
        "trend_strength": features["trend_strength"],
        "momentum": features["momentum"],
        "score": score,

        # stop & trail state
        "stop_price": stop_price,
        "peak_price": price,
        "trailing_active": False,
        "trail_stop": None,
    }

    open_trades.append(trade)

    send_telegram(
        f"BUY {symbol}\n"
        f"Price: ${price:.4f}\n"
        f"Cash Used: ${pos_cash:.2f}\n"
        f"Score: {score:.2f}\n"
        f"Stop: ${stop_price:.4f}"
    )

def close_trade(trade: Dict, exit_price: float, reason: str) -> None:
    global current_balance

    symbol = trade["symbol"]
    entry_price = trade["entry_price"]
    qty = trade["qty"]

    proceeds = qty * exit_price
    cost = qty * entry_price
    profit = proceeds - cost

    # Return cash (paper)
    current_balance += proceeds

    # Update learning memory
    learning["trade_count"] += 1
    learning["total_profit"] += profit
    update_learning_on_close(profit)
    save_learning(learning)

    # Append dataset row for ML
    append_trade_history(trade["rsi"], trade["volume_ratio"], trade["trend_strength"], trade["momentum"], profit)

    send_telegram(
        f"SELL {symbol} ({reason})\n"
        f"Entry: ${entry_price:.4f}\n"
        f"Exit: ${exit_price:.4f}\n"
        f"P/L: ${profit:.2f}\n"
        f"Balance: ${current_balance:.2f}"
    )

def manage_open_trades(latest_prices: Dict[str, float]) -> None:
    global open_trades
    remaining = []

    for t in open_trades:
        symbol = t["symbol"]
        price = latest_prices.get(symbol)
        if price is None:
            remaining.append(t)
            continue

        # Update peak
        if price > t["peak_price"]:
            t["peak_price"] = price

        # Activate trailing once price >= entry*(1+TRAILING_START)
        if not t["trailing_active"]:
            target = t["entry_price"] * (1 + TRAILING_START_PERCENT / 100.0)
            if price >= target:
                t["trailing_active"] = True

        # Update trail stop if active
        if t["trailing_active"]:
            t["trail_stop"] = t["peak_price"] * (1 - TRAILING_DISTANCE_PERCENT / 100.0)

        # Hard stop OR trailing stop
        hard_stop_hit = price <= t["stop_price"]
        trail_stop_hit = t["trail_stop"] is not None and price <= t["trail_stop"]

        if hard_stop_hit:
            close_trade(t, price, "STOP")
            continue

        if trail_stop_hit:
            close_trade(t, price, "TRAIL")
            continue

        remaining.append(t)

    open_trades = remaining

#########################################
# SIGNAL GENERATION USING REAL CANDLES
#########################################

def build_features_for_symbol(symbol: str) -> Optional[Dict[str, float]]:
    candles = cb_get_candles(symbol, CANDLE_GRANULARITY, CANDLE_POINTS)
    if candles is None or len(candles) < 20:
        return None

    closes = [float(c[4]) for c in candles]
    vols = [float(c[5]) for c in candles]

    rsi_v = calc_rsi(closes, 14)
    vol_r = volume_ratio(vols, 20)
    trend_v = trend_strength(closes)
    mom_v = momentum(closes, 5)

    return {
        "rsi": float(rsi_v),
        "volume_ratio": float(vol_r),
        "trend_strength": float(trend_v),
        "momentum": float(mom_v),
        "last_close": float(closes[-1]),
    }

#########################################
# STATUS
#########################################

def send_status() -> None:
    total = learning.get("trade_count", 0)
    wins = learning.get("win_count", 0)
    win_rate = (wins / total * 100.0) if total > 0 else 0.0

    # Unrealized P/L (paper) using last known peak/price if available
    # We'll compute with last close fetched per symbol in loop; if not available, skip.
    open_count = len(open_trades)

    msg = (
        f"STATUS\n"
        f"Balance: ${current_balance:.2f}\n"
        f"Profit: ${learning.get('total_profit', 0.0):.2f}\n"
        f"Open trades: {open_count}\n"
        f"Trades: {total} | Wins: {wins} | Win rate: {win_rate:.1f}%\n"
        f"ML: {'ON' if model is not None else 'OFF'}"
    )
    send_telegram(msg)

#########################################
# MAIN LOOP
#########################################

send_telegram(
    "Coin Sniper (REAL Coinbase prices) started\n"
    f"Mode: PAPER\n"
    f"Coins: {', '.join(COIN_LIST)}"
)

train_model_if_ready()
last_status = time.time()
last_train = time.time()

while True:
    try:
        # 1) Pull real prices for all coins (ticker)
        latest_prices: Dict[str, float] = {}
        for sym in COIN_LIST:
            tick = cb_get_ticker(sym)
            if tick and "price" in tick:
                latest_prices[sym] = float(tick["price"])

        # 2) Manage open trades with real prices (STOP / TRAIL)
        if latest_prices:
            manage_open_trades(latest_prices)

        # 3) Scan for new entries
        for sym in COIN_LIST:
            if not can_open_more():
                break
            if already_open(sym):
                continue
            price = latest_prices.get(sym)
            if price is None:
                continue

            feats = build_features_for_symbol(sym)
            if feats is None:
                continue

            score = calculate_score(feats["rsi"], feats["volume_ratio"], feats["trend_strength"], feats["momentum"])
            features_vec = [feats["rsi"], feats["volume_ratio"], feats["trend_strength"], feats["momentum"]]

            if score >= MIN_SCORE and ml_allows(features_vec):
                open_trade(sym, price, feats, score)

        # 4) Periodic status
        if time.time() - last_status >= STATUS_INTERVAL:
            send_status()
            last_status = time.time()

        # 5) Periodic ML retrain
        if time.time() - last_train >= 600:  # every 10 minutes
            train_model_if_ready()
            last_train = time.time()

        time.sleep(SCAN_INTERVAL)

    except Exception as e:
        send_telegram(f"Error: {str(e)}")
        time.sleep(10)
