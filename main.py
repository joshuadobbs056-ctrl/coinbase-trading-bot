import os
import time
import json
import csv
import requests
import numpy as np
from sklearn.ensemble import RandomForestClassifier

#########################################
# VARIABLES (Railway)
#########################################

COINS = os.getenv("COINS", "AUTO")
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 5))
STATUS_INTERVAL = int(os.getenv("STATUS_INTERVAL", 60))

MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", 200))
MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", 50))

MIN_POSITION_SIZE_PERCENT = float(os.getenv("MIN_POSITION_SIZE_PERCENT", 0.5))
MAX_POSITION_SIZE_PERCENT = float(os.getenv("MAX_POSITION_SIZE_PERCENT", 2))

STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", 1.8))
TRAILING_START_PERCENT = float(os.getenv("TRAILING_START_PERCENT", 0.35))
TRAILING_DISTANCE_PERCENT = float(os.getenv("TRAILING_DISTANCE_PERCENT", 0.20))

MAX_TRADE_DURATION_MINUTES = int(os.getenv("MAX_TRADE_DURATION_MINUTES", 12))
MIN_SCORE = float(os.getenv("MIN_SCORE", 2))

START_BALANCE = float(os.getenv("START_BALANCE", 1000))

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

BASE_URL = "https://api.exchange.coinbase.com"

LEARNING_FILE = "learning.json"
HISTORY_FILE = "trade_history.csv"

#########################################
# TELEGRAM
#########################################

def notify(msg):
    print(msg, flush=True)

    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
                timeout=10
            )
        except:
            pass

#########################################
# LEARNING FILE
#########################################

def load_learning():
    data = {}

    if os.path.exists(LEARNING_FILE):
        with open(LEARNING_FILE, "r") as f:
            data = json.load(f)

    data.setdefault("trade_count", 0)
    data.setdefault("win_count", 0)
    data.setdefault("loss_count", 0)
    data.setdefault("total_profit", 0.0)

    return data

def save_learning():
    with open(LEARNING_FILE, "w") as f:
        json.dump(learning, f)

learning = load_learning()

#########################################
# HISTORY FILE
#########################################

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rsi", "volume_ratio", "trend_strength", "momentum", "profit"])

#########################################
# ML MODEL
#########################################

model = None

def train_model():
    global model

    try:
        data = np.loadtxt(HISTORY_FILE, delimiter=",", skiprows=1)

        if data.shape[0] < 50:
            return

        X = data[:, 0:4]
        y = data[:, 4] > 0

        model = RandomForestClassifier(n_estimators=200)
        model.fit(X, y)

        notify("ML MODEL ACTIVATED")

    except:
        pass

#########################################
# MARKET DATA
#########################################

def get_symbols():
    try:
        r = requests.get(BASE_URL + "/products", timeout=10)
        data = r.json()

        symbols = [
            p["id"]
            for p in data
            if "-USD" in p["id"]
            and p.get("status") == "online"
        ]

        return symbols[:MAX_SYMBOLS]

    except:
        return ["BTC-USD", "ETH-USD"]

def get_ticker(sym):
    try:
        r = requests.get(BASE_URL + f"/products/{sym}/ticker", timeout=10)
        data = r.json()

        if "price" not in data:
            return None

        price = float(data["price"])

        if price <= 0:
            return None

        return price

    except:
        return None

def get_candles(sym):
    try:
        r = requests.get(
            BASE_URL + f"/products/{sym}/candles",
            params={"granularity": 60},
            timeout=10
        )

        data = r.json()

        if not isinstance(data, list) or len(data) < 20:
            return None

        data.reverse()

        return data[-60:]

    except:
        return None

#########################################
# INDICATORS
#########################################

def calc_rsi(closes):
    if len(closes) < 15:
        return 50

    deltas = np.diff(closes)
    gains = np.maximum(deltas, 0)
    losses = -np.minimum(deltas, 0)

    avg_gain = np.mean(gains[-14:])
    avg_loss = np.mean(losses[-14:])

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def trend_strength(closes):
    avg = np.mean(closes)
    if avg == 0:
        return 0
    return (closes[-1] - avg) / avg

def momentum(closes):
    if len(closes) < 6:
        return 0
    return (closes[-1] - closes[-5]) / closes[-5]

def volume_ratio(vols):
    if len(vols) < 20:
        return 1
    return vols[-1] / np.mean(vols[-20:])

#########################################
# PORTFOLIO
#########################################

cash = START_BALANCE
open_trades = []

#########################################
# POSITION SIZE
#########################################

def position_size():
    percent = MIN_POSITION_SIZE_PERCENT

    if learning["trade_count"] > 50:
        percent = MAX_POSITION_SIZE_PERCENT

    size = cash * percent / 100

    return min(size, cash)

#########################################
# RECORD TRADE
#########################################

def record_trade(t, profit):
    with open(HISTORY_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            t["rsi"],
            t["volume_ratio"],
            t["trend_strength"],
            t["momentum"],
            profit
        ])

#########################################
# CLOSE TRADE
#########################################

def close_trade(t, price, reason):

    global cash

    proceeds = t["qty"] * price
    profit = proceeds - (t["qty"] * t["entry"])

    cash += proceeds

    learning["trade_count"] += 1
    learning["total_profit"] += profit

    if profit > 0:
        learning["win_count"] += 1
    else:
        learning["loss_count"] += 1

    save_learning()
    record_trade(t, profit)

    winrate = learning["win_count"] / max(1, learning["trade_count"]) * 100

    notify(
f"""SELL {t['sym']} ({reason})
Profit: {profit:.4f}
Balance: {cash:.2f}

Trades: {learning['trade_count']}
Wins: {learning['win_count']}
Losses: {learning['loss_count']}
Winrate: {winrate:.1f}%"""
)

#########################################
# OPEN TRADE
#########################################

def open_trade(sym, price, features):

    global cash

    if sym not in prices:
        return

    if len(open_trades) >= MAX_OPEN_TRADES:
        return

    size = position_size()

    if size <= 0 or cash < size:
        return

    qty = size / price
    cash -= size

    trade = {
        "sym": sym,
        "entry": price,
        "qty": qty,
        "time": time.time(),
        "peak": price,
        "stop": price * (1 - STOP_LOSS_PERCENT / 100),
        "trail": None,
        **features
    }

    open_trades.append(trade)

    notify(f"BUY {sym} | Balance {cash:.2f}")

#########################################
# MANAGE TRADES
#########################################

def manage_trades():

    global open_trades

    remaining = []

    for t in open_trades:

        sym = t["sym"]

        if sym not in prices:
            remaining.append(t)
            continue

        price = prices[sym]

        if price > t["peak"]:
            t["peak"] = price

        if price >= t["entry"] * (1 + TRAILING_START_PERCENT / 100):
            t["trail"] = t["peak"] * (1 - TRAILING_DISTANCE_PERCENT / 100)

        age_min = (time.time() - t["time"]) / 60
        profit_pct = (price - t["entry"]) / t["entry"] * 100

        if age_min >= MAX_TRADE_DURATION_MINUTES and profit_pct <= 0:
            close_trade(t, price, "STAGNATION")
            continue

        if price <= t["stop"]:
            close_trade(t, price, "STOP")
            continue

        if t["trail"] and price <= t["trail"]:
            close_trade(t, price, "TRAIL")
            continue

        remaining.append(t)

    open_trades = remaining

#########################################
# START
#########################################

notify("BOT STARTED")

symbols = get_symbols()

last_status = time.time()
last_train = time.time()

while True:

    try:

        prices = {}

        for sym in symbols:
            price = get_ticker(sym)

            if price is not None:
                prices[sym] = price

        manage_trades()

        for sym in symbols:

            if sym not in prices:
                continue

            if sym in [t["sym"] for t in open_trades]:
                continue

            candles = get_candles(sym)

            if candles is None:
                continue

            closes = [c[4] for c in candles]
            vols = [c[5] for c in candles]

            features = {
                "rsi": calc_rsi(closes),
                "volume_ratio": volume_ratio(vols),
                "trend_strength": trend_strength(closes),
                "momentum": momentum(closes)
            }

            score = sum([
                features["rsi"] > 55,
                features["volume_ratio"] > 1.2,
                features["trend_strength"] > 0,
                features["momentum"] > 0
            ])

            if score >= MIN_SCORE:
                open_trade(sym, prices[sym], features)

        if time.time() - last_train > 300:
            train_model()
            last_train = time.time()

        if time.time() - last_status > STATUS_INTERVAL:

            notify(
f"""STATUS
Cash {cash:.2f}
Open {len(open_trades)}
Trades {learning['trade_count']}
Wins {learning['win_count']}
Losses {learning['loss_count']}
Profit {learning['total_profit']:.2f}"""
)

            last_status = time.time()

        time.sleep(SCAN_INTERVAL)

    except Exception as e:

        notify(f"ERROR {str(e)}")
        time.sleep(5)
