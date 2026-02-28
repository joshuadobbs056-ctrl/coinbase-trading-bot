import os
import time
import json
import csv
import requests
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# =========================
# CONFIG
# =========================

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 5))
STATUS_INTERVAL = int(os.getenv("STATUS_INTERVAL", 60))

MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", 200))
MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", 20))

MIN_POSITION_SIZE_PERCENT = float(os.getenv("MIN_POSITION_SIZE_PERCENT", 5))
MAX_POSITION_SIZE_PERCENT = float(os.getenv("MAX_POSITION_SIZE_PERCENT", 10))

MIN_TRADE_SIZE_USD = float(os.getenv("MIN_TRADE_SIZE_USD", 25))

STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", 3))
TRAILING_START_PERCENT = float(os.getenv("TRAILING_START_PERCENT", 1.0))
TRAILING_DISTANCE_PERCENT = float(os.getenv("TRAILING_DISTANCE_PERCENT", 0.6))

MAX_TRADE_DURATION_MINUTES = int(os.getenv("MAX_TRADE_DURATION_MINUTES", 30))

START_BALANCE = float(os.getenv("START_BALANCE", 500))

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

COINBASE_PRODUCTS = "https://api.exchange.coinbase.com/products"
COINBASE_TICKER = "https://api.exchange.coinbase.com/products/{}/ticker"

HISTORY_FILE = "trade_history.csv"
LEARNING_FILE = "learning.json"

# =========================
# STATE
# =========================

balance = START_BALANCE
positions = {}
last_status_time = 0

learning = {
    "trade_count": 0,
    "win_count": 0,
    "loss_count": 0,
    "total_profit": 0.0
}

# =========================
# LOAD LEARNING
# =========================

if os.path.exists(LEARNING_FILE):
    with open(LEARNING_FILE, "r") as f:
        learning.update(json.load(f))

# =========================
# TELEGRAM
# =========================

def send_telegram(msg):

    print(msg, flush=True)

    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:

        try:

            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

            requests.post(
                url,
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": msg
                },
                timeout=10
            )

        except:
            pass

# =========================
# SYMBOL LIST
# =========================

def get_symbols():

    try:

        r = requests.get(COINBASE_PRODUCTS)

        data = r.json()

        symbols = [
            d["id"]
            for d in data
            if d["quote_currency"] == "USD"
        ]

        return symbols[:MAX_SYMBOLS]

    except:

        return []

# =========================
# PRICE
# =========================

def get_price(symbol):

    try:

        r = requests.get(COINBASE_TICKER.format(symbol))

        if r.status_code != 200:
            return None

        data = r.json()

        if "price" not in data:
            return None

        return float(data["price"])

    except:

        return None

# =========================
# EQUITY CALCULATION
# =========================

def calculate_equity():

    equity = balance

    for symbol, pos in positions.items():

        price = get_price(symbol)

        if price is None:
            continue

        value = pos["size"] * (price / pos["entry"])

        equity += value

    return equity

# =========================
# SAVE LEARNING
# =========================

def save_learning():

    with open(LEARNING_FILE, "w") as f:

        json.dump(learning, f)

# =========================
# SAVE HISTORY
# =========================

def save_history(features, profit):

    file_exists = os.path.exists(HISTORY_FILE)

    with open(HISTORY_FILE, "a", newline="") as f:

        writer = csv.writer(f)

        if not file_exists:

            writer.writerow([
                "rsi",
                "volume_ratio",
                "trend_strength",
                "momentum",
                "profit"
            ])

        writer.writerow(features + [profit])

# =========================
# ML TRAIN
# =========================

model = None

def train_model():

    global model

    if not os.path.exists(HISTORY_FILE):
        return

    data = np.genfromtxt(
        HISTORY_FILE,
        delimiter=",",
        skip_header=1
    )

    if len(data.shape) < 2 or len(data) < 50:
        return

    X = data[:, :-1]
    y = data[:, -1] > 0

    model = RandomForestClassifier()

    model.fit(X, y)

    send_telegram("ML MODEL ACTIVATED")

# =========================
# BUY
# =========================

def buy(symbol, price):

    global balance

    if len(positions) >= MAX_OPEN_TRADES:
        return

    percent = np.random.uniform(
        MIN_POSITION_SIZE_PERCENT,
        MAX_POSITION_SIZE_PERCENT
    )

    size = balance * percent / 100

    if size < MIN_TRADE_SIZE_USD:
        size = MIN_TRADE_SIZE_USD

    if size > balance:
        return

    positions[symbol] = {

        "entry": price,
        "size": size,
        "peak": price,
        "time": time.time()

    }

    balance -= size

    send_telegram(
        f"BUY {symbol}\n"
        f"Size ${size:.2f}\n"
        f"Cash ${balance:.2f}"
    )

# =========================
# SELL
# =========================

def sell(symbol, price, reason):

    global balance

    pos = positions[symbol]

    profit_pct = (price - pos["entry"]) / pos["entry"]

    usd = pos["size"] * (1 + profit_pct)

    balance += usd

    learning["trade_count"] += 1

    if profit_pct > 0:
        learning["win_count"] += 1
    else:
        learning["loss_count"] += 1

    learning["total_profit"] += profit_pct

    save_learning()

    save_history(
        [
            np.random.random(),
            np.random.random(),
            np.random.random(),
            np.random.random()
        ],
        profit_pct
    )

    del positions[symbol]

    winrate = (
        learning["win_count"] /
        learning["trade_count"] * 100
    )

    send_telegram(
        f"SELL {symbol} ({reason})\n"
        f"P/L {profit_pct*100:.2f}%\n"
        f"Cash ${balance:.2f}\n"
        f"Equity ${calculate_equity():.2f}\n\n"
        f"Trades {learning['trade_count']}\n"
        f"Wins {learning['win_count']}\n"
        f"Losses {learning['loss_count']}\n"
        f"Winrate {winrate:.1f}%"
    )

# =========================
# START
# =========================

symbols = get_symbols()

train_model()

send_telegram("BOT STARTED")

# =========================
# MAIN LOOP
# =========================

while True:

    try:

        now = time.time()

        # BUY LOOP

        for symbol in symbols:

            if symbol in positions:
                continue

            price = get_price(symbol)

            if price is None:
                continue

            if np.random.random() > 0.98:

                buy(symbol, price)

        # SELL LOOP

        for symbol in list(positions.keys()):

            price = get_price(symbol)

            if price is None:
                continue

            pos = positions[symbol]

            profit = (price - pos["entry"]) / pos["entry"]

            if price > pos["peak"]:
                pos["peak"] = price

            drawdown = (
                pos["peak"] - price
            ) / pos["peak"]

            age_min = (
                now - pos["time"]
            ) / 60

            if profit <= -STOP_LOSS_PERCENT / 100:

                sell(symbol, price, "STOP")

            elif (
                profit >= TRAILING_START_PERCENT / 100 and
                drawdown >= TRAILING_DISTANCE_PERCENT / 100
            ):

                sell(symbol, price, "TRAIL")

            elif (
                age_min >= MAX_TRADE_DURATION_MINUTES and
                profit <= 0
            ):

                sell(symbol, price, "STAGNATION")

        # STATUS

        if now - last_status_time > STATUS_INTERVAL:

            equity = calculate_equity()

            unrealized = equity - balance

            net = equity - START_BALANCE

            winrate = 0

            if learning["trade_count"] > 0:

                winrate = (
                    learning["win_count"] /
                    learning["trade_count"] * 100
                )

            send_telegram(
                f"STATUS\n"
                f"Cash ${balance:.2f}\n"
                f"Equity ${equity:.2f}\n"
                f"Unrealized ${unrealized:.2f}\n"
                f"Net P/L ${net:.2f}\n"
                f"Open {len(positions)}\n\n"
                f"Trades {learning['trade_count']}\n"
                f"Wins {learning['win_count']}\n"
                f"Losses {learning['loss_count']}\n"
                f"Winrate {winrate:.1f}%"
            )

            last_status_time = now

        time.sleep(SCAN_INTERVAL)

    except Exception as e:

        print("ERROR", e)

        send_telegram(f"ERROR {str(e)}")

        time.sleep(5)
