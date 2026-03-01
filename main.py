# main.py
# Coin Sniper — Savage Mode (Stable, Persistent, No Loop Version)

import os
import time
import json
import csv
import traceback
import requests
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# =========================
# CONFIG
# =========================

START_BALANCE = float(os.getenv("START_BALANCE", 1000))
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 15))
STATUS_INTERVAL = int(os.getenv("STATUS_INTERVAL", 60))

MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", 20))
MIN_TRADE_SIZE = float(os.getenv("MIN_TRADE_SIZE", 25))

STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", 4.0))
TRAIL_START = float(os.getenv("TRAILING_START_PERCENT", 1.2))
TRAIL_DIST = float(os.getenv("TRAILING_DISTANCE_PERCENT", 0.9))

COOLDOWN_SECONDS = 180  # prevents instant rebuy

ML_ENABLE_AFTER = 50

BASE_URL = "https://api.exchange.coinbase.com/products"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")

# =========================
# FILES
# =========================

LEARNING_FILE = "learning.json"
POSITIONS_FILE = "positions.json"
HISTORY_FILE = "trade_history.csv"
COOLDOWN_FILE = "cooldown.json"

# =========================
# TELEGRAM
# =========================

def notify(msg):
    print(msg)
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
                timeout=10,
            )
        except:
            pass

# =========================
# SAFE LOAD/SAVE
# =========================

def load_json(file, default):
    if os.path.exists(file):
        try:
            with open(file, "r") as f:
                return json.load(f)
        except:
            pass
    return default

def save_json(file, data):
    with open(file, "w") as f:
        json.dump(data, f)

# =========================
# STATE
# =========================

learning = load_json(LEARNING_FILE, {
    "cash": START_BALANCE,
    "start_balance": START_BALANCE,
    "trade_count": 0,
    "win_count": 0,
    "loss_count": 0,
    "total_profit": 0
})

positions = load_json(POSITIONS_FILE, {})
cooldown = load_json(COOLDOWN_FILE, {})

cash = learning["cash"]

# =========================
# HISTORY FILE
# =========================

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        f.write("profit\n")

# =========================
# MARKET DATA SAFE
# =========================

def get_price(symbol):
    try:
        r = requests.get(f"{BASE_URL}/{symbol}/ticker", timeout=10)
        data = r.json()
        if "price" not in data:
            return None
        return float(data["price"])
    except:
        return None

def get_symbols():
    try:
        r = requests.get(BASE_URL, timeout=10)
        data = r.json()
        return [
            p["id"] for p in data
            if p["quote_currency"] == "USD"
        ][:60]
    except:
        return []

symbols = get_symbols()

# =========================
# ML MODEL
# =========================

model = None

def train_model():
    global model
    try:
        data = np.genfromtxt(HISTORY_FILE, delimiter=",", skip_header=1)
        if len(data) < ML_ENABLE_AFTER:
            return
        X = np.arange(len(data)).reshape(-1,1)
        y = (data > 0).astype(int)
        model = RandomForestClassifier()
        model.fit(X,y)
    except:
        pass

# =========================
# EQUITY
# =========================

def equity(prices):
    total = cash
    for sym,pos in positions.items():
        if sym in prices:
            total += pos["qty"] * prices[sym]
    return total

# =========================
# OPEN TRADE
# =========================

def open_trade(sym, price):

    global cash

    if sym in positions:
        return

    if len(positions) >= MAX_OPEN_TRADES:
        return

    now = time.time()

    if sym in cooldown and now - cooldown[sym] < COOLDOWN_SECONDS:
        return

    size = min(MIN_TRADE_SIZE, cash)

    if size < MIN_TRADE_SIZE:
        return

    qty = size / price

    positions[sym] = {
        "entry": price,
        "qty": qty,
        "peak": price,
        "stop": price*(1-STOP_LOSS_PERCENT/100)
    }

    cash -= size

    learning["cash"] = cash

    notify(
        f"BUY {sym}\n"
        f"Price: {price:.6f}\n"
        f"Size: ${size:.2f}\n"
        f"Cash: ${cash:.2f}"
    )

# =========================
# SELL TRADE
# =========================

def sell_trade(sym, price):

    global cash

    pos = positions[sym]

    proceeds = pos["qty"] * price

    size = pos["qty"] * pos["entry"]

    profit = proceeds - size

    cash += proceeds

    learning["cash"] = cash
    learning["trade_count"] += 1

    if profit > 0:
        learning["win_count"] += 1
    else:
        learning["loss_count"] += 1

    learning["total_profit"] += profit

    with open(HISTORY_FILE,"a") as f:
        f.write(f"{profit}\n")

    cooldown[sym] = time.time()

    del positions[sym]

    notify(
        f"SELL {sym}\n"
        f"Profit: {profit:.2f}\n"
        f"Cash: ${cash:.2f}"
    )

# =========================
# SAVE
# =========================

def save_all():
    save_json(LEARNING_FILE, learning)
    save_json(POSITIONS_FILE, positions)
    save_json(COOLDOWN_FILE, cooldown)

# =========================
# START
# =========================

notify(
    f"BOT STARTED (SAVAGE MODE)\n"
    f"Cash: ${cash:.2f}\n"
    f"Open Trades: {len(positions)}"
)

last_status = time.time()

# =========================
# LOOP
# =========================

while True:

    try:

        prices = {}

        for sym in symbols:

            price = get_price(sym)

            if price:
                prices[sym] = price

        # Manage sells
        for sym in list(positions):

            price = prices.get(sym)

            if not price:
                continue

            pos = positions[sym]

            if price > pos["peak"]:
                pos["peak"] = price

            trail = pos["peak"]*(1-TRAIL_DIST/100)

            if price <= pos["stop"] or price <= trail:
                sell_trade(sym, price)

        # Open new trades
        for sym in symbols:

            if sym in prices:
                open_trade(sym, prices[sym])

        # Status
        if time.time() - last_status > STATUS_INTERVAL:

            eq = equity(prices)

            trades = learning["trade_count"]
            wins = learning["win_count"]
            losses = learning["loss_count"]

            winrate = (wins/trades*100) if trades>0 else 0

            notify(
                f"STATUS\n"
                f"Cash: ${cash:.2f}\n"
                f"Equity: ${eq:.2f}\n"
                f"Profit: ${learning['total_profit']:.2f}\n"
                f"Open Trades: {len(positions)}\n"
                f"Trades: {trades}\n"
                f"Wins: {wins}\n"
                f"Losses: {losses}\n"
                f"Winrate: {winrate:.1f}%"
            )

            last_status = time.time()

        save_all()

        time.sleep(SCAN_INTERVAL)

    except Exception as e:

        notify(
            f"ERROR\n{str(e)}\n{traceback.format_exc()}"
        )

        time.sleep(10)
