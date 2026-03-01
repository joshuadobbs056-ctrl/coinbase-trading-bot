# ============================================================
# COIN SNIPER BOT — STABLE BUILD (NO LOOP VERSION)
# JD SAFE EDITION
# ============================================================

import os
import time
import json
import csv
import requests
import traceback
import numpy as np
from typing import Dict, List, Optional
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# CONFIG
# ============================================================

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 15))
STATUS_INTERVAL = int(os.getenv("STATUS_INTERVAL", 300))

START_BALANCE = float(os.getenv("START_BALANCE", 1000))
MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", 20))

MIN_TRADE_SIZE = float(os.getenv("MIN_TRADE_SIZE", 25))
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", 4.0))
TRAIL_START = float(os.getenv("TRAIL_START", 1.2))
TRAIL_DISTANCE = float(os.getenv("TRAIL_DISTANCE", 0.9))

ML_MIN_TRADES = int(os.getenv("ML_MIN_TRADES", 50))

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

BASE_URL = "https://api.exchange.coinbase.com"

LEARNING_FILE = "learning.json"
POSITIONS_FILE = "positions.json"
HISTORY_FILE = "trade_history.csv"

COOLDOWN_SECONDS = 1800

# ============================================================
# TELEGRAM
# ============================================================

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


# ============================================================
# FILE CREATION
# ============================================================

def ensure_learning():

    if not os.path.exists(LEARNING_FILE):

        data = {
            "cash": START_BALANCE,
            "start_balance": START_BALANCE,
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "total_profit": 0.0
        }

        with open(LEARNING_FILE, "w") as f:
            json.dump(data, f)

        return data

    with open(LEARNING_FILE, "r") as f:
        return json.load(f)


def save_learning(data):

    with open(LEARNING_FILE, "w") as f:
        json.dump(data, f)


def ensure_positions():

    if not os.path.exists(POSITIONS_FILE):

        with open(POSITIONS_FILE, "w") as f:
            json.dump({}, f)

        return {}

    with open(POSITIONS_FILE, "r") as f:
        return json.load(f)


def save_positions(data):

    with open(POSITIONS_FILE, "w") as f:
        json.dump(data, f)


def ensure_history():

    if not os.path.exists(HISTORY_FILE):

        with open(HISTORY_FILE, "w", newline="") as f:

            writer = csv.writer(f)
            writer.writerow(["profit"])


def append_history(profit):

    with open(HISTORY_FILE, "a", newline="") as f:

        writer = csv.writer(f)
        writer.writerow([profit])


# ============================================================
# MARKET DATA
# ============================================================

def get_symbols():

    r = requests.get(BASE_URL + "/products")

    return [x["id"] for x in r.json() if x["quote_currency"] == "USD"][:60]


def get_price(symbol):

    r = requests.get(BASE_URL + f"/products/{symbol}/ticker")

    return float(r.json()["price"])


# ============================================================
# LEARNING STATE
# ============================================================

learning = ensure_learning()

cash = learning["cash"]

positions = ensure_positions()

ensure_history()

last_exit = {}

model = None


# ============================================================
# ML TRAIN
# ============================================================

def train_model():

    global model

    if learning["trade_count"] < ML_MIN_TRADES:

        model = None
        return

    data = np.genfromtxt(HISTORY_FILE, delimiter=",", skip_header=1)

    X = np.arange(len(data)).reshape(-1,1)

    y = (data > 0).astype(int)

    model = RandomForestClassifier()

    model.fit(X,y)

    notify("ML ACTIVATED")


# ============================================================
# EQUITY
# ============================================================

def equity(prices):

    total = cash

    for sym in positions:

        total += positions[sym]["qty"] * prices.get(sym,0)

    return total


# ============================================================
# BUY
# ============================================================

def buy(sym, price):

    global cash

    if sym in last_exit:

        if time.time() - last_exit[sym] < COOLDOWN_SECONDS:

            return

    if len(positions) >= MAX_OPEN_TRADES:

        return

    if cash < MIN_TRADE_SIZE:

        return

    size = MIN_TRADE_SIZE

    qty = size / price

    positions[sym] = {
        "entry": price,
        "qty": qty,
        "peak": price,
        "stop": price*(1-STOP_LOSS_PERCENT/100),
        "time": time.time()
    }

    cash -= size

    learning["cash"] = cash

    save_learning(learning)
    save_positions(positions)

    notify(f"BUY {sym}\nPrice: {price:.4f}\nCash: {cash:.2f}")


# ============================================================
# SELL
# ============================================================

def sell(sym, price):

    global cash

    pos = positions[sym]

    proceeds = pos["qty"]*price

    profit = proceeds - (pos["qty"]*pos["entry"])

    cash += proceeds

    learning["cash"] = cash
    learning["trade_count"] += 1
    learning["total_profit"] += profit

    if profit>0:
        learning["win_count"] += 1
    else:
        learning["loss_count"] += 1

    append_history(profit)

    last_exit[sym] = time.time()

    del positions[sym]

    save_learning(learning)
    save_positions(positions)

    notify(f"SELL {sym}\nProfit: {profit:.2f}\nCash: {cash:.2f}")


# ============================================================
# START
# ============================================================

notify(f"BOT STARTED\nCash: {cash:.2f}\nOpen Trades: {len(positions)}")

symbols = get_symbols()

last_status = time.time()

# ============================================================
# MAIN LOOP
# ============================================================

while True:

    try:

        prices = {}

        for sym in symbols:

            prices[sym] = get_price(sym)

        for sym in list(positions):

            pos = positions[sym]

            price = prices[sym]

            if price>pos["peak"]:
                pos["peak"]=price

            if price<=pos["stop"]:
                sell(sym,price)
                continue

            trail = pos["peak"]*(1-TRAIL_DISTANCE/100)

            if price<=trail:
                sell(sym,price)

        for sym in symbols:

            if sym not in positions:

                buy(sym,prices[sym])

        if time.time()-last_status>STATUS_INTERVAL:

            eq = equity(prices)

            wins=learning["win_count"]
            losses=learning["loss_count"]
            trades=learning["trade_count"]

            winrate=(wins/trades*100) if trades>0 else 0

            notify(
                f"STATUS\n"
                f"Cash: {cash:.2f}\n"
                f"Equity: {eq:.2f}\n"
                f"Open Trades: {len(positions)}\n"
                f"Trades: {trades}\n"
                f"Wins: {wins}\n"
                f"Losses: {losses}\n"
                f"Winrate: {winrate:.2f}%"
            )

            last_status=time.time()

        time.sleep(SCAN_INTERVAL)

    except Exception as e:

        notify(f"ERROR\n{str(e)}\n{traceback.format_exc()}")

        time.sleep(10)
