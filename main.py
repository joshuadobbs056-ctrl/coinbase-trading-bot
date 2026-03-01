# COIN SNIPER BOT — STABLE BUILD
# Persistent balance, positions, history
# Full status reporting
# No GitHub sync loop
# No boot loop

import os
import time
import json
import csv
import requests
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ========================
# CONFIG
# ========================

START_BALANCE = float(os.getenv("START_BALANCE", 1000))
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 12))
STATUS_INTERVAL = int(os.getenv("STATUS_INTERVAL", 60))

MIN_TRADE = float(os.getenv("MIN_TRADE_SIZE_USD", 25))
MAX_OPEN = int(os.getenv("MAX_OPEN_TRADES", 20))

STOP_LOSS = float(os.getenv("STOP_LOSS_PERCENT", 4.0)) / 100
TRAIL_START = float(os.getenv("TRAILING_START_PERCENT", 1.2)) / 100
TRAIL_DIST = float(os.getenv("TRAILING_DISTANCE_PERCENT", 0.9)) / 100

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT = os.getenv("TELEGRAM_CHAT_ID")

LEARNING_FILE = "learning.json"
POSITIONS_FILE = "positions.json"
HISTORY_FILE = "trade_history.csv"

BASE = "https://api.exchange.coinbase.com"

session = requests.Session()

# ========================
# TELEGRAM
# ========================

def notify(msg):

    print(msg, flush=True)

    if TELEGRAM_TOKEN and TELEGRAM_CHAT:

        try:

            session.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT, "text": msg},
                timeout=10
            )

        except:
            pass

# ========================
# FILE INIT
# ========================

def ensure_files():

    if not os.path.exists(LEARNING_FILE):

        learning = {
            "cash": START_BALANCE,
            "start_balance": START_BALANCE,
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "total_profit": 0
        }

        json.dump(learning, open(LEARNING_FILE, "w"))

    if not os.path.exists(POSITIONS_FILE):

        json.dump({}, open(POSITIONS_FILE, "w"))

    if not os.path.exists(HISTORY_FILE):

        csv.writer(open(HISTORY_FILE,"w")).writerow(
            ["profit"]
        )

ensure_files()

# ========================
# LOAD STATE
# ========================

learning = json.load(open(LEARNING_FILE))
positions = json.load(open(POSITIONS_FILE))

cash = learning["cash"]

# ========================
# SAVE STATE
# ========================

def save():

    learning["cash"] = cash

    json.dump(learning, open(LEARNING_FILE,"w"))
    json.dump(positions, open(POSITIONS_FILE,"w"))

# ========================
# MARKET DATA
# ========================

def get_price(symbol):

    try:

        r = session.get(f"{BASE}/products/{symbol}/ticker", timeout=10)

        return float(r.json()["price"])

    except:

        return None

def get_symbols():

    r = session.get(f"{BASE}/products", timeout=10)

    return [
        x["id"]
        for x in r.json()
        if x["quote_currency"] == "USD"
    ][:60]

symbols = get_symbols()

# ========================
# EQUITY
# ========================

def equity():

    eq = cash

    for sym,pos in positions.items():

        p = get_price(sym)

        if p:

            eq += pos["qty"] * p

    return eq

# ========================
# BUY
# ========================

def buy(sym, price):

    global cash

    if cash < MIN_TRADE:
        return

    size = MIN_TRADE

    qty = size / price

    positions[sym] = {

        "entry": price,
        "qty": qty,
        "peak": price,
        "stop": price * (1-STOP_LOSS)

    }

    cash -= size

    save()

    notify(
        f"BUY {sym}\n"
        f"Price: {price:.2f}\n"
        f"Cash: {cash:.2f}"
    )

# ========================
# SELL
# ========================

def sell(sym, price):

    global cash

    pos = positions[sym]

    proceeds = pos["qty"] * price

    profit = proceeds - (pos["qty"] * pos["entry"])

    cash += proceeds

    learning["trade_count"] += 1

    if profit > 0:
        learning["win_count"] += 1
    else:
        learning["loss_count"] += 1

    learning["total_profit"] += profit

    del positions[sym]

    save()

    notify(
        f"SELL {sym}\n"
        f"Profit: {profit:.2f}\n"
        f"Cash: {cash:.2f}"
    )

# ========================
# BOT START
# ========================

notify(
    f"BOT STARTED\n"
    f"Cash: {cash:.2f}\n"
    f"Open Positions: {len(positions)}"
)

last_status = 0

# ========================
# MAIN LOOP
# ========================

while True:

    try:

        prices = {}

        for sym in list(positions.keys()):

            p = get_price(sym)

            if not p:
                continue

            prices[sym] = p

            pos = positions[sym]

            if p > pos["peak"]:
                pos["peak"] = p

            trail = pos["peak"] * (1-TRAIL_DIST)

            if p < pos["stop"] or p < trail:

                sell(sym,p)

        for sym in symbols:

            if sym in positions:
                continue

            if len(positions) >= MAX_OPEN:
                break

            p = get_price(sym)

            if p:
                buy(sym,p)

        if time.time() - last_status > STATUS_INTERVAL:

            eq = equity()

            trades = learning["trade_count"]
            wins = learning["win_count"]

            winrate = (wins/trades*100) if trades else 0

            notify(
                f"STATUS\n"
                f"Cash: {cash:.2f}\n"
                f"Equity: {eq:.2f}\n"
                f"Open Trades: {len(positions)}\n"
                f"Trades: {trades}\n"
                f"Winrate: {winrate:.1f}%"
            )

            last_status = time.time()

        time.sleep(SCAN_INTERVAL)

    except Exception as e:

        notify(f"ERROR {e}")

        time.sleep(5)
