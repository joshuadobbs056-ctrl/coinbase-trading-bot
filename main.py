import os
import time
import json
import csv
import traceback
import requests
import numpy as np
from sklearn.ensemble import RandomForestClassifier

BASE_URL = "https://api.exchange.coinbase.com"

LEARNING_FILE = "learning.json"
POSITIONS_FILE = "positions.json"
HISTORY_FILE = "trade_history.csv"

START_BALANCE = float(os.getenv("START_BALANCE", "1000"))
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "15"))
STATUS_INTERVAL = int(os.getenv("STATUS_INTERVAL", "60"))

STOP_LOSS = float(os.getenv("STOP_LOSS_PERCENT", "2.8")) / 100
TRAIL_START = float(os.getenv("TRAILING_START_PERCENT", "1.2")) / 100
TRAIL_DIST = float(os.getenv("TRAILING_DISTANCE_PERCENT", "0.9")) / 100

MIN_TRADE = float(os.getenv("MIN_TRADE_SIZE_USD", "25"))
MAX_OPEN = int(os.getenv("MAX_OPEN_TRADES", "20"))

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT = os.getenv("TELEGRAM_CHAT_ID")

session = requests.Session()


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


def load_json(file, default):

    if not os.path.exists(file):
        return default

    try:
        with open(file) as f:
            return json.load(f)
    except:
        return default


def save_json(file, data):

    with open(file, "w") as f:
        json.dump(data, f)


learning = load_json(LEARNING_FILE, {
    "cash": START_BALANCE,
    "trade_count": 0,
    "win_count": 0,
    "loss_count": 0,
    "total_profit": 0
})

positions = load_json(POSITIONS_FILE, {})

cash = learning["cash"]


def save_state():

    learning["cash"] = cash

    save_json(LEARNING_FILE, learning)
    save_json(POSITIONS_FILE, positions)


def get_symbols():

    try:

        r = session.get(BASE_URL + "/products", timeout=10)

        if r.status_code != 200:
            return []

        data = r.json()

        return [
            x["id"] for x in data
            if x["quote_currency"] == "USD"
        ][:60]

    except:
        return []


def get_price(symbol):

    try:

        r = session.get(
            BASE_URL + f"/products/{symbol}/ticker",
            timeout=10
        )

        if r.status_code != 200:
            return None

        data = r.json()

        if "price" not in data:
            return None

        return float(data["price"])

    except:
        return None


def compute_equity(prices):

    total = cash

    for sym in positions:

        if sym in prices:
            total += positions[sym]["qty"] * prices[sym]

    return total


def open_trade(sym, price):

    global cash

    if sym in positions:
        return

    if len(positions) >= MAX_OPEN:
        return

    if cash < MIN_TRADE:
        return

    qty = MIN_TRADE / price

    positions[sym] = {
        "entry": price,
        "qty": qty,
        "peak": price
    }

    cash -= MIN_TRADE

    notify(
        f"BUY {sym}\n"
        f"Price: {price:.4f}\n"
        f"Cash: {cash:.2f}"
    )

    save_state()


def close_trade(sym, price):

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

    notify(
        f"SELL {sym}\n"
        f"Profit: {profit:.2f}\n"
        f"Cash: {cash:.2f}"
    )

    del positions[sym]

    save_state()


def manage_trades(prices):

    for sym in list(positions.keys()):

        price = prices.get(sym)

        if price is None:
            continue

        pos = positions[sym]

        entry = pos["entry"]

        if price > pos["peak"]:
            pos["peak"] = price

        trail = pos["peak"] * (1 - TRAIL_DIST)

        stop = entry * (1 - STOP_LOSS)

        if price <= stop or price <= trail:
            close_trade(sym, price)


def status(prices):

    equity = compute_equity(prices)

    trades = learning["trade_count"]
    wins = learning["win_count"]
    losses = learning["loss_count"]

    wr = (wins / trades * 100) if trades > 0 else 0

    notify(
        f"STATUS\n"
        f"Cash: {cash:.2f}\n"
        f"Equity: {equity:.2f}\n"
        f"Open Trades: {len(positions)}\n"
        f"Trades: {trades}\n"
        f"Wins: {wins}\n"
        f"Losses: {losses}\n"
        f"Winrate: {wr:.1f}%"
    )


symbols = get_symbols()

notify(
    f"BOT STARTED\n"
    f"Cash: {cash:.2f}\n"
    f"Open Trades: {len(positions)}"
)

last_status = 0

while True:

    try:

        prices = {}

        for sym in symbols:

            price = get_price(sym)

            if price is not None:
                prices[sym] = price

        manage_trades(prices)

        for sym in symbols:

            if sym not in positions and sym in prices:
                open_trade(sym, prices[sym])

        if time.time() - last_status > STATUS_INTERVAL:

            status(prices)

            last_status = time.time()

        time.sleep(SCAN_INTERVAL)

    except Exception as e:

        notify(
            "ERROR\n"
            + str(e)
            + "\n"
            + traceback.format_exc()
        )

        time.sleep(10)
