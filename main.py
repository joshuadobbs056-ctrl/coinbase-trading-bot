import os
import time
import json
import csv
import base64
import requests
import numpy as np
from sklearn.ensemble import RandomForestClassifier

#########################################
# CONFIG
#########################################

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 5))
STATUS_INTERVAL = int(os.getenv("STATUS_INTERVAL", 120))

COINS = os.getenv("COINS", "AUTO")

MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", 60))
MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", 12))

MIN_SCORE = float(os.getenv("MIN_SCORE", 3))

MIN_POSITION_SIZE_PERCENT = float(os.getenv("MIN_POSITION_SIZE_PERCENT", 3))
MAX_POSITION_SIZE_PERCENT = float(os.getenv("MAX_POSITION_SIZE_PERCENT", 8))
MIN_POSITION_SIZE_USD = float(os.getenv("MIN_POSITION_SIZE_USD", 25))

STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", 1.4))

TRAILING_START_PERCENT = float(os.getenv("TRAILING_START_PERCENT", 0.5))
TRAILING_DISTANCE_PERCENT = float(os.getenv("TRAILING_DISTANCE_PERCENT", 0.3))

MAX_TRADE_DURATION_MINUTES = int(os.getenv("MAX_TRADE_DURATION_MINUTES", 30))

MIN_CASH_RESERVE_PERCENT = float(os.getenv("MIN_CASH_RESERVE_PERCENT", 20))

START_BALANCE = float(os.getenv("START_BALANCE", 1000))

BASE_URL = "https://api.exchange.coinbase.com"

#########################################
# GITHUB STORAGE
#########################################

def github_get_file(path, default):

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"

    headers = {"Authorization": f"token {GITHUB_TOKEN}"}

    r = requests.get(url, headers=headers)

    if r.status_code == 200:

        data = r.json()

        content = base64.b64decode(data["content"]).decode()

        return json.loads(content), data["sha"]

    return default, None


def github_save_file(path, data, sha=None):

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"

    headers = {"Authorization": f"token {GITHUB_TOKEN}"}

    content = base64.b64encode(json.dumps(data, indent=2).encode()).decode()

    payload = {

        "message": "bot update",

        "content": content,

        "branch": GITHUB_BRANCH

    }

    if sha:

        payload["sha"] = sha

    requests.put(url, headers=headers, json=payload)

#########################################
# LOAD STATE
#########################################

state, state_sha = github_get_file(

    "state.json",

    {

        "cash": START_BALANCE,

        "trade_count": 0,

        "win_count": 0,

        "loss_count": 0,

        "total_profit": 0

    }

)

cash = state["cash"]

trade_count = state["trade_count"]

win_count = state["win_count"]

loss_count = state["loss_count"]

total_profit = state["total_profit"]

#########################################
# TRADE HISTORY FOR ML
#########################################

history, history_sha = github_get_file("learning.json", [])

model = None

ml_active = False

#########################################
# TELEGRAM
#########################################

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def notify(msg):

    print(msg, flush=True)

    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:

        try:

            requests.post(

                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",

                json={

                    "chat_id": TELEGRAM_CHAT_ID,

                    "text": msg

                }

            )

        except:

            pass

#########################################
# ML TRAINING
#########################################

def train_model():

    global model, ml_active

    if len(history) < 50:

        return

    X = []

    y = []

    for row in history:

        X.append(row["features"])

        y.append(1 if row["profit"] > 0 else 0)

    model = RandomForestClassifier(n_estimators=100)

    model.fit(X, y)

    ml_active = True

    notify("ML MODEL ACTIVATED")


train_model()

#########################################
# MARKET DATA
#########################################

def get_symbols():

    if COINS != "AUTO":

        return COINS.split(",")

    r = requests.get(f"{BASE_URL}/products")

    data = r.json()

    usd = [

        x["id"]

        for x in data

        if x["quote_currency"] == "USD"

    ]

    return usd[:MAX_SYMBOLS]


def get_ticker(sym):

    r = requests.get(f"{BASE_URL}/products/{sym}/ticker")

    return r.json()


def get_candles(sym):

    r = requests.get(

        f"{BASE_URL}/products/{sym}/candles",

        params={"granularity": 60}

    )

    data = r.json()

    data.reverse()

    return data[-60:]

#########################################
# INDICATORS
#########################################

def calc_features(candles):

    closes = np.array([c[4] for c in candles])

    volumes = np.array([c[5] for c in candles])

    rsi = np.mean(closes[-14:])

    vol_ratio = volumes[-1] / np.mean(volumes)

    trend = closes[-1] - closes[-20]

    momentum = closes[-1] - closes[-5]

    return [rsi, vol_ratio, trend, momentum]

#########################################
# POSITION SIZE
#########################################

def position_size():

    global cash

    percent = MIN_POSITION_SIZE_PERCENT

    size = cash * percent / 100

    size = max(size, MIN_POSITION_SIZE_USD)

    return min(size, cash)

#########################################
# TRADING
#########################################

open_trades = []


def open_trade(sym, price, features):

    global cash

    size = position_size()

    if size < MIN_POSITION_SIZE_USD:

        return

    qty = size / price

    cash -= size

    open_trades.append({

        "sym": sym,

        "entry": price,

        "qty": qty,

        "peak": price,

        "features": features,

        "time": time.time()

    })

    notify(f"BUY {sym} | Size ${size:.2f} | Cash ${cash:.2f}")

    save_state()


def close_trade(trade, price, reason):

    global cash, trade_count, win_count, loss_count, total_profit

    proceeds = trade["qty"] * price

    profit = proceeds - trade["qty"] * trade["entry"]

    cash += proceeds

    trade_count += 1

    total_profit += profit

    if profit > 0:

        win_count += 1

    else:

        loss_count += 1

    history.append({

        "features": trade["features"],

        "profit": profit

    })

    notify(

        f"SELL {trade['sym']} ({reason})\n"

        f"Profit: {profit:.2f}\n"

        f"Balance: {cash:.2f}\n"

        f"Trades: {trade_count}\n"

        f"Wins: {win_count}\n"

        f"Losses: {loss_count}"

    )

    save_state()

    github_save_file("learning.json", history, history_sha)

#########################################
# SAVE STATE
#########################################

def save_state():

    global state_sha

    state = {

        "cash": cash,

        "trade_count": trade_count,

        "win_count": win_count,

        "loss_count": loss_count,

        "total_profit": total_profit

    }

    github_save_file("state.json", state, state_sha)

#########################################
# MAIN LOOP
#########################################

notify("BOT STARTED")

symbols = get_symbols()

last_status = 0

while True:

    try:

        for trade in open_trades[:]:

            ticker = get_ticker(trade["sym"])

            price = float(ticker["price"])

            if price > trade["peak"]:

                trade["peak"] = price

            trail = trade["peak"] * (1 - TRAILING_DISTANCE_PERCENT / 100)

            if price <= trail:

                close_trade(trade, price, "TRAIL")

                open_trades.remove(trade)

        if len(open_trades) < MAX_OPEN_TRADES:

            for sym in symbols:

                ticker = get_ticker(sym)

                price = float(ticker["price"])

                candles = get_candles(sym)

                features = calc_features(candles)

                score = sum(features)

                if ml_active:

                    prob = model.predict_proba([features])[0][1]

                    if prob < 0.55:

                        continue

                if score >= MIN_SCORE:

                    open_trade(sym, price, features)

                    break

        if time.time() - last_status > STATUS_INTERVAL:

            notify(

                f"STATUS\n"

                f"Cash {cash:.2f}\n"

                f"Open {len(open_trades)}\n"

                f"Trades {trade_count}\n"

                f"Wins {win_count}\n"

                f"Losses {loss_count}\n"

                f"Profit {total_profit:.2f}"

            )

            last_status = time.time()

        time.sleep(SCAN_INTERVAL)

    except Exception as e:

        notify(f"ERROR {str(e)}")

        time.sleep(5)
