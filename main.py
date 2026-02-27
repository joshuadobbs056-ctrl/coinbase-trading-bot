import os
import time
import json
import csv
import requests
import numpy as np
from sklearn.ensemble import RandomForestClassifier

#########################################
# CONFIG
#########################################

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 60))
MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", 6))
MIN_SCORE = float(os.getenv("MIN_SCORE", 3))

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

#########################################
# TELEGRAM
#########################################

def send_telegram(msg):

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

        requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": msg
        })

    except:
        pass


#########################################
# LEARNING SYSTEM (LEVEL 1)
#########################################

LEARNING_FILE = "learning.json"
HISTORY_FILE = "trade_history.csv"

def load_learning():

    if os.path.exists(LEARNING_FILE):

        with open(LEARNING_FILE, "r") as f:
            return json.load(f)

    return {
        "weights": {
            "rsi": 1.0,
            "volume": 1.0,
            "trend": 1.0,
            "momentum": 1.0
        },
        "trade_count": 0,
        "win_count": 0
    }


def save_learning():

    with open(LEARNING_FILE, "w") as f:
        json.dump(learning, f)


learning = load_learning()


#########################################
# HISTORY FILE INIT
#########################################

if not os.path.exists(HISTORY_FILE):

    with open(HISTORY_FILE, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "rsi",
            "volume_ratio",
            "trend_strength",
            "momentum",
            "profit"
        ])


#########################################
# MACHINE LEARNING (LEVEL 2)
#########################################

model = None


def train_model():

    global model

    try:

        data = np.loadtxt(HISTORY_FILE, delimiter=",", skiprows=1)

        if len(data) < 20:
            return

        X = data[:, :-1]

        y = (data[:, -1] > 0).astype(int)

        model = RandomForestClassifier(n_estimators=100)

        model.fit(X, y)

        send_telegram("AI model trained on trade history")

    except:
        pass


#########################################
# SCORING
#########################################

def calculate_score(rsi, volume_ratio, trend_strength, momentum):

    weights = learning["weights"]

    score = 0

    if rsi > 55:
        score += weights["rsi"]

    if volume_ratio > 1.5:
        score += weights["volume"]

    if trend_strength > 0.6:
        score += weights["trend"]

    if momentum > 0:
        score += weights["momentum"]

    return score


#########################################
# ML FILTER
#########################################

def ml_allows(features):

    if model is None:
        return True

    try:

        prob = model.predict_proba([features])[0][1]

        return prob > 0.6

    except:
        return True


#########################################
# RECORD TRADE RESULT
#########################################

def record_trade(rsi, volume_ratio, trend_strength, momentum, profit):

    with open(HISTORY_FILE, "a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            rsi,
            volume_ratio,
            trend_strength,
            momentum,
            profit
        ])

    learning["trade_count"] += 1

    if profit > 0:

        learning["win_count"] += 1

        learning["weights"]["volume"] += 0.02
        learning["weights"]["trend"] += 0.02

    else:

        learning["weights"]["volume"] -= 0.01

    save_learning()

    train_model()


#########################################
# MOCK MARKET DATA (REPLACE WITH REAL)
#########################################

def get_market_data():

    # Replace this with real exchange API

    rsi = np.random.uniform(40, 70)
    volume_ratio = np.random.uniform(1.0, 2.5)
    trend_strength = np.random.uniform(0.4, 1.0)
    momentum = np.random.uniform(-1, 1)

    return rsi, volume_ratio, trend_strength, momentum


#########################################
# TRADE STORAGE
#########################################

open_trades = []


#########################################
# BUY
#########################################

def open_trade():

    global open_trades

    if len(open_trades) >= MAX_OPEN_TRADES:
        return

    rsi, volume_ratio, trend_strength, momentum = get_market_data()

    score = calculate_score(
        rsi,
        volume_ratio,
        trend_strength,
        momentum
    )

    features = [
        rsi,
        volume_ratio,
        trend_strength,
        momentum
    ]

    if score >= MIN_SCORE and ml_allows(features):

        trade = {
            "rsi": rsi,
            "volume_ratio": volume_ratio,
            "trend_strength": trend_strength,
            "momentum": momentum,
            "entry_time": time.time(),
            "entry_price": np.random.uniform(90, 110)
        }

        open_trades.append(trade)

        send_telegram(f"BUY opened | score {round(score,2)}")


#########################################
# SELL SIMULATION
#########################################

def manage_trades():

    global open_trades

    remaining = []

    for trade in open_trades:

        if time.time() - trade["entry_time"] > 120:

            exit_price = np.random.uniform(90, 110)

            profit = exit_price - trade["entry_price"]

            record_trade(
                trade["rsi"],
                trade["volume_ratio"],
                trade["trend_strength"],
                trade["momentum"],
                profit
            )

            send_telegram(
                f"SELL closed | profit {round(profit,2)}"
            )

        else:

            remaining.append(trade)

    open_trades = remaining


#########################################
# MAIN LOOP
#########################################

send_telegram("AI adaptive bot started")

train_model()

while True:

    try:

        open_trade()

        manage_trades()

        time.sleep(SCAN_INTERVAL)

    except Exception as e:

        send_telegram(f"Error: {str(e)}")

        time.sleep(10)
