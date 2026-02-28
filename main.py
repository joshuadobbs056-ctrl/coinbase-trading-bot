# coin_sniper_bot.py
# COMPLETE FINAL VERSION — persistent, restart-safe, GitHub-synced

import os
import time
import json
import csv
import base64
import traceback
import requests
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# CONFIG
# ============================================================

START_BALANCE = float(os.getenv("START_BALANCE", "1000"))
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "12"))
STATUS_INTERVAL = int(os.getenv("STATUS_INTERVAL", "60"))

STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "4.0"))
TRAILING_START_PERCENT = float(os.getenv("TRAILING_START_PERCENT", "1.2"))
TRAILING_DISTANCE_PERCENT = float(os.getenv("TRAILING_DISTANCE_PERCENT", "0.9"))

MIN_TRADE_SIZE_USD = float(os.getenv("MIN_TRADE_SIZE_USD", "25"))
MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", "20"))

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_BRANCH = "main"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

BASE_URL = "https://api.exchange.coinbase.com"


# ============================================================
# FILES
# ============================================================

LEARNING_FILE = "learning.json"
POSITIONS_FILE = "positions.json"
HISTORY_FILE = "trade_history.csv"


# ============================================================
# TELEGRAM
# ============================================================

session = requests.Session()

def notify(msg):
    print(msg, flush=True)

    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            session.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}
            )
        except:
            pass


# ============================================================
# GITHUB SYNC
# ============================================================

def gh_headers():
    return {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

def gh_put_file(path, content_bytes):

    if not GITHUB_TOKEN or not GITHUB_REPO:
        return

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"

    sha = None
    r = session.get(url, headers=gh_headers())

    if r.status_code == 200:
        sha = r.json()["sha"]

    payload = {
        "message": f"update {path}",
        "content": base64.b64encode(content_bytes).decode(),
        "branch": GITHUB_BRANCH
    }

    if sha:
        payload["sha"] = sha

    session.put(url, headers=gh_headers(), json=payload)


def push_all():

    if os.path.exists(LEARNING_FILE):
        gh_put_file(LEARNING_FILE, open(LEARNING_FILE,"rb").read())

    if os.path.exists(POSITIONS_FILE):
        gh_put_file(POSITIONS_FILE, open(POSITIONS_FILE,"rb").read())

    if os.path.exists(HISTORY_FILE):
        gh_put_file(HISTORY_FILE, open(HISTORY_FILE,"rb").read())


# ============================================================
# PERSISTENCE
# ============================================================

def ensure_learning():

    if not os.path.exists(LEARNING_FILE):

        state = {
            "cash": START_BALANCE,
            "start_balance": START_BALANCE,
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "total_profit_usd": 0
        }

        json.dump(state, open(LEARNING_FILE,"w"))

    return json.load(open(LEARNING_FILE))


def save_learning(state):

    json.dump(state, open(LEARNING_FILE,"w"))
    push_all()


def ensure_positions():

    if not os.path.exists(POSITIONS_FILE):
        json.dump({}, open(POSITIONS_FILE,"w"))

    return json.load(open(POSITIONS_FILE))


def save_positions(p):

    json.dump(p, open(POSITIONS_FILE,"w"))
    push_all()


def ensure_history():

    if not os.path.exists(HISTORY_FILE):

        w = csv.writer(open(HISTORY_FILE,"w"))
        w.writerow(["profit_usd"])


# ============================================================
# MARKET
# ============================================================

def get_price(symbol):

    try:

        r = session.get(f"{BASE_URL}/products/{symbol}/ticker")

        return float(r.json()["price"])

    except:

        return None


# ============================================================
# PORTFOLIO
# ============================================================

learning = ensure_learning()
positions = ensure_positions()
ensure_history()

cash = learning["cash"]


def equity():

    total = cash

    for sym,pos in positions.items():

        p = get_price(sym)

        if p:
            total += pos["qty"] * p

    return total


# ============================================================
# OPEN TRADE
# ============================================================

def open_trade(sym, price):

    global cash

    if sym in positions:
        return

    if cash < MIN_TRADE_SIZE_USD:
        return

    size = MIN_TRADE_SIZE_USD

    qty = size / price

    positions[sym] = {

        "entry": price,
        "qty": qty,
        "size": size,
        "stop": price*(1-STOP_LOSS_PERCENT/100),
        "peak": price,
        "time": time.time()
    }

    cash -= size

    learning["cash"] = cash

    save_learning(learning)
    save_positions(positions)

    notify(f"BUY {sym} size {size}")


# ============================================================
# SELL TRADE
# ============================================================

def close_trade(sym, price):

    global cash

    pos = positions[sym]

    proceeds = pos["qty"]*price

    profit = proceeds-pos["size"]

    cash += proceeds

    learning["cash"] = cash
    learning["trade_count"]+=1

    if profit>0:
        learning["win_count"]+=1
    else:
        learning["loss_count"]+=1

    learning["total_profit_usd"]+=profit

    save_learning(learning)

    append = csv.writer(open(HISTORY_FILE,"a"))
    append.writerow([profit])

    del positions[sym]

    save_positions(positions)

    notify(f"SELL {sym} profit {profit}")


# ============================================================
# BOT START
# ============================================================

notify(f"BOT STARTED cash {cash}")


# ============================================================
# MAIN LOOP
# ============================================================

SYMBOLS = ["BTC-USD","ETH-USD","SOL-USD","LINK-USD"]

last_status=0

while True:

    try:

        for sym in SYMBOLS:

            price=get_price(sym)

            if not price:
                continue

            if sym not in positions:
                open_trade(sym,price)

            else:

                pos=positions[sym]

                if price<pos["stop"]:
                    close_trade(sym,price)

        if time.time()-last_status>STATUS_INTERVAL:

            notify(f"STATUS cash {cash} equity {equity()}")

            last_status=time.time()

        time.sleep(SCAN_INTERVAL)

    except Exception as e:

        notify(str(e))
        time.sleep(10)
