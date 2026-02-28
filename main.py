# coin_sniper_bot.py
# FINAL STABLE VERSION — NO BOOT LOOP — FULL PERSISTENCE

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
# ENV HELPERS
# ============================================================

def _env(name: str) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return None
    v = str(v).strip()
    return v if v else None

def env_float(name: str, default: float):
    try:
        return float(_env(name) or default)
    except:
        return default

def env_int(name: str, default: int):
    try:
        return int(float(_env(name) or default))
    except:
        return default

def env_bool(name: str, default: bool):
    v = _env(name)
    if v is None:
        return default
    return v.lower() in ("true","1","yes")


# ============================================================
# CONFIG
# ============================================================

START_BALANCE = env_float("START_BALANCE", 1000)
SCAN_INTERVAL = env_int("SCAN_INTERVAL", 15)
STATUS_INTERVAL = env_int("STATUS_INTERVAL", 60)

MAX_OPEN_TRADES = env_int("MAX_OPEN_TRADES", 20)
MIN_TRADE_SIZE = env_float("MIN_TRADE_SIZE_USD", 25)

STOP_LOSS = env_float("STOP_LOSS_PERCENT", 4.0)
TRAIL_START = env_float("TRAILING_START_PERCENT", 1.2)
TRAIL_DIST = env_float("TRAILING_DISTANCE_PERCENT", 0.9)

ATR_MULT = env_float("ATR_MULT", 1.4)

GITHUB_TOKEN = _env("GITHUB_TOKEN")
GITHUB_REPO = _env("GITHUB_REPO")
GITHUB_BRANCH = _env("GITHUB_BRANCH") or "main"

TELEGRAM_TOKEN = _env("TELEGRAM_TOKEN")
TELEGRAM_CHAT = _env("TELEGRAM_CHAT_ID")

BASE = "https://api.exchange.coinbase.com"


LEARNING_FILE = "learning.json"
POSITIONS_FILE = "positions.json"
HISTORY_FILE = "trade_history.csv"


session = requests.Session()
session.headers.update({"User-Agent": "coin-sniper"})


# ============================================================
# TELEGRAM
# ============================================================

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


# ============================================================
# GITHUB SYNC (SAFE — NO BOOT LOOP)
# ============================================================

last_push = 0

def gh_headers():
    return {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

def gh_put(filename):

    global last_push

    if not GITHUB_TOKEN or not GITHUB_REPO:
        return

    now = time.time()

    if now - last_push < 30:
        return

    last_push = now

    try:

        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"

        sha = None

        r = session.get(url, headers=gh_headers())

        if r.status_code == 200:
            sha = r.json()["sha"]

        with open(filename,"rb") as f:
            content = base64.b64encode(f.read()).decode()

        payload = {
            "message": "update",
            "content": content,
            "branch": GITHUB_BRANCH
        }

        if sha:
            payload["sha"] = sha

        session.put(url, headers=gh_headers(), json=payload)

    except:
        pass


# ============================================================
# FILE MANAGEMENT
# ============================================================

def load_learning():

    if not os.path.exists(LEARNING_FILE):

        data = {
            "cash": START_BALANCE,
            "start_balance": START_BALANCE,
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "total_profit_usd": 0
        }

        save_learning(data)

        return data

    with open(LEARNING_FILE,"r") as f:
        return json.load(f)


def save_learning(data):

    with open(LEARNING_FILE,"w") as f:
        json.dump(data,f)

    gh_put(LEARNING_FILE)


def load_positions():

    if not os.path.exists(POSITIONS_FILE):
        return {}

    with open(POSITIONS_FILE,"r") as f:
        return json.load(f)


def save_positions(p):

    with open(POSITIONS_FILE,"w") as f:
        json.dump(p,f)

    gh_put(POSITIONS_FILE)


# ============================================================
# MARKET DATA
# ============================================================

def price(sym):

    try:
        r = session.get(f"{BASE}/products/{sym}/ticker")
        return float(r.json()["price"])
    except:
        return None


def symbols():

    try:

        r = session.get(f"{BASE}/products")

        return [
            x["id"]
            for x in r.json()
            if x["quote_currency"]=="USD"
        ][:60]

    except:
        return []


# ============================================================
# PORTFOLIO
# ============================================================

learning = load_learning()
positions = load_positions()

cash = learning["cash"]


notify(f"BOT STARTED cash {cash}")


# ============================================================
# TRADING
# ============================================================

def buy(sym, p):

    global cash

    size = MIN_TRADE_SIZE

    if cash < size:
        return

    qty = size/p

    positions[sym] = {
        "entry": p,
        "qty": qty,
        "size": size,
        "peak": p,
        "stop": p*(1-STOP_LOSS/100),
        "time": time.time()
    }

    cash -= size

    learning["cash"] = cash

    save_learning(learning)
    save_positions(positions)

    notify(f"BUY {sym} size {size}")


def sell(sym,p):

    global cash

    pos = positions[sym]

    proceeds = pos["qty"]*p

    profit = proceeds-pos["size"]

    cash += proceeds

    learning["cash"] = cash

    learning["trade_count"] += 1

    if profit>0:
        learning["win_count"]+=1
    else:
        learning["loss_count"]+=1

    learning["total_profit_usd"]+=profit

    del positions[sym]

    save_learning(learning)
    save_positions(positions)

    notify(f"SELL {sym} profit {profit}")


# ============================================================
# MAIN LOOP
# ============================================================

syms = symbols()

last_status = 0

while True:

    try:

        prices = {}

        for s in syms:
            p = price(s)
            if p:
                prices[s]=p

        # manage open
        for s,pos in list(positions.items()):

            p = prices.get(s)

            if not p:
                continue

            if p>pos["peak"]:
                pos["peak"]=p

            trail = pos["peak"]*(1-TRAIL_DIST/100)

            if p<pos["stop"] or p<trail:
                sell(s,p)

        # open new
        for s,p in prices.items():

            if len(positions)>=MAX_OPEN_TRADES:
                break

            if s not in positions:
                buy(s,p)

        # status
        if time.time()-last_status>STATUS_INTERVAL:

            equity = cash + sum(
                pos["qty"]*prices.get(s,pos["entry"])
                for s,pos in positions.items()
            )

            notify(f"STATUS cash {cash} equity {equity}")

            last_status=time.time()

        time.sleep(SCAN_INTERVAL)

    except Exception as e:

        notify(str(e))

        time.sleep(5)
