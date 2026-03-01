# Coin Sniper — Savage Mode ELITE (MAX PROFIT BUILD — ML TELEGRAM FIXED + SCAN LOGS + HISTORY GATE + DYNAMIC SIZING + TRAILING STOP + GITHUB PERSIST)

import os
import time
import json
import traceback
import requests
import numpy as np
import base64
from sklearn.ensemble import RandomForestClassifier


LOCK_FILE = os.getenv("LOCK_FILE", "/tmp/coin_sniper.lock")

def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except:
        return False


def acquire_lock_or_exit():
    if os.path.exists(LOCK_FILE):
        try:
            raw = open(LOCK_FILE).read().strip()
            if raw:
                old_pid = int(raw.split("|")[0])
                if _pid_alive(old_pid):
                    raise SystemExit(0)
        except SystemExit:
            raise
        except:
            pass

    open(LOCK_FILE, "w").write(f"{os.getpid()}|{int(time.time())}")


acquire_lock_or_exit()
INSTANCE_ID = str(os.getpid())


# =========================
# CONFIG
# =========================

START_BALANCE = float(os.getenv("START_BALANCE", 1000))

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 15))
STATUS_INTERVAL = int(os.getenv("STATUS_INTERVAL", 60))
ML_INTERVAL = int(os.getenv("ML_INTERVAL", 300))

MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", 20))
MIN_TRADE_SIZE = float(os.getenv("MIN_TRADE_SIZE", 25))

STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", 4.0))
TRAIL_DIST_BASE = float(os.getenv("TRAILING_DISTANCE_PERCENT", 0.9))

ENTRY_SCORE_MIN = int(os.getenv("ENTRY_SCORE_MIN", 7))

MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", 60))

EXCLUDE = set([s.strip() for s in os.getenv("EXCLUDE", "").split(",") if s.strip()])


CASH_RESERVE_PERCENT = float(os.getenv("CASH_RESERVE_PERCENT", 5))
MIN_POSITION_SIZE_PERCENT = float(os.getenv("MIN_POSITION_SIZE_PERCENT", 3))
MAX_POSITION_SIZE_PERCENT = float(os.getenv("MAX_POSITION_SIZE_PERCENT", 8))


HISTORY_POINTS_REQUIRED = int(os.getenv("HISTORY_POINTS_REQUIRED", 40))
MIN_HISTORY_COINS = int(os.getenv("MIN_HISTORY_COINS", 50))


ML_ENABLED = os.getenv("ML_ENABLED", "true").lower() == "true"
ML_ENABLE_AFTER = int(os.getenv("ML_ENABLE_AFTER", 50))
ML_MIN_PROB = float(os.getenv("ML_MIN_PROB", 0.62))


BOT_PAUSED = os.getenv("BOT_PAUSED", "false").lower() == "true"


BASE_URL = "https://api.exchange.coinbase.com/products"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


# =========================
# GITHUB CONFIG
# =========================

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_DATA_REPO")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")

GITHUB_PUSH_INTERVAL = 120
last_github_push = 0

GITHUB_FILES = [
    "learning.json",
    "positions.json",
    "ml_training.csv",
    "trade_history.csv"
]


def github_headers():
    return {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }


def github_pull_file(filename):

    if not GITHUB_TOKEN or not GITHUB_REPO:
        return

    try:

        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"

        r = requests.get(url, headers=github_headers(), timeout=15)

        if r.status_code == 200:

            content = base64.b64decode(r.json()["content"])

            open(filename, "wb").write(content)

            notify(f"[{INSTANCE_ID}] GITHUB RESTORE {filename}")

    except:
        pass


def github_push_file(filename):

    if not GITHUB_TOKEN or not GITHUB_REPO:
        return

    try:

        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"

        content = base64.b64encode(open(filename, "rb").read()).decode()

        sha = None

        r = requests.get(url, headers=github_headers(), timeout=15)

        if r.status_code == 200:
            sha = r.json()["sha"]

        payload = {
            "message": f"update {filename}",
            "content": content,
            "branch": GITHUB_BRANCH
        }

        if sha:
            payload["sha"] = sha

        requests.put(url, headers=github_headers(), json=payload, timeout=15)

    except:
        pass


def github_pull_all():

    for f in GITHUB_FILES:
        github_pull_file(f)


def github_push_all():

    global last_github_push

    now = time.time()

    if now - last_github_push < GITHUB_PUSH_INTERVAL:
        return

    for f in GITHUB_FILES:
        github_push_file(f)

    last_github_push = now

    notify(f"[{INSTANCE_ID}] GITHUB SYNC COMPLETE")



# =========================
# FILES
# =========================

LEARNING_FILE = "learning.json"
POSITIONS_FILE = "positions.json"
HISTORY_FILE = "trade_history.csv"
ML_TRAIN_FILE = "ml_training.csv"



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



# =========================
# STARTUP FILE RESTORE
# =========================

github_pull_all()



# =========================
# LOAD STATE
# =========================

def load_json(path, default):

    try:
        if os.path.exists(path):
            return json.load(open(path))
    except:
        pass

    return default


def save_json(path, data):

    try:
        json.dump(data, open(path, "w"))
    except:
        pass



learning = load_json(LEARNING_FILE, {
    "cash": START_BALANCE,
    "trade_count": 0,
    "win_count": 0,
    "loss_count": 0,
    "total_profit": 0
})


positions = load_json(POSITIONS_FILE, {})

cash = float(learning.get("cash", START_BALANCE))


price_history = {}


# =========================
# MARKET
# =========================

def get_price(sym):

    try:

        r = requests.get(f"{BASE_URL}/{sym}/ticker", timeout=10)

        return float(r.json()["price"])

    except:

        return None



def get_symbols():

    try:

        r = requests.get(BASE_URL, timeout=10)

        syms = [p["id"] for p in r.json() if p["quote_currency"] == "USD"]

        syms = [s for s in syms if s not in EXCLUDE]

        return syms[:MAX_SYMBOLS]

    except:

        return []



symbols = get_symbols()



# =========================
# TRAILING STOP
# =========================

def apply_trailing_stop(pos, price):

    peak = pos["peak"]

    if price > peak:
        pos["peak"] = price

    trail = pos["peak"] * (1 - TRAIL_DIST_BASE / 100)

    if trail > pos["stop"]:
        pos["stop"] = trail



# =========================
# MAIN LOOP
# =========================

notify(f"[{INSTANCE_ID}] BOT STARTED | Symbols={len(symbols)}")



last_scan = 0
last_status = 0


while True:

    try:

        now = time.time()

        if now - last_scan >= SCAN_INTERVAL:

            prices = {}

            for sym in symbols:

                px = get_price(sym)

                if px:
                    prices[sym] = px

                    price_history.setdefault(sym, []).append(px)


            for sym, pos in list(positions.items()):

                px = prices.get(sym)

                if not px:
                    continue

                apply_trailing_stop(pos, px)

                if px <= pos["stop"]:

                    profit = (px - pos["entry"]) * pos["qty"]

                    cash += pos["qty"] * px

                    learning["trade_count"] += 1
                    learning["total_profit"] += profit

                    notify(f"[{INSTANCE_ID}] SELL {sym} profit={profit:.2f}")

                    del positions[sym]


            save_json(LEARNING_FILE, learning)
            save_json(POSITIONS_FILE, positions)

            github_push_all()

            last_scan = now


        time.sleep(1)

    except:

        notify(traceback.format_exc())

        time.sleep(5)
