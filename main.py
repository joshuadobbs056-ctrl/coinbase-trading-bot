# Coin Sniper — Savage Mode ELITE (STABLE)
# ✅ Single-instance lock
# ✅ Persistent files
# ✅ GitHub backup to DATA repo (NOT deploy repo)
# ✅ Correct stats: wins/losses/open/profit/equity
# ✅ Pause support

import os
import time
import json
import traceback
import requests
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# =========================
# SINGLE INSTANCE LOCK
# =========================

LOCK_FILE = os.getenv("LOCK_FILE", "/tmp/coin_sniper.lock")

def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False

def acquire_lock_or_exit():
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, "r") as f:
                raw = f.read().strip()
            if raw:
                old_pid = int(raw.split("|")[0])
                if old_pid and _pid_alive(old_pid):
                    print(f"[LOCK] Instance already running (pid={old_pid})")
                    raise SystemExit(0)
        except SystemExit:
            raise
        except:
            pass

    with open(LOCK_FILE, "w") as f:
        f.write(f"{os.getpid()}|{int(time.time())}")

acquire_lock_or_exit()

INSTANCE_ID = str(os.getpid())

# =========================
# CONFIG
# =========================

START_BALANCE = float(os.getenv("START_BALANCE", 1000))

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 15))
STATUS_INTERVAL = int(os.getenv("STATUS_INTERVAL", 60))

SAVE_INTERVAL = int(os.getenv("SAVE_INTERVAL", 60))
ML_INTERVAL = int(os.getenv("ML_INTERVAL", 300))
GITHUB_INTERVAL = int(os.getenv("GITHUB_INTERVAL", 300))

MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", 20))
MIN_TRADE_SIZE = float(os.getenv("MIN_TRADE_SIZE", 25))

STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", 4.0))
TRAIL_DIST = float(os.getenv("TRAILING_DISTANCE_PERCENT", 0.9))

COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", 180))

ML_ENABLED = os.getenv("ML_ENABLED", "true").lower() == "true"
ML_ENABLE_AFTER = int(os.getenv("ML_ENABLE_AFTER", 25))

BOT_PAUSED = os.getenv("BOT_PAUSED", "false").lower() == "true"

BASE_URL = "https://api.exchange.coinbase.com/products"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ✅ THIS IS THE IMPORTANT FIX
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_DATA_REPO")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")

# =========================
# FILES
# =========================

LEARNING_FILE = "learning.json"
POSITIONS_FILE = "positions.json"
COOLDOWN_FILE = "cooldown.json"
HISTORY_FILE = "trade_history.csv"

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
                timeout=10
            )
        except:
            pass

# =========================
# FILE INIT
# =========================

def ensure_files():

    if not os.path.exists(LEARNING_FILE):
        with open(LEARNING_FILE,"w") as f:
            json.dump({
                "cash": START_BALANCE,
                "trade_count": 0,
                "win_count": 0,
                "loss_count": 0,
                "total_profit": 0
            }, f)

    if not os.path.exists(POSITIONS_FILE):
        with open(POSITIONS_FILE,"w") as f:
            json.dump({}, f)

    if not os.path.exists(COOLDOWN_FILE):
        with open(COOLDOWN_FILE,"w") as f:
            json.dump({}, f)

    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE,"w") as f:
            f.write("profit\n")

ensure_files()

# =========================
# LOAD STATE
# =========================

def load_json(path, default):
    try:
        with open(path,"r") as f:
            return json.load(f)
    except:
        return default

learning = load_json(LEARNING_FILE, {})
positions = load_json(POSITIONS_FILE, {})
cooldown = load_json(COOLDOWN_FILE, {})

cash = float(learning.get("cash", START_BALANCE))
learning["cash"] = cash

# =========================
# GITHUB BACKUP
# =========================

def github_upload(filename):

    if not GITHUB_TOKEN or not GITHUB_REPO:
        return

    try:

        import base64

        with open(filename,"r") as f:
            content = f.read()

        encoded = base64.b64encode(content.encode()).decode()

        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"

        headers = {"Authorization": f"token {GITHUB_TOKEN}"}

        r = requests.get(url, headers=headers)

        sha = None
        if r.status_code == 200:
            sha = r.json()["sha"]

        payload = {
            "message": f"update {filename}",
            "content": encoded,
            "branch": GITHUB_BRANCH
        }

        if sha:
            payload["sha"] = sha

        requests.put(url, headers=headers, json=payload)

    except:
        pass

def github_backup_all():

    github_upload(LEARNING_FILE)
    github_upload(POSITIONS_FILE)
    github_upload(COOLDOWN_FILE)
    github_upload(HISTORY_FILE)

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
        r = requests.get(BASE_URL)
        data = r.json()
        return [x["id"] for x in data if x["quote_currency"]=="USD"][:60]
    except:
        return []

symbols = get_symbols()

# =========================
# STATS
# =========================

def equity(prices):

    total = cash

    for sym,pos in positions.items():

        if sym in prices:
            total += pos["qty"] * prices[sym]

    return total

# =========================
# TRADE
# =========================

def open_trade(sym,price):

    global cash

    if sym in positions:
        return

    if len(positions)>=MAX_OPEN_TRADES:
        return

    size = min(MIN_TRADE_SIZE,cash)

    if size<MIN_TRADE_SIZE:
        return

    qty=size/price

    positions[sym]={
        "entry":price,
        "qty":qty,
        "peak":price,
        "stop":price*(1-STOP_LOSS_PERCENT/100)
    }

    cash-=size
    learning["cash"]=cash

    notify(f"BUY {sym} ${price:.4f}")

def sell_trade(sym,price):

    global cash

    pos=positions[sym]

    proceeds=pos["qty"]*price
    cost=pos["qty"]*pos["entry"]

    profit=proceeds-cost

    cash+=proceeds

    learning["cash"]=cash
    learning["trade_count"]+=1

    if profit>0:
        learning["win_count"]+=1
    else:
        learning["loss_count"]+=1

    learning["total_profit"]+=profit

    with open(HISTORY_FILE,"a") as f:
        f.write(f"{profit}\n")

    cooldown[sym]=time.time()

    del positions[sym]

    notify(f"SELL {sym} profit ${profit:.2f}")

# =========================
# SAVE
# =========================

def save_all():

    with open(LEARNING_FILE,"w") as f:
        json.dump(learning,f)

    with open(POSITIONS_FILE,"w") as f:
        json.dump(positions,f)

    with open(COOLDOWN_FILE,"w") as f:
        json.dump(cooldown,f)

# =========================
# START
# =========================

notify(f"BOT STARTED | Repo: {GITHUB_REPO}")

# =========================
# LOOP
# =========================

last_scan=0
last_status=0
last_save=0
last_github=0

prices={}

while True:

    try:

        now=time.time()

        if now-last_scan>=SCAN_INTERVAL:

            for sym in symbols:

                p=get_price(sym)

                if p:
                    prices[sym]=p

            for sym in list(positions):

                price=prices.get(sym)

                if not price:
                    continue

                pos=positions[sym]

                if price>pos["peak"]:
                    pos["peak"]=price

                trail=pos["peak"]*(1-TRAIL_DIST/100)

                if price<=pos["stop"] or price<=trail:
                    sell_trade(sym,price)

            if not BOT_PAUSED:
                for sym in symbols:
                    if sym in prices:
                        open_trade(sym,prices[sym])

            last_scan=now

        if now-last_status>=STATUS_INTERVAL:

            eq=equity(prices)

            notify(
                f"STATUS | Cash ${cash:.2f} | Equity ${eq:.2f} | "
                f"Profit ${learning['total_profit']:.2f} | "
                f"Wins {learning['win_count']} | "
                f"Losses {learning['loss_count']} | "
                f"Open {len(positions)}"
            )

            last_status=now

        if now-last_save>=SAVE_INTERVAL:

            save_all()
            last_save=now

        if now-last_github>=GITHUB_INTERVAL:

            github_backup_all()
            last_github=now

        time.sleep(1)

    except Exception as e:

        notify(traceback.format_exc())
        time.sleep(5)
