# Coin Sniper — Savage Mode ELITE
# ML Enabled + GitHub Persistence + SINGLE INSTANCE LOCK (prevents multiple running copies)

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
        os.kill(pid, 0)  # does not kill; checks signal permission/existence
        return True
    except Exception:
        return False

def acquire_lock_or_exit():
    # If lock exists and PID alive -> exit
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, "r") as f:
                raw = f.read().strip()
            if raw:
                old_pid = int(raw.split("|")[0])
                if old_pid and _pid_alive(old_pid):
                    print(f"[LOCK] Another instance is already running (pid={old_pid}). Exiting.")
                    raise SystemExit(0)
        except SystemExit:
            raise
        except Exception:
            # If lock is corrupt, overwrite it below
            pass

    # Write our PID + timestamp
    try:
        with open(LOCK_FILE, "w") as f:
            f.write(f"{os.getpid()}|{int(time.time())}")
    except Exception:
        # If we cannot lock, safest is to exit (prevents duplicates)
        print("[LOCK] Unable to create lock file. Exiting to avoid duplicates.")
        raise SystemExit(0)

acquire_lock_or_exit()

INSTANCE_ID = os.getenv("INSTANCE_ID", f"{os.getpid()}")

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
COOLDOWN_FILE = "cooldown.json"
HISTORY_FILE = "trade_history.csv"

# =========================
# TELEGRAM
# =========================

def notify(msg: str):
    print(msg)
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
                timeout=10,
            )
        except Exception:
            pass

# =========================
# FILE CREATION
# =========================

def ensure_files():
    if not os.path.exists(LEARNING_FILE):
        with open(LEARNING_FILE, "w") as f:
            json.dump({
                "cash": START_BALANCE,
                "start_balance": START_BALANCE,
                "trade_count": 0,
                "win_count": 0,
                "loss_count": 0,
                "total_profit": 0
            }, f)

    if not os.path.exists(POSITIONS_FILE):
        with open(POSITIONS_FILE, "w") as f:
            json.dump({}, f)

    if not os.path.exists(COOLDOWN_FILE):
        with open(COOLDOWN_FILE, "w") as f:
            json.dump({}, f)

    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w") as f:
            f.write("profit\n")

ensure_files()

# =========================
# LOAD / SAVE
# =========================

def load_json(file):
    with open(file, "r") as f:
        return json.load(f)

def save_json(file, data):
    with open(file, "w") as f:
        json.dump(data, f)

# =========================
# GITHUB BACKUP
# =========================

def github_upload(filename):
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return

    try:
        import base64

        with open(filename, "r") as f:
            content = f.read()

        encoded = base64.b64encode(content.encode()).decode()

        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"
        headers = {"Authorization": f"token {GITHUB_TOKEN}"}

        r = requests.get(url, headers=headers, timeout=15)
        sha = None
        if r.status_code == 200:
            sha = r.json().get("sha")

        data = {
            "message": f"Update {filename}",
            "content": encoded,
            "branch": "main"
        }
        if sha:
            data["sha"] = sha

        requests.put(url, headers=headers, json=data, timeout=20)

    except Exception:
        pass

def github_backup_all():
    github_upload(LEARNING_FILE)
    github_upload(POSITIONS_FILE)
    github_upload(COOLDOWN_FILE)
    github_upload(HISTORY_FILE)

# =========================
# STATE
# =========================

learning = load_json(LEARNING_FILE)
positions = load_json(POSITIONS_FILE)
cooldown = load_json(COOLDOWN_FILE)

cash = float(learning.get("cash", START_BALANCE))

# =========================
# MARKET DATA
# =========================

def get_price(symbol):
    try:
        r = requests.get(f"{BASE_URL}/{symbol}/ticker", timeout=10)
        data = r.json()
        if "price" not in data:
            return None
        return float(data["price"])
    except Exception:
        return None

def get_symbols():
    try:
        r = requests.get(BASE_URL, timeout=10)
        data = r.json()
        return [
            p["id"]
            for p in data
            if p.get("quote_currency") == "USD"
        ][:60]
    except Exception:
        return []

symbols = get_symbols()

# =========================
# ML
# =========================

model = None

def train_model():
    global model
    if not ML_ENABLED:
        return
    try:
        data = np.genfromtxt(HISTORY_FILE, delimiter=",", skip_header=1)
        # If only one row, genfromtxt returns scalar; normalize it
        if np.isscalar(data):
            data = np.array([float(data)])

        if len(data) < ML_ENABLE_AFTER:
            return

        X = np.arange(len(data)).reshape(-1, 1)
        y = (data > 0).astype(int)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        notify(f"[{INSTANCE_ID}] ML TRAINED on {len(data)} trades")

    except Exception:
        pass

# =========================
# EQUITY
# =========================

def equity(prices):
    total = cash
    for sym, pos in positions.items():
        px = prices.get(sym)
        if px:
            total += pos["qty"] * px
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
    if sym in cooldown and (now - cooldown[sym]) < COOLDOWN_SECONDS:
        return

    size = min(MIN_TRADE_SIZE, cash)

    if size < MIN_TRADE_SIZE:
        return

    qty = size / price

    positions[sym] = {
        "entry": price,
        "qty": qty,
        "peak": price,
        "stop": price * (1 - STOP_LOSS_PERCENT / 100)
    }

    cash -= size
    learning["cash"] = cash

    notify(f"[{INSTANCE_ID}] BUY {sym} @ {price}")

# =========================
# SELL TRADE
# =========================

def sell_trade(sym, price):
    global cash

    pos = positions[sym]
    proceeds = pos["qty"] * price
    entry_value = pos["qty"] * pos["entry"]
    profit = proceeds - entry_value

    cash += proceeds
    learning["cash"] = cash
    learning["trade_count"] = int(learning.get("trade_count", 0)) + 1

    if profit > 0:
        learning["win_count"] = int(learning.get("win_count", 0)) + 1
    else:
        learning["loss_count"] = int(learning.get("loss_count", 0)) + 1

    learning["total_profit"] = float(learning.get("total_profit", 0)) + profit

    with open(HISTORY_FILE, "a") as f:
        f.write(f"{profit}\n")

    cooldown[sym] = time.time()
    del positions[sym]

    notify(f"[{INSTANCE_ID}] SELL {sym} Profit: {profit:.2f}")

# =========================
# TIMERS
# =========================

last_scan = 0
last_status = 0
last_save = 0
last_ml = 0
last_github = 0

notify(f"[{INSTANCE_ID}] BOT STARTED\nCash: {cash:.2f}\nLock: {LOCK_FILE}")

# =========================
# CONTROLLED LOOP
# =========================

while True:
    try:
        now = time.time()

        # keep last known prices for status/equity even between scans
        if "prices" not in globals():
            prices = {}

        # SCAN
        if now - last_scan >= SCAN_INTERVAL:
            prices = {}

            for sym in symbols:
                price = get_price(sym)
                if price:
                    prices[sym] = price

            # sells
            for sym in list(positions):
                price = prices.get(sym)
                if not price:
                    continue

                pos = positions[sym]
                if price > pos["peak"]:
                    pos["peak"] = price

                trail = pos["peak"] * (1 - TRAIL_DIST / 100)

                if price <= pos["stop"] or price <= trail:
                    sell_trade(sym, price)

            # buys
            for sym in symbols:
                px = prices.get(sym)
                if px:
                    open_trade(sym, px)

            last_scan = now

        # STATUS
        if now - last_status >= STATUS_INTERVAL:
            eq = equity(prices if isinstance(prices, dict) else {})
            notify(f"[{INSTANCE_ID}] STATUS Equity: {eq:.2f} Profit: {learning.get('total_profit', 0):.2f} Open: {len(positions)}")
            last_status = now

        # SAVE
        if now - last_save >= SAVE_INTERVAL:
            save_json(LEARNING_FILE, learning)
            save_json(POSITIONS_FILE, positions)
            save_json(COOLDOWN_FILE, cooldown)
            last_save = now

        # ML
        if now - last_ml >= ML_INTERVAL:
            train_model()
            last_ml = now

        # GITHUB
        if now - last_github >= GITHUB_INTERVAL:
            github_backup_all()
            last_github = now

        time.sleep(1)

    except SystemExit:
        raise
    except Exception:
        notify(f"[{INSTANCE_ID}] ERROR\n{traceback.format_exc()}")
        time.sleep(5)
