# Coin Sniper — Savage Mode ELITE (STABLE)
# ✅ Single-instance lock
# ✅ Persistent files
# ✅ GitHub backup on timer (no spam)
# ✅ Correct stats: wins/losses/open/profit/equity
# ✅ Pause support (pause buys, keep sell protection)

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
    # If lock exists and PID alive -> exit to prevent duplicates (within same container)
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, "r") as f:
                raw = f.read().strip()
            if raw:
                old_pid = int(raw.split("|")[0])
                if old_pid and _pid_alive(old_pid):
                    print(f"[LOCK] Another instance running (pid={old_pid}). Exiting.")
                    raise SystemExit(0)
        except SystemExit:
            raise
        except Exception:
            pass

    # Write our PID
    try:
        with open(LOCK_FILE, "w") as f:
            f.write(f"{os.getpid()}|{int(time.time())}")
    except Exception:
        print("[LOCK] Unable to create lock file. Exiting.")
        raise SystemExit(0)

acquire_lock_or_exit()

INSTANCE_ID = os.getenv("INSTANCE_ID", str(os.getpid()))

# =========================
# CONFIG (ENV)
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

BOT_PAUSED = os.getenv("BOT_PAUSED", "false").lower() == "true"  # pauses NEW buys only

BASE_URL = "https://api.exchange.coinbase.com/products"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")
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
# AUTO FILE CREATION
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
                "total_profit": 0.0
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

def load_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def save_json(path, data):
    try:
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception:
        pass

# =========================
# GITHUB BACKUP (TIMED)
# =========================

def github_upload(filename: str):
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

        payload = {
            "message": f"Auto update {filename}",
            "content": encoded,
            "branch": GITHUB_BRANCH
        }
        if sha:
            payload["sha"] = sha

        requests.put(url, headers=headers, json=payload, timeout=20)

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

learning = load_json(LEARNING_FILE, {
    "cash": START_BALANCE,
    "start_balance": START_BALANCE,
    "trade_count": 0,
    "win_count": 0,
    "loss_count": 0,
    "total_profit": 0.0
})

positions = load_json(POSITIONS_FILE, {})
cooldown = load_json(COOLDOWN_FILE, {})

# Keep cash in ONE place (global), always sync with learning["cash"]
cash = float(learning.get("cash", START_BALANCE))
learning["cash"] = cash

# =========================
# MARKET DATA
# =========================

def get_price(symbol: str):
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
        # USD quote only
        syms = [
            p["id"]
            for p in data
            if p.get("quote_currency") == "USD"
        ]
        return syms[:60]
    except Exception:
        return []

symbols = get_symbols()

# =========================
# ML (TRAINING ONLY)
# =========================

model = None

def train_model():
    global model
    if not ML_ENABLED:
        return
    try:
        data = np.genfromtxt(HISTORY_FILE, delimiter=",", skip_header=1)
        if data is None:
            return
        if np.isscalar(data):
            data = np.array([float(data)])

        if len(data) < ML_ENABLE_AFTER:
            return

        X = np.arange(len(data)).reshape(-1, 1)
        y = (data > 0).astype(int)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        notify(f"[{INSTANCE_ID}] ML TRAINED | trades={len(data)}")

    except Exception:
        pass

# =========================
# STATS
# =========================

def equity(prices: dict) -> float:
    total = cash
    for sym, pos in positions.items():
        px = prices.get(sym)
        if px:
            total += float(pos["qty"]) * float(px)
    return float(total)

def winrate() -> float:
    t = int(learning.get("trade_count", 0))
    w = int(learning.get("win_count", 0))
    return (w / t * 100.0) if t > 0 else 0.0

# =========================
# TRADING
# =========================

def open_trade(sym: str, price: float):
    global cash

    if sym in positions:
        return
    if len(positions) >= MAX_OPEN_TRADES:
        return

    now = time.time()
    last = cooldown.get(sym)
    if last and (now - float(last)) < COOLDOWN_SECONDS:
        return

    # fixed-size for now (min trade size), safe
    size = min(MIN_TRADE_SIZE, cash)
    if size < MIN_TRADE_SIZE:
        return

    qty = size / price

    positions[sym] = {
        "entry": float(price),
        "qty": float(qty),
        "peak": float(price),
        "stop": float(price) * (1 - STOP_LOSS_PERCENT / 100.0)
    }

    cash -= size
    learning["cash"] = cash

    notify(f"[{INSTANCE_ID}] BUY {sym} @ {price:.6f} | size=${size:.2f} | cash=${cash:.2f} | open={len(positions)}")

def sell_trade(sym: str, price: float):
    global cash

    if sym not in positions:
        return

    pos = positions[sym]

    entry = float(pos["entry"])
    qty = float(pos["qty"])

    proceeds = qty * float(price)
    cost = qty * entry
    profit = proceeds - cost

    cash += proceeds
    learning["cash"] = cash

    learning["trade_count"] = int(learning.get("trade_count", 0)) + 1
    if profit > 0:
        learning["win_count"] = int(learning.get("win_count", 0)) + 1
    else:
        learning["loss_count"] = int(learning.get("loss_count", 0)) + 1

    learning["total_profit"] = float(learning.get("total_profit", 0.0)) + float(profit)

    # write history for ML
    try:
        with open(HISTORY_FILE, "a") as f:
            f.write(f"{profit}\n")
    except Exception:
        pass

    cooldown[sym] = time.time()
    del positions[sym]

    notify(f"[{INSTANCE_ID}] SELL {sym} @ {price:.6f} | profit=${profit:.2f} | cash=${cash:.2f} | open={len(positions)}")

# =========================
# SAVE
# =========================

def save_all_local():
    save_json(LEARNING_FILE, learning)
    save_json(POSITIONS_FILE, positions)
    save_json(COOLDOWN_FILE, cooldown)

# =========================
# STARTUP MESSAGE
# =========================

notify(
    f"[{INSTANCE_ID}] BOT STARTED\n"
    f"Cash: ${cash:.2f}\n"
    f"Open: {len(positions)}\n"
    f"Paused(buys): {BOT_PAUSED}\n"
    f"Lock: {LOCK_FILE}"
)

# =========================
# TIMERS
# =========================

last_scan = 0
last_status = 0
last_save = 0
last_ml = 0
last_github = 0

# keep last prices for equity between scans
last_prices = {}

# =========================
# MAIN LOOP (CONTROLLED)
# =========================

while True:
    try:
        now = time.time()

        # Refresh pause flag without redeploy (Railway vars apply on restart,
        # but this also supports environments that live-update env)
        BOT_PAUSED = os.getenv("BOT_PAUSED", "false").lower() == "true"

        # SCAN + TRADE
        if now - last_scan >= SCAN_INTERVAL:
            prices = {}

            for sym in symbols:
                px = get_price(sym)
                if px:
                    prices[sym] = px

            # store for status/equity
            if prices:
                last_prices = prices

            # SELL MANAGEMENT (always runs, even when paused)
            for sym in list(positions.keys()):
                px = prices.get(sym) or last_prices.get(sym)
                if not px:
                    continue

                pos = positions[sym]
                px = float(px)

                # update peak
                if px > float(pos["peak"]):
                    pos["peak"] = px

                # trailing stop
                trail = float(pos["peak"]) * (1 - TRAIL_DIST / 100.0)

                if px <= float(pos["stop"]) or px <= float(trail):
                    sell_trade(sym, px)

            # BUY LOGIC (disabled when paused)
            if not BOT_PAUSED:
                for sym in symbols:
                    px = prices.get(sym)
                    if px:
                        open_trade(sym, float(px))

            last_scan = now

        # STATUS
        if now - last_status >= STATUS_INTERVAL:
            eq = equity(last_prices if isinstance(last_prices, dict) else {})
            t = int(learning.get("trade_count", 0))
            w = int(learning.get("win_count", 0))
            l = int(learning.get("loss_count", 0))
            p = float(learning.get("total_profit", 0.0))
            o = len(positions)
            wr = winrate()

            notify(
                f"[{INSTANCE_ID}] STATUS\n"
                f"Cash: ${cash:.2f}\n"
                f"Equity: ${eq:.2f}\n"
                f"Profit: ${p:.2f}\n"
                f"Open: {o}\n"
                f"Trades: {t} | Wins: {w} | Losses: {l} | Winrate: {wr:.1f}%\n"
                f"Paused(buys): {BOT_PAUSED}"
            )
            last_status = now

        # SAVE LOCAL
        if now - last_save >= SAVE_INTERVAL:
            save_all_local()
            last_save = now

        # ML TRAIN (timed)
        if now - last_ml >= ML_INTERVAL:
            train_model()
            last_ml = now

        # GITHUB BACKUP (timed)
        if now - last_github >= GITHUB_INTERVAL:
            github_backup_all()
            last_github = now

        time.sleep(1)

    except SystemExit:
        raise
    except Exception:
        notify(f"[{INSTANCE_ID}] ERROR\n{traceback.format_exc()}")
        time.sleep(5)
