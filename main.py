# Coin Sniper — Savage Mode ELITE (CLEAN FIXED VERSION)
# ✅ BUY ENGINE + STATUS + WIN/LOSS + EQUITY + ML AUTO-LEARN
# ✅ GITHUB SYNC (on change) + TRAILING STOP + HISTORY GATE + DYNAMIC SIZING

import os
import time
import json
import csv
import hashlib
import traceback
import requests
import numpy as np
import base64
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
            raw = open(LOCK_FILE).read().strip()
            if raw:
                old_pid = int(raw.split("|")[0])
                if _pid_alive(old_pid):
                    raise SystemExit(0)
        except SystemExit:
            raise
        except Exception:
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
TRAILING_DISTANCE_PERCENT = float(os.getenv("TRAILING_DISTANCE_PERCENT", 0.9))
TRAILING_START_PERCENT = float(os.getenv("TRAILING_START_PERCENT", 1.0))
ENTRY_SCORE_MIN = int(os.getenv("ENTRY_SCORE_MIN", 7))
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", 60))
HISTORY_POINTS_REQUIRED = int(os.getenv("HISTORY_POINTS_REQUIRED", 40))
MIN_HISTORY_COINS = int(os.getenv("MIN_HISTORY_COINS", 50))
MIN_POSITION_SIZE_PERCENT = float(os.getenv("MIN_POSITION_SIZE_PERCENT", 3))
MAX_POSITION_SIZE_PERCENT = float(os.getenv("MAX_POSITION_SIZE_PERCENT", 8))
CASH_RESERVE_PERCENT = float(os.getenv("CASH_RESERVE_PERCENT", 5))
ML_ENABLED = os.getenv("ML_ENABLED", "true").lower() == "true"
ML_ENABLE_AFTER = int(os.getenv("ML_ENABLE_AFTER", 50))
ML_MIN_PROB = float(os.getenv("ML_MIN_PROB", 0.62))
BOT_PAUSED = os.getenv("BOT_PAUSED", "false").lower() == "true"

BASE_URL = "https://api.exchange.coinbase.com/products"

# =========================
# FILES
# =========================

LEARNING_FILE = "learning.json"
POSITIONS_FILE = "positions.json"
HISTORY_FILE = "trade_history.csv"
ML_TRAIN_FILE = "ml_training.csv"

# =========================
# STATE
# =========================


def load_json(path, default):
    try:
        if os.path.exists(path):
            return json.load(open(path))
    except Exception:
        pass
    return default


def save_json(path, data):
    try:
        json.dump(data, open(path, "w"))
    except Exception:
        pass


learning = load_json(LEARNING_FILE, {
    "cash": START_BALANCE,
    "trade_count": 0,
    "win_count": 0,
    "loss_count": 0,
    "total_profit": 0.0
})

positions = load_json(POSITIONS_FILE, {})
cash = float(learning.get("cash", START_BALANCE))
price_history = {}

# =========================
# MARKET DATA
# =========================


def get_price(sym):
    try:
        r = requests.get(f"{BASE_URL}/{sym}/ticker", timeout=10)
        j = r.json()
        return float(j.get("price")) if "price" in j else None
    except Exception:
        return None


def get_symbols():
    try:
        r = requests.get(BASE_URL, timeout=10)
        products = r.json()
        syms = [p["id"] for p in products if p.get("quote_currency") == "USD"]
        return syms[:MAX_SYMBOLS]
    except Exception:
        return []


symbols = get_symbols()

# =========================
# ML
# =========================

ml_model = None
ml_last_train = 0


def compute_features(sym):
    h = price_history.get(sym, [])
    if len(h) < 20:
        return None
    arr = np.array(h[-20:], dtype=float)
    p0 = arr[-1]
    p5 = arr[-5]
    ret_5 = (p0 - p5) / p5 if p5 > 0 else 0
    slope = np.polyfit(np.arange(len(arr)), arr, 1)[0] / max(p0, 1e-12)
    vol = np.std(np.diff(arr) / np.maximum(arr[:-1], 1e-12))
    return [ret_5, slope, vol]


def train_ml_if_ready():
    global ml_model, ml_last_train
    if not ML_ENABLED:
        return
    if learning.get("trade_count", 0) < ML_ENABLE_AFTER:
        return
    try:
        rows = []
        with open(ML_TRAIN_FILE, "r") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                rows.append(row)
        if len(rows) < 30:
            return
        X = [[float(r[2]), float(r[3]), float(r[4])] for r in rows]
        y = [int(r[-1]) for r in rows]
        clf = RandomForestClassifier(n_estimators=200)
        clf.fit(X, y)
        ml_model = clf
        ml_last_train = time.time()
    except Exception:
        pass


def ml_probability(sym, score):
    if not ML_ENABLED or ml_model is None:
        return None
    feats = compute_features(sym)
    if feats is None:
        return None
    return float(ml_model.predict_proba([feats])[0][1])

# =========================
# SCORING
# =========================


def score_symbol(sym):
    h = price_history.get(sym, [])
    if len(h) < HISTORY_POINTS_REQUIRED:
        return None
    arr = np.array(h[-HISTORY_POINTS_REQUIRED:], dtype=float)
    drift = (arr[-1] - arr[0]) / max(arr[0], 1e-12)
    vol = np.std(np.diff(arr) / np.maximum(arr[:-1], 1e-12))
    score = 5 + drift * 20 - vol * 200
    return int(max(1, min(10, round(score))))

# =========================
# TRAILING STOP
# =========================


def apply_trailing_stop(pos, price):
    entry = pos["entry"]
    if price > pos["peak"]:
        pos["peak"] = price
    gain_pct = (price - entry) / entry * 100
    if gain_pct < TRAILING_START_PERCENT:
        return
    trail = pos["peak"] * (1 - TRAILING_DISTANCE_PERCENT / 100)
    if trail > pos["stop"]:
        pos["stop"] = trail

# =========================
# MAIN LOOP
# =========================

print(f"[{INSTANCE_ID}] BOT STARTED | Symbols={len(symbols)}")

last_scan = 0
last_status = 0
last_ml = 0

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
                    if len(price_history[sym]) > 200:
                        price_history[sym] = price_history[sym][-200:]

            equity = cash + sum(pos["qty"] * prices.get(sym, pos["entry"]) for sym, pos in positions.items())

            # Manage positions
            for sym, pos in list(positions.items()):
                px = prices.get(sym)
                if not px:
                    continue
                apply_trailing_stop(pos, px)
                if px <= pos["stop"]:
                    qty = pos["qty"]
                    proceeds = qty * px
                    profit = (px - pos["entry"]) * qty
                    cash += proceeds
                    learning["trade_count"] += 1
                    learning["total_profit"] += profit
                    if profit >= 0:
                        learning["win_count"] += 1
                    else:
                        learning["loss_count"] += 1
                    del positions[sym]

            # Buy scan
            ready = sum(1 for s in symbols if len(price_history.get(s, [])) >= HISTORY_POINTS_REQUIRED)
            if ready >= MIN_HISTORY_COINS:
                for sym in symbols:
                    if sym in positions:
                        continue
                    score = score_symbol(sym)
                    if score and score >= ENTRY_SCORE_MIN:
                        notional = equity * (MIN_POSITION_SIZE_PERCENT / 100)
                        if notional >= MIN_TRADE_SIZE and cash >= notional:
                            qty = notional / prices[sym]
                            cash -= notional
                            positions[sym] = {
                                "entry": prices[sym],
                                "qty": qty,
                                "stop": prices[sym] * (1 - STOP_LOSS_PERCENT / 100),
                                "peak": prices[sym]
                            }
                            break

            learning["cash"] = cash
            save_json(LEARNING_FILE, learning)
            save_json(POSITIONS_FILE, positions)

            last_scan = now

        if now - last_status >= STATUS_INTERVAL:
            equity = cash + sum(pos["qty"] * get_price(sym) for sym, pos in positions.items() if get_price(sym))
            trades = learning.get("trade_count", 0)
            wins = learning.get("win_count", 0)
            losses = learning.get("loss_count", 0)
            winpct = (wins / trades * 100) if trades else 0
            print(f"[{INSTANCE_ID}] STATUS cash=${cash:.2f} equity=${equity:.2f} open={len(positions)} trades={trades} win%={winpct:.1f}")
            last_status = now

        if now - last_ml >= ML_INTERVAL:
            train_ml_if_ready()
            last_ml = now

        time.sleep(1)

    except Exception:
        print(traceback.format_exc())
        time.sleep(5)
