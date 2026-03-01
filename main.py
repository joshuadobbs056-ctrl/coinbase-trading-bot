# Coin Sniper — Savage Mode ELITE (MAX PROFIT BUILD — ML TELEGRAM FIXED + SCAN LOGS + HISTORY GATE)

import os
import time
import json
import traceback
import requests
import numpy as np
from sklearn.ensemble import RandomForestClassifier

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

START_BALANCE = float(os.getenv("START_BALANCE", 1000))

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 15))
STATUS_INTERVAL = int(os.getenv("STATUS_INTERVAL", 60))
SAVE_INTERVAL = int(os.getenv("SAVE_INTERVAL", 60))
ML_INTERVAL = int(os.getenv("ML_INTERVAL", 300))

MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", 20))
MIN_TRADE_SIZE = float(os.getenv("MIN_TRADE_SIZE", 25))

STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", 4.0))
TRAIL_DIST_BASE = float(os.getenv("TRAILING_DISTANCE_PERCENT", 0.9))

COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", 180))

ENTRY_SCORE_MIN = int(os.getenv("ENTRY_SCORE_MIN", 7))

# History requirements (so we don't trade until we have enough data)
HISTORY_POINTS_REQUIRED = int(os.getenv("HISTORY_POINTS_REQUIRED", 40))   # score needs 40 points
MIN_HISTORY_COINS = int(os.getenv("MIN_HISTORY_COINS", 50))              # must have >= this many coins "ready"

# Scan log throttle (prevents Telegram spam)
SCAN_LOG_INTERVAL = int(os.getenv("SCAN_LOG_INTERVAL", 60))              # seconds between SCANNING/HISTORY messages

ML_ENABLED = os.getenv("ML_ENABLED", "true").lower() == "true"
ML_ENABLE_AFTER = int(os.getenv("ML_ENABLE_AFTER", 50))
ML_MIN_PROB = float(os.getenv("ML_MIN_PROB", 0.62))

BOT_PAUSED = os.getenv("BOT_PAUSED", "false").lower() == "true"

BASE_URL = "https://api.exchange.coinbase.com/products"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

LEARNING_FILE = "learning.json"
POSITIONS_FILE = "positions.json"
COOLDOWN_FILE = "cooldown.json"
HISTORY_FILE = "trade_history.csv"
ML_TRAIN_FILE = "ml_training.csv"

ml_model = None
ml_active = False

def notify(msg: str):
    print(msg, flush=True)
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
                timeout=10
            )
        except Exception:
            pass

def ensure_files():
    if not os.path.exists(LEARNING_FILE):
        json.dump({
            "cash": START_BALANCE,
            "start_balance": START_BALANCE,
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "total_profit": 0.0
        }, open(LEARNING_FILE, "w"))

    for f, d in [
        (POSITIONS_FILE, {}),
        (COOLDOWN_FILE, {})
    ]:
        if not os.path.exists(f):
            json.dump(d, open(f, "w"))

    if not os.path.exists(HISTORY_FILE):
        open(HISTORY_FILE, "w").write("profit\n")

    if not os.path.exists(ML_TRAIN_FILE):
        open(ML_TRAIN_FILE, "w").write(
            "score,momentum,volatility,trend,range_pos,breakout,outcome\n"
        )

ensure_files()

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
    "total_profit": 0
})

positions = load_json(POSITIONS_FILE, {})
cooldown = load_json(COOLDOWN_FILE, {})

cash = float(learning.get("cash", START_BALANCE))

price_history = {}
symbols = []

def get_price(sym):
    try:
        r = requests.get(f"{BASE_URL}/{sym}/ticker", timeout=10)
        j = r.json()
        return float(j["price"])
    except Exception:
        return None

def get_symbols():
    try:
        r = requests.get(BASE_URL, timeout=10)
        syms = [p["id"] for p in r.json() if p.get("quote_currency") == "USD"]
        return syms[:60]
    except Exception:
        return []

symbols = get_symbols()

def update_price_history(sym, px):
    if sym not in price_history:
        price_history[sym] = []
    price_history[sym].append(float(px))
    if len(price_history[sym]) > 120:
        price_history[sym].pop(0)

def get_hist(sym, n):
    h = price_history.get(sym, [])
    if len(h) < n:
        return None
    return np.array(h[-n:], dtype=float)

def score_coin(sym, price):
    a20 = get_hist(sym, 20)
    a40 = get_hist(sym, 40)
    if a20 is None or a40 is None:
        return 0
    mean20 = a20.mean()
    mean40 = a40.mean()
    trend = (mean20 - mean40) / mean40
    mom = (price - mean20) / mean20
    score = 0
    if trend > 0.002: score += 1
    if trend > 0.005: score += 1
    if trend > 0.01:  score += 1
    if mom > 0.002:   score += 1
    if mom > 0.005:   score += 1
    if mom > 0.01:    score += 1
    return score

def extract_features(sym, price, score):
    a20 = get_hist(sym, 20)
    a40 = get_hist(sym, 40)
    if a20 is None or a40 is None:
        return None
    mean20 = a20.mean()
    mean40 = a40.mean()
    mom = (price - mean20) / mean20
    vol = a20.std() / mean20
    trend = (mean20 - mean40) / mean40
    # placeholders for future expansion:
    return [float(score), float(mom), float(vol), float(trend), 0.0, 0.0]

last_ml_learning_note = 0

def train_model():
    global ml_model, ml_active, last_ml_learning_note
    if not ML_ENABLED:
        return
    try:
        data = np.genfromtxt(ML_TRAIN_FILE, delimiter=",", skip_header=1)

        # handle empty file safely
        if data is None or (hasattr(data, "size") and data.size == 0):
            now = time.time()
            if now - last_ml_learning_note >= 300:
                notify(f"ML LEARNING (0/{ML_ENABLE_AFTER})")
                last_ml_learning_note = now
            return

        # genfromtxt returns 1D for one row, 2D for many rows
        if isinstance(data, np.ndarray) and data.ndim == 1:
            data = np.array([data], dtype=float)

        if len(data) < ML_ENABLE_AFTER:
            now = time.time()
            if now - last_ml_learning_note >= 300:
                notify(f"ML LEARNING ({len(data)}/{ML_ENABLE_AFTER})")
                last_ml_learning_note = now
            return

        X = data[:, :-1]
        y = data[:, -1]

        # needs at least 2 classes
        if len(np.unique(y)) < 2:
            return

        model = RandomForestClassifier(n_estimators=300, random_state=42)
        model.fit(X, y)
        ml_model = model

        if not ml_active:
            ml_active = True
            notify("ML ACTIVATED — probability-gated entries now LIVE")

        notify(f"ML TRAINED samples={len(data)}")

    except Exception:
        pass

def equity(prices):
    total = cash
    for sym, pos in positions.items():
        px = prices.get(sym)
        if px:
            total += pos["qty"] * px
    return float(total)

def winrate():
    t = int(learning.get("trade_count", 0))
    w = int(learning.get("win_count", 0))
    return (w / t * 100.0) if t > 0 else 0.0

def open_trade(sym, price, score):
    global cash

    if BOT_PAUSED:
        return
    if sym in positions:
        return
    if len(positions) >= MAX_OPEN_TRADES:
        return
    if cash < MIN_TRADE_SIZE:
        return

    features = extract_features(sym, price, score)
    if not features:
        return

    # ML gate (only if model is trained)
    if ml_model is not None:
        try:
            prob = float(ml_model.predict_proba([features])[0][1])
            if prob < ML_MIN_PROB:
                return
        except Exception:
            return

    size = min(MIN_TRADE_SIZE, cash)
    qty = size / price

    positions[sym] = {
        "entry": float(price),
        "qty": float(qty),
        "peak": float(price),
        "stop": float(price) * (1 - STOP_LOSS_PERCENT / 100.0),
        "features": features
    }

    cash -= size
    learning["cash"] = cash

    notify(f"BUY {sym} score={score} cash={cash:.2f} open={len(positions)}")

def sell_trade(sym, price, reason):
    global cash

    pos = positions[sym]
    entry = float(pos["entry"])
    qty = float(pos["qty"])
    profit = (float(price) - entry) * qty

    cash += qty * float(price)

    learning["cash"] = cash
    learning["trade_count"] = int(learning.get("trade_count", 0)) + 1
    if profit > 0:
        learning["win_count"] = int(learning.get("win_count", 0)) + 1
    else:
        learning["loss_count"] = int(learning.get("loss_count", 0)) + 1
    learning["total_profit"] = float(learning.get("total_profit", 0.0)) + float(profit)

    open(HISTORY_FILE, "a").write(f"{profit}\n")

    # write ML row
    outcome = 1 if profit > 0 else 0
    open(ML_TRAIN_FILE, "a").write(",".join(map(str, pos["features"])) + f",{outcome}\n")

    del positions[sym]

    notify(f"SELL {sym} reason={reason} profit={profit:.2f} cash={cash:.2f} open={len(positions)}")

notify(f"BOT STARTED cash={cash:.2f} symbols={len(symbols)} min_history_coins={MIN_HISTORY_COINS} history_points_required={HISTORY_POINTS_REQUIRED}")

last_prices = {}
last_scan = 0
last_status = 0
last_ml = 0
last_scanlog = 0

while True:
    try:
        now = time.time()

        if now - last_scan >= SCAN_INTERVAL:
            prices = {}

            # fetch prices + build history
            for sym in symbols:
                px = get_price(sym)
                if px:
                    prices[sym] = px
                    update_price_history(sym, px)

            if prices:
                last_prices = prices

            # stops
            for sym, pos in list(positions.items()):
                px = prices.get(sym)
                if px and float(px) <= float(pos["stop"]):
                    sell_trade(sym, px, "STOP")

            # history readiness
            ready = sum(1 for s in price_history if len(price_history[s]) >= HISTORY_POINTS_REQUIRED)
            history_ok = (ready >= MIN_HISTORY_COINS)

            # SCAN LOGS (throttled)
            if now - last_scanlog >= SCAN_LOG_INTERVAL:
                notify(f"SCANNING {len(symbols)} symbols | history_coins={len(price_history)}")
                notify(f"HISTORY READY: {ready}/{len(symbols)} coins (need >= {MIN_HISTORY_COINS})")
                last_scanlog = now

            # only trade once enough history exists (prevents blind / premature entries)
            if history_ok:
                for sym, px in prices.items():
                    sc = score_coin(sym, px)
                    if sc >= ENTRY_SCORE_MIN:
                        open_trade(sym, px, sc)

            last_scan = now

        if now - last_status >= STATUS_INTERVAL:
            eq = equity(last_prices)

            notify(
                f"STATUS\n"
                f"Cash: {cash:.2f}\n"
                f"Equity: {eq:.2f}\n"
                f"Profit: {float(learning.get('total_profit', 0.0)):.2f}\n\n"
                f"Open: {len(positions)}\n\n"
                f"Trades: {int(learning.get('trade_count', 0))}\n"
                f"Wins: {int(learning.get('win_count', 0))}\n"
                f"Losses: {int(learning.get('loss_count', 0))}\n"
                f"Winrate: {winrate():.1f}%\n\n"
                f"ML: {'ACTIVE' if ml_active else 'LEARNING'}"
            )

            last_status = now

        if now - last_ml >= ML_INTERVAL:
            train_model()
            last_ml = now

        # persist
        save_json(LEARNING_FILE, learning)
        save_json(POSITIONS_FILE, positions)
        save_json(COOLDOWN_FILE, cooldown)

        time.sleep(1)

    except Exception:
        notify(traceback.format_exc())
        time.sleep(5)
