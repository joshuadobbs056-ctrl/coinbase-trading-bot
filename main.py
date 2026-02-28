import os
import time
import json
import csv
import requests
from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

#########################################
# VARIABLES (Railway env)
#########################################

# Loop
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 10))
STATUS_INTERVAL = int(os.getenv("STATUS_INTERVAL", 60))

# Universe
# Set COINS=AUTO to scan all USD pairs on Coinbase Exchange.
COINS = os.getenv("COINS", "AUTO").replace(" ", "")
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", 120))  # cap AUTO list to avoid rate-limits

# Paper trading (explicit flag; this bot is paper-only regardless)
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() == "true"

# Portfolio
START_BALANCE = float(os.getenv("START_BALANCE", 1000))
MIN_CASH_RESERVE_PERCENT = float(os.getenv("MIN_CASH_RESERVE_PERCENT", 2.0))
MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", 30))

# Position sizing (micro-trades)
MIN_POSITION_SIZE_PERCENT = float(os.getenv("MIN_POSITION_SIZE_PERCENT", 1.0))
MAX_POSITION_SIZE_PERCENT = float(os.getenv("MAX_POSITION_SIZE_PERCENT", 6.0))

# Stops
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", 2.0))
TRAILING_START_PERCENT = float(os.getenv("TRAILING_START_PERCENT", 0.6))
TRAILING_DISTANCE_PERCENT = float(os.getenv("TRAILING_DISTANCE_PERCENT", 0.35))

# Capital recycling (sniper exit)
MAX_TRADE_DURATION_MINUTES = int(os.getenv("MAX_TRADE_DURATION_MINUTES", 5))
MIN_PROFIT_KEEP_PERCENT = float(os.getenv("MIN_PROFIT_KEEP_PERCENT", 0.12))

# Replacement
REPLACE_WITH_BETTER = os.getenv("REPLACE_WITH_BETTER", "true").lower() == "true"
REPLACE_SCORE_MARGIN = float(os.getenv("REPLACE_SCORE_MARGIN", 0.4))

# Signal threshold
MIN_SCORE = float(os.getenv("MIN_SCORE", 2.0))

# Candles
CANDLE_GRANULARITY = int(os.getenv("CANDLE_GRANULARITY", 60))
CANDLE_POINTS = int(os.getenv("CANDLE_POINTS", 60))

# ML / AI
ML_ENABLED = os.getenv("ML_ENABLED", "true").lower() == "true"
ML_MIN_TRADES_TO_ENABLE = int(os.getenv("ML_MIN_TRADES_TO_ENABLE", 50))
ML_MIN_PROB = float(os.getenv("ML_MIN_PROB", 0.60))
ML_RETRAIN_SECONDS = int(os.getenv("ML_RETRAIN_SECONDS", 600))

# Telegram (optional)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Coinbase Exchange REST
BASE_URL = "https://api.exchange.coinbase.com"

#########################################
# FILES
#########################################

DATA_DIR = os.getenv("DATA_DIR", ".")
LEARNING_FILE = os.path.join(DATA_DIR, "learning.json")
HISTORY_FILE = os.path.join(DATA_DIR, "trade_history.csv")

#########################################
# HTTP SESSION
#########################################

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "coin-sniper-bot/1.0"})

#########################################
# TELEGRAM + LOG
#########################################

def notify(msg: str) -> None:
    print(msg, flush=True)

    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            SESSION.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=10)
        except Exception:
            pass

#########################################
# LEARNING STATE (lightweight stats)
#########################################

def load_learning() -> Dict:
    default = {"trade_count": 0, "win_count": 0, "total_profit": 0.0}
    if not os.path.exists(LEARNING_FILE):
        return default
    try:
        with open(LEARNING_FILE, "r") as f:
            data = json.load(f)
        for k, v in default.items():
            if k not in data:
                data[k] = v
        return data
    except Exception:
        return default

def save_learning(state: Dict) -> None:
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(LEARNING_FILE, "w") as f:
            json.dump(state, f)
    except Exception:
        pass

learning = load_learning()

#########################################
# HISTORY DATASET (features + profit)
#########################################

def ensure_history_file() -> None:
    if os.path.exists(HISTORY_FILE):
        return
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(HISTORY_FILE, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rsi", "volume_ratio", "trend_strength", "momentum", "profit"])
    except Exception:
        pass

def append_history_row(rsi: float, volr: float, trend: float, mom: float, profit: float) -> None:
    try:
        with open(HISTORY_FILE, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([rsi, volr, trend, mom, profit])
    except Exception:
        pass

ensure_history_file()

#########################################
# COINBASE DATA
#########################################

def get_ticker(sym: str) -> Optional[Dict]:
    try:
        r = SESSION.get(f"{BASE_URL}/products/{sym}/ticker", timeout=10)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def get_candles(sym: str) -> Optional[List[List[float]]]:
    # Coinbase returns newest-first; we'll reverse
    try:
        r = SESSION.get(
            f"{BASE_URL}/products/{sym}/candles",
            params={"granularity": str(CANDLE_GRANULARITY)},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        if not isinstance(data, list) or not data:
            return None
        data = list(reversed(data))
        return data[-min(CANDLE_POINTS, len(data)):]
    except Exception:
        return None

def get_all_usd_pairs() -> List[str]:
    try:
        r = SESSION.get(f"{BASE_URL}/products", timeout=15)
        if r.status_code != 200:
            return []
        products = r.json()
        out: List[str] = []
        for p in products:
            try:
                if p.get("quote_currency") == "USD" and p.get("status") == "online":
                    out.append(p.get("id"))
            except Exception:
                continue
        # de-dupe and keep stable order
        out = sorted(list(set([x for x in out if x])))
        return out
    except Exception:
        return []

def resolve_coin_list() -> List[str]:
    if COINS.upper() == "AUTO":
        pairs = get_all_usd_pairs()
        if not pairs:
            return ["BTC-USD", "ETH-USD", "SOL-USD"]
        return pairs[:max(5, MAX_SYMBOLS)]
    else:
        lst = [c for c in COINS.split(",") if c]
        return lst if lst else ["BTC-USD", "ETH-USD", "SOL-USD"]

COIN_LIST = resolve_coin_list()

#########################################
# INDICATORS
#########################################

def calc_rsi(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    arr = np.array(closes, dtype=float)
    deltas = np.diff(arr)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = float(np.mean(gains[-period:]))
    avg_loss = float(np.mean(losses[-period:]))

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0

    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))

def ema(values: List[float], period: int = 20) -> float:
    if not values:
        return 0.0
    if len(values) < period:
        return float(np.mean(values))
    k = 2.0 / (period + 1.0)
    e = float(values[0])
    for v in values[1:]:
        e = float(v) * k + e * (1.0 - k)
    return float(e)

def trend_strength(closes: List[float]) -> float:
    if len(closes) < 20:
        return 0.0
    e = ema(closes, 20)
    if e <= 0:
        return 0.0
    return float((closes[-1] - e) / e)

def momentum(closes: List[float], lookback: int = 5) -> float:
    if len(closes) < lookback + 1:
        return 0.0
    base = closes[-1 - lookback]
    if base == 0:
        return 0.0
    return float((closes[-1] - base) / base)

def volume_ratio(vols: List[float], period: int = 20) -> float:
    if not vols:
        return 1.0
    if len(vols) < period:
        avg = float(np.mean(vols))
    else:
        avg = float(np.mean(vols[-period:]))
    if avg <= 0:
        return 1.0
    return float(vols[-1] / avg)

#########################################
# SCORING (sniper)
#########################################

def score_trade(rsi: float, volr: float, trend: float, mom: float) -> float:
    score = 0.0
    if 55 <= rsi <= 75:
        score += 1.0
    elif rsi > 75:
        score += 0.5

    if volr > 1.2:
        score += 1.0
    if trend > 0.0:
        score += 1.0
    if mom > 0.0:
        score += 1.0

    return float(score)

#########################################
# ML (RandomForest)
#########################################

model: Optional[RandomForestClassifier] = None
ml_rows = 0

def load_history_matrix() -> Optional[np.ndarray]:
    try:
        if not os.path.exists(HISTORY_FILE):
            return None
        data = np.loadtxt(HISTORY_FILE, delimiter=",", skiprows=1)
        if data.size == 0:
            return None
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        return data
    except Exception:
        return None

def train_model_if_ready() -> None:
    global model, ml_rows
    if not ML_ENABLED:
        model = None
        ml_rows = 0
        return

    data = load_history_matrix()
    if data is None:
        model = None
        ml_rows = 0
        return

    ml_rows = int(len(data))
    if ml_rows < ML_MIN_TRADES_TO_ENABLE:
        model = None
        return

    try:
        X = data[:, :-1]
        y = (data[:, -1] > 0).astype(int)
        m = RandomForestClassifier(n_estimators=250, random_state=42)
        m.fit(X, y)
        model = m
        notify(f"ML TRAINED | trades={ml_rows} | min_prob={ML_MIN_PROB}")
    except Exception:
        model = None

def ml_allows(features_vec: List[float]) -> bool:
    if not ML_ENABLED:
        return True
    if model is None:
        return True  # allow trading while collecting data
    try:
        prob = float(model.predict_proba([features_vec])[0][1])
        return prob >= ML_MIN_PROB
    except Exception:
        return True

#########################################
# PORTFOLIO (paper)
#########################################

cash = START_BALANCE
open_trades: List[Dict] = []

def reserve_cash() -> float:
    return cash * (MIN_CASH_RESERVE_PERCENT / 100.0)

def dynamic_position_percent() -> float:
    trades = learning.get("trade_count", 0)
    wins = learning.get("win_count", 0)

    if trades < 5:
        return float(MIN_POSITION_SIZE_PERCENT)

    win_rate = (wins / trades) if trades > 0 else 0.0
    growth = (cash - START_BALANCE) / START_BALANCE if START_BALANCE > 0 else 0.0

    pct = float(MIN_POSITION_SIZE_PERCENT)

    if win_rate > 0.55: pct += 1
    if win_rate > 0.60: pct += 1
    if win_rate > 0.65: pct += 1

    if growth > 0.05: pct += 1
    if growth > 0.10: pct += 1
    if growth > 0.20: pct += 1

    return float(min(pct, MAX_POSITION_SIZE_PERCENT))

def position_size_cash() -> float:
    pct = dynamic_position_percent()
    desired = cash * (pct / 100.0)
    available = max(0.0, cash - reserve_cash())
    return float(min(desired, available))

def already_open(sym: str) -> bool:
    return any(t["sym"] == sym for t in open_trades)

#########################################
# FEATURES
#########################################

def build_features(sym: str) -> Optional[Dict[str, float]]:
    candles = get_candles(sym)
    if not candles or len(candles) < 20:
        return None

    closes = [float(c[4]) for c in candles]
    vols = [float(c[5]) for c in candles]

    rsi = calc_rsi(closes, 14)
    volr = volume_ratio(vols, 20)
    tr = trend_strength(closes)
    mom = momentum(closes, 5)

    return {
        "rsi": float(rsi),
        "volume_ratio": float(volr),
        "trend_strength": float(tr),
        "momentum": float(mom),
    }

#########################################
# TRADING (paper)
#########################################

def close_trade(t: Dict, price: float, reason: str) -> None:
    global cash

    proceeds = t["qty"] * price
    cost = t["qty"] * t["entry"]
    profit = proceeds - cost

    cash += proceeds

    learning["trade_count"] = learning.get("trade_count", 0) + 1
    learning["total_profit"] = learning.get("total_profit", 0.0) + float(profit)
    if profit > 0:
        learning["win_count"] = learning.get("win_count", 0) + 1
    save_learning(learning)

    # feed ML dataset (features at entry, profit at exit)
    append_history_row(
        float(t.get("rsi", 50.0)),
        float(t.get("volume_ratio", 1.0)),
        float(t.get("trend_strength", 0.0)),
        float(t.get("momentum", 0.0)),
        float(profit),
    )

    notify(
        f"SELL {t['sym']} ({reason})\n"
        f"Entry {t['entry']:.6f}\n"
        f"Exit  {price:.6f}\n"
        f"P/L   {profit:+.2f}\n"
        f"Cash  {cash:.2f}"
    )

def open_trade(sym: str, price: float, score: float, feats: Dict[str, float]) -> None:
    global cash

    if len(open_trades) >= MAX_OPEN_TRADES:
        return
    if already_open(sym):
        return

    size = position_size_cash()
    if size <= 0:
        return

    qty = size / price
    if qty <= 0:
        return

    cash -= size

    t = {
        "sym": sym,
        "entry": price,
        "qty": qty,
        "score": score,
        "time": time.time(),

        # store entry features for ML
        "rsi": feats["rsi"],
        "volume_ratio": feats["volume_ratio"],
        "trend_strength": feats["trend_strength"],
        "momentum": feats["momentum"],

        # stops
        "stop": price * (1.0 - STOP_LOSS_PERCENT / 100.0),
        "peak": price,
        "trail": None,
        "trailing_active": False,
    }

    open_trades.append(t)

    notify(
        f"BUY {sym}\n"
        f"Score {score:.2f}\n"
        f"Size  {size:.2f}\n"
        f"Cash  {cash:.2f}"
    )

def replace_if_better(sym: str, sym_price: float, score: float, feats: Dict[str, float], prices: Dict[str, float]) -> bool:
    if not REPLACE_WITH_BETTER:
        return False
    if len(open_trades) < MAX_OPEN_TRADES:
        return False

    weakest = min(open_trades, key=lambda x: float(x.get("score", 0.0)))
    weakest_score = float(weakest.get("score", 0.0))
    if score < weakest_score + REPLACE_SCORE_MARGIN:
        return False

    weakest_price = prices.get(weakest["sym"])
    if weakest_price is None:
        return False  # can't safely replace without price

    close_trade(weakest, weakest_price, "REPLACED")
    try:
        open_trades.remove(weakest)
    except Exception:
        pass

    open_trade(sym, sym_price, score, feats)
    return True

def manage_trades(prices: Dict[str, float]) -> None:
    global open_trades

    now = time.time()
    remaining: List[Dict] = []

    for t in open_trades:
        sym = t["sym"]
        p = prices.get(sym)
        if p is None:
            remaining.append(t)
            continue

        # update peak
        if p > t["peak"]:
            t["peak"] = p

        # activate trailing
        if not t["trailing_active"]:
            if p >= t["entry"] * (1.0 + TRAILING_START_PERCENT / 100.0):
                t["trailing_active"] = True

        # update trailing stop
        if t["trailing_active"]:
            t["trail"] = t["peak"] * (1.0 - TRAILING_DISTANCE_PERCENT / 100.0)

        age_min = (now - t["time"]) / 60.0
        profit_pct = ((p - t["entry"]) / t["entry"]) * 100.0 if t["entry"] > 0 else 0.0

        # stagnation exit: free capital unless it's doing well
        if age_min >= MAX_TRADE_DURATION_MINUTES and profit_pct < MIN_PROFIT_KEEP_PERCENT:
            close_trade(t, p, "STAGNATION")
            continue

        # hard stop
        if p <= t["stop"]:
            close_trade(t, p, "STOP")
            continue

        # trailing stop only if activated
        if t["trail"] is not None and p <= t["trail"]:
            close_trade(t, p, "TRAIL")
            continue

        remaining.append(t)

    open_trades = remaining

#########################################
# STATUS
#########################################

def send_status() -> None:
    trades = int(learning.get("trade_count", 0))
    wins = int(learning.get("win_count", 0))
    win_rate = (wins / trades * 100.0) if trades > 0 else 0.0

    ml_state = "OFF"
    if ML_ENABLED:
        ml_state = "ON" if model is not None else f"WARMUP {ml_rows}/{ML_MIN_TRADES_TO_ENABLE}"

    notify(
        "STATUS\n"
        f"Cash {cash:.2f}\n"
        f"Open trades {len(open_trades)}/{MAX_OPEN_TRADES}\n"
        f"Position size {dynamic_position_percent():.1f}%\n"
        f"Profit {learning.get('total_profit', 0.0):.2f}\n"
        f"Trades {trades} | Wins {wins} | WR {win_rate:.1f}%\n"
        f"ML {ml_state}"
    )

#########################################
# MAIN LOOP
#########################################

notify(
    "Coin Sniper started\n"
    f"Mode: {'PAPER' if PAPER_TRADING else 'REAL'} (this code is paper-only)\n"
    f"Universe: {'AUTO' if COINS.upper()=='AUTO' else COINS}\n"
    f"Symbols: {len(COIN_LIST)} (cap {MAX_SYMBOLS})\n"
    f"Micro pos%: {MIN_POSITION_SIZE_PERCENT}-{MAX_POSITION_SIZE_PERCENT}\n"
    f"Max open: {MAX_OPEN_TRADES}\n"
    f"MIN_SCORE: {MIN_SCORE}\n"
)

train_model_if_ready()
last_status = time.time()
last_train = time.time()

while True:
    try:
        # 1) pull prices
        prices: Dict[str, float] = {}
        for sym in COIN_LIST:
            tick = get_ticker(sym)
            if tick and "price" in tick:
                try:
                    prices[sym] = float(tick["price"])
                except Exception:
                    pass

        # 2) manage exits first (free capital)
        if prices:
            manage_trades(prices)

        # 3) scan entries (keep trades flowing)
        for sym in COIN_LIST:
            if len(open_trades) >= MAX_OPEN_TRADES:
                break
            if already_open(sym):
                continue

            price = prices.get(sym)
            if price is None:
                continue

            feats = build_features(sym)
            if feats is None:
                continue

            sc = score_trade(feats["rsi"], feats["volume_ratio"], feats["trend_strength"], feats["momentum"])
            if sc < MIN_SCORE:
                continue

            vec = [feats["rsi"], feats["volume_ratio"], feats["trend_strength"], feats["momentum"]]
            if not ml_allows(vec):
                continue

            # if full, try replace weakest
            if len(open_trades) >= MAX_OPEN_TRADES:
                replace_if_better(sym, price, sc, feats, prices)
            else:
                open_trade(sym, price, sc, feats)

        # 4) status
        if time.time() - last_status >= STATUS_INTERVAL:
            send_status()
            last_status = time.time()

        # 5) retrain ML periodically
        if time.time() - last_train >= ML_RETRAIN_SECONDS:
            train_model_if_ready()
            last_train = time.time()

        time.sleep(SCAN_INTERVAL)

    except Exception as e:
        notify(f"ERROR {e}")
        time.sleep(5)
