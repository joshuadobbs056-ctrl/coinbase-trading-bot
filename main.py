
# Coin Sniper — MACD + Golden Cross Runner (Savage ELITE)
# ✅ Runner scan (MACD cross + EMA golden cross + volume expansion)
# ✅ ATR-based adaptive trailing (starts after TRAILING_START_PERCENT)
# ✅ Market guard (BTC crash blocks NEW buys)
# ✅ Cooldowns + max new buys per scan
# ✅ Paper trading ledger + win/loss + equity + trade history CSV
# ✅ Optional GitHub persistence + Telegram alerts
#
# Coinbase Exchange public endpoints (no API keys needed): /products, /ticker, /candles
# NOTE: This is PAPER trading only (simulated fills). Not live trading.

import os, time, json, csv, math, hashlib, traceback, base64
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
# CONFIG (env)
# =========================
START_BALANCE = float(os.getenv("START_BALANCE", "1000"))

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "10"))            # faster scans for runners
STATUS_INTERVAL = int(os.getenv("STATUS_INTERVAL", "60"))
ML_INTERVAL = int(os.getenv("ML_INTERVAL", "300"))

MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", "12"))
MAX_NEW_BUYS_PER_SCAN = int(os.getenv("MAX_NEW_BUYS_PER_SCAN", "2"))

MIN_TRADE_SIZE = float(os.getenv("MIN_TRADE_SIZE", os.getenv("MIN_TRADE_SIZE_USD", "35")))

# Hard stop always active
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "3.5"))

# Trailing start + distance controls (distance will become ATR-based)
TRAILING_START_PERCENT = float(os.getenv("TRAILING_START_PERCENT", "1.2"))   # must be in profit before trailing tightens
TRAIL_DIST_BASE = float(os.getenv("TRAILING_DISTANCE_PERCENT", "0.9"))       # floor trail distance (%)
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_MULT = float(os.getenv("ATR_MULT", "1.4"))                               # ATR% * mult = trail distance (%)

# Entry gate
ENTRY_SCORE_MIN = int(os.getenv("ENTRY_SCORE_MIN", os.getenv("MIN_ENTRY_SCORE", "7")))

# Symbol universe
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "80"))
COINS = os.getenv("COINS", "").strip()                                       # optional comma list pin
EXCLUDE = set([s.strip() for s in os.getenv("EXCLUDE", "").split(",") if s.strip()])

SYMBOL_REFRESH_INTERVAL = int(os.getenv("SYMBOL_REFRESH_INTERVAL", "3600"))

# Candles for indicators/volume
CANDLE_GRANULARITY = int(os.getenv("CANDLE_GRANULARITY", "60"))              # 60s candles
CANDLE_POINTS = int(os.getenv("CANDLE_POINTS", "200"))

# Runner filters
MIN_VOLUME_RATIO = float(os.getenv("MIN_VOLUME_RATIO", "1.25"))              # last vol / mean prev
MIN_TREND_STRENGTH = float(os.getenv("MIN_TREND_STRENGTH", "0.00035"))       # slope gate (normalized)
MACD_PRE_CROSS_MAX = float(os.getenv("MACD_PRE_CROSS_MAX", 0.002))
MACD_PRE_CROSS_BONUS = float(os.getenv("MACD_PRE_CROSS_BONUS", 1.0))
EMA_GOLDEN_BONUS = float(os.getenv("EMA_GOLDEN_BONUS", 1.0))

COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "120"))

# Risk sizing
CASH_RESERVE_PERCENT = float(os.getenv("CASH_RESERVE_PERCENT", "5"))
MIN_POSITION_SIZE_PERCENT = float(os.getenv("MIN_POSITION_SIZE_PERCENT", "3"))
MAX_POSITION_SIZE_PERCENT = float(os.getenv("MAX_POSITION_SIZE_PERCENT", "10"))

# Trade duration safety (0 disables)
MAX_TRADE_DURATION_MINUTES = int(os.getenv("MAX_TRADE_DURATION_MINUTES", "720"))  # 12h default
# Optional profit target (0 disables)
PROFIT_TARGET_PERCENT = float(os.getenv("PROFIT_TARGET_PERCENT", "0"))

# Market guard
MARKET_GUARD = os.getenv("MARKET_GUARD", "true").lower() == "true"
BTC_SYMBOL = os.getenv("BTC_SYMBOL", "BTC-USD")
BTC_WINDOW = int(os.getenv("BTC_WINDOW", "20"))
BTC_CRASH_PCT = float(os.getenv("BTC_CRASH_PCT", "2.0"))

# ML gating
ML_ENABLED = os.getenv("ML_ENABLED", "true").lower() == "true"
ML_ENABLE_AFTER = int(os.getenv("ML_ENABLE_AFTER", os.getenv("ML_MIN_TRADES_TO_ENABLE", "50")))
ML_MIN_PROB = float(os.getenv("ML_MIN_PROB", "0.62"))

BOT_PAUSED = os.getenv("BOT_PAUSED", "false").lower() == "true"

# Notifications
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# GitHub persistence (optional)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_DATA_REPO")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")
GITHUB_PUSH_INTERVAL = int(os.getenv("GITHUB_PUSH_INTERVAL", "180"))

# =========================
# FILES
# =========================
LEARNING_FILE = "learning.json"
POSITIONS_FILE = "positions.json"
HISTORY_FILE = "trade_history.csv"
ML_TRAIN_FILE = "ml_training.csv"

# Coinbase Exchange base
BASE_URL = "https://api.exchange.coinbase.com/products"

# =========================
# NOTIFY
# =========================
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

# =========================
# GITHUB HELPERS
# =========================
GITHUB_FILES = [LEARNING_FILE, POSITIONS_FILE, ML_TRAIN_FILE, HISTORY_FILE]

def _gh_headers():
    return {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}

def github_pull_file(filename: str):
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return
    try:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"
        r = requests.get(url, headers=_gh_headers(), timeout=15)
        j = r.json()
        if r.status_code == 200 and isinstance(j, dict) and "content" in j:
            content = base64.b64decode(j["content"])
            open(filename, "wb").write(content)
            notify(f"[{INSTANCE_ID}] GITHUB RESTORE {filename}")
    except Exception:
        pass

def github_push_file(filename: str):
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return
    try:
        if not os.path.exists(filename):
            return
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"
        content = base64.b64encode(open(filename, "rb").read()).decode("utf-8")

        sha = None
        r = requests.get(url, headers=_gh_headers(), timeout=15)
        j = r.json() if r is not None else {}
        if r.status_code == 200 and isinstance(j, dict) and "sha" in j:
            sha = j["sha"]

        payload = {"message": f"update {filename}", "content": content, "branch": GITHUB_BRANCH}
        if sha:
            payload["sha"] = sha

        requests.put(url, headers=_gh_headers(), json=payload, timeout=15)
    except Exception:
        pass

def github_pull_all():
    for f in GITHUB_FILES:
        github_pull_file(f)

_last_push_time = 0.0
_last_state_hash = None

def _hash_state(learning_obj, positions_obj) -> str:
    try:
        raw = json.dumps({"learning": learning_obj, "positions": positions_obj}, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()
    except Exception:
        return str(time.time())

def github_push_all_if_needed(state_hash: str):
    global _last_push_time, _last_state_hash
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return
    now = time.time()
    interval_ok = (now - _last_push_time) >= GITHUB_PUSH_INTERVAL
    changed = (state_hash != _last_state_hash)
    if not (changed or interval_ok):
        return
    for f in GITHUB_FILES:
        github_push_file(f)
    _last_push_time = now
    _last_state_hash = state_hash
    notify(f"[{INSTANCE_ID}] GITHUB SYNC COMPLETE (changed={changed})")

# =========================
# STATE LOAD/SAVE
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

def _ensure_csv_headers(path, headers):
    if os.path.exists(path):
        return
    try:
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(headers)
    except Exception:
        pass

def _append_row(path, row):
    try:
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow(row)
    except Exception:
        pass

# Restore from GitHub first (if configured)
github_pull_all()

learning = load_json(LEARNING_FILE, {
    "cash": START_BALANCE,
    "trade_count": 0,
    "win_count": 0,
    "loss_count": 0,
    "total_profit": 0.0
})
positions = load_json(POSITIONS_FILE, {})

learning["cash"] = float(learning.get("cash", START_BALANCE))
learning["trade_count"] = int(learning.get("trade_count", 0))
learning["win_count"] = int(learning.get("win_count", 0))
learning["loss_count"] = int(learning.get("loss_count", 0))
learning["total_profit"] = float(learning.get("total_profit", 0.0))

cash = float(learning["cash"])

_hist_headers = ["ts","action","sym","price","qty","notional","profit","cash","equity","score","reason"]
_ensure_csv_headers(HISTORY_FILE, _hist_headers)

_train_headers = ["sym","ts","ret_5","ret_15","vol_15","slope_15","dd_15","score","label"]
_ensure_csv_headers(ML_TRAIN_FILE, _train_headers)

def record_history(action, sym, price, qty, notional, profit, cash_now, equity_now, score=None, reason=""):
    _append_row(HISTORY_FILE, [
        int(time.time()),
        action,
        sym,
        f"{price:.10f}",
        f"{qty:.10f}",
        f"{notional:.2f}",
        f"{profit:.2f}",
        f"{cash_now:.2f}",
        f"{equity_now:.2f}",
        "" if score is None else int(score),
        reason
    ])

# =========================
# MARKET DATA
# =========================
def get_price(sym: str):
    try:
        r = requests.get(f"{BASE_URL}/{sym}/ticker", timeout=10)
        j = r.json()
        if "price" not in j:
            return None
        return float(j["price"])
    except Exception:
        return None

def price_from_prices_or_candles(sym: str, prices: dict):
    """Fallback to latest candle close when ticker price isn't available."""
    px = price_from_prices_or_candles(sym, prices)
    if px:
        return float(px)
    try:
        candles = get_candles(sym)
        if isinstance(candles, list) and len(candles) > 0:
            return float(candles[0][4])  # most-recent candle close (Coinbase returns newest-first)
    except Exception:
        pass
    return None


def get_symbols():
    try:
        r = requests.get(BASE_URL, timeout=10)
        products = r.json()
        syms = []
        for p in products:
            if p.get("quote_currency") == "USD":
                sid = p.get("id")
                if sid and sid.endswith("-USD") and sid not in EXCLUDE:
                    syms.append(sid)
        return syms[:MAX_SYMBOLS]
    except Exception:
        return []

# Candles (cached)
_candle_cache = {}
CANDLE_CACHE_TTL = int(os.getenv("CANDLE_CACHE_TTL", "25"))

def get_candles(sym: str):
    """Returns candles [time, low, high, open, close, volume] most-recent first."""
    try:
        now = time.time()
        c = _candle_cache.get(sym)
        if c and (now - c["ts"]) < CANDLE_CACHE_TTL:
            return c["data"]
        url = f"{BASE_URL}/{sym}/candles"
        params = {"granularity": int(CANDLE_GRANULARITY), "limit": int(CANDLE_POINTS)}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if not isinstance(data, list) or len(data) < max(ATR_PERIOD + 5, 50):
            return None
        _candle_cache[sym] = {"ts": now, "data": data}
        return data
    except Exception:
        return None

def _atr_percent(candles) -> float:
    try:
        # convert to oldest->newest for indicator math
        recent = list(reversed(candles[:max(ATR_PERIOD + 20, 50)]))
        highs = [float(c[2]) for c in recent]
        lows  = [float(c[1]) for c in recent]
        closes= [float(c[4]) for c in recent]
        trs = []
        for i in range(1, len(closes)):
            h, l, pc = highs[i], lows[i], closes[i-1]
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        if len(trs) < ATR_PERIOD:
            return float(TRAIL_DIST_BASE)
        atr = float(np.mean(trs[-ATR_PERIOD:]))
        last_close = float(closes[-1])
        if last_close <= 0:
            return float(TRAIL_DIST_BASE)
        return float((atr / last_close) * 100.0)
    except Exception:
        return float(TRAIL_DIST_BASE)

def volume_ratio(candles) -> float:
    try:
        recent = list(reversed(candles[:80]))
        vols = [float(c[5]) for c in recent]
        if len(vols) < 30:
            return 1.0
        last = vols[-1]
        base = float(np.mean(vols[-21:-1])) if float(np.mean(vols[-21:-1])) > 0 else 1.0
        return float(last / base)
    except Exception:
        return 1.0

# =========================
# INDICATORS (EMA / MACD)
# =========================
def _ema(values, period):
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return arr
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
    return out

def macd_pack(closes, fast=12, slow=26, signal=9):
    closes = np.array(closes, dtype=float)
    if len(closes) < slow + signal + 3:
        return None
    ef = _ema(closes, fast)
    es = _ema(closes, slow)
    macd = ef - es
    sig = _ema(macd, signal)
    hist = macd - sig
    # return current/prev diff sign for cross detection
    diff_now = float(macd[-1] - sig[-1])
    diff_prev = float(macd[-2] - sig[-2])
    hist_prev = float(hist[-2])
    return float(macd[-1]), float(sig[-1]), float(hist[-1]), diff_now, diff_prev, hist_prev

def ema_cross(closes, fast=9, slow=21):
    closes = np.array(closes, dtype=float)
    if len(closes) < slow + 3:
        return None
    ef = _ema(closes, fast)
    es = _ema(closes, slow)
    now = float(ef[-1] - es[-1])
    prev = float(ef[-2] - es[-2])
    return now, prev

# =========================
# ML (optional)
# =========================
ml_model = None
ml_last_train = 0.0

# in-memory mini history for ML features (ticker points)
price_history = {}
def compute_features_from_history(sym):
    h = price_history.get(sym, [])
    if len(h) < 20:
        return None
    last = np.array(h[-20:], dtype=float)
    p0 = last[-1]
    p5 = last[-5]
    p15 = last[-15]
    ret_5 = (p0 - p5) / p5 if p5 > 0 else 0.0
    ret_15 = (p0 - p15) / p15 if p15 > 0 else 0.0
    rets = np.diff(last) / np.maximum(last[:-1], 1e-12)
    vol_15 = float(np.std(rets[-15:])) if len(rets) >= 15 else float(np.std(rets))
    x = np.arange(len(last), dtype=float)
    try:
        slope = float(np.polyfit(x, last, 1)[0] / max(p0, 1e-12))
    except Exception:
        slope = 0.0
    peak = float(np.max(last))
    dd_15 = (p0 - peak) / peak if peak > 0 else 0.0
    return [ret_5, ret_15, vol_15, slope, dd_15]

def train_ml_if_ready(force=False):
    global ml_model, ml_last_train
    if not ML_ENABLED:
        return
    now = time.time()
    if not force and (now - ml_last_train) < ML_INTERVAL:
        return
    if learning.get("trade_count", 0) < 10:
        return
    try:
        rows = []
        with open(ML_TRAIN_FILE, "r") as f:
            for row in csv.DictReader(f):
                rows.append(row)
        if len(rows) < 40:
            return
        X, y = [], []
        for row in rows:
            X.append([float(row["ret_5"]), float(row["ret_15"]), float(row["vol_15"]),
                      float(row["slope_15"]), float(row["dd_15"]), float(row["score"])])
            y.append(int(row["label"]))
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X, y)
        ml_model = clf
        ml_last_train = now
        notify(f"[{INSTANCE_ID}] ML TRAINED rows={len(rows)}")
    except Exception:
        notify(f"[{INSTANCE_ID}] ML TRAIN FAIL\n{traceback.format_exc()}")

def ml_probability(sym, score):
    if not ML_ENABLED or ml_model is None:
        return None
    feats = compute_features_from_history(sym)
    if feats is None:
        return None
    X = [[feats[0], feats[1], feats[2], feats[3], feats[4], float(score)]]
    try:
        return float(ml_model.predict_proba(X)[0][1])
    except Exception:
        return None

# =========================
# SIZING / EQUITY
# =========================
def compute_equity(latest_prices):
    eq = float(cash)
    for sym, pos in positions.items():
        px = latest_prices.get(sym)
        eq += float(pos["qty"]) * float(px if px else pos["entry"])
    return float(eq)

def compute_position_notional(score, equity):
    min_pct = MIN_POSITION_SIZE_PERCENT / 100.0
    max_pct = MAX_POSITION_SIZE_PERCENT / 100.0
    t = (float(score) - 1.0) / 9.0
    pct = min_pct + (max_pct - min_pct) * t
    reserve = equity * (CASH_RESERVE_PERCENT / 100.0)
    available = max(0.0, cash - reserve)
    notional = available * pct
    if notional < MIN_TRADE_SIZE:
        return 0.0
    return float(min(notional, available))

# =========================
# TRAILING STOP (ATR adaptive)
# =========================
def apply_trailing_stop(pos, sym: str, price: float):
    # update peak always
    peak = float(pos.get("peak", pos["entry"]))
    if price > peak:
        pos["peak"] = price

    entry = float(pos["entry"])
    if entry <= 0:
        return

    gain_pct = (price - entry) / entry * 100.0
    if gain_pct < TRAILING_START_PERCENT:
        return

    candles = get_candles(sym)
    atr_pct = _atr_percent(candles) if candles else TRAIL_DIST_BASE
    trail_dist = max(TRAIL_DIST_BASE, ATR_MULT * atr_pct)
    trail_dist = min(trail_dist, 6.0)

    trail = float(pos.get("peak", price)) * (1 - trail_dist / 100.0)
    if trail > float(pos["stop"]):
        pos["stop"] = trail
        pos["trail_dist"] = float(trail_dist)

# =========================
# SCORE (MACD + Golden Cross + Runner filter)
# =========================
def score_symbol(sym: str, candles):
    # Need candles
    if not candles:
        return None, "no_candles"

    # oldest->newest
    recent = list(reversed(candles[:CANDLE_POINTS]))
    closes = [float(c[4]) for c in recent]
    if len(closes) < 60:
        return None, "candles_short"

    p0 = closes[-1]
    p15 = closes[-16] if len(closes) > 16 else closes[0]
    p60 = closes[-61] if len(closes) > 61 else closes[0]

    r15 = (p0 - p15) / p15 if p15 > 0 else 0.0
    r60 = (p0 - p60) / p60 if p60 > 0 else 0.0

    # MACD and EMA cross
    mp = macd_pack(closes)
    ep = ema_cross(closes)
    if mp is None or ep is None:
        return None, "indicators_short"

    macd, sig, hist, diff_now, diff_prev, hist_prev = mp
    ema_now, ema_prev = ep

    macd_cross_up = (diff_prev <= 0.0 and diff_now > 0.0)
    golden_cross = (ema_prev <= 0.0 and ema_now > 0.0) or (ema_now > 0.0)

    vratio = volume_ratio(candles)

    # Base score
    score = 5.0

    # Runner: positive short return + volume expansion
    if r15 >= 0.02 and vratio >= MIN_VOLUME_RATIO:
        score += 2.0
    elif r15 >= 0.012 and vratio >= MIN_VOLUME_RATIO:
        score += 1.0

    # MACD / Golden Cross confirmation
    if hist > 0:
        score += 1.0
    else:
        score -= 1.5

    if macd_cross_up:
        score += 1.0

    if golden_cross:
        score += 1
    else:
        score -= 1.0

    # PRE-CROSS ENTRY (your style): MACD still negative but rising toward zero, histogram improving
    # This helps you get in *before* the actual MACD cross.
    try:
        if (diff_now is not None) and (diff_prev is not None) and (hist is not None):
            # diff_now < 0 means MACD is still below signal (no cross yet)
            if diff_now < 0 and diff_now > diff_prev and (hist_prev is not None) and (hist > hist_prev) and abs(diff_now) <= MACD_PRE_CROSS_MAX:
                score += float(MACD_PRE_CROSS_BONUS)
    except Exception:
        pass


    # Avoid chasing late blow-offs
    if r60 >= 0.25:
        score -= 2.0
    elif r60 >= 0.15:
        score -= 1.0

    # Trend strength proxy from close slope (normalized)
    try:
        arr = np.array(closes[-60:], dtype=float)
        x = np.arange(len(arr), dtype=float)
        slope = float(np.polyfit(x, arr, 1)[0] / max(arr[-1], 1e-12))
        if slope < MIN_TREND_STRENGTH:
            score -= 2.0
    except Exception:
        slope = 0.0

    score = int(max(1, min(10, round(score))))
    reason = f"r15={r15:.3f} r60={r60:.3f} vR={vratio:.2f} macdH={hist:.4f} macdX={int(macd_cross_up)} emaX={int(golden_cross)} slope={slope:.5f}"
    return score, reason

# =========================
# BUY/SELL
# =========================
last_buy_by_sym = {}
btc_history = []

def should_buy(sym, score, prices, equity, market_ok):
    if BOT_PAUSED:
        return False, "BOT_PAUSED"
    if not market_ok:
        return False, "MARKET_GUARD"
    if sym in EXCLUDE:
        return False, "EXCLUDED"
    if sym in positions:
        return False, "ALREADY_IN_POSITION"
    if len(positions) >= MAX_OPEN_TRADES:
        return False, f"MAX_OPEN_TRADES({MAX_OPEN_TRADES})"

    last_ts = int(last_buy_by_sym.get(sym, 0))
    if last_ts and (time.time() - last_ts) < COOLDOWN_SECONDS:
        return False, f"COOLDOWN<{COOLDOWN_SECONDS}s"

    px = price_from_prices_or_candles(sym, prices)
    if not px:
        return False, "NO_PRICE"

    if score < ENTRY_SCORE_MIN:
        return False, f"SCORE<{ENTRY_SCORE_MIN}"

    # ML gate (optional)
    if ML_ENABLED and learning.get("trade_count", 0) >= ML_ENABLE_AFTER:
        p = ml_probability(sym, score)
        if p is None:
            return False, "ML_NO_FEATURES"
        if p < ML_MIN_PROB:
            return False, f"ML_PROB<{ML_MIN_PROB:.2f} ({p:.2f})"

    notional = compute_position_notional(score, equity)
    if notional <= 0:
        return False, "SIZE_TOO_SMALL"
    return True, "OK"

def open_position(sym, score, prices, equity, reason):
    global cash
    px = price_from_prices_or_candles(sym, prices)
    if not px:
        return False, "NO_PRICE"
    notional = compute_position_notional(score, equity)
    if notional <= 0:
        return False, "SIZE_TOO_SMALL"
    qty = notional / px
    if qty <= 0:
        return False, "QTY_ZERO"

    cash -= notional
    entry = float(px)
    stop = entry * (1.0 - STOP_LOSS_PERCENT / 100.0)
    positions[sym] = {
        "entry": entry,
        "qty": float(qty),
        "stop": float(stop),
        "peak": float(entry),
        "opened_ts": int(time.time()),
        "score": int(score),
        "reason": reason,
        "trail_dist": None
    }
    last_buy_by_sym[sym] = int(time.time())
    return True, f"BUY qty={qty:.6f} notional={notional:.2f} entry={entry:.6f} stop={stop:.6f}"

def close_position(sym, px, reason):
    global cash
    pos = positions[sym]
    entry = float(pos["entry"])
    qty = float(pos["qty"])
    proceeds = qty * px
    profit = (px - entry) * qty
    cash += proceeds

    learning["trade_count"] += 1
    learning["total_profit"] += float(profit)
    if profit >= 0:
        learning["win_count"] += 1
    else:
        learning["loss_count"] += 1

    return profit, proceeds

# =========================
# SYMBOLS
# =========================
symbols = get_symbols() if (not COINS or COINS.strip().upper() == "AUTO") else [s.strip() for s in COINS.split(",") if s.strip()]

# =========================
# MAIN LOOP
# =========================
notify(f"[{INSTANCE_ID}] START | symbols={len(symbols)} scan={SCAN_INTERVAL}s status={STATUS_INTERVAL}s")

last_scan = 0.0
last_status = 0.0
last_ml = 0.0
last_symbol_refresh = 0.0

while True:
    try:
        now = time.time()

        # refresh symbols (only if not pinned)
        if (not COINS) and (now - last_symbol_refresh >= SYMBOL_REFRESH_INTERVAL):
            symbols = get_symbols()
            last_symbol_refresh = now
            notify(f"[{INSTANCE_ID}] SYMBOL_REFRESH count={len(symbols)}")

        did_state_change = False

        if now - last_scan >= SCAN_INTERVAL:
            prices = {}
            fetched = 0

            # Pull tickers and build mini history (for ML + quick equity)
            for sym in symbols:
                px = get_price(sym)
                if not px:
                    # ticker can fail; fall back to latest candle close
                    try:
                        c = get_candles(sym)
                        if isinstance(c, list) and len(c) > 0:
                            px = float(c[0][4])
                    except Exception:
                        px = None
                if px:
                    prices[sym] = px
                    fetched += 1
                    price_history.setdefault(sym, []).append(px)
                    if len(price_history[sym]) > 200:
                        price_history[sym] = price_history[sym][-200:]

            equity = compute_equity(prices)

            # Market guard (BTC crash blocks NEW buys)
            market_ok = True
            if MARKET_GUARD:
                btc_px = get_price(BTC_SYMBOL)
                if btc_px:
                    btc_history.append(float(btc_px))
                    if len(btc_history) > 200:
                        btc_history = btc_history[-200:]
                    if len(btc_history) >= BTC_WINDOW:
                        btc_ret = (btc_history[-1] - btc_history[-BTC_WINDOW]) / max(btc_history[-BTC_WINDOW], 1e-12) * 100.0
                        if btc_ret <= -abs(BTC_CRASH_PCT):
                            market_ok = False

            # Manage open positions
            for sym in list(positions.keys()):
                px = price_from_prices_or_candles(sym, prices)
                if not px:
                    continue

                pos = positions[sym]
                apply_trailing_stop(pos, sym, px)

                # Optional profit target
                if PROFIT_TARGET_PERCENT > 0:
                    entry = float(pos["entry"])
                    if entry > 0 and px >= entry * (1.0 + PROFIT_TARGET_PERCENT / 100.0):
                        profit, proceeds = close_position(sym, px, "PROFIT_TARGET")
                        eq2 = compute_equity(prices)
                        notify(f"[{INSTANCE_ID}] SELL {sym} px={px:.6f} profit={profit:.2f} reason=PROFIT_TARGET({PROFIT_TARGET_PERCENT:.2f}%)")
                        record_history("SELL", sym, px, pos["qty"], proceeds, profit, cash, eq2, pos.get("score"), "PROFIT_TARGET")
                        # ML row
                        feats = compute_features_from_history(sym)
                        if feats is not None:
                            _append_row(ML_TRAIN_FILE, [sym, int(time.time()),
                                f"{feats[0]:.6f}", f"{feats[1]:.6f}", f"{feats[2]:.6f}", f"{feats[3]:.6f}", f"{feats[4]:.6f}",
                                int(pos.get("score", 0)), 1 if profit > 0 else 0
                            ])
                        del positions[sym]
                        did_state_change = True
                        continue

                # Time-based exit
                if MAX_TRADE_DURATION_MINUTES > 0:
                    age = int(time.time()) - int(pos.get("opened_ts", int(time.time())))
                    if age >= MAX_TRADE_DURATION_MINUTES * 60:
                        profit, proceeds = close_position(sym, px, "MAX_DURATION")
                        eq2 = compute_equity(prices)
                        notify(f"[{INSTANCE_ID}] SELL {sym} px={px:.6f} profit={profit:.2f} reason=MAX_DURATION")
                        record_history("SELL", sym, px, pos["qty"], proceeds, profit, cash, eq2, pos.get("score"), "MAX_DURATION")
                        feats = compute_features_from_history(sym)
                        if feats is not None:
                            _append_row(ML_TRAIN_FILE, [sym, int(time.time()),
                                f"{feats[0]:.6f}", f"{feats[1]:.6f}", f"{feats[2]:.6f}", f"{feats[3]:.6f}", f"{feats[4]:.6f}",
                                int(pos.get("score", 0)), 1 if profit > 0 else 0
                            ])
                        del positions[sym]
                        did_state_change = True
                        continue

                # Stop hit
                if px <= float(pos["stop"]):
                    profit, proceeds = close_position(sym, px, "STOP/TRAIL")
                    eq2 = compute_equity(prices)
                    td = pos.get("trail_dist")
                    td_txt = f" trail={td:.2f}%" if isinstance(td, (int, float)) else ""
                    notify(f"[{INSTANCE_ID}] SELL {sym} px={px:.6f} profit={profit:.2f} stop={pos['stop']:.6f}{td_txt}")
                    record_history("SELL", sym, px, pos["qty"], proceeds, profit, cash, eq2, pos.get("score"), "STOP/TRAIL")
                    feats = compute_features_from_history(sym)
                    if feats is not None:
                        _append_row(ML_TRAIN_FILE, [sym, int(time.time()),
                            f"{feats[0]:.6f}", f"{feats[1]:.6f}", f"{feats[2]:.6f}", f"{feats[3]:.6f}", f"{feats[4]:.6f}",
                            int(pos.get("score", 0)), 1 if profit > 0 else 0
                        ])
                    del positions[sym]
                    did_state_change = True

            # BUY scan (runners)
            if not BOT_PAUSED:
                candidates = []
                for sym in symbols:
                    if sym in positions or sym in EXCLUDE:
                        continue
                    candles = get_candles(sym)
                    sc, reason = score_symbol(sym, candles)
                    if sc is None:
                        continue
                    candidates.append((sc, sym, reason))

                candidates.sort(reverse=True, key=lambda x: x[0])

                buys = 0
                rejects = 0
                equity = compute_equity(prices)

                for sc, sym, reason in candidates[:50]:
                    if buys >= MAX_NEW_BUYS_PER_SCAN:
                        break
                    ok, why = should_buy(sym, sc, prices, equity, market_ok)
                    if not ok:
                        rejects += 1
                        continue
                    ok2, msg = open_position(sym, sc, prices, equity, reason)
                    equity = compute_equity(prices)
                    if ok2:
                        notify(f"[{INSTANCE_ID}] BUY {sym} score={sc} | {msg} | {reason}")
                        record_history("BUY", sym, prices[sym], positions[sym]["qty"], positions[sym]["qty"] * prices[sym], 0.0, cash, equity, sc, reason)
                        did_state_change = True
                        buys += 1
                    else:
                        rejects += 1

                if candidates:
                    top_sc, top_sym, top_reason = candidates[0]
                    notify(f"[{INSTANCE_ID}] SCAN fetched={fetched}/{len(symbols)} cand={len(candidates)} rejects={rejects} buys={buys} top={top_sym} score={top_sc} | {top_reason}")
                else:
                    notify(f"[{INSTANCE_ID}] SCAN fetched={fetched}/{len(symbols)} cand=0 (no runners right now)")

            # Persist
            learning["cash"] = float(cash)
            save_json(LEARNING_FILE, learning)
            save_json(POSITIONS_FILE, positions)

            github_push_all_if_needed(_hash_state(learning, positions))
            last_scan = now

        # STATUS
        if now - last_status >= STATUS_INTERVAL:
            eq = compute_equity(prices if 'prices' in locals() else {})
            trades = int(learning.get("trade_count", 0))
            wins = int(learning.get("win_count", 0))
            losses = int(learning.get("loss_count", 0))
            winpct = (wins / trades * 100.0) if trades > 0 else 0.0
            notify(f"[{INSTANCE_ID}] STATUS cash=${cash:.2f} equity=${eq:.2f} open={len(positions)}/{MAX_OPEN_TRADES} trades={trades} W={wins} L={losses} win%={winpct:.1f} profit=${learning.get('total_profit',0.0):.2f}")
            last_status = now

        # ML train
        if now - last_ml >= ML_INTERVAL:
            train_ml_if_ready()
            last_ml = now

        time.sleep(1)

    except SystemExit:
        raise
    except Exception:
        notify(f"[{INSTANCE_ID}] ERROR\n{traceback.format_exc()}")
        time.sleep(5)
