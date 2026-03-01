# main.py
# Coin Sniper Bot (Elite Paper) — Coinbase public data + persistent learning + (optional) GitHub sync
#
# FIXES / UPGRADES YOU ASKED FOR:
# ✅ Status includes: Cash, Equity, Net, Open Trades, Trades, Wins, Losses, Winrate, ML ON/OFF
# ✅ Auto-creates + auto-updates learning.json, positions.json, trade_history.csv
# ✅ NO “GitHub push loop” by default (Railway autodeploy loop happens if the bot commits to the same repo Railway deploys from)
# ✅ Safe price parsing (prevents KeyError: 'price')
# ✅ Cooldown after SELL to prevent immediate sell-then-rebuy churn
# ✅ Optional GitHub sync modes: OFF / PULL_ONLY / PUSH_ONLY / BOTH (default PULL_ONLY)
#
# IMPORTANT (about your looping):
# If Railway is connected to GitHub with Auto-Deploy ON, and your bot PUSHES commits to that same repo,
# Railway will redeploy again → bot boots → pushes again → redeploy loop.
# To stop that forever:
#   - Set env: GITHUB_SYNC_MODE=PULL_ONLY   (recommended)
#   - OR disable Railway Auto-Deploy from repo
#   - OR sync to a DIFFERENT repo/branch not wired to Railway deploys

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
# Helpers: env parsing
# ============================================================

def _env(name: str) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return None
    v = str(v).strip()
    return v if v != "" else None

def env_str(*names: str, default: str = "") -> str:
    for n in names:
        v = _env(n)
        if v is not None:
            return v
    return default

def env_int(*names: str, default: int) -> int:
    for n in names:
        v = _env(n)
        if v is not None:
            try:
                return int(float(v))
            except Exception:
                pass
    return default

def env_float(*names: str, default: float) -> float:
    for n in names:
        v = _env(n)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    return default

def env_bool(*names: str, default: bool) -> bool:
    for n in names:
        v = _env(n)
        if v is not None:
            return v.lower() in ("1", "true", "yes", "y", "on")
    return default


# ============================================================
# CONFIG
# ============================================================

# Timing
SCAN_INTERVAL = env_int("SCAN_INTERVAL", default=12)
STATUS_INTERVAL = env_int("STATUS_INTERVAL", "STATUS_INTERVA", default=60)

# Universe
COINS = env_str("COINS", default="AUTO").strip()
MAX_SYMBOLS = env_int("MAX_SYMBOLS", default=60)
EXCLUDE_RAW = env_str("EXCLUDE", default="")
EXCLUDE = {s.strip().upper() for s in EXCLUDE_RAW.split(",") if s.strip()}

# Risk / exposure
START_BALANCE = env_float("START_BALANCE", "START_BAL", "START", default=1000.0)
CASH_RESERVE_PERCENT = env_float("CASH_RESERVE_PERCENT", "CASH_RESERVE_P", "MIN_CASH_RESER", default=5.0)
MAX_OPEN_TRADES = env_int("MAX_OPEN_TRADES", "MAX_OPEN_TRADE", "MAX_OPEN_TR", default=20)
MIN_TRADE_SIZE_USD = env_float("MIN_TRADE_SIZE_USD", "MIN_TRADE_SIZE", "MIN_TRADE_SIZ", default=25.0)

# Position sizing (% of cash)
MIN_POSITION_SIZE_PERCENT = env_float("MIN_POSITION_SIZE_PERCENT", "MIN_POSITION_S", "MIN_POSITION_U", default=1.0)
MAX_POSITION_SIZE_PERCENT = env_float("MAX_POSITION_SIZE_PERCENT", "MAX_POSITION_S", default=3.0)

# Stops
STOP_LOSS_PERCENT = env_float("STOP_LOSS_PERCENT", "STOP_LOSS_PERC", "STOP_LOSS_PERCEN", default=2.8)
TRAILING_START_PERCENT = env_float("TRAILING_START_PERCENT", "TRAILING_START", default=1.2)
TRAILING_DISTANCE_PERCENT = env_float(
    "TRAILING_DISTANCE_PERCENT",
    "TRAILING_DISTANCE",
    "TRAILING_DIST",
    "TRAILING_DISTANC",
    "TRAILING_DISTAN",
    "TRAILING_DISTA",
    default=0.9
)

# ATR trailing
ATR_PERIOD = env_int("ATR_PERIOD", "ATR_PERIO", default=14)
ATR_MULT = env_float("ATR_MULT", default=1.4)

# Time exits
MAX_TRADE_DURATION_MINUTES = env_int("MAX_TRADE_DURATION_MINUTES", "MAX_TRADE_DURA", "MAX_TRADE_DUR", default=30)

# Entry strictness
MIN_ENTRY_SCORE = env_float("MIN_ENTRY_SCORE", "MIN_ENTRY_SCOR", default=7.0)
MIN_TREND_STRENGTH = env_float("MIN_TREND_STRENGTH", "MIN_TREND_STRE", default=0.20)
MIN_VOLUME_RATIO = env_float("MIN_VOLUME_RATIO", "MIN_VOLUME_RAT", "MIN_VOL_RATIO", default=1.25)
RSI_MIN = env_float("RSI_MIN", default=52.0)
RSI_MAX = env_float("RSI_MAX", default=72.0)

# Candles
CANDLE_GRANULARITY = env_int("CANDLE_GRANULARITY", "CANDLE_GRANULA", default=60)
CANDLE_POINTS = env_int("CANDLE_POINTS", default=120)

# ML
ML_MIN_TRADES_TO_ENABLE = env_int("ML_MIN_TRADES_TO_ENABLE", "ML_MIN_TRADES", default=50)
ML_MIN_PROB = env_float("ML_MIN_PROB", default=0.58)
ML_RETRAIN_EVERY_SEC = env_int("ML_RETRAIN_EVERY_SEC", "ML_RETRAIN_EVE", default=600)

# Fee estimate (paper)
FEE_PERCENT_PER_SIDE = env_float("FEE_PERCENT_PER_SIDE", "FEE_PERCENT_PE", default=0.20)

# Re-entry cooldown (prevents sell→buy same coin instantly)
REENTRY_COOLDOWN_SEC = env_int("REENTRY_COOLDOWN_SEC", default=300)

# Mode
PAPER_TRADING = env_bool("PAPER_TRADING", default=True)

# Telegram
TELEGRAM_TOKEN = _env("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = _env("TELEGRAM_CHAT_ID") or _env("TELEGRAM_CHAT_")

# GitHub sync (optional)
GITHUB_TOKEN = _env("GITHUB_TOKEN")
GITHUB_REPO = _env("GITHUB_REPO")     # username/repo
GITHUB_BRANCH = _env("GITHUB_BRANCH") or "main"
# OFF / PULL_ONLY / PUSH_ONLY / BOTH  (default PULL_ONLY to prevent Railway auto-deploy loop)
GITHUB_SYNC_MODE = env_str("GITHUB_SYNC_MODE", default="PULL_ONLY").strip().upper()

# Files
LEARNING_FILE = "learning.json"
HISTORY_FILE = "trade_history.csv"
POSITIONS_FILE = "positions.json"
COOLDOWN_FILE = "cooldowns.json"

# Coinbase public endpoints
BASE_URL = "https://api.exchange.coinbase.com"
PRODUCTS_URL = f"{BASE_URL}/products"
TICKER_URL = f"{BASE_URL}/products/{{}}/ticker"
CANDLES_URL = f"{BASE_URL}/products/{{}}/candles"


# ============================================================
# HTTP + notify
# ============================================================

session = requests.Session()
session.headers.update({"User-Agent": "coin-sniper/elite"})

def notify(msg: str) -> None:
    print(msg, flush=True)
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        session.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
            timeout=10
        )
    except Exception:
        pass


# ============================================================
# GitHub Sync (optional) — SAFE MODE to avoid redeploy loops
# ============================================================

def gh_enabled_pull() -> bool:
    return bool(GITHUB_TOKEN and GITHUB_REPO) and GITHUB_SYNC_MODE in ("PULL_ONLY", "BOTH")

def gh_enabled_push() -> bool:
    return bool(GITHUB_TOKEN and GITHUB_REPO) and GITHUB_SYNC_MODE in ("PUSH_ONLY", "BOTH")

def gh_headers() -> Dict[str, str]:
    return {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "coin-sniper/elite"
    }

def gh_get_file(path: str) -> Tuple[Optional[bytes], Optional[str]]:
    if not gh_enabled_pull():
        return None, None
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
    try:
        r = session.get(url, headers=gh_headers(), params={"ref": GITHUB_BRANCH}, timeout=15)
        if r.status_code != 200:
            return None, None
        data = r.json()
        content_b64 = data.get("content")
        sha = data.get("sha")
        if not content_b64:
            return None, sha
        raw = base64.b64decode(content_b64.encode("utf-8"))
        return raw, sha
    except Exception:
        return None, None

def gh_put_file(path: str, content_bytes: bytes, message: str) -> None:
    if not gh_enabled_push():
        return
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
    # get sha if exists
    _, sha = (None, None)
    try:
        # only look up sha if push enabled
        url_get = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
        r = session.get(url_get, headers=gh_headers(), params={"ref": GITHUB_BRANCH}, timeout=15)
        if r.status_code == 200:
            sha = r.json().get("sha")
    except Exception:
        sha = None

    payload = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode("utf-8"),
        "branch": GITHUB_BRANCH
    }
    if sha:
        payload["sha"] = sha
    try:
        session.put(url, headers=gh_headers(), json=payload, timeout=20)
    except Exception:
        pass

def gh_pull_bootstrap() -> None:
    """
    Boot-only pull:
    - If local exists, keep local
    - If local missing, pull from GitHub (if enabled)
    """
    if not gh_enabled_pull():
        return
    for fname in (LEARNING_FILE, HISTORY_FILE, POSITIONS_FILE, COOLDOWN_FILE):
        if os.path.exists(fname):
            continue
        raw, _ = gh_get_file(fname)
        if raw:
            try:
                with open(fname, "wb") as f:
                    f.write(raw)
                notify(f"GITHUB SYNC: Pulled {fname}")
            except Exception:
                pass

_last_gh_push = 0.0
def gh_push_throttled(reason: str) -> None:
    """
    Throttled push (OFF by default unless you set GITHUB_SYNC_MODE=PUSH_ONLY or BOTH).
    """
    global _last_gh_push
    if not gh_enabled_push():
        return
    now = time.time()
    if now - _last_gh_push < 45:
        return
    _last_gh_push = now

    try:
        for fname in (LEARNING_FILE, HISTORY_FILE, POSITIONS_FILE, COOLDOWN_FILE):
            if os.path.exists(fname):
                with open(fname, "rb") as f:
                    gh_put_file(fname, f.read(), f"Update {fname} ({reason})")
        notify("GITHUB SYNC: Pushed state files")
    except Exception:
        pass


# ============================================================
# Persistent state: learning / positions / cooldowns
# ============================================================

def safe_json_load(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default

def safe_json_save(path: str, data) -> None:
    try:
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception:
        pass

def load_learning() -> Dict:
    default = {
        "start_balance": float(START_BALANCE),
        "cash": float(START_BALANCE),
        "trade_count": 0,
        "win_count": 0,
        "loss_count": 0,
        "total_profit_usd": 0.0,
    }
    data = safe_json_load(LEARNING_FILE, default)
    # patch fields
    for k, v in default.items():
        if k not in data:
            data[k] = v
    # backward compat
    if "total_profit" in data and "total_profit_usd" not in data:
        try:
            data["total_profit_usd"] = float(data.get("total_profit", 0.0))
        except Exception:
            data["total_profit_usd"] = 0.0
    # sanity clamp for cash
    try:
        data["cash"] = float(data.get("cash", START_BALANCE))
    except Exception:
        data["cash"] = float(START_BALANCE)
    return data

def load_positions() -> Dict[str, Dict]:
    data = safe_json_load(POSITIONS_FILE, {})
    if not isinstance(data, dict):
        return {}
    out: Dict[str, Dict] = {}
    for sym, pos in data.items():
        if not isinstance(pos, dict):
            continue
        required = ("entry", "qty", "size", "time", "stop", "peak")
        if all(k in pos for k in required):
            out[str(sym)] = pos
    return out

def load_cooldowns() -> Dict[str, float]:
    data = safe_json_load(COOLDOWN_FILE, {})
    if not isinstance(data, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in data.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            pass
    return out

def ensure_history() -> None:
    if os.path.exists(HISTORY_FILE):
        return
    try:
        with open(HISTORY_FILE, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rsi", "volume_ratio", "trend_strength", "momentum", "atr_pct", "profit_usd"])
    except Exception:
        pass

def append_history(row: List[float]) -> None:
    try:
        with open(HISTORY_FILE, "a", newline="") as f:
            csv.writer(f).writerow(row)
    except Exception:
        pass

def persist_all(reason: str) -> None:
    safe_json_save(LEARNING_FILE, learning)
    safe_json_save(POSITIONS_FILE, positions)
    safe_json_save(COOLDOWN_FILE, cooldowns)
    gh_push_throttled(reason)


# ============================================================
# Market data (Coinbase public)
# ============================================================

def get_symbols_auto() -> List[str]:
    try:
        r = session.get(PRODUCTS_URL, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
        out: List[str] = []
        for d in data:
            pid = d.get("id", "")
            if not pid:
                continue
            if pid.upper() in EXCLUDE:
                continue
            if d.get("quote_currency") != "USD":
                continue
            # skip stables
            if pid.endswith("-USDT") or pid.endswith("-USDC") or pid.startswith("USD-"):
                continue
            out.append(pid)
        return out[:MAX_SYMBOLS]
    except Exception:
        return []

def get_symbols() -> List[str]:
    if COINS.upper() == "AUTO":
        return get_symbols_auto()
    coins = [c.strip().upper() for c in COINS.split(",") if c.strip()]
    coins = [c for c in coins if c.upper() not in EXCLUDE]
    return coins[:MAX_SYMBOLS]

def get_price(symbol: str) -> Optional[float]:
    """
    SAFE: avoids KeyError('price') when Coinbase returns an error JSON
    """
    try:
        r = session.get(TICKER_URL.format(symbol), timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        p = data.get("price")
        if p is None:
            return None
        return float(p)
    except Exception:
        return None

def get_candles(symbol: str, granularity: int, points: int) -> Optional[List[List[float]]]:
    try:
        r = session.get(
            CANDLES_URL.format(symbol),
            params={"granularity": str(granularity)},
            timeout=12
        )
        if r.status_code != 200:
            return None
        data = r.json()
        if not isinstance(data, list) or len(data) < 25:
            return None
        data = list(reversed(data))  # oldest -> newest
        return data[-min(points, len(data)):]
    except Exception:
        return None


# ============================================================
# Indicators
# ============================================================

def ema(arr: np.ndarray, period: int) -> float:
    if len(arr) < 2:
        return float(arr[0]) if len(arr) else 0.0
    k = 2.0 / (period + 1.0)
    e = float(arr[0])
    for v in arr[1:]:
        e = float(v) * k + e * (1.0 - k)
    return float(e)

def calc_rsi(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    a = np.array(closes, dtype=float)
    d = np.diff(a)
    gains = np.where(d > 0, d, 0.0)
    losses = np.where(d < 0, -d, 0.0)
    avg_gain = float(np.mean(gains[-period:]))
    avg_loss = float(np.mean(losses[-period:]))
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))

def momentum(closes: List[float], lookback: int = 5) -> float:
    if len(closes) < lookback + 1:
        return 0.0
    prev = closes[-1 - lookback]
    if prev == 0:
        return 0.0
    return float((closes[-1] - prev) / prev)

def volume_ratio(vols: List[float], period: int = 20) -> float:
    if len(vols) < 2:
        return 1.0
    p = min(period, len(vols))
    avg = float(np.mean(vols[-p:]))
    if avg <= 0:
        return 1.0
    return float(vols[-1] / avg)

def atr_percent(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 2:
        return 0.0
    h = np.array(highs, dtype=float)
    l = np.array(lows, dtype=float)
    c = np.array(closes, dtype=float)
    prev_close = c[:-1]
    tr = np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - prev_close), np.abs(l[1:] - prev_close))
    )
    atr = float(np.mean(tr[-period:]))
    last_close = float(c[-1])
    if last_close <= 0:
        return 0.0
    return float(atr / last_close)

def trend_strength(closes: List[float]) -> float:
    if len(closes) < 60:
        return 0.0
    arr = np.array(closes, dtype=float)
    ema_fast = ema(arr[-50:], 20)
    ema_slow = ema(arr[-60:], 50)
    if ema_slow <= 0:
        return 0.0
    spread = (ema_fast - ema_slow) / ema_slow
    ema_fast_10ago = ema(arr[-60:-10], 20)
    slope = (ema_fast - ema_fast_10ago) / ema_slow
    return float(spread + slope)

def build_features(symbol: str) -> Optional[List[float]]:
    candles = get_candles(symbol, CANDLE_GRANULARITY, CANDLE_POINTS)
    if not candles:
        return None
    closes = [float(c[4]) for c in candles]
    highs  = [float(c[2]) for c in candles]
    lows   = [float(c[1]) for c in candles]
    vols   = [float(c[5]) for c in candles]

    rsi_v = calc_rsi(closes, 14)
    vol_r = volume_ratio(vols, 20)
    trend_v = trend_strength(closes)
    mom_v = momentum(closes, 5)
    atr_p = atr_percent(highs, lows, closes, ATR_PERIOD)
    return [float(rsi_v), float(vol_r), float(trend_v), float(mom_v), float(atr_p)]


# ============================================================
# Entry scoring + hard filters
# ============================================================

def passes_hard_filters(feat: List[float]) -> bool:
    rsi_v, vol_r, trend_v, mom_v, atr_p = feat
    if trend_v <= 0:
        return False
    if mom_v <= 0:
        return False
    if vol_r < 1.05:
        return False
    # avoid dead / insanely wild
    if atr_p <= 0.0008:
        return False
    if atr_p >= 0.080:
        return False
    return True

def entry_score(feat: List[float]) -> float:
    rsi_v, vol_r, trend_v, mom_v, atr_p = feat
    score = 0.0

    if trend_v > MIN_TREND_STRENGTH:
        score += 4.0
    elif trend_v > 0:
        score += 2.0

    if mom_v > 0.002:
        score += 2.0
    elif mom_v > 0:
        score += 1.0

    if vol_r >= MIN_VOLUME_RATIO:
        score += 2.5
    elif vol_r >= 1.10:
        score += 1.5

    if RSI_MIN <= rsi_v <= RSI_MAX:
        score += 1.5
    elif 50 <= rsi_v < RSI_MIN:
        score += 0.75

    if 0.002 <= atr_p <= 0.040:
        score += 0.5

    return float(score)


# ============================================================
# ML model
# ============================================================

model: Optional[RandomForestClassifier] = None
_last_train = 0.0

def train_model_if_ready() -> None:
    global model, _last_train
    now = time.time()
    if now - _last_train < ML_RETRAIN_EVERY_SEC:
        return
    _last_train = now

    if not os.path.exists(HISTORY_FILE):
        model = None
        return

    try:
        data = np.genfromtxt(HISTORY_FILE, delimiter=",", skip_header=1)
        if data is None or (hasattr(data, "size") and data.size == 0):
            model = None
            return
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)

        if len(data) < ML_MIN_TRADES_TO_ENABLE:
            model = None
            return

        X = data[:, :-1]
        y = (data[:, -1] > 0).astype(int)

        m = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            max_depth=7,
            min_samples_leaf=3
        )
        m.fit(X, y)
        model = m
        notify(f"ML MODEL ACTIVATED ({len(data)} trades)")
    except Exception:
        model = None

def ml_allows(feat: List[float]) -> Tuple[bool, float]:
    if model is None:
        return True, 0.0
    try:
        prob = float(model.predict_proba([feat])[0][1])
        return prob >= ML_MIN_PROB, prob
    except Exception:
        return True, 0.0


# ============================================================
# Portfolio / paper execution
# ============================================================

learning: Dict = {}
positions: Dict[str, Dict] = {}
cooldowns: Dict[str, float] = {}
cash: float = 0.0

def compute_equity(latest_prices: Dict[str, float]) -> float:
    eq = float(cash)
    for sym, pos in positions.items():
        p = latest_prices.get(sym)
        if p is None:
            continue
        eq += float(pos["qty"] * p)
    return float(eq)

def cash_reserve_amount() -> float:
    return float(cash * (CASH_RESERVE_PERCENT / 100.0))

def position_size_usd(score: float) -> float:
    pct = MIN_POSITION_SIZE_PERCENT + (max(0.0, min(10.0, score)) / 10.0) * (MAX_POSITION_SIZE_PERCENT - MIN_POSITION_SIZE_PERCENT)
    raw = cash * (pct / 100.0)
    size = max(MIN_TRADE_SIZE_USD, raw)
    available = max(0.0, cash - cash_reserve_amount())
    return float(min(size, available))

def in_cooldown(sym: str) -> bool:
    until = float(cooldowns.get(sym, 0.0))
    return time.time() < until

def set_cooldown(sym: str) -> None:
    cooldowns[sym] = time.time() + float(REENTRY_COOLDOWN_SEC)

def open_trade(sym: str, price: float, feat: List[float], score: float, ml_prob: float) -> None:
    global cash

    if sym in positions:
        return
    if len(positions) >= MAX_OPEN_TRADES:
        return
    if in_cooldown(sym):
        return

    size = position_size_usd(score)
    if size <= 0 or size > cash:
        return

    qty = size / price
    if qty <= 0:
        return

    stop_price = price * (1 - STOP_LOSS_PERCENT / 100.0)

    positions[sym] = {
        "entry": float(price),
        "qty": float(qty),
        "size": float(size),
        "time": float(time.time()),
        "peak": float(price),
        "stop": float(stop_price),
        "trail": None,
        "features": feat
    }

    cash -= size
    learning["cash"] = float(cash)

    persist_all("buy")

    notify(
        f"BUY {sym}\n"
        f"Price: {price:.6f}\n"
        f"Size: ${size:.2f}\n"
        f"Score: {score:.2f} | ML: {ml_prob:.2f}\n"
        f"Cash: ${cash:.2f}"
    )

def sell_trade(sym: str, price: float, reason: str, latest_prices: Dict[str, float]) -> None:
    global cash

    pos = positions.get(sym)
    if not pos:
        return

    qty = float(pos["qty"])
    size = float(pos["size"])

    proceeds = qty * price
    gross_profit_usd = proceeds - size

    # Fee estimate per side (entry + exit)
    fee_entry = size * (FEE_PERCENT_PER_SIDE / 100.0)
    fee_exit = proceeds * (FEE_PERCENT_PER_SIDE / 100.0)
    net_profit_usd = gross_profit_usd - fee_entry - fee_exit

    cash += proceeds

    learning["trade_count"] = int(learning.get("trade_count", 0)) + 1
    if net_profit_usd > 0:
        learning["win_count"] = int(learning.get("win_count", 0)) + 1
    else:
        learning["loss_count"] = int(learning.get("loss_count", 0)) + 1
    learning["total_profit_usd"] = float(learning.get("total_profit_usd", 0.0)) + float(net_profit_usd)
    learning["cash"] = float(cash)

    # Persist ML training row with ENTRY features
    f = pos.get("features") or [50.0, 1.0, 0.0, 0.0, 0.01]
    append_history([f[0], f[1], f[2], f[3], f[4], float(net_profit_usd)])

    del positions[sym]

    # cooldown prevents immediate re-buy churn
    set_cooldown(sym)

    trades = int(learning.get("trade_count", 0))
    wins = int(learning.get("win_count", 0))
    losses = int(learning.get("loss_count", 0))
    winrate = (wins / trades * 100.0) if trades > 0 else 0.0
    equity = compute_equity(latest_prices)

    persist_all("sell")

    notify(
        f"SELL {sym} ({reason})\n"
        f"P/L (net): ${net_profit_usd:.2f}\n"
        f"Cash: ${cash:.2f}\n"
        f"Equity: ${equity:.2f}\n\n"
        f"Trades: {trades}\n"
        f"Wins: {wins}\n"
        f"Losses: {losses}\n"
        f"Winrate: {winrate:.1f}%"
    )


# ============================================================
# BOOT
# ============================================================

# Pull state only if file missing (safe)
gh_pull_bootstrap()

ensure_history()

learning = load_learning()
positions = load_positions()
cooldowns = load_cooldowns()
cash = float(learning.get("cash", START_BALANCE))

# Ensure start_balance exists and is stable for NET
if "start_balance" not in learning:
    learning["start_balance"] = float(START_BALANCE)

# Save immediately so missing files are created
persist_all("boot")

symbols = get_symbols()

notify(
    f"BOT STARTED ({'PAPER' if PAPER_TRADING else 'REAL'})\n"
    f"Symbols: {len(symbols)} | MaxOpen: {MAX_OPEN_TRADES}\n"
    f"Cash: ${cash:.2f} | Loaded open: {len(positions)}\n"
    f"Stop: {STOP_LOSS_PERCENT:.2f}% | TrailStart: {TRAILING_START_PERCENT:.2f}% | "
    f"TrailDist: {TRAILING_DISTANCE_PERCENT:.2f}% | ATRx: {ATR_MULT:.2f}\n"
    f"MinTrade: ${MIN_TRADE_SIZE_USD:.2f} | Fee/side: {FEE_PERCENT_PER_SIDE:.2f}%\n"
    f"GitHubSync: {GITHUB_SYNC_MODE}"
)

last_status_time = time.time()
train_model_if_ready()


# ============================================================
# MAIN LOOP
# ============================================================

while True:
    try:
        if not symbols:
            notify("No symbols found. Check COINS=AUTO and MAX_SYMBOLS, or set COINS=BTC-USD,ETH-USD,...")
            time.sleep(30)
            symbols = get_symbols()
            continue

        now = time.time()

        # Latest prices for open positions first
        latest_prices: Dict[str, float] = {}
        for sym in list(positions.keys()):
            p = get_price(sym)
            if p is not None:
                latest_prices[sym] = p

        # Rotate scan set (reduces rate-limit patterns)
        shift = int(time.time()) % max(1, len(symbols))
        scan_batch = symbols[shift:] + symbols[:shift]
        scan_batch = scan_batch[:min(len(scan_batch), max(25, MAX_OPEN_TRADES * 4))]

        for sym in scan_batch:
            if sym in latest_prices:
                continue
            p = get_price(sym)
            if p is not None:
                latest_prices[sym] = p

        # ---- Manage open trades ----
        for sym in list(positions.keys()):
            price = latest_prices.get(sym)
            if price is None:
                continue

            pos = positions[sym]
            entry = float(pos["entry"])

            profit_pct = (price - entry) / entry if entry > 0 else 0.0
            age_min = (now - float(pos["time"])) / 60.0

            # peak update
            if price > float(pos["peak"]):
                pos["peak"] = float(price)

            # ATR-based dynamic trailing (uses entry-time ATR% for stability)
            atr_p = float((pos.get("features") or [0, 0, 0, 0, 0])[4] or 0.0)
            dynamic_trail = max(TRAILING_DISTANCE_PERCENT / 100.0, ATR_MULT * atr_p)

            # activate/update trail once in profit
            if profit_pct >= (TRAILING_START_PERCENT / 100.0):
                pos["trail"] = float(pos["peak"]) * (1.0 - dynamic_trail)

            # hard stop
            if price <= float(pos["stop"]):
                sell_trade(sym, price, "STOP", latest_prices)
                continue

            # trailing stop
            if pos.get("trail") is not None and price <= float(pos["trail"]):
                sell_trade(sym, price, "TRAIL", latest_prices)
                continue

            # stagnation exit
            if age_min >= MAX_TRADE_DURATION_MINUTES and profit_pct <= 0:
                sell_trade(sym, price, "STAGNATION", latest_prices)
                continue

        # ---- ML retrain ----
        train_model_if_ready()

        # ---- Open new trades ----
        if len(positions) < MAX_OPEN_TRADES:
            for sym in scan_batch:
                if sym in positions:
                    continue
                if len(positions) >= MAX_OPEN_TRADES:
                    break
                if in_cooldown(sym):
                    continue

                price = latest_prices.get(sym)
                if price is None:
                    continue

                feat = build_features(sym)
                if not feat:
                    continue

                if not passes_hard_filters(feat):
                    continue

                score = entry_score(feat)
                if score < MIN_ENTRY_SCORE:
                    continue

                allow, prob = ml_allows(feat)
                if not allow:
                    continue

                open_trade(sym, price, feat, score, prob)

        # ---- Status ----
        if now - last_status_time >= STATUS_INTERVAL:
            equity = compute_equity(latest_prices)
            start_base = float(learning.get("start_balance", START_BALANCE))
            net = equity - start_base

            trades = int(learning.get("trade_count", 0))
            wins = int(learning.get("win_count", 0))
            losses = int(learning.get("loss_count", 0))
            winrate = (wins / trades * 100.0) if trades > 0 else 0.0
            ml_state = "ON" if model is not None else "OFF"

            notify(
                f"STATUS\n"
                f"Cash: ${cash:.2f}\n"
                f"Equity: ${equity:.2f}\n"
                f"Net: ${net:.2f}\n"
                f"Open Trades: {len(positions)}\n"
                f"Trades: {trades}\n"
                f"Wins: {wins}\n"
                f"Losses: {losses}\n"
                f"Winrate: {winrate:.1f}%\n"
                f"ML: {ml_state}"
            )

            persist_all("status")
            last_status_time = now

        time.sleep(SCAN_INTERVAL)

    except Exception as e:
        tb = traceback.format_exc(limit=6)
        notify(f"ERROR: {str(e)}\n{tb}")
        time.sleep(10)
