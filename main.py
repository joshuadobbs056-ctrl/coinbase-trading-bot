# coin_sniper_bot.py
# Elite Paper Trading Bot (Coinbase public data) + Persistent state + OPTIONAL GitHub data sync + Telegram alerts
#
# ✅ Fixes included:
# 1) Prevents “Railway deploy loop” by DEFAULT: will NOT push to a repo that looks like your *code* repo
#    unless you explicitly allow it (GITHUB_ALLOW_SELF_DEPLOY=1).
# 2) Boot pull is SAFE: only pulls missing local files (does NOT overwrite local state).
# 3) Persists OPEN positions across restarts (positions.json) so balances/equity don’t “reset weird”.
# 4) Equity calc won’t show $0 if Coinbase price fetch fails (falls back to entry price).
# 5) Never opens $0 trades (hard guard).
#
# -------------------------
# REQUIRED ENV VARS (for Telegram alerts)
#   TELEGRAM_TOKEN=...
#   TELEGRAM_CHAT_ID=...
#
# OPTIONAL GitHub STATE SYNC (recommended to use a SEPARATE data repo)
#   GITHUB_TOKEN=ghp_...
#   GITHUB_REPO=yourusername/coin-sniper-data     <-- IMPORTANT: make this a DATA repo (not the code repo)
#   GITHUB_BRANCH=main
#
# If you *insist* on pushing state into the same repo Railway deploys from:
#   GITHUB_ALLOW_SELF_DEPLOY=1    (NOT recommended; can cause redeploy loops)
#
# OPTIONAL: completely disable GitHub sync
#   GITHUB_SYNC=0
#
# -------------------------
# Trading settings are env-driven (Railway Variables).
# This file is PAPER by default unless you later add real-execution code.

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
# Helpers: env var parsing + backward-compatible aliases
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
# CONFIG (Railway env vars)
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

# Position sizing (percent of cash)
MIN_POSITION_SIZE_PERCENT = env_float("MIN_POSITION_SIZE_PERCENT", "MIN_POSITION_S", "MIN_POSITION_U", default=1.0)
MAX_POSITION_SIZE_PERCENT = env_float("MAX_POSITION_SIZE_PERCENT", "MAX_POSITION_S", default=3.0)

# Stops
STOP_LOSS_PERCENT = env_float(
    "STOP_LOSS_PERCENT",
    "STOP_LOSS_PERC",
    "STOP_LOSS_PERC_",
    "STOP_LOSS_PERCEN",
    default=2.8
)
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
ML_MIN_TRADES_TO_ENABLE = env_int("ML_MIN_TRADES_TO_ENABLE", "ML_MIN_TRADES_", "ML_MIN_TRADES", default=50)
ML_MIN_PROB = env_float("ML_MIN_PROB", default=0.58)
ML_RETRAIN_EVERY_SEC = env_int("ML_RETRAIN_EVERY_SEC", "ML_RETRAIN_EVE", default=600)

# Fee estimate
FEE_PERCENT_PER_SIDE = env_float("FEE_PERCENT_PER_SIDE", "FEE_PERCENT_PE", default=0.20)

# Mode
PAPER_TRADING = env_bool("PAPER_TRADING", default=True)

# Telegram
TELEGRAM_TOKEN = _env("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = _env("TELEGRAM_CHAT_ID") or _env("TELEGRAM_CHAT_")

# GitHub state sync (optional)
GITHUB_SYNC = env_bool("GITHUB_SYNC", default=True)  # set to 0 to fully disable
GITHUB_ALLOW_SELF_DEPLOY = env_bool("GITHUB_ALLOW_SELF_DEPLOY", default=False)
GITHUB_TOKEN = _env("GITHUB_TOKEN")
GITHUB_REPO = _env("GITHUB_REPO")  # should be DATA repo like: joshuadobbs056-ctrl/coin-sniper-data
GITHUB_BRANCH = _env("GITHUB_BRANCH") or "main"

# Files
LEARNING_FILE = "learning.json"
HISTORY_FILE = "trade_history.csv"
POSITIONS_FILE = "positions.json"

# Coinbase endpoints (public)
BASE_URL = "https://api.exchange.coinbase.com"
PRODUCTS_URL = f"{BASE_URL}/products"
TICKER_URL = f"{BASE_URL}/products/{{}}/ticker"
CANDLES_URL = f"{BASE_URL}/products/{{}}/candles"


# ============================================================
# HTTP session + notify
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
# GitHub Sync (optional) — SAFE by default to prevent redeploy loops
# ============================================================

def _repo_looks_like_code_repo(repo: str) -> bool:
    """
    Heuristic: if your repo name looks like the code repo, pushing state there can trigger Railway redeploys.
    Adjust/extend if you want.
    """
    r = (repo or "").lower()
    # common names you showed
    if "coinbase-trading-bot" in r:
        return True
    if "coin-sniper-bot" in r:
        return True
    return False

def github_enabled() -> bool:
    if not GITHUB_SYNC:
        return False
    if not (GITHUB_TOKEN and GITHUB_REPO):
        return False

    # Guard: do NOT push to a code repo unless explicitly allowed.
    if _repo_looks_like_code_repo(GITHUB_REPO) and not GITHUB_ALLOW_SELF_DEPLOY:
        return False

    return True

def gh_headers() -> Dict[str, str]:
    return {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "coin-sniper/elite"
    }

def gh_get_file(path: str) -> Tuple[Optional[bytes], Optional[str]]:
    if not github_enabled():
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
    if not github_enabled():
        return
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
    _, sha = gh_get_file(path)
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
      - If local file exists, KEEP IT.
      - If local file missing, pull from GitHub if present.
    """
    if not github_enabled():
        # If disabled because it looks like code repo, explain once.
        if GITHUB_SYNC and GITHUB_TOKEN and GITHUB_REPO and _repo_looks_like_code_repo(GITHUB_REPO) and not GITHUB_ALLOW_SELF_DEPLOY:
            notify(
                "GITHUB SYNC: Disabled push to avoid Railway redeploy loop.\n"
                "Use a separate data repo for GITHUB_REPO (recommended), or set GITHUB_ALLOW_SELF_DEPLOY=1."
            )
        return

    for fname in (LEARNING_FILE, HISTORY_FILE, POSITIONS_FILE):
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
    Push state files, throttled so we don't spam GitHub.
    """
    global _last_gh_push
    if not github_enabled():
        return

    now = time.time()
    if now - _last_gh_push < 60:  # <= reduce spam (was 30)
        return
    _last_gh_push = now

    try:
        for fname in (LEARNING_FILE, HISTORY_FILE, POSITIONS_FILE):
            if os.path.exists(fname):
                with open(fname, "rb") as f:
                    gh_put_file(fname, f.read(), f"Update {fname} ({reason})")
        notify("GITHUB SYNC: Pushed learning/history/positions")
    except Exception:
        pass


# ============================================================
# Persistent learning + positions
# ============================================================

def load_learning() -> Dict:
    default = {
        "start_balance": float(START_BALANCE),
        "cash": float(START_BALANCE),
        "trade_count": 0,
        "win_count": 0,
        "loss_count": 0,
        "total_profit_usd": 0.0
    }
    if not os.path.exists(LEARNING_FILE):
        return default
    try:
        with open(LEARNING_FILE, "r") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return default

        for k, v in default.items():
            if k not in data:
                data[k] = v

        # backward compat
        if "total_profit" in data and "total_profit_usd" not in data:
            data["total_profit_usd"] = float(data.get("total_profit", 0.0))

        # sanitize numeric fields
        data["cash"] = float(data.get("cash", default["cash"]))
        data["start_balance"] = float(data.get("start_balance", default["start_balance"]))
        data["trade_count"] = int(data.get("trade_count", 0))
        data["win_count"] = int(data.get("win_count", 0))
        data["loss_count"] = int(data.get("loss_count", 0))
        data["total_profit_usd"] = float(data.get("total_profit_usd", 0.0))

        return data
    except Exception:
        return default

def save_learning(state: Dict) -> None:
    try:
        with open(LEARNING_FILE, "w") as f:
            json.dump(state, f)
    except Exception:
        pass

def load_positions() -> Dict[str, Dict]:
    if not os.path.exists(POSITIONS_FILE):
        return {}
    try:
        with open(POSITIONS_FILE, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        out: Dict[str, Dict] = {}
        for sym, pos in data.items():
            if not isinstance(pos, dict):
                continue
            required = ("entry", "qty", "size", "time", "stop", "peak")
            if all(k in pos for k in required):
                # sanitize
                pos["entry"] = float(pos["entry"])
                pos["qty"] = float(pos["qty"])
                pos["size"] = float(pos["size"])
                pos["time"] = float(pos["time"])
                pos["stop"] = float(pos["stop"])
                pos["peak"] = float(pos["peak"])
                if "trail" in pos and pos["trail"] is not None:
                    pos["trail"] = float(pos["trail"])
                out[str(sym)] = pos
        return out
    except Exception:
        return {}

def save_positions(positions: Dict[str, Dict]) -> None:
    try:
        with open(POSITIONS_FILE, "w") as f:
            json.dump(positions, f)
    except Exception:
        pass


# ============================================================
# Trade history (for ML)
# ============================================================

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


# ============================================================
# Market data
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
            # skip obvious stables
            if pid.endswith("-USDT") or pid.endswith("-USDC") or pid.startswith("USD-"):
                continue
            out.append(pid)
        return out[:MAX_SYMBOLS]
    except Exception:
        return []

def get_symbols() -> List[str]:
    if COINS.upper() == "AUTO":
        return get_symbols_auto()
    coins = [c.strip() for c in COINS.split(",") if c.strip()]
    coins = [c for c in coins if c.upper() not in EXCLUDE]
    return coins[:MAX_SYMBOLS]

def get_price(symbol: str) -> Optional[float]:
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
    # response: [ time, low, high, open, close, volume ]
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
    highs = [float(c[2]) for c in candles]
    lows  = [float(c[1]) for c in candles]
    vols  = [float(c[5]) for c in candles]

    rsi_v = calc_rsi(closes, 14)
    vol_r = volume_ratio(vols, 20)
    trend_v = trend_strength(closes)
    mom_v = momentum(closes, 5)
    atr_p = atr_percent(highs, lows, closes, ATR_PERIOD)

    return [float(rsi_v), float(vol_r), float(trend_v), float(mom_v), float(atr_p)]


# ============================================================
# Entry scoring + filters
# ============================================================

def passes_hard_filters(feat: List[float]) -> bool:
    rsi_v, vol_r, trend_v, mom_v, atr_p = feat
    if trend_v <= 0:
        return False
    if mom_v <= 0:
        return False
    if vol_r < 1.05:
        return False
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
cash: float = 0.0
positions: Dict[str, Dict] = {}

def compute_equity(latest_prices: Dict[str, float]) -> float:
    """
    If Coinbase price fetch fails, fallback to entry price so equity doesn’t show $0.00.
    """
    eq = float(cash)
    for sym, pos in positions.items():
        p = latest_prices.get(sym)
        if p is None:
            p = float(pos.get("entry", 0.0))
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

def persist_all(reason: str) -> None:
    save_learning(learning)
    save_positions(positions)
    gh_push_throttled(reason)

def open_trade(sym: str, price: float, feat: List[float], score: float, ml_prob: float) -> None:
    global cash

    if sym in positions:
        return
    if len(positions) >= MAX_OPEN_TRADES:
        return

    size = position_size_usd(score)

    # HARD guard: never open $0 trades
    if size is None or size <= 0:
        return
    if size > cash:
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

    entry = float(pos["entry"])
    qty = float(pos["qty"])
    size = float(pos["size"])

    proceeds = qty * price
    gross_profit_usd = proceeds - size

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

    # ML training row uses ENTRY features
    f = pos.get("features") or [50.0, 1.0, 0.0, 0.0, 0.01]
    append_history([float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[4]), float(net_profit_usd)])

    del positions[sym]

    trades = int(learning.get("trade_count", 0))
    wins = int(learning.get("win_count", 0))
    losses = int(learning.get("loss_count", 0))
    winrate = (wins / trades * 100.0) if trades > 0 else 0.0
    equity = compute_equity(latest_prices)

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

    persist_all("sell")


# ============================================================
# Boot sequence
# ============================================================

gh_pull_bootstrap()
ensure_history()

learning = load_learning()
positions = load_positions()

cash = float(learning.get("cash", START_BALANCE))

# Ensure start_balance exists for NET calc
if "start_balance" not in learning:
    learning["start_balance"] = float(START_BALANCE)
    save_learning(learning)

symbols = get_symbols()

notify(
    f"BOT STARTED ({'PAPER' if PAPER_TRADING else 'REAL'})\n"
    f"Symbols: {len(symbols)} | MaxOpen: {MAX_OPEN_TRADES}\n"
    f"Loaded cash: ${cash:.2f} | Open positions loaded: {len(positions)}\n"
    f"Stop: {STOP_LOSS_PERCENT:.2f}% | TrailStart: {TRAILING_START_PERCENT:.2f}% | "
    f"TrailDist: {TRAILING_DISTANCE_PERCENT:.2f}% | ATRx: {ATR_MULT:.2f}\n"
    f"MinTrade: ${MIN_TRADE_SIZE_USD:.2f} | Fee/side: {FEE_PERCENT_PER_SIDE:.2f}%"
)

last_status_time = time.time()
train_model_if_ready()

# DO NOT auto-push on boot (this can trigger loops in some setups).
# We only push when actual changes happen (buy/sell/status throttled).


# ============================================================
# Main loop
# ============================================================

while True:
    try:
        if not symbols:
            notify("No symbols found. Check COINS=AUTO and MAX_SYMBOLS, or set COINS=BTC-USD,ETH-USD,...")
            time.sleep(30)
            symbols = get_symbols()
            continue

        now = time.time()

        # Pull latest prices for open positions first
        latest_prices: Dict[str, float] = {}
        for sym in list(positions.keys()):
            p = get_price(sym)
            if p is not None:
                latest_prices[sym] = p

        # Rotate scan set to reduce rate limiting
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
                # can't manage without a price; skip this loop
                continue

            pos = positions[sym]
            entry = float(pos["entry"])

            profit_pct = (price - entry) / entry
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

        # Persist positions periodically so restarts don’t mess your balances
        save_positions(positions)

        # ---- ML retrain periodically ----
        train_model_if_ready()

        # ---- Open new trades ----
        if len(positions) < MAX_OPEN_TRADES:
            for sym in scan_batch:
                if sym in positions:
                    continue
                if len(positions) >= MAX_OPEN_TRADES:
                    break

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
                f"Open: {len(positions)}\n"
                f"Trades: {trades} | Wins: {wins} | Losses: {losses} | WR: {winrate:.1f}%\n"
                f"ML: {ml_state}"
            )

            gh_push_throttled("status")
            last_status_time = now

        time.sleep(SCAN_INTERVAL)

    except Exception as e:
        tb = traceback.format_exc(limit=8)
        notify(f"ERROR: {str(e)}\n{tb}")
        time.sleep(10)
