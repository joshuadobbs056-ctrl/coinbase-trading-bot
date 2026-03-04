# Coin Sniper — Savage ELITE (PAPER) — EXPLOSIVE OPTIMIZED (FULL RUNNING FILE)
# ✅ RVOL spike detection (primary trigger)
# ✅ Momentum acceleration / velocity
# ✅ Extension filter (avoid god-candle tops)
# ✅ ATR-based adaptive trailing distance
# ✅ BTC market guard (blocks NEW buys during sharp drops)
# ✅ Paper trading: ledger CSV + stats (wins/losses/win%) + equity/cash/open trades
# ✅ ML training + auto-activation after enough trades + Telegram notification
# ✅ Optional GitHub persistence (save/load state + ML data)
#
# NOTE: PAPER trading only. No real orders placed.

import os, time, json, csv, math, traceback, base64, hashlib
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

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
# CONFIG
# =========================
START_BALANCE = float(os.getenv("START_BALANCE", "1000"))

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "6"))      # seconds
STATUS_INTERVAL = int(os.getenv("STATUS_INTERVAL", "60")) # seconds
ML_INTERVAL = int(os.getenv("ML_INTERVAL", "300"))        # seconds

MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", "12"))
MAX_NEW_BUYS_PER_SCAN = int(os.getenv("MAX_NEW_BUYS_PER_SCAN", "2"))
MIN_TRADE_SIZE = float(os.getenv("MIN_TRADE_SIZE", "35"))

STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "4.0"))                 # hard stop
TRAILING_START_PERCENT = float(os.getenv("TRAILING_START_PERCENT", "0.8"))       # start trailing after profit >= this %
TRAILING_DISTANCE_PERCENT = float(os.getenv("TRAILING_DISTANCE_PERCENT", "1.0")) # base trail distance (%)

ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_MULT = float(os.getenv("ATR_MULT", "2.0"))                                   # trail distance = max(base, ATR%*ATR_MULT)

ENTRY_SCORE_MIN = int(os.getenv("ENTRY_SCORE_MIN", "7"))
MIN_VOLUME_RATIO = float(os.getenv("MIN_VOLUME_RATIO", "2.0"))                   # RVOL gate (>=)
EXTENSION_MAX = float(os.getenv("EXTENSION_MAX", "0.06"))                        # ext (price over EMA20) max before penalty

COOLDOWN_SECONDS_AFTER_SELL = int(os.getenv("COOLDOWN_SECONDS_AFTER_SELL", "900")) # 15 min default
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "80"))

# If COINS is set, only trade those symbols (comma-separated, e.g. "ETH-USD,SOL-USD")
COINS = os.getenv("COINS", "").strip()
EXCLUDE = set([s.strip() for s in os.getenv("EXCLUDE", "").split(",") if s.strip()])

# Candle settings (Coinbase public)
CANDLE_GRANULARITY = int(os.getenv("CANDLE_GRANULARITY", "60"))  # seconds (60 = 1m)
CANDLE_POINTS = int(os.getenv("CANDLE_POINTS", "200"))

# Market Guard
BTC_GUARD_ENABLED = os.getenv("BTC_GUARD_ENABLED", "1") == "1"
BTC_GUARD_DROP_PCT = float(os.getenv("BTC_GUARD_DROP_PCT", "1.0")) # % drop over window
BTC_GUARD_WINDOW_MIN = int(os.getenv("BTC_GUARD_WINDOW_MIN", "15")) # minutes
BTC_SYMBOL = os.getenv("BTC_SYMBOL", "BTC-USD")

# ML gating
ML_ENABLED = os.getenv("ML_ENABLED", "1") == "1"
ML_MIN_TRADES = int(os.getenv("ML_MIN_TRADES", "25"))
ML_MIN_PROB = float(os.getenv("ML_MIN_PROB", "0.55"))              # require win probability >= this
ML_FEATURE_VERSION = 1

# Persistence
STATE_FILE = os.getenv("STATE_FILE", "coin_sniper_state.json")
ML_FILE = os.getenv("ML_FILE", "coin_sniper_ml.json")
LEDGER_FILE = os.getenv("LEDGER_FILE", "coin_sniper_ledger.csv")

# GitHub persistence (optional)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")          # "owner/repo"
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")
GITHUB_STATE_PATH = os.getenv("GITHUB_STATE_PATH", STATE_FILE)
GITHUB_ML_PATH = os.getenv("GITHUB_ML_PATH", ML_FILE)

# Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# HTTP
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "12"))

# =========================
# UTIL: NOTIFY
# =========================
def notify(msg: str):
    print(msg, flush=True)
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
# COINBASE PUBLIC API HELPERS
# =========================
COINBASE_API = "https://api.exchange.coinbase.com"
HEADERS = {
    "User-Agent": "coin-sniper-paper/1.0",
    "Accept": "application/json",
}

def _http_get(url: str, params: Optional[dict] = None) -> Any:
    r = requests.get(url, params=params, headers=HEADERS, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()

def list_usd_products() -> List[str]:
    # Uses Coinbase Exchange public products. Filters for -USD spot products.
    data = _http_get(f"{COINBASE_API}/products")
    syms = []
    for p in data:
        try:
            pid = p.get("id", "")
            status = p.get("status", "")
            quote = p.get("quote_currency", "")
            if status == "online" and quote == "USD" and pid.endswith("-USD"):
                syms.append(pid)
        except Exception:
            continue
    # Deterministic order
    syms = sorted(set(syms))
    return syms[:MAX_SYMBOLS]

def get_candles(product_id: str, granularity: int, limit_points: int) -> List[list]:
    # Coinbase returns newest-first: [time, low, high, open, close, volume]
    # We'll keep as provided and reverse later where needed.
    # Note: API limit often ~300 points; we request only what we need.
    params = {"granularity": granularity}
    data = _http_get(f"{COINBASE_API}/products/{product_id}/candles", params=params)
    if not isinstance(data, list) or len(data) == 0:
        return []
    # Ensure we only keep recent points
    return data[:limit_points]

def get_last_price_from_candles(candles: List[list]) -> Optional[float]:
    # candles: newest-first [t, low, high, open, close, vol]
    try:
        return float(candles[0][4])
    except Exception:
        return None

# =========================
# INDICATORS
# =========================
def _ema(values: np.ndarray, period: int) -> np.ndarray:
    if len(values) == 0:
        return np.array([])
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out

def macd_pack(closes: List[float]) -> Optional[Tuple[float, float, float]]:
    if len(closes) < 35:
        return None
    arr = np.array(closes, dtype=float)
    ef = _ema(arr, 12)
    es = _ema(arr, 26)
    macd = ef - es
    sig = _ema(macd, 9)
    return float(macd[-1]), float(sig[-1]), float(macd[-1] - sig[-1])

def ema_cross(closes: List[float]) -> Optional[Tuple[float, float]]:
    if len(closes) < 25:
        return None
    arr = np.array(closes, dtype=float)
    fast = _ema(arr, 9)
    slow = _ema(arr, 21)
    return float(fast[-1] - slow[-1]), float(fast[-2] - slow[-2])

def _atr_percent(candles: List[list]) -> float:
    """ATR% using last ~30 candles, returned as percent of last close."""
    try:
        # normalize to oldest->newest for ATR calc
        recent = list(reversed(candles[:max(ATR_PERIOD + 2, 30)]))
        highs = [float(c[2]) for c in recent]
        lows = [float(c[1]) for c in recent]
        closes = [float(c[4]) for c in recent]
        if len(closes) < ATR_PERIOD + 2:
            return 1.5
        trs = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs.append(tr)
        atr = float(np.mean(trs[-ATR_PERIOD:]))
        last_close = float(closes[-1])
        if last_close <= 0:
            return 1.5
        return (atr / last_close) * 100.0
    except Exception:
        return 1.5

def score_symbol(sym: str, candles: List[list]) -> Tuple[Optional[int], str, Dict[str, float]]:
    """
    Returns: (score 1-10 or None, reason, extra)
    extra contains rvol, accel, ext, atrp
    """
    if not candles:
        return None, "no_candles", {}

    recent = list(reversed(candles[:CANDLE_POINTS]))  # oldest->newest
    closes = [float(c[4]) for c in recent]
    vols = [float(c[5]) for c in recent]

    if len(closes) < 60 or len(vols) < 60:
        return None, "short", {}

    p0 = closes[-1]

    # RVOL (last volume / avg of prior 20)
    avg_vol = float(np.mean(vols[-21:-1])) if len(vols) >= 22 else float(np.mean(vols[:-1]))
    rvol = (vols[-1] / avg_vol) if avg_vol > 0 else 1.0

    # Acceleration: delta between last 5-candle return and previous 5-candle return
    ret_5 = (closes[-1] - closes[-6]) / closes[-6] if closes[-6] > 0 else 0.0
    ret_prev_5 = (closes[-6] - closes[-11]) / closes[-11] if closes[-11] > 0 else 0.0
    accel = ret_5 - ret_prev_5

    # Extension filter: price above EMA20
    ema20 = float(_ema(np.array(closes, dtype=float), 20)[-1])
    ext = ((p0 - ema20) / ema20) if ema20 > 0 else 0.0

    # ATR%
    atrp = _atr_percent(candles)

    # Score
    score = 4

    # RVOL engine
    if rvol >= 3.0:
        score += 4
    elif rvol >= MIN_VOLUME_RATIO:
        score += 2
    elif rvol < 1.0:
        score -= 3

    # Acceleration boost
    if accel > 0:
        score += 2

    # Over-extension penalty
    if ext > EXTENSION_MAX:
        score -= 5

    # Confluence (light)
    mp = macd_pack(closes)
    ep = ema_cross(closes)
    if mp and mp[2] > 0:
        score += 1
    if ep and ep[0] > 0:
        score += 1

    score = int(max(1, min(10, round(score))))
    reason = f"RVOL={rvol:.2f} Accel={accel:.4f} Ext={ext:.3f} ATR%={atrp:.2f}"
    extra = {"rvol": float(rvol), "accel": float(accel), "ext": float(ext), "atrp": float(atrp)}
    return score, reason, extra

# =========================
# DATA MODELS
# =========================
@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    entry_time: float
    high_water: float
    stop_price: float
    trailing_active: bool
    trail_dist_pct: float
    last_reason: str
    last_score: int

# =========================
# PERSISTENCE (LOCAL)
# =========================
def _safe_read_json(path: str, default: Any):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def _safe_write_json(path: str, obj: Any):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

# =========================
# OPTIONAL GITHUB PERSISTENCE
# Uses GitHub Contents API
# =========================
def _gh_headers():
    return {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "coin-sniper-paper/1.0",
    }

def github_load(path: str) -> Optional[dict]:
    if not (GITHUB_TOKEN and GITHUB_REPO and path):
        return None
    try:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
        r = requests.get(url, headers=_gh_headers(), params={"ref": GITHUB_BRANCH}, timeout=HTTP_TIMEOUT)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        j = r.json()
        content_b64 = j.get("content", "")
        if not content_b64:
            return None
        raw = base64.b64decode(content_b64).decode("utf-8", errors="ignore")
        return {"sha": j.get("sha"), "data": json.loads(raw)}
    except Exception:
        return None

def github_save(path: str, data: dict):
    if not (GITHUB_TOKEN and GITHUB_REPO and path):
        return
    try:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
        existing = github_load(path)
        payload = {
            "message": f"coin-sniper save {path}",
            "content": base64.b64encode(json.dumps(data, indent=2, sort_keys=True).encode("utf-8")).decode("utf-8"),
            "branch": GITHUB_BRANCH,
        }
        if existing and existing.get("sha"):
            payload["sha"] = existing["sha"]
        r = requests.put(url, headers=_gh_headers(), json=payload, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
    except Exception:
        pass

# =========================
# LEDGER (CSV)
# =========================
def ensure_ledger_header():
    if os.path.exists(LEDGER_FILE):
        return
    with open(LEDGER_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "time", "symbol", "side", "qty", "price",
            "pnl_usd", "pnl_pct",
            "cash_after", "equity_after",
            "score", "reason"
        ])

def append_ledger_row(row: list):
    with open(LEDGER_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

# =========================
# ML: FEATURES + TRAINING + GATING
# =========================
def make_features(extra: Dict[str, float], score: int) -> List[float]:
    # Keep stable ordering and version
    return [
        float(score),
        float(extra.get("rvol", 1.0)),
        float(extra.get("accel", 0.0)),
        float(extra.get("ext", 0.0)),
        float(extra.get("atrp", 1.5)),
        float(MIN_VOLUME_RATIO),
        float(EXTENSION_MAX),
        float(TRAILING_START_PERCENT),
    ]

def load_ml_store() -> dict:
    return _safe_read_json(ML_FILE, {"version": ML_FEATURE_VERSION, "rows": [], "ml_active": False})

def save_ml_store(store: dict):
    _safe_write_json(ML_FILE, store)
    github_save(GITHUB_ML_PATH, store)

def train_model(store: dict) -> Optional[RandomForestClassifier]:
    rows = store.get("rows", [])
    if not rows or len(rows) < max(10, ML_MIN_TRADES):
        return None
    try:
        X = np.array([r["x"] for r in rows], dtype=float)
        y = np.array([r["y"] for r in rows], dtype=int)
        # Basic RF, conservative
        clf = RandomForestClassifier(
            n_estimators=250,
            max_depth=6,
            random_state=42,
            class_weight="balanced_subsample",
        )
        clf.fit(X, y)
        return clf
    except Exception:
        return None

def ml_should_allow_buy(clf: Optional[RandomForestClassifier], store: dict, x: List[float]) -> Tuple[bool, float]:
    if not ML_ENABLED:
        return True, 0.0
    # Activate ML only after ML_MIN_TRADES
    rows = store.get("rows", [])
    if len(rows) < ML_MIN_TRADES or clf is None:
        return True, 0.0
    try:
        proba = float(clf.predict_proba(np.array([x], dtype=float))[0][1])
        return proba >= ML_MIN_PROB, proba
    except Exception:
        return True, 0.0

# =========================
# BOT STATE
# =========================
def default_state() -> dict:
    return {
        "cash": START_BALANCE,
        "positions": {},  # symbol -> Position dict
        "wins": 0,
        "losses": 0,
        "realized_pnl": 0.0,
        "last_status_ts": 0,
        "last_ml_ts": 0,
        "cooldowns": {},  # symbol -> timestamp until
    }

def load_state() -> dict:
    # Prefer GitHub if configured; fallback local
    gh = github_load(GITHUB_STATE_PATH)
    if gh and isinstance(gh.get("data"), dict):
        return gh["data"]
    return _safe_read_json(STATE_FILE, default_state())

def save_state(state: dict):
    _safe_write_json(STATE_FILE, state)
    github_save(GITHUB_STATE_PATH, state)

def positions_from_state(state: dict) -> Dict[str, Position]:
    out = {}
    for sym, pd in (state.get("positions") or {}).items():
        try:
            out[sym] = Position(**pd)
        except Exception:
            continue
    return out

def positions_to_state(pos: Dict[str, Position]) -> Dict[str, dict]:
    return {k: asdict(v) for k, v in pos.items()}

def equity(cash: float, positions: Dict[str, Position], last_prices: Dict[str, float]) -> float:
    eq = float(cash)
    for sym, p in positions.items():
        px = last_prices.get(sym)
        if px is None:
            px = p.entry_price
        eq += float(p.qty) * float(px)
    return eq

# =========================
# MARKET GUARD
# =========================
def btc_guard_ok() -> Tuple[bool, str]:
    if not BTC_GUARD_ENABLED:
        return True, "guard_off"
    try:
        points = max(30, int((BTC_GUARD_WINDOW_MIN * 60) / CANDLE_GRANULARITY) + 5)
        candles = get_candles(BTC_SYMBOL, CANDLE_GRANULARITY, min(points, 250))
        if not candles or len(candles) < 10:
            return True, "guard_no_data"
        # newest-first -> build closes oldest->newest
        recent = list(reversed(candles))
        closes = [float(c[4]) for c in recent]
        if len(closes) < 5:
            return True, "guard_short"
        window = int((BTC_GUARD_WINDOW_MIN * 60) / CANDLE_GRANULARITY)
        window = max(2, min(window, len(closes) - 1))
        now = closes[-1]
        past = closes[-1 - window]
        if past <= 0:
            return True, "guard_past0"
        drop_pct = ((now - past) / past) * 100.0
        if drop_pct <= -abs(BTC_GUARD_DROP_PCT):
            return False, f"BTC_GUARD drop={drop_pct:.2f}%/{BTC_GUARD_WINDOW_MIN}m"
        return True, f"BTC_GUARD ok drop={drop_pct:.2f}%/{BTC_GUARD_WINDOW_MIN}m"
    except Exception:
        return True, "guard_err"

# =========================
# BUY/SELL (PAPER)
# =========================
def can_buy_symbol(state: dict, sym: str) -> bool:
    # cooldown prevents instant re-buy after a sell
    cd = (state.get("cooldowns") or {}).get(sym)
    if cd and time.time() < float(cd):
        return False
    return True

def open_position(state: dict, positions: Dict[str, Position], sym: str, price: float, score: int, reason: str, extra: Dict[str, float]):
    cash = float(state["cash"])
    if cash < MIN_TRADE_SIZE:
        return

    # fixed trade size: MIN_TRADE_SIZE (simple & predictable)
    trade_usd = min(MIN_TRADE_SIZE, cash)
    qty = trade_usd / price

    atrp = float(extra.get("atrp", 1.5))
    # Adaptive trailing distance
    trail_dist = max(float(TRAILING_DISTANCE_PERCENT), atrp * float(ATR_MULT))
    trail_dist = max(0.4, min(8.0, trail_dist))

    # Hard stop
    stop_price = price * (1.0 - STOP_LOSS_PERCENT / 100.0)

    positions[sym] = Position(
        symbol=sym,
        qty=float(qty),
        entry_price=float(price),
        entry_time=float(time.time()),
        high_water=float(price),
        stop_price=float(stop_price),
        trailing_active=False,
        trail_dist_pct=float(trail_dist),
        last_reason=reason,
        last_score=int(score),
    )

    state["cash"] = float(cash - trade_usd)

    ensure_ledger_header()
    # pnl fields are 0 at buy
    last_prices = {sym: price}
    eq = equity(float(state["cash"]), positions, last_prices)
    append_ledger_row([
        int(time.time()), sym, "BUY", f"{qty:.8f}", f"{price:.8f}",
        "0", "0",
        f"{state['cash']:.2f}", f"{eq:.2f}",
        score, reason
    ])

    notify(f"🟢 BUY {sym} ${trade_usd:.2f} @ {price:.6f} | score={score} | {reason} | trail={trail_dist:.2f}% stop={STOP_LOSS_PERCENT:.2f}%")

def close_position(state: dict, positions: Dict[str, Position], sym: str, price: float, reason: str):
    p = positions.get(sym)
    if not p:
        return
    cash = float(state["cash"])
    proceeds = float(p.qty) * float(price)
    cost = float(p.qty) * float(p.entry_price)
    pnl = proceeds - cost
    pnl_pct = (pnl / cost) * 100.0 if cost > 0 else 0.0

    state["cash"] = float(cash + proceeds)
    state["realized_pnl"] = float(state.get("realized_pnl", 0.0) + pnl)

    if pnl >= 0:
        state["wins"] = int(state.get("wins", 0) + 1)
    else:
        state["losses"] = int(state.get("losses", 0) + 1)

    # cooldown so it doesn't rebuy immediately
    cds = state.get("cooldowns") or {}
    cds[sym] = float(time.time() + COOLDOWN_SECONDS_AFTER_SELL)
    state["cooldowns"] = cds

    ensure_ledger_header()
    last_prices = {sym: price}
    eq = equity(float(state["cash"]), {k:v for k,v in positions.items() if k != sym}, last_prices)
    append_ledger_row([
        int(time.time()), sym, "SELL", f"{p.qty:.8f}", f"{price:.8f}",
        f"{pnl:.6f}", f"{pnl_pct:.3f}",
        f"{state['cash']:.2f}", f"{eq:.2f}",
        p.last_score, reason
    ])

    del positions[sym]
    notify(f"🔴 SELL {sym} @ {price:.6f} | PnL=${pnl:.2f} ({pnl_pct:.2f}%) | {reason}")

    # ML training row on trade close
    if ML_ENABLED:
        store = load_ml_store()
        y = 1 if pnl > 0 else 0
        x = make_features(
            extra={"rvol": 0, "accel": 0, "ext": 0, "atrp": p.trail_dist_pct / max(ATR_MULT, 1e-9)},
            score=int(p.last_score),
        )
        # We can't reconstruct exact entry extras unless you store them; so we store a stable approximation.
        # If you want exact entry features, we can store them directly on Position too.
        store["rows"].append({"x": x, "y": y, "t": int(time.time())})
        # Cap store size
        if len(store["rows"]) > 2000:
            store["rows"] = store["rows"][-2000:]
        save_ml_store(store)

# =========================
# POSITION MANAGEMENT
# =========================
def manage_positions(state: dict, positions: Dict[str, Position], last_prices: Dict[str, float]):
    for sym, p in list(positions.items()):
        px = last_prices.get(sym)
        if px is None:
            continue

        # Update high water
        if px > p.high_water:
            p.high_water = float(px)

        # Hard stop loss
        hard_stop = p.entry_price * (1.0 - STOP_LOSS_PERCENT / 100.0)
        if px <= hard_stop:
            close_position(state, positions, sym, px, f"STOP_LOSS {STOP_LOSS_PERCENT:.2f}%")
            continue

        # Activate trailing after minimal profit
        profit_pct = ((px - p.entry_price) / p.entry_price) * 100.0 if p.entry_price > 0 else 0.0
        if (not p.trailing_active) and profit_pct >= TRAILING_START_PERCENT:
            p.trailing_active = True

        # Trailing logic if active:
        if p.trailing_active:
            # Adaptive trail distance already stored in p.trail_dist_pct
            trail_stop = p.high_water * (1.0 - p.trail_dist_pct / 100.0)
            # Never worse than hard stop:
            trail_stop = max(trail_stop, hard_stop)

            # Price dips below trailing stop => sell
            if px <= trail_stop:
                close_position(state, positions, sym, px, f"TRAIL_STOP {p.trail_dist_pct:.2f}% from high")
                continue

        positions[sym] = p

# =========================
# STATUS REPORTING
# =========================
def status_report(state: dict, positions: Dict[str, Position], last_prices: Dict[str, float], guard_msg: str, ml_active: bool):
    cash = float(state["cash"])
    eq = equity(cash, positions, last_prices)
    wins = int(state.get("wins", 0))
    losses = int(state.get("losses", 0))
    total = wins + losses
    winpct = (wins / total) * 100.0 if total > 0 else 0.0
    realized = float(state.get("realized_pnl", 0.0))

    lines = []
    lines.append(f"📊 Coin Sniper PAPER [{INSTANCE_ID}]")
    lines.append(f"Cash: ${cash:.2f} | Equity: ${eq:.2f} | Realized PnL: ${realized:.2f}")
    lines.append(f"W/L: {wins}/{losses} | Win%: {winpct:.1f}% | Open trades: {len(positions)}/{MAX_OPEN_TRADES}")
    lines.append(f"Guard: {guard_msg}")
    lines.append(f"ML: {'ON' if ml_active else 'OFF'} (min_trades={ML_MIN_TRADES}, min_prob={ML_MIN_PROB})")

    if positions:
        lines.append("Open positions:")
        for sym, p in positions.items():
            px = last_prices.get(sym, p.entry_price)
            upct = ((px - p.entry_price) / p.entry_price) * 100.0 if p.entry_price > 0 else 0.0
            lines.append(
                f" - {sym}: qty={p.qty:.6f} entry={p.entry_price:.6f} now={px:.6f} "
                f"pnl%={upct:.2f} high={p.high_water:.6f} trail={'Y' if p.trailing_active else 'N'} dist={p.trail_dist_pct:.2f}%"
            )

    notify("\n".join(lines))

# =========================
# MAIN LOOP
# =========================
def main():
    notify(f"[{INSTANCE_ID}] Coin Sniper PAPER is LIVE. scan={SCAN_INTERVAL}s status={STATUS_INTERVAL}s")

    state = load_state()
    if not isinstance(state, dict):
        state = default_state()

    positions = positions_from_state(state)
    ensure_ledger_header()

    # Load ML store, build model periodically
    ml_store = load_ml_store()
    ml_active = bool(ml_store.get("ml_active", False))
    model = None

    # If user specified COINS, use them; else discover products
    if COINS:
        universe = [s.strip() for s in COINS.split(",") if s.strip()]
    else:
        universe = list_usd_products()

    # Apply EXCLUDE
    universe = [s for s in universe if s not in EXCLUDE]
    universe = universe[:MAX_SYMBOLS]

    last_status = float(state.get("last_status_ts", 0))
    last_ml = float(state.get("last_ml_ts", 0))

    while True:
        try:
            # Market guard
            guard_ok, guard_msg = btc_guard_ok()

            # Pull candles and prices for universe (lightweight: reuse candles for scoring + price)
            last_prices: Dict[str, float] = {}

            # Update open positions prices first (so exits are responsive)
            for sym in list(positions.keys()):
                candles = get_candles(sym, CANDLE_GRANULARITY, min(CANDLE_POINTS, 200))
                px = get_last_price_from_candles(candles)
                if px is not None:
                    last_prices[sym] = px

            # Manage exits
            manage_positions(state, positions, last_prices)

            # Entries only if guard is OK
            new_buys = 0
            if guard_ok and len(positions) < MAX_OPEN_TRADES:
                candidates = []
                for sym in universe:
                    if sym in positions:
                        continue
                    if sym in EXCLUDE:
                        continue
                    if not can_buy_symbol(state, sym):
                        continue

                    candles = get_candles(sym, CANDLE_GRANULARITY, min(CANDLE_POINTS, 200))
                    px = get_last_price_from_candles(candles)
                    if px is None:
                        continue
                    last_prices[sym] = px

                    score, reason, extra = score_symbol(sym, candles)
                    if score is None:
                        continue

                    # Hard RVOL gate (keeps it sniper)
                    if float(extra.get("rvol", 1.0)) < MIN_VOLUME_RATIO:
                        continue

                    # Score gate
                    if int(score) < ENTRY_SCORE_MIN:
                        continue

                    # ML gate (if active enough)
                    x = make_features(extra, int(score))
                    allow, prob = ml_should_allow_buy(model, ml_store, x)
                    if (model is not None) and (len(ml_store.get("rows", [])) >= ML_MIN_TRADES):
                        # If model exists, it is effectively active
                        if not ml_active:
                            ml_active = True
                            ml_store["ml_active"] = True
                            save_ml_store(ml_store)
                            notify(f"🧠 ML ACTIVATED: trades={len(ml_store.get('rows', []))} min_prob={ML_MIN_PROB}")

                    if not allow:
                        continue

                    # Candidate: higher score first, then higher RVOL
                    candidates.append((int(score), float(extra.get("rvol", 1.0)), sym, px, reason, extra, prob))

                # Sort candidates (best first)
                candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

                for sc, rv, sym, px, reason, extra, prob in candidates:
                    if new_buys >= MAX_NEW_BUYS_PER_SCAN:
                        break
                    if len(positions) >= MAX_OPEN_TRADES:
                        break
                    if float(state["cash"]) < MIN_TRADE_SIZE:
                        break

                    # Include prob in reason if ML is active
                    if prob and prob > 0:
                        reason2 = f"{reason} MLp={prob:.2f}"
                    else:
                        reason2 = reason

                    open_position(state, positions, sym, px, sc, reason2, extra)
                    new_buys += 1

            # Periodic ML training
            now = time.time()
            if ML_ENABLED and (now - last_ml) >= ML_INTERVAL:
                ml_store = load_ml_store()
                model = train_model(ml_store)
                last_ml = now
                state["last_ml_ts"] = int(last_ml)
                save_state({**state, "positions": positions_to_state(positions)})

            # Periodic status
            if (now - last_status) >= STATUS_INTERVAL:
                last_status = now
                state["last_status_ts"] = int(last_status)
                # Save state every status tick
                state["positions"] = positions_to_state(positions)
                save_state(state)
                status_report(state, positions, last_prices, guard_msg, ml_active)

            time.sleep(max(1, SCAN_INTERVAL))

        except SystemExit:
            raise
        except Exception as e:
            notify(f"⚠️ ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            # Don't spin too fast
            time.sleep(3)

if __name__ == "__main__":
    main()
