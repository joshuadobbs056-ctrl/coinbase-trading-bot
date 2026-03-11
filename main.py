# Coin Sniper — Savage ELITE (PAPER) — EXPLOSIVE OPTIMIZED (FULL RUNNING FILE)
# ✅ Wins/Losses/Win% Tracking + Realized PnL
# ✅ Exact Status Report Formatting (W/L, Equity, Position List)
# ✅ Micro-cap targeting (Price ≤ $0.50 + Vol Filter)
# ✅ BTC market guard + ATR-based adaptive trailing
# ✅ ML training + Telegram notification
#
# NOTE: PAPER trading only. No real orders placed.

import os, time, json, csv, traceback, base64
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

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "6"))       # seconds
STATUS_INTERVAL = int(os.getenv("STATUS_INTERVAL", "60"))  # seconds
ML_INTERVAL = int(os.getenv("ML_INTERVAL", "300"))         # seconds

MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", "12"))
MAX_NEW_BUYS_PER_SCAN = int(os.getenv("MAX_NEW_BUYS_PER_SCAN", "2"))
MIN_TRADE_SIZE = float(os.getenv("MIN_TRADE_SIZE", "35"))

STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "4.0"))
TRAILING_START_PERCENT = float(os.getenv("TRAILING_START_PERCENT", "0.8"))
TRAILING_DISTANCE_PERCENT = float(os.getenv("TRAILING_DISTANCE_PERCENT", "2.50"))

ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_MULT = float(os.getenv("ATR_MULT", "2.0"))

ENTRY_SCORE_MIN = int(os.getenv("ENTRY_SCORE_MIN", "7"))
MIN_VOLUME_RATIO = float(os.getenv("MIN_VOLUME_RATIO", "2.0"))
EXTENSION_MAX = float(os.getenv("EXTENSION_MAX", "0.06"))

COOLDOWN_SECONDS_AFTER_SELL = int(os.getenv("COOLDOWN_SECONDS_AFTER_SELL", "900"))
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "150"))

# Micro-cap constraints
MICROCAP_MAX_PRICE = 0.50  
MICROCAP_THRESHOLD = float(os.getenv("MICROCAP_THRESHOLD", "50000000")) 

COINS = os.getenv("COINS", "AUTO").strip()
EXCLUDE = set([s.strip() for s in os.getenv("EXCLUDE", "").split(",") if s.strip()])

CANDLE_GRANULARITY = int(os.getenv("CANDLE_GRANULARITY", "60"))
CANDLE_POINTS = int(os.getenv("CANDLE_POINTS", "200"))

BTC_GUARD_ENABLED = os.getenv("BTC_GUARD_ENABLED", "1") == "1"
BTC_GUARD_DROP_PCT = float(os.getenv("BTC_GUARD_DROP_PCT", "1.0"))
BTC_GUARD_WINDOW_MIN = int(os.getenv("BTC_GUARD_WINDOW_MIN", "15"))
BTC_SYMBOL = os.getenv("BTC_SYMBOL", "BTC-USD")

ML_ENABLED = os.getenv("ML_ENABLED", "1") == "1"
ML_MIN_TRADES = int(os.getenv("ML_MIN_TRADES", "25"))
ML_MIN_PROB = float(os.getenv("ML_MIN_PROB", "0.62"))

STATE_FILE = os.getenv("STATE_FILE", "coin_sniper_state.json")
ML_FILE = os.getenv("ML_FILE", "coin_sniper_ml.json")
LEDGER_FILE = os.getenv("LEDGER_FILE", "coin_sniper_ledger.csv")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

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
HEADERS = {"User-Agent": "coin-sniper-paper/1.0", "Accept": "application/json"}

def _http_get(url: str, params: Optional[dict] = None) -> Any:
    r = requests.get(url, params=params, headers=HEADERS, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()

def get_market_cap_proxy(product_id: str) -> float:
    try:
        data = _http_get(f"{COINBASE_API}/products/{product_id}/stats")
        return float(data.get("volume", 0)) * float(data.get("last", 0))
    except:
        return 0.0

def list_usd_products() -> List[str]:
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
    syms = sorted(set(syms))
    return syms[:MAX_SYMBOLS]

def get_candles(product_id: str, granularity: int, limit_points: int) -> List[list]:
    params = {"granularity": granularity}
    try:
        data = _http_get(f"{COINBASE_API}/products/{product_id}/candles", params=params)
        if not isinstance(data, list) or len(data) == 0:
            return []
        return data[:limit_points]
    except:
        return []

def get_last_price_from_candles(candles: List[list]) -> Optional[float]:
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
    try:
        recent = list(reversed(candles[:max(ATR_PERIOD + 2, 30)]))
        highs = [float(c[2]) for c in recent]
        lows = [float(c[1]) for c in recent]
        closes = [float(c[4]) for c in recent]
        if len(closes) < ATR_PERIOD + 2:
            return 1.5
        trs = []
        for i in range(1, len(closes)):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
            trs.append(tr)
        atr = float(np.mean(trs[-ATR_PERIOD:]))
        last_close = float(closes[-1])
        if last_close <= 0:
            return 1.5
        return (atr / last_close) * 100.0
    except Exception:
        return 1.5

def score_symbol(sym: str, candles: List[list]) -> Tuple[Optional[int], str, Dict[str, float]]:
    if not candles:
        return None, "no_candles", {}

    recent = list(reversed(candles[:CANDLE_POINTS]))
    closes = [float(c[4]) for c in recent]
    vols = [float(c[5]) for c in recent]

    if len(closes) < 60 or len(vols) < 60:
        return None, "short", {}

    p0 = closes[-1]

    # Skip if price > MICROCAP_MAX_PRICE
    if p0 > MICROCAP_MAX_PRICE:
        return None, f"price_above_microcap({MICROCAP_MAX_PRICE})", {}

    # Skip if volume proxy is too high (not a micro-cap)
    if get_market_cap_proxy(sym) > MICROCAP_THRESHOLD:
         return None, "exceeds_mcap_threshold", {}

    avg_vol = float(np.mean(vols[-21:-1])) if len(vols) >= 22 else float(np.mean(vols[:-1]))
    rvol = (vols[-1] / avg_vol) if avg_vol > 0 else 1.0

    ret_5 = (closes[-1] - closes[-6]) / closes[-6] if closes[-6] > 0 else 0.0
    ret_prev_5 = (closes[-6] - closes[-11]) / closes[-11] if closes[-11] > 0 else 0.0
    accel = ret_5 - ret_prev_5

    ema20 = float(_ema(np.array(closes, dtype=float), 20)[-1])
    ext = ((p0 - ema20) / ema20) if ema20 > 0 else 0.0

    atrp = _atr_percent(candles)

    score = 4
    if rvol >= 3.0:
        score += 4
    elif rvol >= MIN_VOLUME_RATIO:
        score += 2
    elif rvol < 1.0:
        score -= 3
    if accel > 0:
        score += 2
    if ext > EXTENSION_MAX:
        score -= 5

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
    entry_rvol: float
    entry_accel: float
    entry_ext: float
    entry_atrp: float

# =========================
# STATE & LEDGER
# =========================
def default_state():
    return {
        "cash": START_BALANCE, 
        "realized_pnl": 0.0, 
        "wins": 0, 
        "losses": 0, 
        "positions": {}, 
        "last_status_ts": 0, 
        "last_ml_ts": 0
    }

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except:
            return default_state()
    return default_state()

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def positions_from_state(state) -> Dict[str, Position]:
    pos_dict = {}
    raw = state.get("positions", {})
    for k, v in raw.items():
        pos_dict[k] = Position(**v)
    return pos_dict

def positions_to_state(positions: Dict[str, Position]) -> Dict[str, dict]:
    return {k: asdict(v) for k, v in positions.items()}

def ensure_ledger_header():
    if not os.path.exists(LEDGER_FILE):
        with open(LEDGER_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "symbol", "side", "price", "qty", "reason", "pnl", "pnl_pct"])

def log_trade(symbol, side, price, qty, reason, pnl=0, pnl_pct=0):
    with open(LEDGER_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), symbol, side, price, qty, reason, pnl, pnl_pct])

def can_buy_symbol(state, sym):
    if not os.path.exists(LEDGER_FILE): return True
    now = time.time()
    try:
        with open(LEDGER_FILE, "r") as f:
            lines = list(csv.reader(f))
            for row in reversed(lines):
                if row[1] == sym and row[2] == "SELL":
                    ts = time.mktime(time.strptime(row[0], "%Y-%m-%d %H:%M:%S"))
                    if (now - ts) < COOLDOWN_SECONDS_AFTER_SELL:
                        return False
                    break
    except:
        pass
    return True

# =========================
# ML PERSISTENCE
# =========================
def load_ml_store():
    if os.path.exists(ML_FILE):
        try:
            with open(ML_FILE, "r") as f: return json.load(f)
        except: pass
    return {"trades": [], "ml_active": False}

def make_features(extra, score):
    return [float(score), extra.get("rvol", 1.0), extra.get("accel", 0.0), extra.get("ext", 0.0), extra.get("atrp", 1.5)]

def ml_should_allow_buy(model, ml_store, features):
    if not ML_ENABLED or model is None or not ml_store.get("ml_active"):
        return True, 0.0
    try:
        probs = model.predict_proba([features])[0]
        prob_win = float(probs[1])
        return (prob_win >= ML_MIN_PROB), prob_win
    except:
        return True, 0.0

def train_model(ml_store):
    trades = ml_store.get("trades", [])
    if len(trades) < ML_MIN_TRADES: return None
    X, y = [], []
    for t in trades:
        X.append(t["features"])
        y.append(1 if t["pnl_pct"] > 0 else 0)
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X, y)
    ml_store["ml_active"] = True
    return model

# =========================
# CORE LOGIC
# =========================
def open_position(state, positions, sym, px, score, reason, extra):
    cash = float(state["cash"])
    trade_size = max(MIN_TRADE_SIZE, cash / (MAX_OPEN_TRADES - len(positions) + 1))
    if trade_size > cash: trade_size = cash
    qty = trade_size / px
    state["cash"] = cash - trade_size
    
    atrp = extra.get("atrp", 1.5)
    stop_px = px * (1 - (max(STOP_LOSS_PERCENT, atrp * ATR_MULT) / 100.0))
    
    pos = Position(
        symbol=sym, qty=qty, entry_price=px, entry_time=time.time(),
        high_water=px, stop_price=stop_px, trailing_active=False,
        trail_dist_pct=TRAILING_DISTANCE_PERCENT, last_reason=reason,
        last_score=score, entry_rvol=extra.get("rvol", 0),
        entry_accel=extra.get("accel", 0), entry_ext=extra.get("ext", 0),
        entry_atrp=atrp
    )
    positions[sym] = pos
    log_trade(sym, "BUY", px, qty, reason)
    notify(f"🚀 BUY {sym} @ {px:.6f} (Score:{score})")

def close_position(state, positions, sym, px, reason):
    pos = positions.pop(sym)
    pnl = (px - pos.entry_price) * pos.qty
    pnl_pct = (px / pos.entry_price - 1) * 100
    state["cash"] = float(state["cash"]) + (pos.qty * px)
    
    # Track stats
    state["realized_pnl"] = float(state.get("realized_pnl", 0.0)) + pnl
    if pnl > 0:
        state["wins"] = int(state.get("wins", 0)) + 1
    else:
        state["losses"] = int(state.get("losses", 0)) + 1

    log_trade(sym, "SELL", px, pos.qty, reason, pnl, pnl_pct)
    notify(f"💰 SELL {sym} @ {px:.6f} | PnL: {pnl:.2f} ({pnl_pct:.2f}%) | {reason}")
    
    # Save to ML store
    ml_s = load_ml_store()
    ml_s["trades"].append({
        "features": make_features({"rvol": pos.entry_rvol, "accel": pos.entry_accel, "ext": pos.entry_ext, "atrp": pos.entry_atrp}, pos.last_score),
        "pnl_pct": pnl_pct
    })
    with open(ML_FILE, "w") as f: json.dump(ml_s, f)

def manage_positions(state, positions, last_prices):
    for sym, pos in list(positions.items()):
        px = last_prices.get(sym)
        if not px: continue
        
        if px > pos.high_water:
            pos.high_water = px
            if not pos.trailing_active and (px / pos.entry_price - 1) >= (TRAILING_START_PERCENT / 100.0):
                pos.trailing_active = True
        
        if pos.trailing_active:
            new_stop = pos.high_water * (1 - (pos.trail_dist_pct / 100.0))
            if new_stop > pos.stop_price: pos.stop_price = new_stop
            
        if px <= pos.stop_price:
            reason = "STOP_LOSS" if not pos.trailing_active else "TRAILING_STOP"
            close_position(state, positions, sym, px, reason)

def btc_guard_ok() -> Tuple[bool, str]:
    if not BTC_GUARD_ENABLED: return True, "Disabled"
    try:
        c = get_candles(BTC_SYMBOL, 60, BTC_GUARD_WINDOW_MIN + 1)
        if not c or len(c) < BTC_GUARD_WINDOW_MIN: return True, "Insuff. BTC data"
        p_now = float(c[0][4])
        p_old = float(c[-1][4])
        drop = (p_now / p_old - 1) * 100
        if drop <= -BTC_GUARD_DROP_PCT:
            return False, f"BTC Drop {drop:.2f}%/15m"
        return True, f"BTC_GUARD ok drop={drop:.2f}%/15m"
    except: return True, "Guard Error"

def status_report(state, positions, last_prices, guard_msg, ml_active):
    cash = float(state["cash"])
    wins = int(state.get("wins", 0))
    losses = int(state.get("losses", 0))
    realized = float(state.get("realized_pnl", 0.0))
    
    total_trades = wins + losses
    win_pct = (wins / total_trades * 100) if total_trades > 0 else 0.0
    
    equity = cash
    pos_lines = []
    for sym, pos in positions.items():
        px = last_prices.get(sym, pos.entry_price)
        equity += (px * pos.qty)
        pnl_p = (px / pos.entry_price - 1) * 100
        trail_status = "Y" if pos.trailing_active else "N"
        pos_lines.append(f" - {sym}: qty={pos.qty:.6f} entry={pos.entry_price:.6f} now={px:.6f} pnl%={pnl_p:.2f} high={pos.high_water:.6f} trail={trail_status} dist={pos.trail_dist_pct:.2f}%")

    report = (
        f"📊 Coin Sniper PAPER [{INSTANCE_ID}]\n"
        f"Cash: ${cash:.2f} | Equity: ${equity:.2f} | Realized PnL: ${realized:.2f}\n"
        f"W/L: {wins}/{losses} | Win%: {win_pct:.1f}% | Open trades: {len(positions)}/{MAX_OPEN_TRADES}\n"
        f"Guard: {guard_msg}\n"
        f"ML: {'ON' if ml_active else 'OFF'} (min_trades={ML_MIN_TRADES}, min_prob={ML_MIN_PROB})\n"
        f"Open positions:\n" + ("\n".join(pos_lines) if pos_lines else "None")
    )
    notify(report)

# =========================
# MAIN LOOP
# =========================
def main():
    notify(f"[{INSTANCE_ID}] Coin Sniper PAPER is LIVE. scan={SCAN_INTERVAL}s")

    state = load_state()
    positions = positions_from_state(state)
    ensure_ledger_header()

    ml_store = load_ml_store()
    model = train_model(ml_store)

    if (COINS or "").upper() == "AUTO":
        universe = list_usd_products()
    else:
        universe = [s.strip() for s in COINS.split(",") if s.strip()]

    universe = [s for s in universe if s not in EXCLUDE][:MAX_SYMBOLS]

    last_status = float(state.get("last_status_ts", 0))
    last_ml = float(state.get("last_ml_ts", 0))

    while True:
        try:
            guard_ok, guard_msg = btc_guard_ok()
            last_prices: Dict[str, float] = {}
            
            for sym in list(positions.keys()):
                candles = get_candles(sym, CANDLE_GRANULARITY, 5)
                px = get_last_price_from_candles(candles)
                if px: last_prices[sym] = px

            manage_positions(state, positions, last_prices)
            state["positions"] = positions_to_state(positions)
            save_state(state)

            if guard_ok and len(positions) < MAX_OPEN_TRADES:
                candidates = []
                for sym in universe:
                    if sym in positions or sym in EXCLUDE or not can_buy_symbol(state, sym):
                        continue

                    candles = get_candles(sym, CANDLE_GRANULARITY, CANDLE_POINTS)
                    px = get_last_price_from_candles(candles)
                    if px is None: continue
                    
                    score, reason, extra = score_symbol(sym, candles)
                    if score is None or int(score) < ENTRY_SCORE_MIN:
                        continue

                    x = make_features(extra, int(score))
                    allow, prob = ml_should_allow_buy(model, ml_store, x)
                    if not allow: continue

                    candidates.append((int(score), float(extra.get("rvol", 1.0)), sym, px, reason, extra, prob))

                candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

                new_buys = 0
                for sc, rv, sym, px, reason, extra, prob in candidates:
                    if new_buys >= MAX_NEW_BUYS_PER_SCAN or len(positions) >= MAX_OPEN_TRADES: break
                    if float(state["cash"]) < MIN_TRADE_SIZE: break
                    
                    open_position(state, positions, sym, px, sc, reason, extra)
                    state["positions"] = positions_to_state(positions)
                    save_state(state)
                    new_buys += 1

            now = time.time()
            if ML_ENABLED and (now - last_ml) >= ML_INTERVAL:
                ml_store = load_ml_store()
                model = train_model(ml_store)
                last_ml = now
                state["last_ml_ts"] = int(last_ml)

            if (now - last_status) >= STATUS_INTERVAL:
                status_report(state, positions, last_prices, guard_msg, ml_store.get("ml_active", False))
                last_status = now
                state["last_status_ts"] = int(last_status)
                save_state(state)

            time.sleep(SCAN_INTERVAL)

        except Exception as e:
            notify(f"⚠️ ERROR: {e}\n{traceback.format_exc()}")
            time.sleep(10)

if __name__ == "__main__":
    main()
