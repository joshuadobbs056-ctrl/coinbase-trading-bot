# Coin Sniper — Savage Mode ELITE (TRADING FIXED)
# ✅ BUY ENGINE + SCAN LOGS + STATUS + WIN/LOSS + EQUITY + ML AUTO-LEARN
# ✅ GITHUB PERSIST (sync on change) + TRAILING STOP + HISTORY GATE + DYNAMIC SIZING

import os
import time
import json
import math
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
TRAIL_DIST_BASE = float(os.getenv("TRAILING_DISTANCE_PERCENT", 0.9))

ENTRY_SCORE_MIN = int(os.getenv("ENTRY_SCORE_MIN", 7))
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", 60))

EXCLUDE = set([s.strip() for s in os.getenv("EXCLUDE", "").split(",") if s.strip()])

CASH_RESERVE_PERCENT = float(os.getenv("CASH_RESERVE_PERCENT", 5))
MIN_POSITION_SIZE_PERCENT = float(os.getenv("MIN_POSITION_SIZE_PERCENT", 3))
MAX_POSITION_SIZE_PERCENT = float(os.getenv("MAX_POSITION_SIZE_PERCENT", 8))

HISTORY_POINTS_REQUIRED = int(os.getenv("HISTORY_POINTS_REQUIRED", 40))
MIN_HISTORY_COINS = int(os.getenv("MIN_HISTORY_COINS", 50))

ML_ENABLED = os.getenv("ML_ENABLED", "true").lower() == "true"
ML_ENABLE_AFTER = int(os.getenv("ML_ENABLE_AFTER", 50))  # completed trades required before ML can gate entries
ML_MIN_PROB = float(os.getenv("ML_MIN_PROB", 0.62))

BOT_PAUSED = os.getenv("BOT_PAUSED", "false").lower() == "true"

BASE_URL = "https://api.exchange.coinbase.com/products"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# =========================
# GITHUB CONFIG
# =========================

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_DATA_REPO")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")

GITHUB_PUSH_INTERVAL = int(os.getenv("GITHUB_PUSH_INTERVAL", "180"))  # slow it down by default

GITHUB_FILES = [
    "learning.json",
    "positions.json",
    "ml_training.csv",
    "trade_history.csv"
]

def github_headers():
    return {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

def github_pull_file(filename):
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return
    try:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"
        r = requests.get(url, headers=github_headers(), timeout=15)
        if r.status_code == 200 and "content" in r.json():
            content = base64.b64decode(r.json()["content"])
            open(filename, "wb").write(content)
            notify(f"[{INSTANCE_ID}] GITHUB RESTORE {filename}")
    except Exception:
        pass

def github_push_file(filename):
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return
    try:
        if not os.path.exists(filename):
            return
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"
        content = base64.b64encode(open(filename, "rb").read()).decode()

        sha = None
        r = requests.get(url, headers=github_headers(), timeout=15)
        if r.status_code == 200 and "sha" in r.json():
            sha = r.json()["sha"]

        payload = {
            "message": f"update {filename}",
            "content": content,
            "branch": GITHUB_BRANCH
        }
        if sha:
            payload["sha"] = sha

        requests.put(url, headers=github_headers(), json=payload, timeout=15)
    except Exception:
        pass

def github_pull_all():
    for f in GITHUB_FILES:
        github_pull_file(f)

# Sync only on changes (plus interval)
_last_push_time = 0
_last_state_hash = None

def _hash_state_for_push(learning_obj, positions_obj):
    try:
        payload = {
            "learning": learning_obj,
            "positions": positions_obj,
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
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
# FILES
# =========================

LEARNING_FILE = "learning.json"
POSITIONS_FILE = "positions.json"
HISTORY_FILE = "trade_history.csv"
ML_TRAIN_FILE = "ml_training.csv"

# =========================
# NOTIFY
# =========================

def notify(msg):
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
# STARTUP RESTORE
# =========================

github_pull_all()

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

learning = load_json(LEARNING_FILE, {
    "cash": START_BALANCE,
    "trade_count": 0,
    "win_count": 0,
    "loss_count": 0,
    "total_profit": 0.0
})

positions = load_json(POSITIONS_FILE, {})  # { "SYM": {entry, qty, stop, peak, opened_ts, score, ...} }

# ensure types
learning["cash"] = float(learning.get("cash", START_BALANCE))
learning["trade_count"] = int(learning.get("trade_count", 0))
learning["win_count"] = int(learning.get("win_count", 0))
learning["loss_count"] = int(learning.get("loss_count", 0))
learning["total_profit"] = float(learning.get("total_profit", 0.0))

cash = float(learning["cash"])

# in-memory price history
price_history = {}  # sym -> list[float]

# =========================
# MARKET DATA
# =========================

def get_price(sym):
    try:
        r = requests.get(f"{BASE_URL}/{sym}/ticker", timeout=10)
        j = r.json()
        if "price" not in j:
            return None
        return float(j["price"])
    except Exception:
        return None

def get_symbols():
    try:
        r = requests.get(BASE_URL, timeout=10)
        products = r.json()

        syms = []
        for p in products:
            # Coinbase Exchange products
            if p.get("quote_currency") == "USD":
                sid = p.get("id")
                if sid and sid not in EXCLUDE:
                    syms.append(sid)

        # Avoid obvious non-tradeable / weird pairs (optional)
        syms = [s for s in syms if s.endswith("-USD")]

        return syms[:MAX_SYMBOLS]
    except Exception:
        return []

symbols = get_symbols()

# =========================
# ML (PAPER LEARNING)
# =========================

ml_model = None
ml_last_train = 0

def _ensure_csv_headers(path, headers):
    if os.path.exists(path):
        return
    try:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(headers)
    except Exception:
        pass

def _append_row(path, row):
    try:
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow(row)
    except Exception:
        pass

# training columns:
# sym, ts, ret_5, ret_15, vol_15, slope_15, dd_15, score, label
_train_headers = ["sym","ts","ret_5","ret_15","vol_15","slope_15","dd_15","score","label"]
_ensure_csv_headers(ML_TRAIN_FILE, _train_headers)

def compute_features(sym):
    h = price_history.get(sym, [])
    if len(h) < 20:
        return None

    # Use last N points (each point ~ SCAN_INTERVAL seconds)
    last = np.array(h[-20:], dtype=float)
    if len(last) < 20:
        return None

    # Returns
    p0 = last[-1]
    p5 = last[-5]
    p15 = last[-15] if len(last) >= 15 else last[0]

    ret_5 = (p0 - p5) / p5 if p5 > 0 else 0.0
    ret_15 = (p0 - p15) / p15 if p15 > 0 else 0.0

    # Volatility proxy
    rets = np.diff(last) / np.maximum(last[:-1], 1e-12)
    vol_15 = float(np.std(rets[-15:])) if len(rets) >= 15 else float(np.std(rets))

    # Slope (trend)
    x = np.arange(len(last), dtype=float)
    try:
        slope = float(np.polyfit(x, last, 1)[0] / max(p0, 1e-12))
    except Exception:
        slope = 0.0

    # Drawdown in last window
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

    # Need enough completed trades to learn anything meaningful
    if learning.get("trade_count", 0) < 10:
        return

    # Load CSV rows
    try:
        rows = []
        with open(ML_TRAIN_FILE, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(row)
        if len(rows) < 30:
            return

        X = []
        y = []
        for row in rows:
            X.append([
                float(row["ret_5"]),
                float(row["ret_15"]),
                float(row["vol_15"]),
                float(row["slope_15"]),
                float(row["dd_15"]),
                float(row["score"])
            ])
            y.append(int(row["label"]))

        # Train
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
    feats = compute_features(sym)
    if feats is None:
        return None
    X = [[feats[0], feats[1], feats[2], feats[3], feats[4], float(score)]]
    try:
        p = float(ml_model.predict_proba(X)[0][1])
        return p
    except Exception:
        return None

# =========================
# TRADE HISTORY CSV
# =========================

_hist_headers = ["ts","action","sym","price","qty","notional","profit","cash","equity","score","reason"]
_ensure_csv_headers(HISTORY_FILE, _hist_headers)

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
        "" if score is None else score,
        reason
    ])

# =========================
# TRAILING STOP
# =========================

def apply_trailing_stop(pos, price):
    peak = float(pos.get("peak", pos["entry"]))
    if price > peak:
        pos["peak"] = price
    trail = float(pos["peak"]) * (1 - TRAIL_DIST_BASE / 100.0)
    if trail > float(pos["stop"]):
        pos["stop"] = trail

# =========================
# SCORING + BUY DECISION
# =========================

def score_symbol(sym):
    """
    Returns (score_1_to_10, reason) or (None, reason)
    """
    h = price_history.get(sym, [])
    if len(h) < HISTORY_POINTS_REQUIRED:
        return None, f"history<{HISTORY_POINTS_REQUIRED}"

    arr = np.array(h[-HISTORY_POINTS_REQUIRED:], dtype=float)
    if len(arr) < HISTORY_POINTS_REQUIRED:
        return None, f"history_short"

    p0 = float(arr[-1])
    if p0 <= 0:
        return None, "bad_price"

    # "Accumulation" proxy: low vol + slight upward drift + not far from recent peak (avoid blow-off)
    rets = np.diff(arr) / np.maximum(arr[:-1], 1e-12)
    vol = float(np.std(rets))  # smaller is better
    drift = float((arr[-1] - arr[0]) / max(arr[0], 1e-12))  # positive is better
    peak = float(np.max(arr))
    dd = float((p0 - peak) / peak) if peak > 0 else 0.0  # near 0 means close to peak; very negative means dumped

    # Momentum short-term
    mom = float((arr[-1] - arr[-5]) / max(arr[-5], 1e-12))

    # Build a composite score
    # Target: drift slightly positive, mom positive, vol not crazy, dd not too negative
    score = 5.0

    # Drift contribution
    score += np.clip(drift * 20.0, -2.0, 3.0)
    # Momentum contribution
    score += np.clip(mom * 30.0, -2.0, 3.0)
    # Volatility penalty
    score -= np.clip(vol * 200.0, 0.0, 3.0)
    # Drawdown penalty (avoid deep dumps)
    if dd < -0.08:
        score -= 2.0
    elif dd < -0.04:
        score -= 1.0

    score = int(max(1, min(10, round(score))))
    reason = f"drift={drift:.3f} mom={mom:.3f} vol={vol:.4f} dd={dd:.3f}"
    return score, reason

def compute_equity(latest_prices):
    eq = float(cash)
    for sym, pos in positions.items():
        px = latest_prices.get(sym)
        if px:
            eq += float(pos["qty"]) * float(px)
        else:
            # fallback to entry if missing price this tick
            eq += float(pos["qty"]) * float(pos["entry"])
    return float(eq)

def compute_position_notional(score, equity):
    """
    Dynamic sizing between MIN_POSITION_SIZE_PERCENT and MAX_POSITION_SIZE_PERCENT,
    scaled by score.
    """
    min_pct = MIN_POSITION_SIZE_PERCENT / 100.0
    max_pct = MAX_POSITION_SIZE_PERCENT / 100.0

    # map score 1..10 to 0..1
    t = (float(score) - 1.0) / 9.0
    pct = min_pct + (max_pct - min_pct) * t

    reserve = equity * (CASH_RESERVE_PERCENT / 100.0)
    available = max(0.0, cash - reserve)

    notional = available * pct

    # enforce minimum trade size
    if notional < MIN_TRADE_SIZE:
        notional = 0.0

    # don't exceed available cash
    notional = min(notional, available)

    return float(notional)

def should_buy(sym, score, prices, equity):
    if BOT_PAUSED:
        return False, "BOT_PAUSED"

    if sym in EXCLUDE:
        return False, "EXCLUDED"

    if sym in positions:
        return False, "ALREADY_IN_POSITION"

    if len(positions) >= MAX_OPEN_TRADES:
        return False, f"MAX_OPEN_TRADES({MAX_OPEN_TRADES})"

    px = prices.get(sym)
    if not px:
        return False, "NO_PRICE"

    if score < ENTRY_SCORE_MIN:
        return False, f"SCORE<{ENTRY_SCORE_MIN}"

    # ML gating only after enough completed trades
    if ML_ENABLED and learning.get("trade_count", 0) >= ML_ENABLE_AFTER:
        p = ml_probability(sym, score)
        if p is None:
            return False, "ML_NO_FEATURES"
        if p < ML_MIN_PROB:
            return False, f"ML_PROB<{ML_MIN_PROB:.2f} ({p:.2f})"

    # Sizing
    notional = compute_position_notional(score, equity)
    if notional <= 0:
        return False, "SIZE_ZERO_OR_BELOW_MIN"

    # qty calc
    qty = notional / px
    if qty <= 0:
        return False, "QTY_ZERO"

    return True, "OK"

def open_position(sym, score, prices, equity, score_reason=""):
    global cash

    px = prices.get(sym)
    if not px:
        return False, "NO_PRICE"

    notional = compute_position_notional(score, equity)
    if notional <= 0:
        return False, "SIZE_ZERO_OR_BELOW_MIN"

    qty = notional / px
    if qty <= 0:
        return False, "QTY_ZERO"

    # "paper fill" at current price
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
        "score_reason": score_reason
    }

    return True, f"BUY qty={qty:.6f} notional={notional:.2f} entry={entry:.6f} stop={stop:.6f}"

# =========================
# MAIN LOOP
# =========================

notify(f"[{INSTANCE_ID}] BOT STARTED | Symbols={len(symbols)} | scan={SCAN_INTERVAL}s status={STATUS_INTERVAL}s")

last_scan = 0
last_status = 0
last_ml = 0

while True:
    try:
        now = time.time()

        # refresh symbols occasionally (optional)
        # if int(now) % 3600 == 0: symbols = get_symbols()

        did_state_change = False

        if now - last_scan >= SCAN_INTERVAL:
            prices = {}
            fetched = 0

            # 1) Pull prices + build history
            for sym in symbols:
                px = get_price(sym)
                if px:
                    prices[sym] = px
                    fetched += 1
                    price_history.setdefault(sym, []).append(px)
                    # keep history bounded
                    if len(price_history[sym]) > max(HISTORY_POINTS_REQUIRED * 3, 200):
                        price_history[sym] = price_history[sym][-200:]

            equity = compute_equity(prices)

            # 2) Manage open positions (trailing stop / stop hit)
            for sym, pos in list(positions.items()):
                px = prices.get(sym)
                if not px:
                    continue

                apply_trailing_stop(pos, px)

                if px <= float(pos["stop"]):
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

                    equity = compute_equity(prices)

                    notify(f"[{INSTANCE_ID}] SELL {sym} px={px:.6f} entry={entry:.6f} profit={profit:.2f} stop={pos['stop']:.6f}")
                    record_history("SELL", sym, px, qty, proceeds, profit, cash, equity, pos.get("score"), "STOP/TRAIL")

                    # ML training row from the symbol at close
                    feats = compute_features(sym)
                    if feats is not None:
                        label = 1 if profit > 0 else 0
                        _append_row(ML_TRAIN_FILE, [
                            sym, int(time.time()),
                            f"{feats[0]:.6f}", f"{feats[1]:.6f}", f"{feats[2]:.6f}", f"{feats[3]:.6f}", f"{feats[4]:.6f}",
                            int(pos.get("score", 0)),
                            label
                        ])

                    del positions[sym]
                    did_state_change = True

            # 3) BUY scan (only if enough history across enough coins)
            history_ready = sum(1 for s in symbols if len(price_history.get(s, [])) >= HISTORY_POINTS_REQUIRED)
            if history_ready < MIN_HISTORY_COINS:
                notify(f"[{INSTANCE_ID}] SCAN tick fetched={fetched}/{len(symbols)} history_ready={history_ready}/{MIN_HISTORY_COINS} -> waiting")
            else:
                candidates = []
                rejects = 0

                for sym in symbols:
                    if sym in positions:
                        continue
                    sc, sc_reason = score_symbol(sym)
                    if sc is None:
                        continue
                    candidates.append((sc, sym, sc_reason))

                candidates.sort(reverse=True, key=lambda x: x[0])

                # pick best candidate that passes should_buy
                chosen = None
                chosen_reason = ""
                chosen_score_reason = ""

                equity = compute_equity(prices)

                for sc, sym, sc_reason in candidates[:20]:  # look at top 20
                    ok, reason = should_buy(sym, sc, prices, equity)
                    if ok:
                        chosen = (sc, sym)
                        chosen_reason = reason
                        chosen_score_reason = sc_reason
                        break
                    else:
                        rejects += 1

                if candidates:
                    top_sc, top_sym, top_sc_reason = candidates[0]
                    notify(f"[{INSTANCE_ID}] SCAN candidates={len(candidates)} rejects={rejects} top={top_sym} score={top_sc} | {top_sc_reason}")
                else:
                    notify(f"[{INSTANCE_ID}] SCAN candidates=0 (no symbols meet history/score yet)")

                if chosen and len(positions) < MAX_OPEN_TRADES:
                    sc, sym = chosen
                    ok, msg = open_position(sym, sc, prices, equity, chosen_score_reason)
                    equity = compute_equity(prices)
                    if ok:
                        notify(f"[{INSTANCE_ID}] BUY {sym} score={sc} | {msg}")
                        record_history("BUY", sym, prices[sym], positions[sym]["qty"], positions[sym]["qty"] * prices[sym], 0.0, cash, equity, sc, chosen_score_reason)
                        did_state_change = True
                    else:
                        notify(f"[{INSTANCE_ID}] BUY_FAIL {sym} score={sc} reason={msg}")

            # 4) Persist state (cash MUST be written back)
            learning["cash"] = float(cash)

            save_json(LEARNING_FILE, learning)
            save_json(POSITIONS_FILE, positions)

            # 5) GitHub sync only if state changed (or interval)
            state_hash = _hash_state_for_push(learning, positions)
            if did_state_change:
                github_push_all_if_needed(state_hash)
            else:
                # allow periodic push so you still see it alive
                github_push_all_if_needed(state_hash)

            last_scan = now

        # STATUS message
        if now - last_status >= STATUS_INTERVAL:
            prices_now = {}
            for sym in list(positions.keys())[:25]:
                px = get_price(sym)
                if px:
                    prices_now[sym] = px
            eq = compute_equity(prices_now)
            trades = learning.get("trade_count", 0)
            wins = learning.get("win_count", 0)
            losses = learning.get("loss_count", 0)
            winpct = (wins / trades * 100.0) if trades > 0 else 0.0

            notify(
                f"[{INSTANCE_ID}] STATUS cash=${cash:.2f} equity=${eq:.2f} open_trades={len(positions)}/{MAX_OPEN_TRADES} "
                f"trades={trades} W={wins} L={losses} win%={winpct:.1f} profit=${learning.get('total_profit',0.0):.2f}"
            )
            last_status = now

        # ML train periodically
        if now - last_ml >= ML_INTERVAL:
            train_ml_if_ready()
            last_ml = now

        time.sleep(1)

    except Exception:
        notify(f"[{INSTANCE_ID}] ERROR\n{traceback.format_exc()}")
        time.sleep(5)
