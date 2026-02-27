import os
import time
import json
import random
import threading
import requests
from datetime import datetime, timezone

import websocket  # pip: websocket-client

# ============================================================
# CONFIG (Railway Variables)
# ============================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

STATE_FILE = os.getenv("STATE_FILE", "/data/state.json")

START_BALANCE = float(os.getenv("START_BALANCE", "1000"))

# Base risk
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.02"))
MIN_SPEND = float(os.getenv("MIN_SPEND", "25"))
MAX_SPEND = float(os.getenv("MAX_SPEND", "60"))

MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "12"))

# Risk controls
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.015"))

TRAILING_ACTIVATION = float(os.getenv("TRAILING_ACTIVATION", "0.02"))
TRAILING_DISTANCE = float(os.getenv("TRAILING_DISTANCE", "0.006"))

# Strict entry filters (5m candles)
MIN_TREND = float(os.getenv("MIN_TREND", "0.004"))               # ~0.4% over last ~50 min window below
MIN_VOLUME_MULT = float(os.getenv("MIN_VOLUME_MULT", "1.3"))     # current 5m volume vs avg

# 24/7 runtime settings
WS_URL = os.getenv("COINBASE_WS_URL", "wss://ws-feed.exchange.coinbase.com")
UNIVERSE_SIZE = int(os.getenv("UNIVERSE_SIZE", "120"))           # total symbols to consider
WS_SUBSCRIBE_SIZE = int(os.getenv("WS_SUBSCRIBE_SIZE", "60"))    # how many to subscribe to on WS at once

ENTRY_CHECK_INTERVAL = int(os.getenv("ENTRY_CHECK_INTERVAL", "30"))   # seconds between entry attempts
SYMBOL_ROTATE_SECONDS = int(os.getenv("SYMBOL_ROTATE_SECONDS", "600")) # rotate WS list every 10 min

EQUITY_REFRESH_SECONDS = int(os.getenv("EQUITY_REFRESH_SECONDS", "60")) # refresh equity using REST
HEARTBEAT_SECONDS = int(os.getenv("HEARTBEAT_SECONDS", "120"))

# Profit target mode (all enabled)
DAILY_PROFIT_TARGET = float(os.getenv("DAILY_PROFIT_TARGET", "25"))  # dollars per day target
PAUSE_AFTER_TARGET_SECONDS = int(os.getenv("PAUSE_AFTER_TARGET_SECONDS", "21600"))  # 6 hours
CLOSE_ALL_ON_TARGET = os.getenv("CLOSE_ALL_ON_TARGET", "1") == "1"   # close positions when hit target?

# Optional kill switch
MAX_DRAWDOWN_PCT = float(os.getenv("MAX_DRAWDOWN_PCT", "0.12"))  # e.g. 0.12 = 12% from peak; set high to effectively disable

# Compounding tiers (equity grows -> allow larger spend cap)
# You can change these in Railway if you want later
EQUITY_TIER_1 = float(os.getenv("EQUITY_TIER_1", "1500"))
EQUITY_TIER_2 = float(os.getenv("EQUITY_TIER_2", "2500"))
EQUITY_TIER_3 = float(os.getenv("EQUITY_TIER_3", "5000"))

MAX_SPEND_TIER_1 = float(os.getenv("MAX_SPEND_TIER_1", "80"))
MAX_SPEND_TIER_2 = float(os.getenv("MAX_SPEND_TIER_2", "120"))
MAX_SPEND_TIER_3 = float(os.getenv("MAX_SPEND_TIER_3", "200"))

# ============================================================
# GLOBALS
# ============================================================

STATE_LOCK = threading.Lock()
LATEST_PRICES = {}  # product -> float
WS_PRODUCTS = []
WS_APP = None

# ============================================================
# TELEGRAM
# ============================================================

def send(msg: str):
    print("[TELEGRAM]", msg, flush=True)
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TELEGRAM] Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID", flush=True)
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
            timeout=10
        )
        print(f"[TELEGRAM RESPONSE] {r.status_code}", flush=True)
    except Exception as e:
        print(f"[TELEGRAM ERROR] {e}", flush=True)

# ============================================================
# STATE
# ============================================================

def _utc_day_key(ts=None):
    # Daily profit target resets at UTC midnight
    if ts is None:
        ts = time.time()
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d")

def load_state():
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                s = json.load(f)
                s.setdefault("cash", START_BALANCE)
                s.setdefault("positions", {})
                s.setdefault("equity", START_BALANCE)
                s.setdefault("profit", 0.0)
                s.setdefault("trades", 0)
                s.setdefault("wins", 0)
                s.setdefault("peak_equity", START_BALANCE)
                s.setdefault("pause_until", 0)
                s.setdefault("daily", {"day": _utc_day_key(), "start_equity": START_BALANCE, "pnl": 0.0})
                return s
    except Exception as e:
        print("[STATE LOAD ERROR]", e, flush=True)

    return {
        "cash": START_BALANCE,
        "positions": {},
        "equity": START_BALANCE,
        "profit": 0.0,
        "trades": 0,
        "wins": 0,
        "peak_equity": START_BALANCE,
        "pause_until": 0,
        "daily": {"day": _utc_day_key(), "start_equity": START_BALANCE, "pnl": 0.0},
    }

def save_state(state):
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, STATE_FILE)
    except Exception as e:
        print("[STATE SAVE ERROR]", e, flush=True)

# ============================================================
# COINBASE REST
# ============================================================

def get_products():
    try:
        r = requests.get("https://api.exchange.coinbase.com/products", timeout=10)
        data = r.json()
        pairs = [
            p["id"] for p in data
            if p.get("status") == "online" and (p["id"].endswith("-USD") or p["id"].endswith("-USDC"))
        ]
        if not pairs:
            return []
        return random.sample(pairs, min(len(pairs), UNIVERSE_SIZE))
    except Exception as e:
        print("[PRODUCTS ERROR]", e, flush=True)
        return []

def get_candles_5m(product, limit=30):
    # Coinbase returns newest-first; each candle: [time, low, high, open, close, volume]
    try:
        r = requests.get(
            f"https://api.exchange.coinbase.com/products/{product}/candles?granularity=300",
            timeout=10
        )
        if r.status_code != 200:
            return []
        candles = r.json()
        return candles[:limit] if isinstance(candles, list) else []
    except:
        return []

def refresh_equity(state):
    total = state["cash"]
    for product, pos in state["positions"].items():
        # use latest websocket price if we have it; fallback to REST ticker if missing
        p = LATEST_PRICES.get(product)
        if p is None:
            try:
                rr = requests.get(f"https://api.exchange.coinbase.com/products/{product}/ticker", timeout=10)
                if rr.status_code == 200:
                    p = float(rr.json()["price"])
            except:
                p = None
        if p:
            total += pos["size"] * p

    state["equity"] = total
    state["profit"] = total - START_BALANCE

    if total > state.get("peak_equity", START_BALANCE):
        state["peak_equity"] = total

    # daily bookkeeping
    day = _utc_day_key()
    if state["daily"].get("day") != day:
        state["daily"] = {"day": day, "start_equity": total, "pnl": 0.0}
    state["daily"]["pnl"] = total - float(state["daily"].get("start_equity", total))

# ============================================================
# COMPounding spend / tiers
# ============================================================

def tiered_max_spend(equity: float) -> float:
    if equity >= EQUITY_TIER_3:
        return max(MAX_SPEND, MAX_SPEND_TIER_3)
    if equity >= EQUITY_TIER_2:
        return max(MAX_SPEND, MAX_SPEND_TIER_2)
    if equity >= EQUITY_TIER_1:
        return max(MAX_SPEND, MAX_SPEND_TIER_1)
    return MAX_SPEND

def calc_spend(state) -> float:
    equity = float(state.get("equity", START_BALANCE))
    spend = equity * RISK_PER_TRADE
    spend = max(spend, MIN_SPEND)
    spend = min(spend, tiered_max_spend(equity))
    return spend

# ============================================================
# STRICT ENTRY FILTER
# ============================================================

def valid_entry(product) -> bool:
    candles = get_candles_5m(product, limit=25)
    if len(candles) < 15:
        return False

    # candles newest-first
    closes = [c[4] for c in candles]
    vols = [c[5] for c in candles]

    # trend over ~50 minutes (10 candles of 5m each)
    # closes[0] is newest, closes[10] is ~50 min ago
    base = closes[10]
    if base <= 0:
        return False
    trend = (closes[0] - base) / base
    if trend < MIN_TREND:
        return False

    # volume spike: current candle volume vs avg of prior 12 candles
    v_now = vols[0]
    v_avg = sum(vols[1:13]) / 12.0 if len(vols) >= 13 else 0
    if v_avg <= 0:
        return False
    if v_now < (v_avg * MIN_VOLUME_MULT):
        return False

    return True

# ============================================================
# POSITIONS + trailing stop (tick-driven)
# ============================================================

def fmt_stats(state) -> str:
    eq = float(state.get("equity", START_BALANCE))
    pf = float(state.get("profit", 0.0))
    cash = float(state.get("cash", 0.0))
    dpf = float(state.get("daily", {}).get("pnl", 0.0))
    trades = int(state.get("trades", 0))
    wins = int(state.get("wins", 0))
    winrate = (wins / trades) * 100 if trades else 0.0
    return (
        f"Cash: ${cash:.2f}\n"
        f"Equity: ${eq:.2f}\n"
        f"Profit: ${pf:.2f}\n"
        f"Daily: ${dpf:.2f}\n"
        f"Trades: {trades} | Win: {winrate:.1f}%"
    )

def open_position(state, product, price: float):
    if product in state["positions"]:
        return False
    if len(state["positions"]) >= MAX_OPEN_POSITIONS:
        return False

    # Profit target pause check
    if time.time() < float(state.get("pause_until", 0)):
        return False

    # Max drawdown kill switch
    peak = float(state.get("peak_equity", state.get("equity", START_BALANCE)))
    eq = float(state.get("equity", START_BALANCE))
    if peak > 0 and (peak - eq) / peak >= MAX_DRAWDOWN_PCT:
        # stop opening positions
        state["pause_until"] = time.time() + 86400  # pause 24h
        send("KILL SWITCH: Max drawdown hit. Pausing entries for 24h.")
        return False

    # Strict entry check
    if not valid_entry(product):
        return False

    spend = calc_spend(state)
    if state["cash"] < spend:
        return False

    size = spend / price
    state["cash"] -= spend

    state["positions"][product] = {
        "entry": price,
        "size": size,
        "highest": price,
        "trailing_active": False,
        "trailing_stop": None,
        "opened": time.time(),
    }

    send(
        f"BUY {product}\n"
        f"Entry: ${price:.6f}\n"
        f"Spend: ${spend:.2f}\n"
        f"{fmt_stats(state)}"
    )
    return True

def close_position(state, product, price: float, reason: str):
    pos = state["positions"].get(product)
    if not pos:
        return False

    value = pos["size"] * price
    pnl = value - (pos["size"] * pos["entry"])

    state["cash"] += value

    state["trades"] = int(state.get("trades", 0)) + 1
    if pnl > 0:
        state["wins"] = int(state.get("wins", 0)) + 1

    del state["positions"][product]

    send(
        f"SELL {product}\n"
        f"Exit: ${price:.6f}\n"
        f"P/L: ${pnl:.2f}\n"
        f"Reason: {reason}\n"
        f"{fmt_stats(state)}"
    )
    return True

def on_tick(state, product, price: float):
    pos = state["positions"].get(product)
    if not pos:
        return

    entry = float(pos["entry"])
    change = (price - entry) / entry if entry else 0.0

    # highest update
    if price > float(pos["highest"]):
        pos["highest"] = price

    # activate trailing once in profit
    if (not pos["trailing_active"]) and change >= TRAILING_ACTIVATION:
        pos["trailing_active"] = True
        pos["trailing_stop"] = float(pos["highest"]) * (1 - TRAILING_DISTANCE)
        send(
            f"TRAIL ON {product}\n"
            f"Entry: ${entry:.6f}\n"
            f"Peak: ${float(pos['highest']):.6f}\n"
            f"Stop: ${float(pos['trailing_stop']):.6f}"
        )

    # trailing update + hit
    if pos["trailing_active"]:
        new_stop = float(pos["highest"]) * (1 - TRAILING_DISTANCE)
        if pos["trailing_stop"] is None or new_stop > float(pos["trailing_stop"]):
            pos["trailing_stop"] = new_stop

        if price <= float(pos["trailing_stop"]):
            close_position(state, product, price, "Trailing Stop")
            return

    # hard stop loss
    if change <= -STOP_LOSS_PCT:
        close_position(state, product, price, "Stop Loss")
        return

# ============================================================
# PROFIT TARGET MODE
# ============================================================

def check_profit_target(state):
    # hit daily target => pause entries and optionally close all positions
    dpf = float(state.get("daily", {}).get("pnl", 0.0))
    if dpf >= DAILY_PROFIT_TARGET and time.time() >= float(state.get("pause_until", 0)):
        state["pause_until"] = time.time() + PAUSE_AFTER_TARGET_SECONDS

        send(
            "DAILY TARGET HIT ✅\n"
            f"Daily PnL: ${dpf:.2f} (target ${DAILY_PROFIT_TARGET:.2f})\n"
            f"Pausing new entries for {PAUSE_AFTER_TARGET_SECONDS//3600}h\n"
            f"{fmt_stats(state)}"
        )

        if CLOSE_ALL_ON_TARGET:
            # Close everything at current prices
            for product in list(state["positions"].keys()):
                p = LATEST_PRICES.get(product)
                if p is None:
                    continue
                close_position(state, product, p, "Daily Target Lock")
        return True
    return False

# ============================================================
# WEBSOCKET
# ============================================================

def ws_subscribe(ws, products):
    sub = {"type": "subscribe", "channels": [{"name": "ticker", "product_ids": products}]}
    ws.send(json.dumps(sub))

def ws_unsubscribe(ws, products):
    unsub = {"type": "unsubscribe", "channels": [{"name": "ticker", "product_ids": products}]}
    ws.send(json.dumps(unsub))

def on_open(ws):
    print("[WS] open", flush=True)
    if WS_PRODUCTS:
        ws_subscribe(ws, WS_PRODUCTS)
        print(f"[WS] subscribed {len(WS_PRODUCTS)} products", flush=True)

def on_message(ws, message):
    try:
        msg = json.loads(message)
        if msg.get("type") != "ticker":
            return
        product = msg.get("product_id")
        price = msg.get("price")
        if not product or not price:
            return

        p = float(price)
        LATEST_PRICES[product] = p

        with STATE_LOCK:
            on_tick(GLOBAL_STATE, product, p)

    except Exception as e:
        print("[WS MESSAGE ERROR]", e, flush=True)

def on_error(ws, error):
    print("[WS] error:", error, flush=True)

def on_close(ws, code, reason):
    print(f"[WS] closed: {code} {reason}", flush=True)

def start_ws(products):
    global WS_APP, WS_PRODUCTS
    WS_PRODUCTS = products

    WS_APP = websocket.WebSocketApp(
        WS_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    def run():
        while True:
            try:
                WS_APP.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                print("[WS RUN ERROR]", e, flush=True)
            time.sleep(3)

    t = threading.Thread(target=run, daemon=True)
    t.start()

def rotate_ws_symbols():
    products = get_products()
    if not products:
        return None
    return random.sample(products, min(len(products), WS_SUBSCRIBE_SIZE))

def resubscribe(new_products):
    global WS_PRODUCTS
    old = WS_PRODUCTS
    WS_PRODUCTS = new_products
    try:
        if WS_APP and WS_APP.sock and WS_APP.sock.connected:
            try:
                ws_unsubscribe(WS_APP, old)
            except:
                pass
            ws_subscribe(WS_APP, new_products)
            print(f"[WS] resubscribed {len(new_products)} products", flush=True)
    except Exception as e:
        print("[WS RESUB ERROR]", e, flush=True)

# ============================================================
# ENTRY LOOP (strict)
# ============================================================

def try_entries(state):
    if time.time() < float(state.get("pause_until", 0)):
        return

    if not LATEST_PRICES:
        return

    # Candidates from current WS universe
    candidates = list(WS_PRODUCTS) if WS_PRODUCTS else list(LATEST_PRICES.keys())
    random.shuffle(candidates)

    # attempt only a few per cycle to stay under REST candle limits
    attempts = 0
    for product in candidates:
        if attempts >= 8:
            break
        if product in state["positions"]:
            continue
        price = LATEST_PRICES.get(product)
        if not price:
            continue
        # strict entry calls REST candles (rate-limited), so keep it small
        if open_position(state, product, price):
            pass
        attempts += 1

# ============================================================
# MAIN
# ============================================================

GLOBAL_STATE = None

def main():
    global GLOBAL_STATE

    GLOBAL_STATE = load_state()

    # Startup equity refresh so stats are right
    with STATE_LOCK:
        refresh_equity(GLOBAL_STATE)
        save_state(GLOBAL_STATE)

    send(
        "STRICT WS SNIPER STARTED\n"
        f"Trail act {TRAILING_ACTIVATION*100:.2f}% | dist {TRAILING_DISTANCE*100:.2f}%\n"
        f"SL {STOP_LOSS_PCT*100:.2f}%\n"
        f"Daily target ${DAILY_PROFIT_TARGET:.2f} | CloseAll {int(CLOSE_ALL_ON_TARGET)}\n"
        f"{fmt_stats(GLOBAL_STATE)}"
    )

    initial = rotate_ws_symbols()
    if not initial:
        raise RuntimeError("Could not fetch products from Coinbase.")
    start_ws(initial)

    last_equity = 0
    last_heartbeat = 0
    last_entries = 0
    last_rotate = 0

    while True:
        now = time.time()

        try:
            # equity refresh (REST-safe)
            if now - last_equity >= EQUITY_REFRESH_SECONDS:
                with STATE_LOCK:
                    refresh_equity(GLOBAL_STATE)
                    check_profit_target(GLOBAL_STATE)
                    save_state(GLOBAL_STATE)
                last_equity = now

            # heartbeat
            if now - last_heartbeat >= HEARTBEAT_SECONDS:
                with STATE_LOCK:
                    send("Heartbeat ✅\n" + fmt_stats(GLOBAL_STATE))
                last_heartbeat = now

            # strict entries
            if now - last_entries >= ENTRY_CHECK_INTERVAL:
                with STATE_LOCK:
                    check_profit_target(GLOBAL_STATE)
                    try_entries(GLOBAL_STATE)
                    save_state(GLOBAL_STATE)
                last_entries = now

            # rotate ws universe periodically
            if now - last_rotate >= SYMBOL_ROTATE_SECONDS:
                new_syms = rotate_ws_symbols()
                if new_syms:
                    resubscribe(new_syms)
                last_rotate = now

            time.sleep(1)

        except Exception as e:
            print("[MAIN ERROR]", e, flush=True)
            send(f"ERROR: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
