import os
import time
import json
import random
import requests
from datetime import datetime

# =========================
# CONFIG
# =========================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

STATE_FILE = os.getenv("STATE_FILE", "/data/state.json")

START_BALANCE = float(os.getenv("START_BALANCE", "1000"))

RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.03"))

MIN_SPEND = float(os.getenv("MIN_SPEND", "25"))
MAX_SPEND = float(os.getenv("MAX_SPEND", "300"))

MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "30"))

TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.03"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.02"))
TRAILING_STOP_PCT = float(os.getenv("TRAILING_STOP_PCT", "0.015"))

MIN_TREND_PCT = float(os.getenv("MIN_TREND_PCT", "0.007"))

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "20"))

UNIVERSE_SIZE = 350
SAMPLE_SIZE = 200

COOLDOWN_SECONDS = 600
MAX_PORTFOLIO_EXPOSURE = 0.85

# =========================
# TELEGRAM
# =========================

def send(msg):

    print(f"[TELEGRAM SEND ATTEMPT] {msg}", flush=True)

    if not TELEGRAM_TOKEN:
        print("[TELEGRAM ERROR] Missing TELEGRAM_TOKEN", flush=True)
        return

    if not TELEGRAM_CHAT_ID:
        print("[TELEGRAM ERROR] Missing TELEGRAM_CHAT_ID", flush=True)
        return

    try:

        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

        response = requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": msg
            },
            timeout=10
        )

        print(f"[TELEGRAM RESPONSE] {response.status_code} {response.text}", flush=True)

    except Exception as e:

        print(f"[TELEGRAM EXCEPTION] {e}", flush=True)

# =========================
# STATE
# =========================

def load_state():

    try:

        if os.path.exists(STATE_FILE):

            with open(STATE_FILE, "r") as f:

                state = json.load(f)

                print("[STATE] Loaded", flush=True)

                return state

    except Exception as e:

        print("[STATE LOAD ERROR]", e, flush=True)

    print("[STATE] New state created", flush=True)

    return {

        "cash": START_BALANCE,
        "positions": {},
        "equity": START_BALANCE,
        "cooldowns": {},
        "peak_equity": START_BALANCE

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

# =========================
# MARKET
# =========================

def get_products():

    try:

        r = requests.get(
            "https://api.exchange.coinbase.com/products",
            timeout=10
        )

        data = r.json()

        pairs = [
            p["id"]
            for p in data
            if ("USD" in p["id"] or "USDC" in p["id"])
            and p.get("status") == "online"
        ]

        return random.sample(pairs, min(len(pairs), UNIVERSE_SIZE))

    except Exception as e:

        print("[PRODUCT ERROR]", e, flush=True)

        return []

def get_price(product):

    try:

        r = requests.get(
            f"https://api.exchange.coinbase.com/products/{product}/ticker",
            timeout=10
        )

        return float(r.json()["price"])

    except:

        return None

# =========================
# TREND
# =========================

def get_trend(product):

    try:

        r = requests.get(
            f"https://api.exchange.coinbase.com/products/{product}/candles?granularity=300",
            timeout=10
        )

        candles = r.json()

        if len(candles) < 12:
            return None

        closes = [c[4] for c in candles]

        change = (closes[0] - closes[8]) / closes[8]

        volatility = abs(closes[0] - closes[1]) / closes[1]

        if change < MIN_TREND_PCT:
            return None

        if volatility > 0.05:
            return None

        return change * random.uniform(1.2, 2.2)

    except:

        return None

# =========================
# EQUITY
# =========================

def calculate_equity(state):

    total = state["cash"]

    for product, pos in state["positions"].items():

        price = get_price(product)

        if price:
            total += pos["size"] * price

    state["equity"] = total

    if total > state["peak_equity"]:
        state["peak_equity"] = total

    return total

# =========================
# EXPOSURE
# =========================

def portfolio_exposure(state):

    exposure = 0

    for pos in state["positions"].values():

        exposure += pos["size"] * pos["entry"]

    if state["equity"] == 0:
        return 0

    return exposure / state["equity"]

# =========================
# POSITION SIZE
# =========================

def get_spend(state):

    equity = calculate_equity(state)

    spend = equity * RISK_PER_TRADE

    spend = max(spend, MIN_SPEND)
    spend = min(spend, MAX_SPEND)

    return spend

# =========================
# OPEN POSITION
# =========================

def open_position(state, product):

    now = time.time()

    if product in state["cooldowns"]:

        if now - state["cooldowns"][product] < COOLDOWN_SECONDS:
            return

    if len(state["positions"]) >= MAX_OPEN_POSITIONS:
        return

    if portfolio_exposure(state) > MAX_PORTFOLIO_EXPOSURE:
        return

    price = get_price(product)

    if not price:
        return

    spend = get_spend(state)

    if state["cash"] < spend:
        return

    size = spend / price

    state["cash"] -= spend

    state["positions"][product] = {

        "entry": price,
        "size": size,
        "peak": price,
        "opened": now

    }

    send(f"BUY {product} @ ${price:.5f}")

# =========================
# CLOSE POSITION
# =========================

def close_position(state, product, price, reason):

    pos = state["positions"][product]

    value = pos["size"] * price

    pnl = value - (pos["size"] * pos["entry"])

    state["cash"] += value

    del state["positions"][product]

    state["cooldowns"][product] = time.time()

    send(f"SELL {product} @ ${price:.5f} | PNL ${pnl:.2f} | {reason}")

# =========================
# POSITION MANAGEMENT
# =========================

def manage_positions(state):

    for product in list(state["positions"].keys()):

        pos = state["positions"][product]

        price = get_price(product)

        if not price:
            continue

        entry = pos["entry"]
        peak = pos["peak"]

        if price > peak:
            pos["peak"] = price
            peak = price

        drawdown = (peak - price) / peak
        profit = (price - entry) / entry

        if drawdown >= TRAILING_STOP_PCT:
            close_position(state, product, price, "TRAILING STOP")

        elif profit >= TAKE_PROFIT_PCT:
            close_position(state, product, price, "TAKE PROFIT")

        elif profit <= -STOP_LOSS_PCT:
            close_position(state, product, price, "STOP LOSS")

# =========================
# SCANNER
# =========================

def scanner(state):

    products = get_products()

    sample = random.sample(products, min(len(products), SAMPLE_SIZE))

    for product in sample:

        score = get_trend(product)

        if score:

            print(f"[SIGNAL] {product} score={score:.4f}", flush=True)

            open_position(state, product)

# =========================
# MAIN LOOP
# =========================

def main():

    print("[BOT] STARTED", flush=True)

    state = load_state()

    send("BOT STARTED SUCCESSFULLY")

    while True:

        try:

            calculate_equity(state)

            send(f"Heartbeat: Equity ${state['equity']:.2f}")

            print(
                f"[{datetime.now()}] Equity=${state['equity']:.2f} Cash=${state['cash']:.2f} Positions={len(state['positions'])}",
                flush=True
            )

            manage_positions(state)

            scanner(state)

            save_state(state)

            time.sleep(SCAN_INTERVAL)

        except Exception as e:

            print("[ERROR]", e, flush=True)

            send(f"ERROR {e}")

            time.sleep(10)

# =========================

if __name__ == "__main__":
    main()
