import os
import time
import json
import random
import requests

# ============================================
# ENV CONFIG
# ============================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

STATE_FILE = os.getenv("STATE_FILE", "/data/state.json")

START_BALANCE = float(os.getenv("START_BALANCE", "1000"))

SAVAGE_MODE = os.getenv("SAVAGE_MODE", "0") == "1"

STATUS_INTERVAL = int(os.getenv("STATUS_INTERVAL", "120"))
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "6" if SAVAGE_MODE else "12"))

# Risk settings
BASE_RISK = 0.03
SAVAGE_RISK = 0.06

MIN_SPEND = float(os.getenv("MIN_SPEND", "25"))
MAX_SPEND = float(os.getenv("MAX_SPEND", "300"))

MAX_OPEN_POSITIONS = 20 if SAVAGE_MODE else 12

STOP_LOSS_PCT = 0.02 if SAVAGE_MODE else 0.015
TRAILING_ACTIVATION = 0.015 if SAVAGE_MODE else 0.02
TRAILING_DISTANCE = 0.005 if SAVAGE_MODE else 0.006

COOLDOWN_SECONDS = 600 if SAVAGE_MODE else 1800

# Elite filters
MIN_TREND = 0.012 if SAVAGE_MODE else 0.015
MAX_SPREAD = 0.004 if SAVAGE_MODE else 0.0035
MIN_VOLUME_USD = 250000 if SAVAGE_MODE else 500000

# ============================================
# TELEGRAM
# ============================================

def send(msg):

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return

    try:

        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": msg
            },
            timeout=10
        )

    except Exception as e:
        print("Telegram error:", e)

# ============================================
# STATE MANAGEMENT
# ============================================

def load_state():

    try:

        with open(STATE_FILE, "r") as f:

            return json.load(f)

    except:

        return {

            "cash": START_BALANCE,
            "positions": {},
            "cooldowns": {},
            "start_balance": START_BALANCE

        }

def save_state(state):

    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)

    with open(STATE_FILE, "w") as f:

        json.dump(state, f)

# ============================================
# MARKET DATA
# ============================================

def get_products():

    try:

        r = requests.get(
            "https://api.exchange.coinbase.com/products",
            timeout=10
        )

        pairs = [

            p["id"]

            for p in r.json()

            if p["status"] == "online"
            and p["quote_currency"] == "USD"

        ]

        return random.sample(pairs, min(len(pairs), 150))

    except:

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

def get_spread(product):

    try:

        r = requests.get(
            f"https://api.exchange.coinbase.com/products/{product}/book?level=1",
            timeout=10
        )

        data = r.json()

        bid = float(data["bids"][0][0])
        ask = float(data["asks"][0][0])

        return (ask - bid) / bid

    except:

        return 999

def get_volume(product):

    try:

        r = requests.get(
            f"https://api.exchange.coinbase.com/products/{product}/stats",
            timeout=10
        )

        stats = r.json()

        return float(stats["volume"]) * float(stats["last"])

    except:

        return 0

def get_trend(product):

    try:

        r = requests.get(
            f"https://api.exchange.coinbase.com/products/{product}/candles?granularity=300",
            timeout=10
        )

        candles = r.json()

        closes = [c[4] for c in candles[:12]]

        return (closes[0] - closes[-1]) / closes[-1]

    except:

        return 0

# ============================================
# EQUITY CALCULATION
# ============================================

def equity(state):

    total = state["cash"]

    for product, pos in state["positions"].items():

        price = get_price(product)

        if price:

            total += pos["size"] * price

    return total

# ============================================
# RISK CALCULATION
# ============================================

def risk_percent(state):

    eq = equity(state)

    if SAVAGE_MODE:

        if eq > 5000:
            return 0.10

        if eq > 2000:
            return 0.08

        return SAVAGE_RISK

    return BASE_RISK

# ============================================
# ENTRY FILTER
# ============================================

def elite_entry(state, product):

    now = time.time()

    if product in state["cooldowns"]:

        if now < state["cooldowns"][product]:
            return False

    if get_spread(product) > MAX_SPREAD:
        return False

    if get_volume(product) < MIN_VOLUME_USD:
        return False

    if get_trend(product) < MIN_TREND:
        return False

    return True

# ============================================
# POSITION OPEN
# ============================================

def open_position(state, product):

    if product in state["positions"]:
        return

    if len(state["positions"]) >= MAX_OPEN_POSITIONS:
        return

    if not elite_entry(state, product):
        return

    price = get_price(product)

    if not price:
        return

    spend = equity(state) * risk_percent(state)

    spend = max(spend, MIN_SPEND)
    spend = min(spend, MAX_SPEND)

    if state["cash"] < spend:
        return

    size = spend / price

    state["cash"] -= spend

    state["positions"][product] = {

        "entry": price,
        "size": size,
        "highest": price,
        "trailing": False

    }

    eq = equity(state)
    profit = eq - state["start_balance"]

    send(
        f"BUY {product}\n"
        f"Price ${price:.5f}\n"
        f"Balance ${eq:.2f}\n"
        f"Profit ${profit:.2f}"
    )

# ============================================
# POSITION CLOSE
# ============================================

def close_position(state, product, price, reason):

    pos = state["positions"][product]

    value = pos["size"] * price

    pnl = value - (pos["entry"] * pos["size"])

    state["cash"] += value

    del state["positions"][product]

    state["cooldowns"][product] = time.time() + COOLDOWN_SECONDS

    eq = equity(state)
    profit = eq - state["start_balance"]

    send(
        f"SELL {product} ({reason})\n"
        f"P/L ${pnl:.2f}\n"
        f"Balance ${eq:.2f}\n"
        f"Profit ${profit:.2f}"
    )

# ============================================
# MANAGE POSITIONS
# ============================================

def manage_positions(state):

    for product in list(state["positions"].keys()):

        price = get_price(product)

        if not price:
            continue

        pos = state["positions"][product]

        entry = pos["entry"]

        change = (price - entry) / entry

        if price > pos["highest"]:
            pos["highest"] = price

        if change >= TRAILING_ACTIVATION:
            pos["trailing"] = True

        if pos["trailing"]:

            stop = pos["highest"] * (1 - TRAILING_DISTANCE)

            if price <= stop:

                close_position(state, product, price, "TRAIL")

                continue

        if change <= -STOP_LOSS_PCT:

            close_position(state, product, price, "STOP")

# ============================================
# MAIN LOOP
# ============================================

def main():

    state = load_state()

    send("SAVAGE MODE ACTIVE" if SAVAGE_MODE else "ELITE MODE ACTIVE")

    last_status = 0

    while True:

        try:

            now = time.time()

            manage_positions(state)

            for product in get_products():

                open_position(state, product)

            if now - last_status >= STATUS_INTERVAL:

                eq = equity(state)
                profit = eq - state["start_balance"]

                send(
                    f"STATUS\n"
                    f"Balance ${eq:.2f}\n"
                    f"Profit ${profit:.2f}\n"
                    f"Open trades {len(state['positions'])}"
                )

                last_status = now

            save_state(state)

            time.sleep(SCAN_INTERVAL)

        except Exception as e:

            print("Error:", e)

            time.sleep(5)

# ============================================

if __name__ == "__main__":

    main()
