import os
import time
import json
import random
import requests

# ============================================
# ENV VARIABLES
# ============================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

STATE_FILE = os.getenv("STATE_FILE", "/data/state.json")

START_BALANCE = float(os.getenv("START_BALANCE", "1000"))

RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.02"))

MIN_SPEND = float(os.getenv("MIN_SPEND", "25"))
MAX_SPEND = float(os.getenv("MAX_SPEND", "100"))

MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "20"))

TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.02"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.015"))

TRAILING_ACTIVATION = float(os.getenv("TRAILING_ACTIVATION", "0.02"))
TRAILING_DISTANCE = float(os.getenv("TRAILING_DISTANCE", "0.006"))

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "20"))

HEARTBEAT_INTERVAL = 300

MIN_TREND_PCT = float(os.getenv("MIN_TREND_PCT_30M", "0.009"))

# ============================================
# TELEGRAM
# ============================================

def send(msg):

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return

    try:

        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

        r = requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": msg
            },
            timeout=10
        )

        print("[TELEGRAM]", r.text)

    except Exception as e:

        print("Telegram error:", e)

# ============================================
# STATE
# ============================================

def load_state():

    try:

        if os.path.exists(STATE_FILE):

            with open(STATE_FILE, "r") as f:

                return json.load(f)

    except:
        pass

    return {

        "cash": START_BALANCE,
        "positions": {},
        "start_balance": START_BALANCE,
        "last_heartbeat": 0

    }

def save_state(state):

    try:

        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)

        with open(STATE_FILE, "w") as f:

            json.dump(state, f)

    except:
        pass

# ============================================
# MARKET DATA
# ============================================

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

            if ("USD" in p["id"]) and p["status"] == "online"

        ]

        return random.sample(pairs, min(len(pairs), 200))

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

# ============================================
# EQUITY
# ============================================

def calculate_equity(state):

    total = state["cash"]

    for product, pos in state["positions"].items():

        price = get_price(product)

        if price:

            total += pos["size"] * price

    return total

# ============================================
# POSITION SIZE
# ============================================

def calculate_spend(state):

    equity = calculate_equity(state)

    spend = equity * RISK_PER_TRADE

    spend = max(spend, MIN_SPEND)
    spend = min(spend, MAX_SPEND)

    return spend

# ============================================
# OPEN POSITION
# ============================================

def open_position(state, product):

    if product in state["positions"]:
        return

    if len(state["positions"]) >= MAX_OPEN_POSITIONS:
        return

    price = get_price(product)

    if not price:
        return

    spend = calculate_spend(state)

    if state["cash"] < spend:
        return

    size = spend / price

    state["cash"] -= spend

    state["positions"][product] = {

        "entry": price,
        "size": size,
        "highest": price,
        "trailing_active": False

    }

    equity = calculate_equity(state)
    profit = equity - state["start_balance"]

    send(
        f"BUY {product}\n"
        f"Price ${price:.5f}\n"
        f"Balance ${equity:.2f}\n"
        f"Profit ${profit:.2f}"
    )

# ============================================
# CLOSE POSITION
# ============================================

def close_position(state, product, price, reason):

    pos = state["positions"][product]

    value = pos["size"] * price

    state["cash"] += value

    pnl = value - (pos["entry"] * pos["size"])

    del state["positions"][product]

    equity = calculate_equity(state)
    profit = equity - state["start_balance"]

    send(
        f"SELL {product}\n"
        f"Price ${price:.5f}\n"
        f"P/L ${pnl:.2f}\n"
        f"Balance ${equity:.2f}\n"
        f"Profit ${profit:.2f}\n"
        f"{reason}"
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

            pos["trailing_active"] = True

        if pos["trailing_active"]:

            trail_stop = pos["highest"] * (1 - TRAILING_DISTANCE)

            if price <= trail_stop:

                close_position(state, product, price, "TRAIL STOP")

                continue

        if change >= TAKE_PROFIT_PCT:

            close_position(state, product, price, "TAKE PROFIT")

            continue

        if change <= -STOP_LOSS_PCT:

            close_position(state, product, price, "STOP LOSS")

# ============================================
# SCANNER
# ============================================

def scanner(state):

    products = get_products()

    for product in products:

        price = get_price(product)

        if not price:
            continue

        score = random.random()

        if score > 0.995:

            open_position(state, product)

# ============================================
# HEARTBEAT
# ============================================

def heartbeat(state):

    now = time.time()

    if now - state["last_heartbeat"] < HEARTBEAT_INTERVAL:
        return

    state["last_heartbeat"] = now

    equity = calculate_equity(state)

    profit = equity - state["start_balance"]

    send(
        f"STATUS\n"
        f"Balance ${equity:.2f}\n"
        f"Profit ${profit:.2f}\n"
        f"Open trades {len(state['positions'])}"
    )

# ============================================
# MAIN LOOP
# ============================================

def main():

    state = load_state()

    send("BOT STARTED")

    while True:

        try:

            manage_positions(state)

            scanner(state)

            heartbeat(state)

            save_state(state)

            time.sleep(SCAN_INTERVAL)

        except Exception as e:

            print("Error:", e)

            time.sleep(10)

# ============================================

if __name__ == "__main__":

    main()
