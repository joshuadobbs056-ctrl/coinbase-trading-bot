import os
import time
import json
import random
import requests
from datetime import datetime

# =========================
# ENV
# =========================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

STATE_FILE = os.getenv("STATE_FILE", "/data/state.json")

START_BALANCE = float(os.getenv("START_BALANCE", "1000"))

RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.02"))
MIN_SPEND = float(os.getenv("MIN_SPEND", "25"))
MAX_SPEND = float(os.getenv("MAX_SPEND", "50"))

MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "20"))

TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.02"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.015"))

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "20"))

UNIVERSE_SIZE = int(os.getenv("UNIVERSE_SIZE", "200"))
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "120"))

# =========================
# TELEGRAM
# =========================

def send(msg):

    print("[TELEGRAM]", msg, flush=True)

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
        print("Telegram error:", e, flush=True)

# =========================
# STATE
# =========================

def load_state():

    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except:
        return {
            "cash": START_BALANCE,
            "positions": {},
            "equity": START_BALANCE,
            "profit": 0
        }

def save_state(state):

    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)

    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

# =========================
# MARKET
# =========================

def get_products():

    try:
        r = requests.get(
            "https://api.exchange.coinbase.com/products",
            timeout=10
        )

        products = [
            p["id"]
            for p in r.json()
            if "USD" in p["id"] and p["status"] == "online"
        ]

        return random.sample(products, min(len(products), UNIVERSE_SIZE))

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

# =========================
# EQUITY / PROFIT
# =========================

def calculate_equity(state):

    total = state["cash"]

    for product, pos in state["positions"].items():

        price = get_price(product)

        if price:
            total += pos["size"] * price

    state["equity"] = total
    state["profit"] = total - START_BALANCE

    return total

# =========================
# POSITION
# =========================

def calculate_spend(state):

    equity = calculate_equity(state)

    spend = equity * RISK_PER_TRADE

    spend = max(spend, MIN_SPEND)
    spend = min(spend, MAX_SPEND)

    return spend

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
        "size": size
    }

    calculate_equity(state)

    send(
        f"BUY {product}\n"
        f"Entry: ${price:.5f}\n"
        f"Cash: ${state['cash']:.2f}\n"
        f"Equity: ${state['equity']:.2f}\n"
        f"Profit: ${state['profit']:.2f}"
    )

def close_position(state, product, price):

    pos = state["positions"][product]

    value = pos["size"] * price

    pnl = value - (pos["size"] * pos["entry"])

    state["cash"] += value

    del state["positions"][product]

    calculate_equity(state)

    send(
        f"SELL {product}\n"
        f"Exit: ${price:.5f}\n"
        f"P/L: ${pnl:.2f}\n"
        f"Cash: ${state['cash']:.2f}\n"
        f"Equity: ${state['equity']:.2f}\n"
        f"Profit: ${state['profit']:.2f}"
    )

def manage_positions(state):

    for product in list(state["positions"].keys()):

        price = get_price(product)

        if not price:
            continue

        entry = state["positions"][product]["entry"]

        change = (price - entry) / entry

        if change >= TAKE_PROFIT_PCT or change <= -STOP_LOSS_PCT:
            close_position(state, product, price)

# =========================
# SCANNER
# =========================

def scanner(state):

    products = get_products()

    if not products:
        return

    sample = random.sample(products, min(len(products), SAMPLE_SIZE))

    for product in sample:

        open_position(state, product)

# =========================
# MAIN
# =========================

def main():

    state = load_state()

    calculate_equity(state)

    send(
        f"BOT STARTED\n"
        f"Cash: ${state['cash']:.2f}\n"
        f"Equity: ${state['equity']:.2f}\n"
        f"Profit: ${state['profit']:.2f}"
    )

    while True:

        try:

            manage_positions(state)

            scanner(state)

            calculate_equity(state)

            save_state(state)

            print(
                f"{datetime.now()} "
                f"Equity ${state['equity']:.2f} "
                f"Profit ${state['profit']:.2f}",
                flush=True
            )

            time.sleep(SCAN_INTERVAL)

        except Exception as e:

            print("ERROR", e, flush=True)

            send(f"ERROR: {str(e)}")

            time.sleep(10)

# =========================

if __name__ == "__main__":
    main()
