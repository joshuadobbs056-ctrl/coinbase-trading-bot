import os
import time
import json
import random
import requests
from datetime import datetime

# =========================
# ENV VARIABLES
# =========================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TRADE_MODE = os.getenv("TRADE_MODE", "paper")

STATE_FILE = os.getenv("STATE_FILE", "/data/state.json")

START_BALANCE = float(os.getenv("START_BALANCE", "1000"))

RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.02"))

MIN_SPEND = float(os.getenv("MIN_SPEND", "25"))
MAX_SPEND = float(os.getenv("MAX_SPEND", "50"))

MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "20"))

TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.02"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.015"))

MIN_TREND_PCT_30M = float(os.getenv("MIN_TREND_PCT_30M", "0.009"))
MIN_VOL_SPIKE_MULT = float(os.getenv("MIN_VOL_SPIKE_MULT", "1.2"))

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "20"))

SPREAD_MAX = float(os.getenv("SPREAD_MAX", "0.004"))

UNIVERSE_SIZE = int(os.getenv("UNIVERSE_SIZE", "250"))
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "160"))

# =========================
# TELEGRAM
# =========================

def send(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": msg
        }, timeout=10)
    except:
        pass

# =========================
# STATE
# =========================

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
        "equity": START_BALANCE
    }

def save_state(state):
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, STATE_FILE)
    except:
        pass

# =========================
# MARKET DATA
# =========================

def get_products():
    try:
        r = requests.get("https://api.exchange.coinbase.com/products", timeout=10)
        data = r.json()

        usd_pairs = [
            p["id"]
            for p in data
            if ("USD" in p["id"] or "USDC" in p["id"])
            and p.get("status") == "online"
        ]

        return random.sample(usd_pairs, min(len(usd_pairs), UNIVERSE_SIZE))

    except:
        return []

def get_price(product):
    try:
        url = f"https://api.exchange.coinbase.com/products/{product}/ticker"
        r = requests.get(url, timeout=10)

        if r.status_code != 200:
            return None

        data = r.json()

        if "price" not in data:
            return None

        return float(data["price"])

    except:
        return None

# =========================
# TREND DETECTION (SNIPER)
# =========================

def get_trend_score(product):

    try:

        url = f"https://api.exchange.coinbase.com/products/{product}/candles?granularity=300"
        r = requests.get(url, timeout=10)

        if r.status_code != 200:
            return None

        candles = r.json()

        if len(candles) < 10:
            return None

        closes = [c[4] for c in candles]

        recent = closes[0]
        past = closes[6]

        trend = (recent - past) / past

        if trend < MIN_TREND_PCT_30M:
            return None

        volume_spike = random.uniform(1.0, 2.0)

        if volume_spike < MIN_VOL_SPIKE_MULT:
            return None

        spread = random.uniform(0.0001, 0.005)

        if spread > SPREAD_MAX:
            return None

        score = trend * volume_spike * 100

        return score

    except:
        return None

# =========================
# POSITION MANAGEMENT
# =========================

def calculate_spend(state):

    equity = calculate_equity(state)

    spend = equity * RISK_PER_TRADE

    spend = max(spend, MIN_SPEND)
    spend = min(spend, MAX_SPEND)

    return spend

def open_position(state, product):

    if len(state["positions"]) >= MAX_OPEN_POSITIONS:
        return

    price = get_price(product)

    if price is None:
        return

    spend = calculate_spend(state)

    if state["cash"] < spend:
        return

    size = spend / price

    state["cash"] -= spend

    state["positions"][product] = {
        "entry": price,
        "size": size,
        "time": time.time()
    }

    send(
        f"PAPER BUY — {product}\n"
        f"Entry: ${price:.5f}\n"
        f"Spend: ${spend:.2f}\n"
        f"Equity: ${calculate_equity(state):.2f}\n"
        f"Open: {len(state['positions'])}/{MAX_OPEN_POSITIONS}"
    )

def close_position(state, product, price):

    pos = state["positions"][product]

    value = pos["size"] * price

    state["cash"] += value

    pnl = value - (pos["size"] * pos["entry"])

    del state["positions"][product]

    send(
        f"SELL — {product}\n"
        f"Exit: ${price:.5f}\n"
        f"P/L: ${pnl:.2f}\n"
        f"Equity: ${calculate_equity(state):.2f}"
    )

def manage_positions(state):

    for product in list(state["positions"].keys()):

        price = get_price(product)

        if price is None:
            continue

        pos = state["positions"][product]

        change = (price - pos["entry"]) / pos["entry"]

        if change >= TAKE_PROFIT_PCT or change <= -STOP_LOSS_PCT:
            close_position(state, product, price)

# =========================
# EQUITY
# =========================

def calculate_equity(state):

    total = state["cash"]

    for product, pos in state["positions"].items():

        price = get_price(product)

        if price is None:
            continue

        total += pos["size"] * price

    state["equity"] = total

    return total

# =========================
# SCANNER
# =========================

def scanner(state):

    products = get_products()

    if not products:
        return

    sample = random.sample(products, min(len(products), SAMPLE_SIZE))

    for product in sample:

        score = get_trend_score(product)

        if score is None:
            continue

        open_position(state, product)

# =========================
# MAIN LOOP
# =========================

def main():

    state = load_state()

    send(
        f"SCALPER SNIPER STARTED\n"
        f"Mode: {TRADE_MODE}\n"
        f"Cash: ${state['cash']:.2f}\n"
        f"Max open: {MAX_OPEN_POSITIONS}\n"
        f"TP/SL: {TAKE_PROFIT_PCT*100:.2f}% / {STOP_LOSS_PCT*100:.2f}%"
    )

    while True:

        try:

            manage_positions(state)

            scanner(state)

            calculate_equity(state)

            save_state(state)

            time.sleep(SCAN_INTERVAL)

        except Exception as e:

            send(f"Error: {str(e)}")

            time.sleep(10)

# =========================

if __name__ == "__main__":
    main()
