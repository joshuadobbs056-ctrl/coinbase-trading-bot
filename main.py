import os
import time
import json
import requests
import statistics
import sys

# =========================
# CONFIG
# =========================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

STATE_FILE = os.getenv("STATE_FILE", "state.json")

START_BALANCE = float(os.getenv("PAPER_START_BALANCE", 1000))
TRADE_SIZE = float(os.getenv("QUOTE_PER_TRADE_USD", 25))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", 1))

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL_SECONDS", 60))

TAKE_PROFIT = float(os.getenv("TAKE_PROFIT_PCT", 1.6))
STOP_LOSS = float(os.getenv("STOP_LOSS_PCT", -0.9))

FEE = float(os.getenv("PAPER_FEE_PCT", 0.004))

print("BOOT: Starting bot...", flush=True)

# =========================
# TELEGRAM
# =========================

def telegram(msg):

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return

    try:

        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": msg,
                "parse_mode": "HTML"
            },
            timeout=10
        )

    except Exception as e:
        print("Telegram error:", e, flush=True)

# =========================
# STATE
# =========================

def ensure_dir():

    directory = os.path.dirname(STATE_FILE)

    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def load_state():

    ensure_dir()

    if not os.path.exists(STATE_FILE):

        print("BOOT: Creating new state", flush=True)

        return {
            "cash": START_BALANCE,
            "positions": {},
            "profit": 0
        }

    try:

        print("BOOT: Loading state", flush=True)

        with open(STATE_FILE, "r") as f:
            return json.load(f)

    except Exception as e:

        print("State load error:", e, flush=True)

        return {
            "cash": START_BALANCE,
            "positions": {},
            "profit": 0
        }

def save_state(state):

    ensure_dir()

    tmp = STATE_FILE + ".tmp"

    with open(tmp, "w") as f:
        json.dump(state, f)

    os.replace(tmp, STATE_FILE)

# =========================
# COINBASE
# =========================

def get_products():

    print("SCAN: Fetching products...", flush=True)

    try:

        r = requests.get(
            "https://api.exchange.coinbase.com/products",
            timeout=10
        )

        products = [
            p["id"]
            for p in r.json()
            if p["quote_currency"] == "USD"
            and p["status"] == "online"
        ]

        print(f"SCAN: Found {len(products)} products", flush=True)

        return products

    except Exception as e:

        print("Product fetch error:", e, flush=True)

        return []

def get_price(product):

    try:

        r = requests.get(
            f"https://api.exchange.coinbase.com/products/{product}/ticker",
            timeout=5
        )

        return float(r.json()["price"])

    except:
        return None

# =========================
# HISTORY
# =========================

history = {}

def update_history(product, price):

    if product not in history:
        history[product] = []

    history[product].append(price)

    if len(history[product]) > 20:
        history[product].pop(0)

def score(product):

    if product not in history:
        return 0

    h = history[product]

    if len(h) < 10:
        return 0

    recent = statistics.mean(h[-5:])
    older = statistics.mean(h[:5])

    momentum = (recent - older) / older * 100

    vol = statistics.stdev(h) / recent * 100

    return momentum + vol

# =========================
# TRADING
# =========================

def buy(state, product, price):

    print(f"TRADE: Buying {product}", flush=True)

    if product in state["positions"]:
        return

    if len(state["positions"]) >= MAX_POSITIONS:
        return

    if state["cash"] < TRADE_SIZE:
        return

    cost = TRADE_SIZE
    fee = cost * FEE
    size = (cost - fee) / price

    state["cash"] -= cost

    state["positions"][product] = {
        "size": size,
        "entry": price
    }

    save_state(state)

    telegram(f"PAPER BUY {product} @ ${price:.4f}")

def sell(state, product, price):

    print(f"TRADE: Selling {product}", flush=True)

    pos = state["positions"][product]

    value = pos["size"] * price
    fee = value * FEE
    entry_value = pos["size"] * pos["entry"]

    pnl = value - fee - entry_value

    state["cash"] += value - fee
    state["profit"] += pnl

    del state["positions"][product]

    save_state(state)

    telegram(f"PAPER SELL {product} PnL ${pnl:.2f}")

def manage(state):

    for product in list(state["positions"].keys()):

        price = get_price(product)

        if not price:
            continue

        entry = state["positions"][product]["entry"]

        change = (price - entry) / entry * 100

        if change >= TAKE_PROFIT or change <= STOP_LOSS:

            sell(state, product, price)

# =========================
# MAIN
# =========================

def main():

    state = load_state()

    telegram("BOT STARTED")

    products = get_products()

    if not products:

        print("ERROR: No products found", flush=True)

        return

    print("BOT: Entering main loop", flush=True)

    while True:

        print("SCAN: Running scan cycle...", flush=True)

        manage(state)

        best = None
        best_score = 0

        for p in products:

            price = get_price(p)

            if not price:
                continue

            update_history(p, price)

            s = score(p)

            if s > best_score:

                best_score = s
                best = (p, price)

        if best:

            buy(state, best[0], best[1])

        time.sleep(SCAN_INTERVAL)

# =========================

if __name__ == "__main__":

    main()
