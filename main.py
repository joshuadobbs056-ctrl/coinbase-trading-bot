import os
import time
import json
import requests
import statistics

# =========================
# ENV CONFIG
# =========================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

STATE_FILE = os.getenv("STATE_FILE", "state.json")

START_BALANCE = float(os.getenv("PAPER_START_BALANCE", 1000))
TRADE_SIZE = float(os.getenv("QUOTE_PER_TRADE_USD", 25))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", 1))

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL_SECONDS", 300))

TAKE_PROFIT = float(os.getenv("TAKE_PROFIT_PCT", 1.6))
STOP_LOSS = float(os.getenv("STOP_LOSS_PCT", -0.9))

FEE = float(os.getenv("PAPER_FEE_PCT", 0.004))

# =========================
# TELEGRAM
# =========================

def telegram(msg):

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    try:
        requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": msg,
            "parse_mode": "HTML"
        }, timeout=10)
    except:
        pass

# =========================
# STATE
# =========================

def ensure_dir():

    d = os.path.dirname(STATE_FILE)

    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def load_state():

    ensure_dir()

    if not os.path.exists(STATE_FILE):

        return {
            "cash": START_BALANCE,
            "positions": {},
            "profit": 0
        }

    try:

        with open(STATE_FILE, "r") as f:
            return json.load(f)

    except:

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
# COINBASE DATA
# =========================

def get_products():

    try:

        r = requests.get("https://api.exchange.coinbase.com/products", timeout=10)

        return [
            p["id"]
            for p in r.json()
            if p["quote_currency"] == "USD"
            and p["status"] == "online"
        ]

    except:

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
# SCORING
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

    telegram(
        f"<b>PAPER BUY</b>\n"
        f"{product}\n"
        f"Price: ${price:.4f}\n"
        f"Cash: ${state['cash']:.2f}"
    )

def sell(state, product, price):

    pos = state["positions"][product]

    value = pos["size"] * price

    fee = value * FEE

    entry_value = pos["size"] * pos["entry"]

    pnl = value - fee - entry_value

    state["cash"] += value - fee

    state["profit"] += pnl

    del state["positions"][product]

    save_state(state)

    telegram(
        f"<b>PAPER SELL</b>\n"
        f"{product}\n"
        f"PnL: ${pnl:.2f}\n"
        f"Cash: ${state['cash']:.2f}"
    )

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
# FIND TRADE
# =========================

def find_trade(state, products):

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

    if best and best_score > 0:

        buy(state, best[0], best[1])

# =========================
# MAIN LOOP
# =========================

def main():

    state = load_state()

    telegram(
        f"<b>ACTIVE BOT STARTED</b>\n"
        f"Cash: ${state['cash']:.2f}"
    )

    products = get_products()

    while True:

        manage(state)

        find_trade(state, products)

        time.sleep(SCAN_INTERVAL)

# =========================

if __name__ == "__main__":

    main()
