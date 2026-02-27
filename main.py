import os
import time
import json
import requests
import statistics

# =========================
# CONFIG FROM ENVIRONMENT
# =========================

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

STATE_FILE = os.environ.get("STATE_FILE", "/data/state.json")

PAPER_START_BALANCE = float(os.environ.get("PAPER_START_BALANCE", 1000))
QUOTE_PER_TRADE_USD = float(os.environ.get("QUOTE_PER_TRADE_USD", 25))
MAX_POSITIONS = int(os.environ.get("MAX_POSITIONS", 1))

SCAN_INTERVAL_SECONDS = int(os.environ.get("SCAN_INTERVAL_SECONDS", 300))

TAKE_PROFIT_PCT = float(os.environ.get("TAKE_PROFIT_PCT", 1.6))
STOP_LOSS_PCT = float(os.environ.get("STOP_LOSS_PCT", -0.9))

PAPER_FEE_PCT = float(os.environ.get("PAPER_FEE_PCT", 0.004))

COINBASE_PRODUCTS_URL = "https://api.exchange.coinbase.com/products"
COINBASE_TICKER_URL = "https://api.exchange.coinbase.com/products/{}/ticker"


# =========================
# ENSURE DIRECTORY EXISTS
# =========================

def ensure_state_dir():
    directory = os.path.dirname(STATE_FILE)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


# =========================
# TELEGRAM
# =========================

def telegram_send(msg):

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
# STATE MANAGEMENT
# =========================

def load_state():

    ensure_state_dir()

    if not os.path.exists(STATE_FILE):
        return {
            "cash": PAPER_START_BALANCE,
            "positions": {},
            "realized_profit": 0,
            "fees_paid": 0
        }

    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except:
        return {
            "cash": PAPER_START_BALANCE,
            "positions": {},
            "realized_profit": 0,
            "fees_paid": 0
        }


def save_state(state):

    ensure_state_dir()

    tmp_file = STATE_FILE + ".tmp"

    with open(tmp_file, "w") as f:
        json.dump(state, f)

    os.replace(tmp_file, STATE_FILE)


# =========================
# COINBASE DATA
# =========================

def get_products():

    try:
        r = requests.get(COINBASE_PRODUCTS_URL, timeout=10)
        data = r.json()

        return [
            p["id"]
            for p in data
            if p["quote_currency"] == "USD"
            and p["status"] == "online"
        ]

    except:
        return []


def get_price(product_id):

    try:
        r = requests.get(
            COINBASE_TICKER_URL.format(product_id),
            timeout=5
        )

        return float(r.json()["price"])

    except:
        return None


# =========================
# SCORING LOGIC
# =========================

price_history = {}

def update_price_history(product_id, price):

    if product_id not in price_history:
        price_history[product_id] = []

    price_history[product_id].append(price)

    if len(price_history[product_id]) > 20:
        price_history[product_id].pop(0)


def calculate_score(product_id):

    if product_id not in price_history:
        return 0

    prices = price_history[product_id]

    if len(prices) < 10:
        return 0

    recent = prices[-5:]
    older = prices[:5]

    momentum = (statistics.mean(recent) - statistics.mean(older)) / statistics.mean(older) * 100

    volatility = statistics.stdev(prices) / statistics.mean(prices) * 100

    score = momentum + volatility

    return score


# =========================
# TRADING LOGIC
# =========================

def buy(state, product_id, price):

    if product_id in state["positions"]:
        return

    if len(state["positions"]) >= MAX_POSITIONS:
        return

    if state["cash"] < QUOTE_PER_TRADE_USD:
        return

    cost = QUOTE_PER_TRADE_USD
    fee = cost * PAPER_FEE_PCT
    size = (cost - fee) / price

    state["cash"] -= cost
    state["fees_paid"] += fee

    state["positions"][product_id] = {
        "size": size,
        "entry_price": price
    }

    save_state(state)

    telegram_send(
        f"<b>PAPER BUY</b>\n"
        f"{product_id}\n"
        f"Price: ${price:.4f}\n"
        f"Cash: ${state['cash']:.2f}"
    )


def sell(state, product_id, price):

    if product_id not in state["positions"]:
        return

    pos = state["positions"][product_id]

    value = pos["size"] * price
    fee = value * PAPER_FEE_PCT

    pnl = value - fee - (pos["size"] * pos["entry_price"])

    state["cash"] += value - fee
    state["fees_paid"] += fee
    state["realized_profit"] += pnl

    del state["positions"][product_id]

    save_state(state)

    telegram_send(
        f"<b>P
