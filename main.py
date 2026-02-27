import os
import time
import json
import requests
import statistics

# =========================
# ENV CONFIG
# =========================

STATE_FILE = os.getenv("STATE_FILE", "state.json")

SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "300"))
MAX_PRODUCTS = int(os.getenv("MAX_PRODUCTS", "250"))

TRADE_MODE = os.getenv("TRADE_MODE", "paper")
QUOTE_PER_TRADE_USD = float(os.getenv("QUOTE_PER_TRADE_USD", "100"))
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "1"))
MIN_SCORE_TO_BUY = int(os.getenv("MIN_SCORE_TO_BUY", "7"))

PAPER_START_BALANCE = float(os.getenv("PAPER_START_BALANCE", "1000"))
PAPER_FEE_PCT = float(os.getenv("PAPER_FEE_PCT", "0.004"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

BASE_URL = "https://api.coinbase.com"

# =========================
# TELEGRAM
# =========================

def telegram_send(text):

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(text)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML"
    }

    try:
        requests.post(url, json=payload, timeout=10)
    except:
        pass

# =========================
# STATE MANAGEMENT
# =========================

def load_state():

    if not os.path.exists(STATE_FILE):

        return {
            "cash": PAPER_START_BALANCE,
            "positions": {},
            "realized_profit": 0,
            "fees_paid": 0
        }

    with open(STATE_FILE, "r") as f:
        return json.load(f)


def save_state(state):

    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

# =========================
# COINBASE DATA
# =========================

def get_products():

    url = f"{BASE_URL}/api/v3/brokerage/market/products"

    r = requests.get(url, timeout=15)

    data = r.json()

    products = []

    for p in data["products"]:

        if p["quote_currency_id"] != "USD":
            continue

        if p["trading_disabled"]:
            continue

        products.append(p["product_id"])

    return products[:MAX_PRODUCTS]


def get_candles(product_id):

    url = f"{BASE_URL}/api/v3/brokerage/market/products/{product_id}/candles"

    params = {
        "granularity": "ONE_HOUR"
    }

    r = requests.get(url, params=params, timeout=15)

    data = r.json()

    candles = []

    for c in data["candles"]:

        candles.append(float(c["close"]))

    candles.reverse()

    return candles


def get_price(product_id):

    url = f"{BASE_URL}/api/v3/brokerage/market/products/{product_id}/ticker"

    r = requests.get(url, timeout=15)

    return float(r.json()["price"])

# =========================
# ACCUMULATION DETECTION
# =========================

def accumulation_score(prices):

    if len(prices) < 48:
        return 0

    high = max(prices[-48:])
    low = min(prices[-48:])

    range_pct = (high - low) / low * 100

    volatility = statistics.pstdev(prices[-24:]) / prices[-1] * 100

    score = 0

    if range_pct < 8:
        score += 4

    if volatility < 2:
        score += 3

    if prices[-1] > prices[-24]:
        score += 3

    return score

# =========================
# PAPER TRADING
# =========================

def buy(state, product_id, price):

    cost = QUOTE_PER_TRADE_USD

    if state["cash"] < cost:
        return

    fee = cost * PAPER_FEE_PCT

    size = (cost - fee) / price

    state["cash"] -= cost

    state["fees_paid"] += fee

    state["positions"][product_id] = {
        "size": size,
        "entry_price": price
    }

    telegram_send(
        f"<b>PAPER BUY</b>\n"
        f"{product_id}\n"
        f"Price: ${price:.4f}\n"
        f"Cash: ${state['cash']:.2f}"
    )


def sell(state, product_id, price):

    pos = state["positions"][product_id]

    value = pos["size"] * price

    fee = value * PAPER_FEE_PCT

    proceeds = value - fee

    cost = pos["size"] * pos["entry_price"]

    profit = proceeds - cost

    state["cash"] += proceeds

    state["realized_profit"] += profit

    state["fees_paid"] += fee

    del state["positions"][product_id]

    telegram_send(
        f"<b>PAPER SELL</b>\n"
        f"{product_id}\n"
        f"Profit: ${profit:.2f}\n"
        f"Cash: ${state['cash']:.2f}"
    )

# =========================
# EQUITY CALCULATION
# =========================

def equity(state):

    total = state["cash"]

    for pid in state["positions"]:

        price = get_price(pid)

        total += state["positions"][pid]["size"] * price

    return total

# =========================
# MAIN LOOP
# =========================

def main():

    state = load_state()

    telegram_send(
        f"<b>BOT STARTED</b>\n"
        f"Cash: ${state['cash']:.2f}"
    )

    while True:

        try:

            products = get_products()

            for pid in products:

                prices = get_candles(pid)

                score = accumulation_score(prices)

                if score >= MIN_SCORE_TO_BUY:

                    if pid not in state["positions"]:

                        if len(state["positions"]) < MAX_OPEN_POSITIONS:

                            price = prices[-1]

                            buy(state, pid, price)

                if pid in state["positions"]:

                    price = prices[-1]

                    entry = state["positions"][pid]["entry_price"]

                    change = (price - entry) / entry * 100

                    if change > 10 or change < -5:

                        sell(state, pid, price)

            save_state(state)

            eq = equity(state)

            telegram_send(
                f"Equity: ${eq:.2f} | Cash: ${state['cash']:.2f}"
            )

        except Exception as e:

            print(e)

        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":

    main()
