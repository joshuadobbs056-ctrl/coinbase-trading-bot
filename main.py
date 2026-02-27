import os
import time
import json
import math
import requests
import traceback
from datetime import datetime

# =========================
# CONFIG FROM ENV
# =========================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TRADE_MODE = os.getenv("TRADE_MODE", "paper")

STATE_FILE = os.getenv("STATE_FILE", "/data/state.json")

START_BALANCE = float(os.getenv("START_BALANCE", "1000"))

MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "20"))

MIN_SPEND = float(os.getenv("MIN_SPEND", "25"))
MAX_SPEND = float(os.getenv("MAX_SPEND", "50"))

RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.02"))

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "20"))

TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.02"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.015"))

TRAILING_ACTIVATION = float(os.getenv("TRAILING_ACTIVATION", "0.02"))
TRAILING_DISTANCE = float(os.getenv("TRAILING_DISTANCE", "0.006"))

MIN_TREND_PCT_30M = float(os.getenv("MIN_TREND_PCT_30M", "0.009"))

MIN_VOL_SPIKE_MULT = float(os.getenv("MIN_VOL_SPIKE_MULT", "1.2"))

SPREAD_MAX = float(os.getenv("SPREAD_MAX", "0.004"))

UNIVERSE_SIZE = int(os.getenv("UNIVERSE_SIZE", "250"))

# =========================
# TELEGRAM
# =========================

def send(msg):

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return

    try:

        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

        requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": msg,
            "parse_mode": "HTML"
        })

    except:
        pass


# =========================
# STATE
# =========================

def load_state():

    if os.path.exists(STATE_FILE):

        with open(STATE_FILE, "r") as f:
            return json.load(f)

    return {
        "cash": START_BALANCE,
        "positions": [],
        "equity": START_BALANCE
    }


def save_state(state):

    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)

    tmp = STATE_FILE + ".tmp"

    with open(tmp, "w") as f:
        json.dump(state, f)

    os.replace(tmp, STATE_FILE)


# =========================
# COINBASE DATA
# =========================

def get_products():

    url = "https://api.exchange.coinbase.com/products"

    r = requests.get(url)

    products = r.json()

    usd = [p["id"] for p in products if "-USD" in p["id"]]

    return usd[:UNIVERSE_SIZE]


def get_price(product):

    url = f"https://api.exchange.coinbase.com/products/{product}/ticker"

    r = requests.get(url)

    return float(r.json()["price"])


def get_candles(product):

    url = f"https://api.exchange.coinbase.com/products/{product}/candles?granularity=60"

    r = requests.get(url)

    return r.json()


# =========================
# INDICATORS
# =========================

def ema(data, period):

    k = 2 / (period + 1)

    ema_val = data[0]

    for price in data:

        ema_val = price * k + ema_val * (1 - k)

    return ema_val


# =========================
# ENTRY SCORE
# =========================

def entry_signal(product):

    candles = get_candles(product)

    closes = [c[4] for c in candles]

    if len(closes) < 30:
        return False, None

    ema9 = ema(closes[-9:], 9)
    ema21 = ema(closes[-21:], 21)

    trend = (ema9 - ema21) / ema21

    if trend < MIN_TREND_PCT_30M:
        return False, None

    volume_now = candles[-1][5]
    volume_avg = sum(c[5] for c in candles[-20:]) / 20

    vol_mult = volume_now / volume_avg

    if vol_mult < MIN_VOL_SPIKE_MULT:
        return False, None

    return True, trend


# =========================
# POSITION SIZE
# =========================

def calc_spend(state):

    equity = state["equity"]

    spend = equity * RISK_PER_TRADE

    spend = max(spend, MIN_SPEND)

    spend = min(spend, MAX_SPEND)

    return spend


# =========================
# BUY
# =========================

def open_position(state, product):

    price = get_price(product)

    spend = calc_spend(state)

    if state["cash"] < spend:
        return

    size = spend / price

    pos = {

        "product": product,

        "entry": price,

        "size": size,

        "spend": spend,

        "peak": price,

        "time": time.time()

    }

    state["cash"] -= spend

    state["positions"].append(pos)

    send(
        f"<b>SNIPER BUY</b>\n"
        f"{product}\n"
        f"Entry: ${price:.5f}\n"
        f"Spend: ${spend:.2f}\n"
        f"Cash: ${state['cash']:.2f}"
    )


# =========================
# SELL LOGIC WITH TRAILING
# =========================

def manage_positions(state):

    for pos in state["positions"][:]:

        price = get_price(pos["product"])

        entry = pos["entry"]

        pnl = (price - entry) / entry

        if price > pos["peak"]:
            pos["peak"] = price

        peak = pos["peak"]

        drawdown = (peak - price) / peak

        if pnl <= -STOP_LOSS_PCT:

            close_position(state, pos, price, "STOP")

        elif pnl >= TRAILING_ACTIVATION and drawdown >= TRAILING_DISTANCE:

            close_position(state, pos, price, "TRAIL")


# =========================
# CLOSE POSITION
# =========================

def close_position(state, pos, price, reason):

    value = pos["size"] * price

    profit = value - pos["spend"]

    state["cash"] += value

    state["positions"].remove(pos)

    send(
        f"<b>SNIPER SELL ({reason})</b>\n"
        f"{pos['product']}\n"
        f"Exit: ${price:.5f}\n"
        f"P/L: ${profit:.2f}\n"
        f"Cash: ${state['cash']:.2f}"
    )


# =========================
# EQUITY
# =========================

def update_equity(state):

    total = state["cash"]

    for pos in state["positions"]:

        price = get_price(pos["product"])

        total += pos["size"] * price

    state["equity"] = total


# =========================
# LOOP
# =========================

def run():

    state = load_state()

    send(
        f"<b>SNIPER v4 STARTED</b>\n"
        f"Cash: ${state['cash']:.2f}\n"
        f"Equity: ${state['equity']:.2f}\n"
        f"Max pos: {MAX_OPEN_POSITIONS}"
    )

    products = get_products()

    while True:

        try:

            update_equity(state)

            manage_positions(state)

            if len(state["positions"]) < MAX_OPEN_POSITIONS:

                for product in products:

                    ok, trend = entry_signal(product)

                    if ok:

                        open_position(state, product)

                        break

            save_state(state)

            time.sleep(SCAN_INTERVAL)

        except Exception as e:

            send(f"ERROR: {e}")

            traceback.print_exc()

            time.sleep(5)


# =========================
# START
# =========================

run()
