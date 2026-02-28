import os
import time
import json
import csv
import requests
from typing import Dict, List

import numpy as np

#########################################
# VARIABLES
#########################################

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 5))
STATUS_INTERVAL = int(os.getenv("STATUS_INTERVAL", 60))

COINS = os.getenv("COINS", "AUTO")
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", 200))

START_BALANCE = float(os.getenv("START_BALANCE", 1000))

MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", 50))

MIN_POSITION_SIZE_PERCENT = float(os.getenv("MIN_POSITION_SIZE_PERCENT", 0.5))

STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", 1.8))

TRAILING_START_PERCENT = float(os.getenv("TRAILING_START_PERCENT", 0.18))
TRAILING_DISTANCE_PERCENT = float(os.getenv("TRAILING_DISTANCE_PERCENT", 0.12))

MAX_TRADE_DURATION_MINUTES = int(os.getenv("MAX_TRADE_DURATION_MINUTES", 12))

MIN_SCORE = float(os.getenv("MIN_SCORE", 2))

CANDLE_GRANULARITY = 60
CANDLE_POINTS = 60

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

BASE_URL = "https://api.exchange.coinbase.com"

#########################################
# TELEGRAM
#########################################

session = requests.Session()

def notify(msg):

    print(msg, flush=True)

    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:

        try:

            session.post(

                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",

                json={"chat_id": TELEGRAM_CHAT_ID, "text": msg},

                timeout=10

            )

        except:

            pass

#########################################
# LEARNING STATE
#########################################

LEARNING_FILE = "learning.json"

def load_learning():

    if os.path.exists(LEARNING_FILE):

        with open(LEARNING_FILE,"r") as f:

            return json.load(f)

    return {

        "trade_count":0,
        "win_count":0,
        "loss_count":0,
        "total_profit":0

    }

def save_learning():

    with open(LEARNING_FILE,"w") as f:

        json.dump(learning,f)

learning = load_learning()

#########################################
# AUTO COINS
#########################################

def get_auto_coins():

    try:

        r=session.get(f"{BASE_URL}/products",timeout=15)

        data=r.json()

        coins=[p["id"] for p in data if p.get("quote_currency")=="USD"]

        return coins[:MAX_SYMBOLS]

    except:

        return ["BTC-USD","ETH-USD"]

COIN_LIST = get_auto_coins() if COINS=="AUTO" else COINS.split(",")

#########################################
# MARKET DATA
#########################################

def get_price(sym):

    try:

        r=session.get(f"{BASE_URL}/products/{sym}/ticker",timeout=10)

        data=r.json()

        if "price" not in data:

            return None

        return float(data["price"])

    except:

        return None

def get_candles(sym):

    try:

        r=session.get(

            f"{BASE_URL}/products/{sym}/candles",

            params={"granularity":CANDLE_GRANULARITY},

            timeout=10

        )

        data=r.json()

        data.reverse()

        return data[-CANDLE_POINTS:]

    except:

        return None

#########################################
# INDICATORS
#########################################

def rsi(closes):

    if len(closes)<15: return 50

    d=np.diff(closes)

    gain=np.maximum(d,0)

    loss=-np.minimum(d,0)

    ag=np.mean(gain[-14:])
    al=np.mean(loss[-14:])

    if al==0: return 100

    rs=ag/al

    return 100-(100/(1+rs))

def trend(closes):

    return (closes[-1]-np.mean(closes[-20:]))/closes[-20]

def momentum(closes):

    return (closes[-1]-closes[-5])/closes[-5]

def vol_ratio(vols):

    return vols[-1]/np.mean(vols[-20:])

#########################################
# SCORE
#########################################

def score(r,v,t,m):

    s=0

    if r>55: s+=1
    if v>1.1: s+=1
    if t>0: s+=1
    if m>0: s+=1

    return s

#########################################
# PORTFOLIO
#########################################

cash=START_BALANCE
open_trades=[]

#########################################
# POSITION SIZE
#########################################

def position_size():

    return cash*(MIN_POSITION_SIZE_PERCENT/100)

#########################################
# CLOSE TRADE WITH WIN/LOSS STATS
#########################################

def close_trade(t,price,reason):

    global cash

    proceeds=t["qty"]*price

    profit=proceeds-(t["qty"]*t["entry"])

    cash+=proceeds

    learning["trade_count"]+=1
    learning["total_profit"]+=profit

    if profit>0:
        learning["win_count"]+=1
    else:
        learning["loss_count"]+=1

    save_learning()

    trades=learning["trade_count"]
    wins=learning["win_count"]
    losses=learning["loss_count"]

    winrate=(wins/trades)*100 if trades>0 else 0

    notify(
f"""SELL {t['sym']} ({reason})
Profit: {profit:.4f}
Balance: {cash:.2f}

Trades: {trades}
Wins: {wins}
Losses: {losses}
Winrate: {winrate:.1f}%"""
)

#########################################
# OPEN TRADE
#########################################

def open_trade(sym,price):

    global cash

    if len(open_trades)>=MAX_OPEN_TRADES: return

    size=position_size()

    if size<=0: return

    qty=size/price

    cash-=size

    trade={

        "sym":sym,
        "entry":price,
        "qty":qty,
        "time":time.time(),
        "peak":price,
        "trail":None,
        "active":False,
        "stop":price*(1-STOP_LOSS_PERCENT/100)

    }

    open_trades.append(trade)

    notify(f"BUY {sym} | Balance {cash:.2f}")

#########################################
# MANAGE
#########################################

def manage(prices):

    global open_trades

    remaining=[]

    for t in open_trades:

        p=prices.get(t["sym"])

        if not p:
            remaining.append(t)
            continue

        if p>t["peak"]:
            t["peak"]=p

        if not t["active"] and p>=t["entry"]*(1+TRAILING_START_PERCENT/100):

            t["active"]=True

        if t["active"]:

            t["trail"]=t["peak"]*(1-TRAILING_DISTANCE_PERCENT/100)

        age=(time.time()-t["time"])/60

        profit=(p-t["entry"])/t["entry"]*100

        if age>=MAX_TRADE_DURATION_MINUTES and profit<=0:

            close_trade(t,p,"STAGNATION")
            continue

        if p<=t["stop"]:

            close_trade(t,p,"STOP")
            continue

        if t["trail"] and p<=t["trail"]:

            close_trade(t,p,"TRAIL")
            continue

        remaining.append(t)

    open_trades=remaining

#########################################
# MAIN LOOP
#########################################

notify("SAVAGE MODE STARTED")

last_status=time.time()

while True:

    try:

        prices={}

        for sym in COIN_LIST:

            price=get_price(sym)

            if price:
                prices[sym]=price

        manage(prices)

        for sym,price in prices.items():

            if sym in [t["sym"] for t in open_trades]:
                continue

            candles=get_candles(sym)

            if not candles:
                continue

            closes=[c[4] for c in candles]
            vols=[c[5] for c in candles]

            sc=score(
                rsi(closes),
                vol_ratio(vols),
                trend(closes),
                momentum(closes)
            )

            if sc>=MIN_SCORE:

                open_trade(sym,price)

        if time.time()-last_status>STATUS_INTERVAL:

            notify(f"STATUS Balance {cash:.2f} Open {len(open_trades)}")

            last_status=time.time()

        time.sleep(SCAN_INTERVAL)

    except Exception as e:

        notify(f"ERROR {e}")
        time.sleep(5)
