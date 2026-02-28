import os
import time
import json
import csv
import requests
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier

#########################################
# VARIABLES
#########################################

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 30))
STATUS_INTERVAL = int(os.getenv("STATUS_INTERVAL", 120))

COINS = os.getenv("COINS", "BTC-USD,ETH-USD,SOL-USD").replace(" ", "")
COIN_LIST = [c for c in COINS.split(",") if c]

START_BALANCE = float(os.getenv("START_BALANCE", 1000))

MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", 6))

MIN_POSITION_SIZE_PERCENT = float(os.getenv("MIN_POSITION_SIZE_PERCENT", 5))
MAX_POSITION_SIZE_PERCENT = float(os.getenv("MAX_POSITION_SIZE_PERCENT", 25))

MIN_CASH_RESERVE_PERCENT = float(os.getenv("MIN_CASH_RESERVE_PERCENT", 5))

STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", 2.0))
TRAILING_START_PERCENT = float(os.getenv("TRAILING_START_PERCENT", 1.2))
TRAILING_DISTANCE_PERCENT = float(os.getenv("TRAILING_DISTANCE_PERCENT", 0.7))

MAX_TRADE_DURATION_MINUTES = int(os.getenv("MAX_TRADE_DURATION_MINUTES", 20))
MIN_PROFIT_KEEP_PERCENT = float(os.getenv("MIN_PROFIT_KEEP_PERCENT", 0.5))

REPLACE_WITH_BETTER = os.getenv("REPLACE_WITH_BETTER", "true").lower() == "true"
REPLACE_SCORE_MARGIN = float(os.getenv("REPLACE_SCORE_MARGIN", 0.75))

MIN_SCORE = float(os.getenv("MIN_SCORE", 3.0))

CANDLE_GRANULARITY = int(os.getenv("CANDLE_GRANULARITY", 60))
CANDLE_POINTS = int(os.getenv("CANDLE_POINTS", 60))

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

BASE_URL = "https://api.exchange.coinbase.com"

#########################################
# FILES
#########################################

DATA_DIR = "."
LEARNING_FILE = os.path.join(DATA_DIR, "learning.json")
HISTORY_FILE = os.path.join(DATA_DIR, "trade_history.csv")

#########################################
# TELEGRAM
#########################################

def notify(msg):
    print(msg, flush=True)

    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
                timeout=10
            )
        except:
            pass

#########################################
# LEARNING MEMORY
#########################################

def load_learning():

    if os.path.exists(LEARNING_FILE):

        with open(LEARNING_FILE,"r") as f:

            return json.load(f)

    return {

        "trade_count":0,

        "win_count":0,

        "total_profit":0

    }

def save_learning():

    with open(LEARNING_FILE,"w") as f:

        json.dump(learning,f)

learning=load_learning()

#########################################
# HISTORY FILE
#########################################

if not os.path.exists(HISTORY_FILE):

    with open(HISTORY_FILE,"w",newline="") as f:

        writer=csv.writer(f)

        writer.writerow(["profit"])

#########################################
# MARKET DATA
#########################################

def get_ticker(sym):

    try:

        r=requests.get(

            f"{BASE_URL}/products/{sym}/ticker",

            timeout=10

        )

        return r.json()

    except:

        return None

def get_candles(sym):

    try:

        r=requests.get(

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

def calc_rsi(closes):

    if len(closes)<15:

        return 50

    deltas=np.diff(closes)

    gains=np.maximum(deltas,0)

    losses=-np.minimum(deltas,0)

    avg_gain=np.mean(gains[-14:])

    avg_loss=np.mean(losses[-14:])

    if avg_loss==0:

        return 100

    rs=avg_gain/avg_loss

    return 100-(100/(1+rs))

def ema(vals,period=20):

    if len(vals)<period:

        return np.mean(vals)

    k=2/(period+1)

    e=vals[0]

    for v in vals:

        e=v*k+e*(1-k)

    return e

def trend_strength(closes):

    e=ema(closes)

    return (closes[-1]-e)/e

def momentum(closes):

    return (closes[-1]-closes[-5])/closes[-5]

def volume_ratio(vols):

    return vols[-1]/np.mean(vols[-20:])

#########################################
# SCORING
#########################################

def score_trade(rsi,vol,trend,mom):

    score=0

    if 55<=rsi<=75: score+=1

    if vol>1.3: score+=1

    if trend>0: score+=1

    if mom>0: score+=1

    return score

#########################################
# PORTFOLIO
#########################################

cash=START_BALANCE

open_trades=[]

#########################################
# DYNAMIC POSITION SIZING
#########################################

def reserve():

    return cash*(MIN_CASH_RESERVE_PERCENT/100)

def dynamic_position_percent():

    trades=learning["trade_count"]

    wins=learning["win_count"]

    if trades<5:

        return MIN_POSITION_SIZE_PERCENT

    win_rate=wins/trades

    growth=(cash-START_BALANCE)/START_BALANCE

    percent=MIN_POSITION_SIZE_PERCENT

    if win_rate>0.55: percent+=3

    if win_rate>0.60: percent+=4

    if win_rate>0.65: percent+=5

    if growth>0.05: percent+=3

    if growth>0.10: percent+=4

    if growth>0.20: percent+=5

    return min(percent,MAX_POSITION_SIZE_PERCENT)

def position_size():

    percent=dynamic_position_percent()

    size=cash*(percent/100)

    return min(size,cash-reserve())

#########################################
# CLOSE TRADE
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

    save_learning()

    notify(

f"""SELL {t["sym"]} ({reason})
Entry {t["entry"]:.2f}
Exit {price:.2f}
Profit {profit:.2f}
Cash {cash:.2f}"""
)

#########################################
# OPEN TRADE
#########################################

def open_trade(sym,price,score):

    global cash

    size=position_size()

    if size<=0:

        return

    qty=size/price

    cash-=size

    trade={

        "sym":sym,

        "entry":price,

        "qty":qty,

        "score":score,

        "time":time.time(),

        "stop":price*(1-STOP_LOSS_PERCENT/100),

        "peak":price,

        "trail":None

    }

    open_trades.append(trade)

    notify(

f"""BUY {sym}
Score {score}
Size {size:.2f}
Cash {cash:.2f}"""
)

#########################################
# REPLACE WEAK TRADE
#########################################

def replace_if_better(sym,price,score):

    if len(open_trades)<MAX_OPEN_TRADES:

        return False

    weakest=min(open_trades,key=lambda x:x["score"])

    if score<weakest["score"]+REPLACE_SCORE_MARGIN:

        return False

    close_trade(weakest,price,"REPLACED")

    open_trades.remove(weakest)

    open_trade(sym,price,score)

    return True

#########################################
# MANAGE TRADES
#########################################

def manage_trades(prices):

    global open_trades

    remaining=[]

    for t in open_trades:

        p=prices[t["sym"]]

        if p>t["peak"]:

            t["peak"]=p

        if p>=t["entry"]*(1+TRAILING_START_PERCENT/100):

            t["trail"]=t["peak"]*(1-TRAILING_DISTANCE_PERCENT/100)

        age=(time.time()-t["time"])/60

        profit=(p-t["entry"])/t["entry"]*100

        if age>MAX_TRADE_DURATION_MINUTES and profit<MIN_PROFIT_KEEP_PERCENT:

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
# STATUS
#########################################

def send_status():

    percent=dynamic_position_percent()

    notify(

f"""STATUS
Cash {cash:.2f}
Open trades {len(open_trades)}
Position size {percent:.1f}%
Profit {learning["total_profit"]:.2f}
Trades {learning["trade_count"]}
Wins {learning["win_count"]}
"""
)

#########################################
# MAIN LOOP
#########################################

notify("BOT STARTED")

last_status=time.time()

while True:

    try:

        prices={}

        for sym in COIN_LIST:

            tick=get_ticker(sym)

            if tick:

                prices[sym]=float(tick["price"])

        manage_trades(prices)

        for sym in COIN_LIST:

            candles=get_candles(sym)

            if not candles:

                continue

            closes=[c[4] for c in candles]

            vols=[c[5] for c in candles]

            rsi=calc_rsi(closes)

            vol=volume_ratio(vols)

            trend=trend_strength(closes)

            mom=momentum(closes)

            score=score_trade(rsi,vol,trend,mom)

            price=prices[sym]

            if score>=MIN_SCORE:

                if sym not in [t["sym"] for t in open_trades]:

                    if not replace_if_better(sym,price,score):

                        if len(open_trades)<MAX_OPEN_TRADES:

                            open_trade(sym,price,score)

        if time.time()-last_status>STATUS_INTERVAL:

            send_status()

            last_status=time.time()

        time.sleep(SCAN_INTERVAL)

    except Exception as e:

        notify(f"ERROR {e}")

        time.sleep(5)
