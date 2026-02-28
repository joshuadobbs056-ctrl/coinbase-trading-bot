import os
import time
import json
import csv
import requests
from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

#########################################
# VARIABLES
#########################################

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 10))
STATUS_INTERVAL = int(os.getenv("STATUS_INTERVAL", 60))

COINS = os.getenv("COINS", "AUTO")
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", 120))

PAPER_TRADING = True

START_BALANCE = float(os.getenv("START_BALANCE", 1000))
MIN_CASH_RESERVE_PERCENT = float(os.getenv("MIN_CASH_RESERVE_PERCENT", 2))
MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", 30))

MIN_POSITION_SIZE_PERCENT = float(os.getenv("MIN_POSITION_SIZE_PERCENT", 1))
MAX_POSITION_SIZE_PERCENT = float(os.getenv("MAX_POSITION_SIZE_PERCENT", 5))

STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", 2.0))

TRAILING_START_PERCENT = float(os.getenv("TRAILING_START_PERCENT", 0.25))
TRAILING_DISTANCE_PERCENT = float(os.getenv("TRAILING_DISTANCE_PERCENT", 0.20))

MAX_TRADE_DURATION_MINUTES = int(os.getenv("MAX_TRADE_DURATION_MINUTES", 20))

MIN_SCORE = float(os.getenv("MIN_SCORE", 2.0))

CANDLE_GRANULARITY = int(os.getenv("CANDLE_GRANULARITY", 60))
CANDLE_POINTS = int(os.getenv("CANDLE_POINTS", 60))

ML_ENABLED = True
ML_MIN_TRADES = 50

BASE_URL = "https://api.exchange.coinbase.com"

#########################################
# FILES
#########################################

LEARNING_FILE = "learning.json"
HISTORY_FILE = "trade_history.csv"

#########################################
# LEARNING STATE
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

        writer.writerow(["rsi","volume_ratio","trend_strength","momentum","profit"])

#########################################
# HTTP
#########################################

session=requests.Session()

#########################################
# COIN DISCOVERY
#########################################

def get_auto_coins():

    try:

        r=session.get(f"{BASE_URL}/products",timeout=15)

        data=r.json()

        coins=[]

        for p in data:

            if p["quote_currency"]=="USD":

                coins.append(p["id"])

        return coins[:MAX_SYMBOLS]

    except:

        return ["BTC-USD","ETH-USD","SOL-USD"]

if COINS=="AUTO":

    COIN_LIST=get_auto_coins()

else:

    COIN_LIST=COINS.split(",")

#########################################
# MARKET DATA
#########################################

def get_ticker(sym):

    try:

        r=session.get(f"{BASE_URL}/products/{sym}/ticker",timeout=10)

        return r.json()

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

    if vol>1.2: score+=1

    if trend>0: score+=1

    if mom>0: score+=1

    return score

#########################################
# ML MODEL
#########################################

model=None

def train_model():

    global model

    try:

        data=np.loadtxt(HISTORY_FILE,delimiter=",",skiprows=1)

        if len(data)<ML_MIN_TRADES:

            return

        X=data[:,:-1]

        y=(data[:,-1]>0).astype(int)

        model=RandomForestClassifier(n_estimators=200)

        model.fit(X,y)

        print("ML MODEL TRAINED")

    except:

        pass

def ml_allows(features):

    if model is None:

        return True

    prob=model.predict_proba([features])[0][1]

    return prob>=0.55

#########################################
# PORTFOLIO
#########################################

cash=START_BALANCE

open_trades=[]

#########################################
# POSITION SIZE
#########################################

def reserve():

    return cash*(MIN_CASH_RESERVE_PERCENT/100)

def position_size():

    percent=MIN_POSITION_SIZE_PERCENT

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

    with open(HISTORY_FILE,"a",newline="") as f:

        writer=csv.writer(f)

        writer.writerow([

            t["rsi"],

            t["vol"],

            t["trend"],

            t["mom"],

            profit

        ])

    print("SELL",t["sym"],reason,"Profit",profit,"Cash",cash)

#########################################
# OPEN TRADE
#########################################

def open_trade(sym,price,score,features):

    global cash

    if len(open_trades)>=MAX_OPEN_TRADES:

        return

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

        "rsi":features[0],
        "vol":features[1],
        "trend":features[2],
        "mom":features[3],

        "stop":price*(1-STOP_LOSS_PERCENT/100),
        "peak":price,
        "trail":None,
        "trailing_active":False

    }

    open_trades.append(trade)

    print("BUY",sym,"Score",score,"Cash",cash)

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

        if not t["trailing_active"]:

            if p>=t["entry"]*(1+TRAILING_START_PERCENT/100):

                t["trailing_active"]=True

        if t["trailing_active"]:

            t["trail"]=t["peak"]*(1-TRAILING_DISTANCE_PERCENT/100)

        age_min=(time.time()-t["time"])/60

        profit_pct=(p-t["entry"])/t["entry"]*100

        # FIXED STAGNATION LOGIC (ONLY EXIT LOSERS)
        if age_min>=MAX_TRADE_DURATION_MINUTES and profit_pct<=0:

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

print("BOT STARTED")

last_status=time.time()

train_model()

while True:

    try:

        prices={}

        for sym in COIN_LIST:

            tick=get_ticker(sym)

            if tick:

                prices[sym]=float(tick["price"])

        manage_trades(prices)

        for sym in COIN_LIST:

            if sym in [t["sym"] for t in open_trades]:

                continue

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

            features=[rsi,vol,trend,mom]

            if score>=MIN_SCORE and ml_allows(features):

                open_trade(sym,prices[sym],score,features)

        if time.time()-last_status>STATUS_INTERVAL:

            print("STATUS Cash",cash,"Open",len(open_trades))

            last_status=time.time()

        time.sleep(SCAN_INTERVAL)

    except Exception as e:

        print("ERROR",e)

        time.sleep(5)
