# Coin Sniper — Savage Mode ELITE (MAX PROFIT BUILD — ML TELEGRAM FIXED + SCAN LOGS)

import os
import time
import json
import traceback
import requests
import numpy as np
from sklearn.ensemble import RandomForestClassifier

LOCK_FILE = os.getenv("LOCK_FILE", "/tmp/coin_sniper.lock")

def _pid_alive(pid:int)->bool:
    try:
        os.kill(pid,0)
        return True
    except:
        return False

def acquire_lock_or_exit():
    if os.path.exists(LOCK_FILE):
        try:
            raw=open(LOCK_FILE).read().strip()
            if raw:
                old_pid=int(raw.split("|")[0])
                if _pid_alive(old_pid):
                    raise SystemExit(0)
        except SystemExit:
            raise
        except:
            pass
    open(LOCK_FILE,"w").write(f"{os.getpid()}|{int(time.time())}")

acquire_lock_or_exit()
INSTANCE_ID=str(os.getpid())

START_BALANCE=float(os.getenv("START_BALANCE",1000))

SCAN_INTERVAL=int(os.getenv("SCAN_INTERVAL",15))
STATUS_INTERVAL=int(os.getenv("STATUS_INTERVAL",60))
SAVE_INTERVAL=int(os.getenv("SAVE_INTERVAL",60))
ML_INTERVAL=int(os.getenv("ML_INTERVAL",300))

MAX_OPEN_TRADES=int(os.getenv("MAX_OPEN_TRADES",20))
MIN_TRADE_SIZE=float(os.getenv("MIN_TRADE_SIZE",25))

STOP_LOSS_PERCENT=float(os.getenv("STOP_LOSS_PERCENT",4.0))
TRAIL_DIST_BASE=float(os.getenv("TRAILING_DISTANCE_PERCENT",0.9))

COOLDOWN_SECONDS=int(os.getenv("COOLDOWN_SECONDS",180))

ENTRY_SCORE_MIN=int(os.getenv("ENTRY_SCORE_MIN",7))

ML_ENABLED=os.getenv("ML_ENABLED","true").lower()=="true"
ML_ENABLE_AFTER=int(os.getenv("ML_ENABLE_AFTER",50))
ML_MIN_PROB=float(os.getenv("ML_MIN_PROB",0.62))

BOT_PAUSED=os.getenv("BOT_PAUSED","false").lower()=="true"

BASE_URL="https://api.exchange.coinbase.com/products"

TELEGRAM_TOKEN=os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID=os.getenv("TELEGRAM_CHAT_ID")

LEARNING_FILE="learning.json"
POSITIONS_FILE="positions.json"
COOLDOWN_FILE="cooldown.json"
HISTORY_FILE="trade_history.csv"
ML_TRAIN_FILE="ml_training.csv"

ml_model=None
ml_active=False

def notify(msg:str):
    print(msg, flush=True)
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id":TELEGRAM_CHAT_ID,"text":msg},
                timeout=10)
        except:
            pass

def ensure_files():
    if not os.path.exists(LEARNING_FILE):
        json.dump({
            "cash":START_BALANCE,
            "start_balance":START_BALANCE,
            "trade_count":0,
            "win_count":0,
            "loss_count":0,
            "total_profit":0.0
        },open(LEARNING_FILE,"w"))

    for f,d in [
        (POSITIONS_FILE,{}),
        (COOLDOWN_FILE,{})
    ]:
        if not os.path.exists(f):
            json.dump(d,open(f,"w"))

    if not os.path.exists(HISTORY_FILE):
        open(HISTORY_FILE,"w").write("profit\n")

    if not os.path.exists(ML_TRAIN_FILE):
        open(ML_TRAIN_FILE,"w").write(
            "score,momentum,volatility,trend,range_pos,breakout,outcome\n")

ensure_files()

def load_json(path,default):
    try:
        if os.path.exists(path):
            return json.load(open(path))
    except:
        pass
    return default

def save_json(path,data):
    try:
        json.dump(data,open(path,"w"))
    except:
        pass

learning=load_json(LEARNING_FILE,{
    "cash":START_BALANCE,
    "trade_count":0,
    "win_count":0,
    "loss_count":0,
    "total_profit":0
})

positions=load_json(POSITIONS_FILE,{})
cooldown=load_json(COOLDOWN_FILE,{})

cash=float(learning.get("cash",START_BALANCE))

price_history={}
symbols=[]

def get_price(sym):
    try:
        r=requests.get(f"{BASE_URL}/{sym}/ticker",timeout=10)
        j=r.json()
        return float(j["price"])
    except:
        return None

def get_symbols():
    try:
        r=requests.get(BASE_URL,timeout=10)
        syms=[p["id"] for p in r.json() if p.get("quote_currency")=="USD"]
        return syms[:60]
    except:
        return []

symbols=get_symbols()

def update_price_history(sym,px):
    if sym not in price_history:
        price_history[sym]=[]
    price_history[sym].append(px)
    if len(price_history[sym])>120:
        price_history[sym].pop(0)

def get_hist(sym,n):
    h=price_history.get(sym,[])
    if len(h)<n:
        return None
    return np.array(h[-n:],dtype=float)

def score_coin(sym,price):
    a20=get_hist(sym,20)
    a40=get_hist(sym,40)
    if a20 is None or a40 is None:
        return 0
    mean20=a20.mean()
    mean40=a40.mean()
    trend=(mean20-mean40)/mean40
    mom=(price-mean20)/mean20
    score=0
    if trend>0.002:score+=1
    if trend>0.005:score+=1
    if trend>0.01:score+=1
    if mom>0.002:score+=1
    if mom>0.005:score+=1
    if mom>0.01:score+=1
    return score

def extract_features(sym,price,score):
    a20=get_hist(sym,20)
    a40=get_hist(sym,40)
    if a20 is None:return None
    mean20=a20.mean()
    mean40=a40.mean()
    mom=(price-mean20)/mean20
    vol=a20.std()/mean20
    trend=(mean20-mean40)/mean40
    return [score,mom,vol,trend,0,0]

def train_model():
    global ml_model,ml_active
    if not ML_ENABLED:return
    try:
        data=np.genfromtxt(ML_TRAIN_FILE,delimiter=",",skip_header=1)
        if data is None or len(data)<ML_ENABLE_AFTER:
            notify(f"ML LEARNING ({0 if data is None else len(data)}/{ML_ENABLE_AFTER})")
            return
        X=data[:,:-1]
        y=data[:,-1]
        model=RandomForestClassifier(n_estimators=300)
        model.fit(X,y)
        ml_model=model
        if not ml_active:
            ml_active=True
            notify(f"ML ACTIVATED — LIVE TRADING MODE")
        notify(f"ML TRAINED samples={len(data)}")
    except:
        pass

def equity(prices):
    total=cash
    for sym,pos in positions.items():
        px=prices.get(sym)
        if px:
            total+=pos["qty"]*px
    return total

def winrate():
    t=learning["trade_count"]
    w=learning["win_count"]
    return w/t*100 if t>0 else 0

def open_trade(sym,price,score):
    global cash
    if sym in positions:return
    if cash<MIN_TRADE_SIZE:return

    features=extract_features(sym,price,score)
    if not features:return

    if ml_model:
        prob=ml_model.predict_proba([features])[0][1]
        if prob<ML_MIN_PROB:return

    size=min(MIN_TRADE_SIZE,cash)
    qty=size/price

    positions[sym]={
        "entry":price,
        "qty":qty,
        "peak":price,
        "stop":price*(1-STOP_LOSS_PERCENT/100),
        "features":features
    }

    cash-=size
    learning["cash"]=cash

    notify(f"BUY {sym} score={score} cash={cash}")

def sell_trade(sym,price,reason):
    global cash
    pos=positions[sym]
    entry=pos["entry"]
    qty=pos["qty"]
    profit=(price-entry)*qty
    cash+=qty*price

    learning["cash"]=cash
    learning["trade_count"]+=1

    if profit>0: learning["win_count"]+=1
    else: learning["loss_count"]+=1

    learning["total_profit"]+=profit

    open(HISTORY_FILE,"a").write(f"{profit}\n")

    outcome=1 if profit>0 else 0
    open(ML_TRAIN_FILE,"a").write(",".join(map(str,pos["features"]))+f",{outcome}\n")

    del positions[sym]

    notify(f"SELL {sym} profit={profit}")

notify(f"BOT STARTED cash={cash}")

last_prices={}
last_scan=0
last_status=0
last_ml=0

while True:
    try:
        now=time.time()

        if now-last_scan>=SCAN_INTERVAL:

            notify(f"SCANNING {len(symbols)} symbols | history coins={len(price_history)}")

            prices={}

            for sym in symbols:
                px=get_price(sym)
                if px:
                    prices[sym]=px
                    update_price_history(sym,px)

            last_prices=prices

            for sym,pos in list(positions.items()):
                px=prices.get(sym)
                if px and px<=pos["stop"]:
                    sell_trade(sym,px,"STOP")

            ready=sum(1 for s in price_history if len(price_history[s])>=40)

            notify(f"HISTORY READY: {ready}/{len(symbols)} coins")

            for sym,px in prices.items():
                score=score_coin(sym,px)
                if score>=ENTRY_SCORE_MIN:
                    open_trade(sym,px,score)

            last_scan=now

        if now-last_status>=STATUS_INTERVAL:

            eq=equity(last_prices)

            notify(f"""
STATUS
Cash: {cash}
Equity: {eq}
Profit: {learning["total_profit"]}

Open: {len(positions)}

Trades: {learning["trade_count"]}
Wins: {learning["win_count"]}
Losses: {learning["loss_count"]}
Winrate: {winrate():.1f}%

ML: {"ACTIVE" if ml_active else "LEARNING"}
""")

            last_status=now

        if now-last_ml>=ML_INTERVAL:
            train_model()
            last_ml=now

        save_json(LEARNING_FILE,learning)
        save_json(POSITIONS_FILE,positions)
        save_json(COOLDOWN_FILE,cooldown)

        time.sleep(1)

    except:
        notify(traceback.format_exc())
        time.sleep(5)
