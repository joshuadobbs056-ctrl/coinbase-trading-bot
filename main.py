import os
import time
import json
import csv
import requests
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# =========================
# CONFIG
# =========================

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 12))
STATUS_INTERVAL = int(os.getenv("STATUS_INTERVAL", 60))

MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", 60))
EXCLUDE = set(os.getenv("EXCLUDE", "USDT-USD,USDC-USD").split(","))

START_BALANCE = float(os.getenv("START_BALANCE", 1000))
CASH_RESERVE_PERCENT = float(os.getenv("CASH_RESERVE_PERCENT", 2))

MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", 20))
MIN_TRADE_SIZE_USD = float(os.getenv("MIN_TRADE_SIZE_USD", 25))

STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", 2.8))
TRAILING_START_PERCENT = float(os.getenv("TRAILING_START_PERCENT", 1.2))
TRAILING_DISTANCE_PERCENT = float(os.getenv("TRAILING_DISTANCE_PERCENT", 0.9))

ATR_MULT = float(os.getenv("ATR_MULT", 1.4))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", 14))

MAX_TRADE_DURATION_MINUTES = int(os.getenv("MAX_TRADE_DURATION_MINUTES", 30))

ML_MIN_TRADES_TO_ENABLE = int(os.getenv("ML_MIN_TRADES_TO_ENABLE", 50))
ML_MIN_PROB = float(os.getenv("ML_MIN_PROB", 0.58))
ML_RETRAIN_EVERY_SEC = int(os.getenv("ML_RETRAIN_EVERY_SEC", 600))

FEE_PERCENT_PER_SIDE = float(os.getenv("FEE_PERCENT_PER_SIDE", 0.20))

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

LEARNING_FILE = "learning.json"
HISTORY_FILE = "trade_history.csv"

BASE = "https://api.exchange.coinbase.com/products"

# =========================
# TELEGRAM
# =========================

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

# =========================
# FILE INIT
# =========================

def init_learning():

    if os.path.exists(LEARNING_FILE):

        with open(LEARNING_FILE,"r") as f:

            data=json.load(f)

    else:

        data={}

    data.setdefault("cash",START_BALANCE)
    data.setdefault("trade_count",0)
    data.setdefault("win_count",0)
    data.setdefault("loss_count",0)
    data.setdefault("total_profit_usd",0)

    save_learning(data)

    return data

def save_learning(data):

    with open(LEARNING_FILE,"w") as f:

        json.dump(data,f)

def init_history():

    if not os.path.exists(HISTORY_FILE):

        with open(HISTORY_FILE,"w",newline="") as f:

            w=csv.writer(f)

            w.writerow([
                "rsi",
                "volume",
                "trend",
                "momentum",
                "atr",
                "profit"
            ])

def append_history(row):

    with open(HISTORY_FILE,"a",newline="") as f:

        csv.writer(f).writerow(row)

learning=init_learning()
init_history()

cash=float(learning["cash"])
positions={}

# =========================
# MARKET
# =========================

def get_symbols():

    r=requests.get(BASE)

    data=r.json()

    out=[]

    for d in data:

        sym=d["id"]

        if d["quote_currency"]!="USD":
            continue

        if sym in EXCLUDE:
            continue

        out.append(sym)

    return out[:MAX_SYMBOLS]

def price(sym):

    try:

        r=requests.get(f"{BASE}/{sym}/ticker")

        return float(r.json()["price"])

    except:

        return None

def candles(sym):

    try:

        r=requests.get(f"{BASE}/{sym}/candles",params={"granularity":60})

        data=list(reversed(r.json()))

        closes=[x[4] for x in data]
        highs=[x[2] for x in data]
        lows=[x[1] for x in data]
        vols=[x[5] for x in data]

        return closes,highs,lows,vols

    except:

        return None

# =========================
# INDICATORS
# =========================

def rsi(closes):

    if len(closes)<15:return 50

    d=np.diff(closes)

    up=np.where(d>0,d,0)

    dn=np.where(d<0,-d,0)

    rs=np.mean(up[-14:])/max(np.mean(dn[-14:]),1e-9)

    return 100-(100/(1+rs))

def trend(closes):

    return (np.mean(closes[-10:])-np.mean(closes[-40:]))/np.mean(closes[-40:])

def momentum(closes):

    return (closes[-1]-closes[-5])/closes[-5]

def volume(vols):

    return vols[-1]/np.mean(vols[-20:])

def atr(highs,lows,closes):

    tr=[max(highs[i]-lows[i],abs(highs[i]-closes[i-1]),abs(lows[i]-closes[i-1])) for i in range(1,len(closes))]

    return np.mean(tr[-ATR_PERIOD:])/closes[-1]

# =========================
# ML
# =========================

model=None
last_train=0

def train():

    global model,last_train

    if time.time()-last_train<ML_RETRAIN_EVERY_SEC:return

    last_train=time.time()

    try:

        data=np.genfromtxt(HISTORY_FILE,delimiter=",",skip_header=1)

        if len(data)<ML_MIN_TRADES_TO_ENABLE:

            model=None

            return

        X=data[:,:-1]
        y=data[:,-1]>0

        model=RandomForestClassifier(n_estimators=300,max_depth=6)

        model.fit(X,y)

        notify("ML ACTIVATED")

    except:

        model=None

def ml_allow(f):

    if model is None:return True,0

    p=model.predict_proba([f])[0][1]

    return p>=ML_MIN_PROB,p

# =========================
# EQUITY
# =========================

def equity(prices):

    e=cash

    for s,p in positions.items():

        if s in prices:

            e+=p["qty"]*prices[s]

    return e

# =========================
# EXECUTION
# =========================

def buy(sym,price,f):

    global cash

    if len(positions)>=MAX_OPEN_TRADES:return

    size=max(MIN_TRADE_SIZE_USD,cash*0.05)

    if size>cash:return

    qty=size/price

    positions[sym]={

        "entry":price,
        "qty":qty,
        "size":size,
        "peak":price,
        "time":time.time(),
        "features":f

    }

    cash-=size

    learning["cash"]=cash

    save_learning(learning)

    notify(f"BUY {sym}\n${size:.2f}\nCash ${cash:.2f}")

def sell(sym,price,reason,prices):

    global cash

    p=positions[sym]

    proceeds=p["qty"]*price

    profit=proceeds-p["size"]

    fee=(p["size"]+proceeds)*(FEE_PERCENT_PER_SIDE/100)

    net=profit-fee

    cash+=proceeds

    learning["cash"]=cash
    learning["trade_count"]+=1
    learning["total_profit_usd"]+=net

    if net>0:
        learning["win_count"]+=1
    else:
        learning["loss_count"]+=1

    save_learning(learning)

    append_history(p["features"]+[net])

    del positions[sym]

    wr=learning["win_count"]/learning["trade_count"]*100

    notify(
f"""SELL {sym} ({reason})
P/L ${net:.2f}
Cash ${cash:.2f}
Equity ${equity(prices):.2f}

Trades {learning['trade_count']}
Wins {learning['win_count']}
Losses {learning['loss_count']}
WR {wr:.1f}%
ML {"ON" if model else "OFF"}"""
)

# =========================
# MAIN
# =========================

symbols=get_symbols()

notify("BOT STARTED")

while True:

    try:

        train()

        prices={}

        for s in symbols[:40]:

            p=price(s)

            if p:prices[s]=p

        for s,p in list(positions.items()):

            pr=prices.get(s)

            if not pr:continue

            profit=(pr-p["entry"])/p["entry"]

            age=(time.time()-p["time"])/60

            if pr>p["peak"]:p["peak"]=pr

            trail=max(TRAILING_DISTANCE_PERCENT/100,ATR_MULT*atr(*candles(s)))

            if profit>=TRAILING_START_PERCENT/100:

                stop=p["peak"]*(1-trail)

                if pr<=stop:sell(s,pr,"TRAIL",prices)

            elif profit<=-STOP_LOSS_PERCENT/100:

                sell(s,pr,"STOP",prices)

            elif age>=MAX_TRADE_DURATION_MINUTES and profit<=0:

                sell(s,pr,"STAGNATION",prices)

        for s in symbols:

            if s in positions:continue

            c=candles(s)

            if not c:continue

            closes,highs,lows,vols=c

            f=[
                rsi(closes),
                volume(vols),
                trend(closes),
                momentum(closes),
                atr(highs,lows,closes)
            ]

            allow,p=ml_allow(f)

            if allow and f[1]>1.2 and f[2]>0 and f[3]>0:

                buy(s,prices.get(s),f)

        time.sleep(SCAN_INTERVAL)

    except Exception as e:

        notify(str(e))

        time.sleep(5)
