import os
import time
import json
import random
from typing import Dict, List, Optional, Tuple

import requests

# ============================================================
# SCALPER SNIPER (PAPER) — Trend + Continuation (2% scalps)
# ============================================================
# What this bot does:
# - Scans Coinbase USD pairs using LIVE market data (no Coinbase keys needed in paper)
# - Enters ONLY when trend is bullish AND a continuation trigger happens
# - Auto-sells each position independently on TP / SL / Max hold
# - Supports compounding position sizing (equity-based sizing)
# - Avoids duplicate exposure to the same base coin (e.g., ESP-USD and ESP-USDC won't both be traded)
#
# Railway env variables supported (exact names):
# TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
# TRADE_MODE=paper
# STATE_FILE=/data/state.json (or state.json)
# PAPER_START_BALANCE=1000
# PAPER_FEE_PCT=0.006
# MAX_OPEN_POSITIONS=20
# SCAN_INTERVAL_SECONDS=30
# MAX_PRODUCTS=400
# SCAN_SAMPLE_SIZE=160
# TAKE_PROFIT_PCT=2.0
# STOP_LOSS_PCT=1.0
# MAX_HOLD_SECONDS=3600
#
# Trend/entry tuning:
# EMA_FAST=9
# EMA_SLOW=21
# MIN_TREND_PCT_30M=0.9
# MIN_MOMENTUM_PCT_5M=0.25
# MIN_VOL_SPIKE_MULT=1.2
# MAX_SPREAD_PCT=0.40
# PULLBACK_TOL_PCT=0.35
#
# Compounding:
# USE_COMPOUNDING=true
# RISK_PCT_PER_TRADE=2.0
# MIN_TRADE_USD=25
# MAX_TRADE_USD=50
# (If USE_COMPOUNDING=false, uses QUOTE_PER_TRADE_USD)
# ============================================================

# -------------------------
# ENV / CONFIG
# -------------------------

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

TRADE_MODE = os.getenv("TRADE_MODE", "paper").strip().lower()

STATE_FILE = os.getenv("STATE_FILE", "state.json").strip()

PAPER_START_BALANCE = float(os.getenv("PAPER_START_BALANCE", "1000"))
PAPER_FEE_PCT = float(os.getenv("PAPER_FEE_PCT", "0.006"))  # 0.6% per side simulation

MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "20"))
SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "30"))

MAX_PRODUCTS = int(os.getenv("MAX_PRODUCTS", "400"))          # cap how many products we consider
SCAN_SAMPLE_SIZE = int(os.getenv("SCAN_SAMPLE_SIZE", "160"))  # how many we actually scan each cycle

TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "2.0"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "1.0"))
MAX_HOLD_SECONDS = int(os.getenv("MAX_HOLD_SECONDS", "3600"))

# Trend logic
EMA_FAST = int(os.getenv("EMA_FAST", "9"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "21"))

MIN_TREND_PCT_30M = float(os.getenv("MIN_TREND_PCT_30M", "0.9"))     # 30m drift up
MIN_MOMENTUM_PCT_5M = float(os.getenv("MIN_MOMENTUM_PCT_5M", "0.25")) # last candle green enough
MIN_VOL_SPIKE_MULT = float(os.getenv("MIN_VOL_SPIKE_MULT", "1.2"))    # last volume vs avg
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "0.40"))           # avoid wide spreads
PULLBACK_TOL_PCT = float(os.getenv("PULLBACK_TOL_PCT", "0.35"))        # how close to EMA_FAST to count as pullback

# Compounding
USE_COMPOUNDING = os.getenv("USE_COMPOUNDING", "true").lower() in ("1", "true", "yes", "y")
RISK_PCT_PER_TRADE = float(os.getenv("RISK_PCT_PER_TRADE", "2.0"))
MIN_TRADE_USD = float(os.getenv("MIN_TRADE_USD", "25"))
MAX_TRADE_USD = float(os.getenv("MAX_TRADE_USD", "50"))
QUOTE_PER_TRADE_USD = float(os.getenv("QUOTE_PER_TRADE_USD", "25"))  # used only if USE_COMPOUNDING=false

# Operational
HTTP_TIMEOUT = 15
USER_AGENT = "scalper-sniper/1.0"
HEARTBEAT_EVERY_CYCLES = int(os.getenv("HEARTBEAT_EVERY_CYCLES", "5"))  # Telegram heartbeat frequency (cycles)

# Coinbase public endpoints (brokerage)
BASE_URL = "https://api.coinbase.com"
PRODUCTS_URL = f"{BASE_URL}/api/v3/brokerage/market/products"
CANDLES_URL = f"{BASE_URL}/api/v3/brokerage/market/products/{{product_id}}/candles"
BOOK_URL = f"{BASE_URL}/api/v3/brokerage/market/product_book"

GRANULARITY = "FIVE_MINUTE"
LOOKBACK = 60  # 60 x 5m = 5 hours

# -------------------------
# UTILS
# -------------------------

def now_ts() -> int:
    return int(time.time())

def fmt_money(x: float) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    return f"{sign}${x:,.2f}"

def pct_change(a: float, b: float) -> float:
    if a <= 0:
        return 0.0
    return (b - a) / a * 100.0

def http_get_json(url: str, params: Optional[Dict] = None) -> Dict:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    r = requests.get(url, params=params or {}, headers=headers, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()

def telegram_send(msg: str) -> None:
    # Always print to logs too
    print(msg.replace("<b>", "").replace("</b>", ""), flush=True)

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        requests.post(url, json=payload, timeout=HTTP_TIMEOUT).raise_for_status()
    except Exception:
        print("[warn] telegram send failed", flush=True)

# -------------------------
# STATE
# -------------------------

def fresh_state() -> Dict:
    return {
        "paper": {
            "start_balance": PAPER_START_BALANCE,
            "cash": PAPER_START_BALANCE,
            "realized_pnl": 0.0,
            "fees_paid": 0.0,
            "wins": 0,
            "losses": 0,
            "trades": 0
        },
        "positions": {},  # product_id -> dict
        "cycle": 0
    }

def ensure_state_dir() -> None:
    d = os.path.dirname(STATE_FILE)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def load_state() -> Dict:
    ensure_state_dir()
    if not os.path.exists(STATE_FILE):
        return fresh_state()
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            s = json.load(f)
        if "paper" not in s or "positions" not in s:
            return fresh_state()

        s["paper"].setdefault("start_balance", PAPER_START_BALANCE)
        s["paper"].setdefault("cash", s["paper"]["start_balance"])
        s["paper"].setdefault("realized_pnl", 0.0)
        s["paper"].setdefault("fees_paid", 0.0)
        s["paper"].setdefault("wins", 0)
        s["paper"].setdefault("losses", 0)
        s["paper"].setdefault("trades", 0)
        s.setdefault("positions", {})
        s.setdefault("cycle", 0)
        return s
    except Exception:
        return fresh_state()

def save_state(state: Dict) -> None:
    ensure_state_dir()
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, STATE_FILE)

def cash(state: Dict) -> float:
    return float(state["paper"].get("cash", 0.0))

def apply_fee(amount: float) -> Tuple[float, float]:
    fee = abs(amount) * PAPER_FEE_PCT
    return amount - fee, fee

def current_equity(state: Dict) -> float:
    eq = cash(state)
    for _, pos in state["positions"].items():
        base = float(pos.get("base_size", 0.0))
        px = float(pos.get("last_price", pos.get("entry_price", 0.0)))
        if base > 0 and px > 0:
            eq += base * px
    return eq

def dynamic_trade_size(state: Dict) -> float:
    if not USE_COMPOUNDING:
        return QUOTE_PER_TRADE_USD
    eq = current_equity(state)
    spend = eq * (RISK_PCT_PER_TRADE / 100.0)
    spend = max(MIN_TRADE_USD, min(MAX_TRADE_USD, spend))
    return spend

def base_symbol_from_product(product_id: str) -> str:
    # "SOL-USD" -> "SOL"
    return product_id.split("-")[0].strip().upper()

def currently_held_bases(state: Dict) -> set:
    return {base_symbol_from_product(pid) for pid in state["positions"].keys()}

# -------------------------
# MARKET DATA
# -------------------------

def list_usd_products(limit: int) -> List[str]:
    products: List[str] = []
    cursor = None
    while len(products) < limit:
        params = {}
        if cursor:
            params["cursor"] = cursor
        data = http_get_json(PRODUCTS_URL, params=params)

        for p in (data.get("products") or []):
            pid = p.get("product_id")
            if not pid:
                continue

            quote = (p.get("quote_display_symbol") or "").upper()
            if quote != "USD":
                continue

            if bool(p.get("is_disabled", False)) or bool(p.get("trading_disabled", False)):
                continue

            # Avoid derivatives/perps (display_name often includes PERP)
            name = (p.get("display_name") or "").upper()
            if "PERP" in name:
                continue

            products.append(pid)
            if len(products) >= limit:
                break

        pag = data.get("pagination") or {}
        cursor = pag.get("next_cursor")
        if not cursor or not pag.get("has_next", False):
            break

    return products

def get_spread_pct(product_id: str) -> Optional[float]:
    try:
        data = http_get_json(BOOK_URL, params={"product_id": product_id, "limit": "1"})
        pb = data.get("pricebook") or {}
        bids = pb.get("bids") or []
        asks = pb.get("asks") or []
        if not bids or not asks:
            return None
        bid = float(bids[0]["price"])
        ask = float(asks[0]["price"])
        if ask <= 0:
            return None
        return ((ask - bid) / ask) * 100.0
    except Exception:
        return None

def get_candles_close_vol(product_id: str, lookback: int) -> Optional[Tuple[List[float], List[float]]]:
    end_ts = now_ts()
    # 5m candles, add buffer
    start_ts = end_ts - (lookback + 10) * 300
    url = CANDLES_URL.format(product_id=product_id)
    params = {"start": str(start_ts), "end": str(end_ts), "granularity": GRANULARITY}

    data = http_get_json(url, params=params)
    raw = data.get("candles") or []
    if len(raw) < lookback:
        return None

    # Sort by candle start time
    raw_sorted = sorted(raw, key=lambda c: int(c["start"]))[-lookback:]
    closes = [float(c["close"]) for c in raw_sorted]
    vols = [float(c["volume"]) for c in raw_sorted]
    if not closes or closes[-1] <= 0:
        return None
    return closes, vols

# -------------------------
# INDICATORS
# -------------------------

def ema(values: List[float], period: int) -> Optional[float]:
    if period <= 1 or len(values) < period:
        return None
    k = 2 / (period + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e

def ema_series(values: List[float], period: int) -> Optional[List[float]]:
    if period <= 1 or len(values) < period:
        return None
    k = 2 / (period + 1)
    out = []
    e = values[0]
    out.append(e)
    for v in values[1:]:
        e = v * k + e * (1 - k)
        out.append(e)
    return out

# -------------------------
# ENTRY: TREND + CONTINUATION "SNIPER"
# -------------------------

def continuation_signal(product_id: str) -> Optional[Dict]:
    # Spread filter first (cheap call) — if too wide, skip
    sp = get_spread_pct(product_id)
    if sp is None or sp > MAX_SPREAD_PCT:
        return None

    data = get_candles_close_vol(product_id, LOOKBACK)
    if not data:
        return None

    closes, vols = data

    # Need enough data for EMAs + trend windows
    if len(closes) < max(EMA_SLOW + 5, 12):
        return None

    # Compute EMAs (use series so we can see slope)
    ema_fast_series = ema_series(closes, EMA_FAST)
    ema_slow_series = ema_series(closes, EMA_SLOW)
    if not ema_fast_series or not ema_slow_series:
        return None

    px = closes[-1]
    ema_fast_now = ema_fast_series[-1]
    ema_slow_now = ema_slow_series[-1]

    # 1) Bull trend confirmation: price above EMAs and fast above slow
    if not (px > ema_fast_now > 0 and px > ema_slow_now > 0):
        return None
    if ema_fast_now <= ema_slow_now:
        return None

    # 2) EMA slope: rising (fast and slow)
    # Compare now vs 6 candles ago (~30m)
    if len(ema_fast_series) < 7 or len(ema_slow_series) < 7:
        return None
    ema_fast_slope = pct_change(ema_fast_series[-7], ema_fast_now)
    ema_slow_slope = pct_change(ema_slow_series[-7], ema_slow_now)
    if ema_fast_slope <= 0 or ema_slow_slope <= 0:
        return None

    # 3) Trend drift over 30 minutes (6 candles)
    trend_30m = pct_change(closes[-7], closes[-1])  # from 6 candles back to now
    if trend_30m < MIN_TREND_PCT_30M:
        return None

    # 4) Continuation trigger: "pullback near EMA_FAST then green reclaim"
    # We use: previous close near/below EMA_FAST and current close above EMA_FAST with momentum.
    prev_close = closes[-2]
    curr_close = closes[-1]
    curr_mom_5m = pct_change(prev_close, curr_close)

    if curr_mom_5m < MIN_MOMENTUM_PCT_5M:
        return None

    # Pullback
