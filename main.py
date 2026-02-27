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

    # Pullback proximity to EMA_FAST (within tolerance)
    # distance = (prev_close - ema_fast_prev)/ema_fast_prev
    ema_fast_prev = ema_fast_series[-2]
    if ema_fast_prev <= 0:
        return None

    prev_dist_pct = abs(pct_change(ema_fast_prev, prev_close))  # how far prev_close from EMA_FAST
    # Require it was close to EMA_FAST recently (a pullback), AND now it's reclaiming above EMA_FAST
    if prev_dist_pct > PULLBACK_TOL_PCT:
        return None
    if curr_close <= ema_fast_now:
        return None

    # 5) Volume confirmation: last candle vol above average
    # Compare last candle vol to average of prior 12
    if len(vols) < 20:
        return None
    base_vol = sum(vols[-13:-1]) / 12.0
    vol_mult = (vols[-1] / base_vol) if base_vol > 0 else 0.0
    if vol_mult < MIN_VOL_SPIKE_MULT:
        return None

    # Score: prefer stronger trend + momentum + vol, penalize spread
    score = (
        (trend_30m * 1.2) +
        (curr_mom_5m * 2.0) +
        (vol_mult * 1.0) +
        (ema_fast_slope * 0.8) +
        (ema_slow_slope * 0.4) -
        (sp * 1.8)
    )

    return {
        "product_id": product_id,
        "price": curr_close,
        "spread": sp,
        "trend_30m": trend_30m,
        "mom_5m": curr_mom_5m,
        "vol_mult": vol_mult,
        "ema_fast": ema_fast_now,
        "ema_slow": ema_slow_now,
        "score": score
    }

# -------------------------
# PAPER TRADING (AUTO SELL)
# -------------------------

def open_position(state: Dict, sig: Dict) -> None:
    pid = sig["product_id"]
    base = base_symbol_from_product(pid)

    # Avoid duplicates by base coin
    held_bases = currently_held_bases(state)
    if base in held_bases:
        return

    if pid in state["positions"]:
        return
    if len(state["positions"]) >= MAX_OPEN_POSITIONS:
        return

    spend = dynamic_trade_size(state)
    if cash(state) < spend:
        return

    entry = float(sig["price"])
    net_spend, fee = apply_fee(spend)
    size_base = net_spend / entry if entry > 0 else 0.0
    if size_base <= 0:
        return

    state["paper"]["cash"] = cash(state) - spend
    state["paper"]["fees_paid"] = float(state["paper"]["fees_paid"]) + fee
    state["paper"]["trades"] = int(state["paper"]["trades"]) + 1

    state["positions"][pid] = {
        "base": base,
        "entry_price": entry,
        "entry_ts": now_ts(),
        "base_size": size_base,
        "last_price": entry
    }
    save_state(state)

    telegram_send(
        f"<b>PAPER BUY</b> — <b>{pid}</b>\n"
        f"Entry: <b>${entry:.6g}</b> | Spend: <b>{fmt_money(spend)}</b>\n"
        f"Trend30m: <b>+{sig['trend_30m']:.2f}%</b> | Mom5m: <b>+{sig['mom_5m']:.2f}%</b>\n"
        f"Vol: <b>{sig['vol_mult']:.2f}x</b> | Spread: <b>{sig['spread']:.2f}%</b>\n"
        f"Open: <b>{len(state['positions'])}/{MAX_OPEN_POSITIONS}</b> | Equity: <b>{fmt_money(current_equity(state))}</b>"
    )

def close_position(state: Dict, pid: str, price: float, reason: str) -> None:
    pos = state["positions"].get(pid)
    if not pos:
        return

    size_base = float(pos["base_size"])
    entry = float(pos["entry_price"])

    gross = size_base * price
    net, fee = apply_fee(gross)
    cost = size_base * entry

    pnl = net - cost
    pnl_pct = (pnl / cost) * 100.0 if cost > 0 else 0.0

    state["paper"]["cash"] = cash(state) + net
    state["paper"]["fees_paid"] = float(state["paper"]["fees_paid"]) + fee
    state["paper"]["realized_pnl"] = float(state["paper"]["realized_pnl"]) + pnl
    state["paper"]["trades"] = int(state["paper"]["trades"]) + 1

    if pnl >= 0:
        state["paper"]["wins"] = int(state["paper"]["wins"]) + 1
    else:
        state["paper"]["losses"] = int(state["paper"]["losses"]) + 1

    state["positions"].pop(pid, None)
    save_state(state)

    telegram_send(
        f"<b>PAPER SELL</b> — <b>{pid}</b>\n"
        f"Reason: <b>{reason}</b>\n"
        f"Exit: <b>${price:.6g}</b>\n"
        f"PnL: <b>{fmt_money(pnl)}</b> ({pnl_pct:+.2f}%)\n"
        f"Open: <b>{len(state['positions'])}/{MAX_OPEN_POSITIONS}</b> | Equity: <b>{fmt_money(current_equity(state))}</b>"
    )

def manage_positions(state: Dict) -> None:
    # For each open position, check TP/SL/timeouts
    for pid in list(state["positions"].keys()):
        pos = state["positions"][pid]

        # Pull a small candle window for last price
        data = get_candles_close_vol(pid, 12)  # last hour
        if not data:
            continue

        price = data[0][-1]
        pos["last_price"] = price

        entry = float(pos["entry_price"])
        change = pct_change(entry, price)
        age = now_ts() - int(pos["entry_ts"])

        if change >= TAKE_PROFIT_PCT:
            close_position(state, pid, price, f"TP {TAKE_PROFIT_PCT:.2f}%")
        elif change <= -STOP_LOSS_PCT:
            close_position(state, pid, price, f"SL {STOP_LOSS_PCT:.2f}%")
        elif age >= MAX_HOLD_SECONDS:
            close_position(state, pid, price, f"MAX_HOLD {MAX_HOLD_SECONDS//60}m")

# -------------------------
# MAIN LOOP
# -------------------------

def main() -> None:
    if TRADE_MODE != "paper":
        telegram_send(
            "<b>WARNING</b>\n"
            "This file currently supports <b>paper</b> execution only.\n"
            "Live trading requires authenticated Coinbase order endpoints.\n"
            "Leave TRADE_MODE=paper for now."
        )

    state = load_state()

    telegram_send(
        f"<b>SCALPER SNIPER STARTED</b>\n"
        f"Mode: <b>{TRADE_MODE.upper()}</b>\n"
        f"State: <b>{STATE_FILE}</b>\n"
        f"Cash: <b>{fmt_money(cash(state))}</b> | Equity: <b>{fmt_money(current_equity(state))}</b>\n"
        f"Max open: <b>{MAX_OPEN_POSITIONS}</b> | Scan: <b>{SCAN_INTERVAL_SECONDS}s</b>\n"
        f"Universe: <b>{MAX_PRODUCTS}</b> | Sample/cycle: <b>{SCAN_SAMPLE_SIZE}</b>\n"
        f"TP/SL: <b>+{TAKE_PROFIT_PCT:.2f}%</b> / <b>-{STOP_LOSS_PCT:.2f}%</b>\n"
        f"Trend: EMA{EMA_FAST}/EMA{EMA_SLOW} | 30m min: <b>{MIN_TREND_PCT_30M:.2f}%</b>\n"
        f"VolSpike: <b>{MIN_VOL_SPIKE_MULT:.2f}x</b> | SpreadMax: <b>{MAX_SPREAD_PCT:.2f}%</b>\n"
        f"Compounding: <b>{'ON' if USE_COMPOUNDING else 'OFF'}</b> | Risk: <b>{RISK_PCT_PER_TRADE:.2f}%</b>\n"
        f"Min/Max spend: <b>{fmt_money(MIN_TRADE_USD)}</b> / <b>{fmt_money(MAX_TRADE_USD)}</b>"
    )

    products: List[str] = []
    last_refresh = 0

    while True:
        try:
            state["cycle"] = int(state.get("cycle", 0)) + 1

            # Refresh products every 30 minutes
            if not products or (now_ts() - last_refresh) > 1800:
                products = list_usd_products(MAX_PRODUCTS)
                random.shuffle(products)
                last_refresh = now_ts()
                print(f"[info] products loaded: {len(products)}", flush=True)

            # 1) Exits first
            manage_positions(state)

            # 2) Fill slots with best continuation setups
            slots = MAX_OPEN_POSITIONS - len(state["positions"])
            if slots > 0 and products:
                # sample a subset each cycle for speed + to avoid hammering APIs
                sample_size = min(SCAN_SAMPLE_SIZE, len(products))
                universe = random.sample(products, sample_size)

                held_bases = currently_held_bases(state)

                candidates: List[Dict] = []
                for pid in universe:
                    # skip if base already held
                    if base_symbol_from_product(pid) in held_bases:
                        continue
                    # skip if already in positions
                    if pid in state["positions"]:
                        continue

                    sig = continuation_signal(pid)
                    if sig:
                        candidates.append(sig)

                candidates.sort(key=lambda x: x["score"], reverse=True)

                # Attempt to open up to "slots" positions
                opened = 0
                for sig in candidates:
                    if opened >= slots:
                        break
                    before = len(state["positions"])
                    open_position(state, sig)
                    after = len(state["positions"])
                    if after > before:
                        opened += 1
                        held_bases.add(base_symbol_from_product(sig["product_id"]))

            # Heartbeat (so you know it's alive)
            if HEARTBEAT_EVERY_CYCLES > 0 and (state["cycle"] % HEARTBEAT_EVERY_CYCLES == 0):
                eq = current_equity(state)
                p = state["paper"]
                wins = int(p.get("wins", 0))
                losses = int(p.get("losses", 0))
                total_closed = wins + losses
                winrate = (wins / total_closed * 100.0) if total_closed > 0 else 0.0
                telegram_send(
                    f"<b>HEARTBEAT</b>\n"
                    f"Open: <b>{len(state['positions'])}/{MAX_OPEN_POSITIONS}</b>\n"
                    f"Cash: <b>{fmt_money(cash(state))}</b> | Equity: <b>{fmt_money(eq)}</b>\n"
                    f"RealizedPnL: <b>{fmt_money(float(p.get('realized_pnl', 0.0)))}</b>\n"
                    f"Fees: <b>{fmt_money(float(p.get('fees_paid', 0.0)))}</b>\n"
                    f"Win rate: <b>{winrate:.1f}%</b> | Trades: <b>{int(p.get('trades', 0))}</b>"
                )

            save_state(state)
            time.sleep(SCAN_INTERVAL_SECONDS)

        except Exception as e:
            print(f"[error] {type(e).__name__}: {e}", flush=True)
            time.sleep(max(5, SCAN_INTERVAL_SECONDS))

if __name__ == "__main__":
    main()
