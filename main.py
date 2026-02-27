import os
import time
import json
from typing import Dict, List, Optional, Tuple

import requests

# =========================
# ENV / CONFIG
# =========================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

TRADE_MODE = os.getenv("TRADE_MODE", "paper").lower()

STATE_FILE = os.getenv("STATE_FILE", "state.json")

# Paper account & fee simulation (Coinbase taker-ish)
PAPER_START_BALANCE = float(os.getenv("PAPER_START_BALANCE", "1000"))
PAPER_FEE_PCT = float(os.getenv("PAPER_FEE_PCT", "0.006"))  # 0.6% per side simulation

# Position count + scanning
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "20"))
SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "30"))
MAX_PRODUCTS = int(os.getenv("MAX_PRODUCTS", "400"))

# Exits (scalping)
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "2.0"))  # recommended for $25-ish trades on fees
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "1.0"))
MAX_HOLD_SECONDS = int(os.getenv("MAX_HOLD_SECONDS", "3600"))  # 1 hour

# Entry filters (more trades = lower thresholds; fewer trades = higher thresholds)
MIN_MOMENTUM_PCT_5M = float(os.getenv("MIN_MOMENTUM_PCT_5M", "0.35"))
MIN_VOL_SPIKE_MULT = float(os.getenv("MIN_VOL_SPIKE_MULT", "1.3"))
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "0.40"))

# ---- Compounding / dynamic trade sizing ----
# If USE_COMPOUNDING=true, trade size scales with equity:
# spend = equity * (RISK_PCT_PER_TRADE/100), clamped to MIN/MAX.
USE_COMPOUNDING = os.getenv("USE_COMPOUNDING", "true").lower() in ("1", "true", "yes", "y")
RISK_PCT_PER_TRADE = float(os.getenv("RISK_PCT_PER_TRADE", "2.0"))   # percent of equity per trade
MIN_TRADE_USD = float(os.getenv("MIN_TRADE_USD", "25"))
MAX_TRADE_USD = float(os.getenv("MAX_TRADE_USD", "150"))

# If you want fixed trade size instead of compounding, set USE_COMPOUNDING=false and use this:
QUOTE_PER_TRADE_USD = float(os.getenv("QUOTE_PER_TRADE_USD", "25"))

HTTP_TIMEOUT = 15
USER_AGENT = "scalper-bot/3.0"

# Coinbase Advanced Trade public endpoints (no keys needed)
BASE_URL = "https://api.coinbase.com"
PRODUCTS_URL = BASE_URL + "/api/v3/brokerage/market/products"
CANDLES_URL = BASE_URL + "/api/v3/brokerage/market/products/{product_id}/candles"
BOOK_URL = BASE_URL + "/api/v3/brokerage/market/product_book"

GRANULARITY = "FIVE_MINUTE"
LOOKBACK = 40  # 40 x 5m = ~3h20m

# =========================
# HELPERS
# =========================

def now_ts() -> int:
    return int(time.time())

def pct_change(a: float, b: float) -> float:
    if a <= 0:
        return 0.0
    return (b - a) / a * 100.0

def fmt_money(x: float) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    return f"{sign}${x:,.2f}"

def http_get_json(url: str, params: Optional[Dict] = None) -> Dict:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    r = requests.get(url, params=params or {}, headers=headers, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()

def telegram_send(msg: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(msg, flush=True)
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

# =========================
# STATE
# =========================

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
        # backfill
        s["paper"].setdefault("start_balance", PAPER_START_BALANCE)
        s["paper"].setdefault("cash", s["paper"]["start_balance"])
        s["paper"].setdefault("realized_pnl", 0.0)
        s["paper"].setdefault("fees_paid", 0.0)
        s["paper"].setdefault("wins", 0)
        s["paper"].setdefault("losses", 0)
        s["paper"].setdefault("trades", 0)
        s.setdefault("positions", {})
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
    """
    Equity = cash + value of open positions using last_price (or entry_price fallback).
    This is enough for compounding sizing.
    """
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
    if spend < MIN_TRADE_USD:
        spend = MIN_TRADE_USD
    if spend > MAX_TRADE_USD:
        spend = MAX_TRADE_USD
    return spend

# =========================
# COINBASE DATA
# =========================

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

def get_candles_close_vol(product_id: str, lookback: int) -> Optional[Tuple[List[float], List[float]]]:
    end_ts = now_ts()
    start_ts = end_ts - (lookback + 10) * 300  # 5m candles
    url = CANDLES_URL.format(product_id=product_id)
    params = {"start": str(start_ts), "end": str(end_ts), "granularity": GRANULARITY}
    data = http_get_json(url, params=params)
    raw = data.get("candles") or []
    if len(raw) < lookback:
        return None

    raw_sorted = sorted(raw, key=lambda c: int(c["start"]))[-lookback:]
    closes = [float(c["close"]) for c in raw_sorted]
    vols = [float(c["volume"]) for c in raw_sorted]

    if not closes or closes[-1] <= 0:
        return None

    return closes, vols

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

# =========================
# ENTRY SIGNAL (SCALPING)
# =========================

def entry_signal(product_id: str) -> Optional[Dict]:
    data = get_candles_close_vol(product_id, LOOKBACK)
    if not data:
        return None

    closes, vols = data

    # momentum 5m
    mom_5m = pct_change(closes[-2], closes[-1])

    # simple short trend: last 3 candles net move
    trend = pct_change(closes[-4], closes[-1])

    # volume spike: last vol vs avg prior 10
    base_vol = sum(vols[-11:-1]) / 10.0 if len(vols) >= 11 else 0.0
    vol_mult = (vols[-1] / base_vol) if base_vol > 0 else 0.0

    if mom_5m < MIN_MOMENTUM_PCT_5M:
        return None
    if trend < 0:
        return None
    if vol_mult < MIN_VOL_SPIKE_MULT:
        return None

    sp = get_spread_pct(product_id)
    if sp is None or sp > MAX_SPREAD_PCT:
        return None

    # ranking score
    score = (mom_5m * 2.0) + (trend * 0.7) + (vol_mult * 0.8) - (sp * 1.5)

    return {
        "product_id": product_id,
        "price": closes[-1],
        "mom_5m": mom_5m,
        "trend": trend,
        "vol_mult": vol_mult,
        "spread": sp,
        "score": score
    }

# =========================
# PAPER TRADING
# =========================

def open_position(state: Dict, sig: Dict) -> None:
    pid = sig["product_id"]
    if pid in state["positions"]:
        return
    if len(state["positions"]) >= MAX_OPEN_POSITIONS:
        return

    spend = dynamic_trade_size(state)
    if cash(state) < spend:
        return

    entry = float(sig["price"])
    net_spend, fee = apply_fee(spend)

    base = net_spend / entry if entry > 0 else 0.0
    if base <= 0:
        return

    state["paper"]["cash"] = cash(state) - spend
    state["paper"]["fees_paid"] = float(state["paper"]["fees_paid"]) + fee
    state["paper"]["trades"] = int(state["paper"]["trades"]) + 1

    state["positions"][pid] = {
        "entry_price": entry,
        "entry_ts": now_ts(),
        "base_size": base,
        "last_price": entry
    }

    save_state(state)

    telegram_send(
        f"<b>PAPER BUY</b> — <b>{pid}</b>\n"
        f"Entry: <b>${entry:.6g}</b> | Spend: <b>{fmt_money(spend)}</b>\n"
        f"Mom5m: <b>+{sig['mom_5m']:.2f}%</b> | Trend: <b>+{sig['trend']:.2f}%</b>\n"
        f"Vol: <b>{sig['vol_mult']:.2f}x</b> | Spread: <b>{sig['spread']:.2f}%</b>\n"
        f"Equity: <b>{fmt_money(current_equity(state))}</b> | Open: <b>{len(state['positions'])}/{MAX_OPEN_POSITIONS}</b>"
    )

def close_position(state: Dict, pid: str, price: float, reason: str) -> None:
    pos = state["positions"].get(pid)
    if not pos:
        return

    base = float(pos["base_size"])
    entry = float(pos["entry_price"])

    gross = base * price
    net, fee = apply_fee(gross)
    cost = base * entry

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
        f"Equity: <b>{fmt_money(current_equity(state))}</b> | Open: <b>{len(state['positions'])}/{MAX_OPEN_POSITIONS}</b>"
    )

def manage_positions(state: Dict) -> None:
    # check exits for each open position
    for pid in list(state["positions"].keys()):
        pos = state["positions"][pid]

        data = get_candles_close_vol(pid, 8)  # ~40 minutes
        if not data:
            continue

        price = data[0][-1]
        pos["last_price"] = price  # important for equity + compounding sizing

        entry = float(pos["entry_price"])
        change = pct_change(entry, price)
        age = now_ts() - int(pos["entry_ts"])

        if change >= TAKE_PROFIT_PCT:
            close_position(state, pid, price, f"TP {TAKE_PROFIT_PCT:.2f}%")
        elif change <= -STOP_LOSS_PCT:
            close_position(state, pid, price, f"SL {STOP_LOSS_PCT:.2f}%")
        elif age >= MAX_HOLD_SECONDS:
            close_position(state, pid, price, f"MAX_HOLD {MAX_HOLD_SECONDS//60}m")

# =========================
# MAIN
# =========================

def main():
    state = load_state()

    telegram_send(
        f"<b>SCALPER BOT STARTED</b>\n"
        f"Mode: <b>{TRADE_MODE.upper()}</b>\n"
        f"State: <b>{STATE_FILE}</b>\n"
        f"Cash: <b>{fmt_money(cash(state))}</b>\n"
        f"Equity: <b>{fmt_money(current_equity(state))}</b>\n"
        f"Max open: <b>{MAX_OPEN_POSITIONS}</b> | Scan: <b>{SCAN_INTERVAL_SECONDS}s</b>\n"
        f"TP/SL: <b>+{TAKE_PROFIT_PCT:.2f}%</b> / <b>-{STOP_LOSS_PCT:.2f}%</b>\n"
        f"Compounding: <b>{'ON' if USE_COMPOUNDING else 'OFF'}</b> | Risk: <b>{RISK_PCT_PER_TRADE:.2f}%</b>\n"
        f"Min/Max spend: <b>{fmt_money(MIN_TRADE_USD)}</b> / <b>{fmt_money(MAX_TRADE_USD)}</b>"
    )

    products: List[str] = []
    last_refresh = 0

    while True:
        try:
            # refresh product list every 30 minutes
            if not products or (now_ts() - last_refresh) > 1800:
                products = list_usd_products(MAX_PRODUCTS)
                last_refresh = now_ts()
                print(f"[info] products loaded: {len(products)}", flush=True)

            # 1) exits first (free slots)
            manage_positions(state)

            # 2) fill slots with best signals
            slots = MAX_OPEN_POSITIONS - len(state["positions"])
            if slots > 0:
                candidates = []
                for pid in products:
                    if pid in state["positions"]:
                        continue
                    sig = entry_signal(pid)
                    if sig:
                        candidates.append(sig)

                candidates.sort(key=lambda x: x["score"], reverse=True)

                # attempt to open as many as we have slots for
                for sig in candidates[:slots]:
                    open_position(state, sig)

            time.sleep(SCAN_INTERVAL_SECONDS)

        except Exception as e:
            print(f"[error] {type(e).__name__}: {e}", flush=True)
            time.sleep(max(5, SCAN_INTERVAL_SECONDS))

if __name__ == "__main__":
    main()
