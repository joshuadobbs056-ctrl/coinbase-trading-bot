import os
import time
import json
import uuid
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests

# ======================
# ENV / CONFIG
# ======================

SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "300"))
MAX_PRODUCTS = int(os.getenv("MAX_PRODUCTS", "250"))
STATE_FILE = os.getenv("STATE_FILE", "state.json")

TRADE_MODE = os.getenv("TRADE_MODE", "paper").lower()  # paper | live (live not enabled in this file)
QUOTE_PER_TRADE_USD = float(os.getenv("QUOTE_PER_TRADE_USD", "25"))
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "1"))
MIN_SCORE_TO_BUY = int(os.getenv("MIN_SCORE_TO_BUY", "7"))

PAPER_START_BALANCE = float(os.getenv("PAPER_START_BALANCE", "1000"))
PAPER_FEE_PCT = float(os.getenv("PAPER_FEE_PCT", "0.004"))  # 0.4% default

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

HTTP_TIMEOUT = 20
USER_AGENT = "accumulation-trading-bot/1.1"

COINBASE_BASE = "https://api.coinbase.com"
# Public market data (no keys needed)
CB_PRODUCTS_URL = COINBASE_BASE + "/api/v3/brokerage/market/products"
CB_CANDLES_URL = COINBASE_BASE + "/api/v3/brokerage/market/products/{product_id}/candles"
CB_BOOK_URL = COINBASE_BASE + "/api/v3/brokerage/market/product_book"

GRANULARITY = "ONE_HOUR"

# ======================
# ACCUMULATION GATES
# ======================
MAX_RANGE_PCT_72H = 8.0
MIN_VOLUME_RATIO_6H_OVER_PRIOR24H = 1.3
MAX_VOLATILITY_PCT_24H = 3.0
MAX_CHANGE_PCT_24H = 5.0
MIN_QUOTE_VOL_24H_USD = 250_000.0  # degen min liquidity
MIN_MARKET_CAP_USD = 10_000_000.0  # only enforced if market cap is present

# Optional huge filter (order-book)
USE_SLIPPAGE_FILTER = True
SIMULATED_ORDER_SIZE_USD = 200.0
MAX_SPREAD_PCT = 0.6
MAX_SLIPPAGE_PCT = 0.8
BOOK_LIMIT_LEVELS = 50

# ======================
# Exit Logic (Auto management)
# ======================
TP1_PCT = 12.0            # sell 30% at +12%
TP2_PCT = 20.0            # sell 30% at +20%
TRAIL_PCT = 6.0           # trail remaining 40% by 6% from peak
HARD_STOP_BELOW_RANGE = 7.0  # emergency stop if price drops 7% below 72h low at entry

# ======================
# DATA MODELS
# ======================

@dataclass
class Candle:
    start: int
    low: float
    high: float
    open: float
    close: float
    volume_base: float

@dataclass
class Metrics:
    product_id: str
    price: float
    lo_72: float
    hi_72: float
    range_pct_72h: float
    volatility_pct_24h: float
    change_pct_24h: float
    quote_vol_24h_usd: float
    volume_ratio: float
    spread_pct: Optional[float] = None
    slippage_pct: Optional[float] = None
    market_cap_usd: Optional[float] = None
    score: int = 0

# ======================
# UTILITIES
# ======================

def now_ts() -> int:
    return int(time.time())

def safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return default
        return float(str(x).strip())
    except Exception:
        return default

def clamp(n: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, n))

def fmt_money(x: float) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1_000_000_000:
        return f"{sign}${x/1_000_000_000:.2f}B"
    if x >= 1_000_000:
        return f"{sign}${x/1_000_000:.2f}M"
    if x >= 1_000:
        return f"{sign}${x/1_000:.2f}K"
    return f"{sign}${x:.2f}"

def fmt_pct(x: float) -> str:
    return f"{x:.2f}%"

def load_state() -> Dict:
    if not os.path.exists(STATE_FILE):
        return {
            "paper": {
                "start_balance": PAPER_START_BALANCE,
                "cash": PAPER_START_BALANCE,
                "realized_pnl": 0.0,
                "fees_paid": 0.0,
                "trades": 0
            },
            "positions": {},  # product_id -> position dict
            "last_scan_ts": 0
        }
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            s = json.load(f)
        # backfill if missing
        if "paper" not in s:
            s["paper"] = {
                "start_balance": PAPER_START_BALANCE,
                "cash": PAPER_START_BALANCE,
                "realized_pnl": 0.0,
                "fees_paid": 0.0,
                "trades": 0
            }
        if "positions" not in s:
            s["positions"] = {}
        return s
    except Exception:
        return {
            "paper": {
                "start_balance": PAPER_START_BALANCE,
                "cash": PAPER_START_BALANCE,
                "realized_pnl": 0.0,
                "fees_paid": 0.0,
                "trades": 0
            },
            "positions": {},
            "last_scan_ts": 0
        }

def save_state(state: Dict) -> None:
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, STATE_FILE)

def telegram_send(msg: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(msg)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    r = requests.post(url, json=payload, timeout=HTTP_TIMEOUT)
    r.raise_for_status()

def http_get_json(url: str, params: Optional[Dict] = None) -> Dict:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    r = requests.get(url, params=params or {}, headers=headers, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()

# ======================
# MARKET DATA (PUBLIC)
# ======================

def list_usd_products(max_products: int) -> List[Dict]:
    products: List[Dict] = []
    cursor = None
    while len(products) < max_products:
        params = {}
        if cursor:
            params["cursor"] = cursor
        data = http_get_json(CB_PRODUCTS_URL, params=params)
        chunk = data.get("products", []) or []
        for p in chunk:
            product_id = p.get("product_id") or ""
            quote = (p.get("quote_display_symbol") or "").upper()
            if not product_id or quote != "USD":
                continue
            if bool(p.get("is_disabled", False)) or bool(p.get("trading_disabled", False)):
                continue
            name = (p.get("display_name") or "").upper()
            if "PERP" in name:
                continue
            products.append(p)
            if len(products) >= max_products:
                break

        pagination = data.get("pagination") or {}
        cursor = pagination.get("next_cursor")
        if not cursor or not pagination.get("has_next", False):
            break
    return products

def get_candles(product_id: str, start_ts: int, end_ts: int) -> List[Candle]:
    url = CB_CANDLES_URL.format(product_id=product_id)
    params = {"start": str(start_ts), "end": str(end_ts), "granularity": GRANULARITY}
    data = http_get_json(url, params=params)
    raw = data.get("candles", []) or []
    candles: List[Candle] = []
    for c in raw:
        candles.append(Candle(
            start=int(c["start"]),
            low=safe_float(c["low"]),
            high=safe_float(c["high"]),
            open=safe_float(c["open"]),
            close=safe_float(c["close"]),
            volume_base=safe_float(c["volume"]),
        ))
    candles.sort(key=lambda x: x.start)
    return candles

def get_product_book(product_id: str) -> Dict:
    return http_get_json(CB_BOOK_URL, params={"product_id": product_id, "limit": str(BOOK_LIMIT_LEVELS)})

# ======================
# INDICATORS
# ======================

def compute_quote_volumes(candles: List[Candle]) -> List[float]:
    # Approx quote volume per candle: base_volume * close_price
    return [max(0.0, c.volume_base) * max(0.0, c.close) for c in candles]

def pct_change(a: float, b: float) -> float:
    if a <= 0:
        return 0.0
    return (b - a) / a * 100.0

def calc_spread_and_slippage(product_id: str, order_size_usd: float) -> Tuple[Optional[float], Optional[float]]:
    book = get_product_book(product_id)
    pb = book.get("pricebook") or {}
    bids = pb.get("bids") or []
    asks = pb.get("asks") or []
    if not bids or not asks:
        return None, None

    best_bid = safe_float(bids[0].get("price"))
    best_ask = safe_float(asks[0].get("price"))
    if best_bid <= 0 or best_ask <= 0:
        return None, None

    spread_pct = ((best_ask - best_bid) / best_ask) * 100.0

    remaining = float(order_size_usd)
    total_cost = 0.0
    total_base = 0.0

    # Walk the asks to simulate a market buy
    for lvl in asks:
        price = safe_float(lvl.get("price"))
        size = safe_float(lvl.get("size"))
        if price <= 0 or size <= 0:
            continue
        lvl_cost = price * size
        take_cost = min(remaining, lvl_cost)
        take_base = take_cost / price
        total_cost += take_cost
        total_base += take_base
        remaining -= take_cost
        if remaining <= 1e-9:
            break

    if total_base <= 0 or remaining > 1e-6:
        return spread_pct, None

    avg_fill = total_cost / total_base
    slippage_pct = ((avg_fill - best_ask) / best_ask) * 100.0
    return spread_pct, slippage_pct

def score_metrics(m: Metrics) -> int:
    score = 0

    # Range
    if m.range_pct_72h <= 5.0:
        score += 2
    elif m.range_pct_72h <= 8.0:
        score += 1

    # Volume ratio
    if m.volume_ratio >= 1.5:
        score += 2
    elif m.volume_ratio >= 1.3:
        score += 1

    # Volatility
    if m.volatility_pct_24h <= 2.0:
        score += 2
    elif m.volatility_pct_24h <= 3.0:
        score += 1

    # Liquidity
    if m.quote_vol_24h_usd >= 5_000_000:
        score += 2
    elif m.quote_vol_24h_usd >= 1_000_000:
        score += 1

    # Market cap (if available)
    if m.market_cap_usd is not None:
        if m.market_cap_usd >= 100_000_000:
            score += 2
        elif m.market_cap_usd >= 25_000_000:
            score += 1

    # Execution bonus
    if m.spread_pct is not None and m.spread_pct <= 0.3:
        score += 1
    if m.slippage_pct is not None and m.slippage_pct <= 0.4:
        score += 1

    return int(clamp(score, 1, 10)) if score > 0 else 1

def passes_accumulation_gates(m: Metrics) -> Tuple[bool, List[str]]:
    reasons = []

    if m.range_pct_72h > MAX_RANGE_PCT_72H:
        reasons.append("range")
    if m.volume_ratio < MIN_VOLUME_RATIO_6H_OVER_PRIOR24H:
        reasons.append("vol_ratio")
    if m.volatility_pct_24h > MAX_VOLATILITY_PCT_24H:
        reasons.append("volatility")
    if m.change_pct_24h > MAX_CHANGE_PCT_24H:
        reasons.append("pump")
    if m.quote_vol_24h_usd < MIN_QUOTE_VOL_24H_USD:
        reasons.append("liquidity")
    if m.market_cap_usd is not None and m.market_cap_usd < MIN_MARKET_CAP_USD:
        reasons.append("mcap")

    if USE_SLIPPAGE_FILTER:
        if m.spread_pct is None or m.spread_pct > MAX_SPREAD_PCT:
            reasons.append("spread")
        if m.slippage_pct is None or m.slippage_pct > MAX_SLIPPAGE_PCT:
            reasons.append("slippage")

    return (len(reasons) == 0, reasons)

def compute_metrics(product: Dict, candles_72h: List[Candle]) -> Optional[Metrics]:
    if len(candles_72h) < 60:
        return None

    price = candles_72h[-1].close
    if price <= 0:
        return None

    hi_72 = max(c.high for c in candles_72h)
    lo_72 = min(c.low for c in candles_72h)
    range_pct_72h = ((hi_72 - lo_72) / price) * 100.0

    candles_24h = candles_72h[-24:]
    closes_24h = [c.close for c in candles_24h if c.close > 0]
    if len(closes_24h) < 18:
        return None

    vol = statistics.pstdev(closes_24h) if len(closes_24h) > 1 else 0.0
    volatility_pct_24h = (vol / price) * 100.0
    change_pct_24h = pct_change(candles_24h[0].close, candles_24h[-1].close)

    quote_vols = compute_quote_volumes(candles_72h)
    quote_vol_24h_usd = sum(quote_vols[-24:])

    recent_6 = quote_vols[-6:]
    prev_24 = quote_vols[-30:-6]
    if len(recent_6) < 6 or len(prev_24) < 18:
        return None

    avg_recent_6 = sum(recent_6) / len(recent_6)
    avg_prev_24 = sum(prev_24) / len(prev_24)
    volume_ratio = (avg_recent_6 / avg_prev_24) if avg_prev_24 > 0 else 0.0

    mcap_raw = product.get("market_cap")
    market_cap_usd = safe_float(mcap_raw, 0.0) if mcap_raw is not None else None
    if market_cap_usd is not None and market_cap_usd <= 0:
        market_cap_usd = None

    return Metrics(
        product_id=product["product_id"],
        price=price,
        lo_72=lo_72,
        hi_72=hi_72,
        range_pct_72h=range_pct_72h,
        volatility_pct_24h=volatility_pct_24h,
        change_pct_24h=change_pct_24h,
        quote_vol_24h_usd=quote_vol_24h_usd,
        volume_ratio=volume_ratio,
        market_cap_usd=market_cap_usd,
    )

# ======================
# PAPER ACCOUNTING
# ======================

def get_open_positions(state: Dict) -> Dict[str, Dict]:
    return state.get("positions", {})

def open_positions_count(state: Dict) -> int:
    return len(get_open_positions(state))

def paper_cash(state: Dict) -> float:
    return float(state["paper"].get("cash", 0.0))

def paper_equity(state: Dict, prices: Dict[str, float]) -> float:
    cash = paper_cash(state)
    pos = get_open_positions(state)
    value = 0.0
    for pid, p in pos.items():
        base = float(p.get("base_size", 0.0))
        px = prices.get(pid, float(p.get("last_price", 0.0)))
        if px > 0:
            value += base * px
    return cash + value

def paper_unrealized_pnl(state: Dict, prices: Dict[str, float]) -> float:
    pos = get_open_positions(state)
    pnl = 0.0
    for pid, p in pos.items():
        base = float(p.get("base_size", 0.0))
        entry = float(p.get("entry_price", 0.0))
        px = prices.get(pid, float(p.get("last_price", 0.0)))
        if entry > 0 and px > 0:
            pnl += base * (px - entry)
    return pnl

def apply_fee(amount: float) -> Tuple[float, float]:
    """
    Returns (net_amount_after_fee, fee_amount)
    """
    fee = abs(amount) * PAPER_FEE_PCT
    net = amount - fee
    return net, fee

def paper_buy(state: Dict, m: Metrics) -> Optional[str]:
    """
    Simulated market buy spending QUOTE_PER_TRADE_USD.
    Uses slippage estimate (if present) to worsen fill slightly.
    """
    if paper_cash(state) < QUOTE_PER_TRADE_USD:
        return "insufficient_cash"

    # Simulated fill price: worse than last price by slippage if available
    fill_price = m.price
    if m.slippage_pct is not None:
        fill_price = m.price * (1.0 + max(0.0, m.slippage_pct) / 100.0)

    if fill_price <= 0:
        return "bad_price"

    quote_spend = QUOTE_PER_TRADE_USD
    net_spend, fee = apply_fee(quote_spend)  # fee taken from spend
    base_bought = net_spend / fill_price

    state["paper"]["cash"] = paper_cash(state) - quote_spend
    state["paper"]["fees_paid"] = float(state["paper"].get("fees_paid", 0.0)) + fee
    state["paper"]["trades"] = int(state["paper"].get("trades", 0)) + 1

    state["positions"][m.product_id] = {
        "product_id": m.product_id,
        "entry_price": fill_price,
        "entry_ts": now_ts(),
        "base_size": base_bought,
        "quote_spent": quote_spend,
        "tp1_done": False,
        "tp2_done": False,
        "peak_price": fill_price,
        "accum_low": m.lo_72,
        "last_price": m.price,
    }
    return None

def paper_sell_fraction(state: Dict, product_id: str, fraction: float, price: float, reason: str) -> Optional[Dict]:
    """
    Sell a fraction of base_size at given price. Updates cash + realized pnl.
    """
    pos = state["positions"].get(product_id)
    if not pos:
        return None

    base_total = float(pos.get("base_size", 0.0))
    if base_total <= 0:
        return None

    fraction = clamp(fraction, 0.0, 1.0)
    base_to_sell = base_total * fraction
    if base_to_sell <= 0:
        return None

    entry = float(pos.get("entry_price", 0.0))
    if entry <= 0 or price <= 0:
        return None

    gross_proceeds = base_to_sell * price
    net_proceeds, fee = apply_fee(gross_proceeds)

    cost_basis = base_to_sell * entry
    realized = net_proceeds - cost_basis

    # Update state
    pos["base_size"] = base_total - base_to_sell
    state["paper"]["cash"] = paper_cash(state) + net_proceeds
    state["paper"]["realized_pnl"] = float(state["paper"].get("realized_pnl", 0.0)) + realized
    state["paper"]["fees_paid"] = float(state["paper"].get("fees_paid", 0.0)) + fee
    state["paper"]["trades"] = int(state["paper"].get("trades", 0)) + 1

    if pos["base_size"] <= 1e-12:
        # close position completely
        state["positions"].pop(product_id, None)

    return {
        "product_id": product_id,
        "fraction": fraction,
        "base_sold": base_to_sell,
        "price": price,
        "net_proceeds": net_proceeds,
        "fee": fee,
        "realized": realized,
        "reason": reason,
    }

# ======================
# POSITION MANAGEMENT
# ======================

def manage_positions(state: Dict, prices: Dict[str, float]) -> None:
    """
    Apply TP1, TP2, trailing stop, hard stop for each position.
    """
    positions = list(get_open_positions(state).items())

    for pid, pos in positions:
        px = prices.get(pid)
        if px is None or px <= 0:
            continue

        entry = float(pos.get("entry_price", 0.0))
        if entry <= 0:
            continue

        # Update peak
        peak = float(pos.get("peak_price", entry))
        if px > peak:
            peak = px
            pos["peak_price"] = peak

        pos["last_price"] = px

        # Hard stop: price below accum_low by HARD_STOP_BELOW_RANGE%
        accum_low = float(pos.get("accum_low", 0.0))
        if accum_low > 0:
            hard_stop_price = accum_low * (1.0 - HARD_STOP_BELOW_RANGE / 100.0)
            if px <= hard_stop_price:
                sale = paper_sell_fraction(state, pid, 1.0, px, reason="HARD_STOP")
                if sale:
                    send_trade_receipt(state, sale, prices)
                continue

        # TP1
        if not bool(pos.get("tp1_done", False)) and px >= entry * (1.0 + TP1_PCT / 100.0):
            sale = paper_sell_fraction(state, pid, 0.30, px, reason=f"TP1(+{TP1_PCT:.0f}%)")
            if sale:
                pos = state["positions"].get(pid, {})  # refresh after partial
                if pos:
                    pos["tp1_done"] = True
                send_trade_receipt(state, sale, prices)

        # TP2
        if pid in state["positions"]:
            pos = state["positions"][pid]
            if not bool(pos.get("tp2_done", False)) and px >= entry * (1.0 + TP2_PCT / 100.0):
                sale = paper_sell_fraction(state, pid, 0.30, px, reason=f"TP2(+{TP2_PCT:.0f}%)")
                if sale:
                    pos = state["positions"].get(pid, {})
                    if pos:
                        pos["tp2_done"] = True
                    send_trade_receipt(state, sale, prices)

        # Trailing stop for remaining
        if pid in state["positions"]:
            pos = state["positions"][pid]
            peak = float(pos.get("peak_price", entry))
            trail_stop = peak * (1.0 - TRAIL_PCT / 100.0)
            # Only start trailing once we have at least some profit cushion:
            # if peak >= entry*(1+TP1/2) OR tp1 already done
            if peak >= entry * (1.0 + (TP1_PCT / 2.0) / 100.0) or bool(pos.get("tp1_done", False)):
                if px <= trail_stop:
                    sale = paper_sell_fraction(state, pid, 1.0, px, reason=f"TRAIL_STOP(-{TRAIL_PCT:.0f}% from peak)")
                    if sale:
                        send_trade_receipt(state, sale, prices)

def send_buy_receipt(state: Dict, m: Metrics, fill_price: float) -> None:
    prices = {m.product_id: m.price}
    eq = paper_equity(state, prices)
    cash = paper_cash(state)
    unreal = paper_unrealized_pnl(state, prices)
    realized = float(state["paper"].get("realized_pnl", 0.0))
    fees = float(state["paper"].get("fees_paid", 0.0))

    msg = (
        f"<b>PAPER BUY</b> — <b>{m.product_id}</b>\n"
        f"Score: <b>{m.score}/10</b>\n"
        f"Fill: <b>${fill_price:.6g}</b> (last: ${m.price:.6g})\n"
        f"Spent: <b>{fmt_money(QUOTE_PER_TRADE_USD)}</b>\n\n"
        f"Cash: <b>{fmt_money(cash)}</b>\n"
        f"Equity: <b>{fmt_money(eq)}</b>\n"
        f"Unrealized: <b>{fmt_money(unreal)}</b>\n"
        f"Realized: <b>{fmt_money(realized)}</b>\n"
        f"Fees paid: <b>{fmt_money(fees)}</b>"
    )
    telegram_send(msg)

def send_trade_receipt(state: Dict, sale: Dict, prices: Dict[str, float]) -> None:
    eq = paper_equity(state, prices)
    cash = paper_cash(state)
    unreal = paper_unrealized_pnl(state, prices)
    realized = float(state["paper"].get("realized_pnl", 0.0))
    fees = float(state["paper"].get("fees_paid", 0.0))

    msg = (
        f"<b>PAPER SELL</b> — <b>{sale['product_id']}</b>\n"
        f"Reason: <b>{sale['reason']}</b>\n"
        f"Sold: <b>{sale['fraction']*100:.0f}%</b>\n"
        f"Price: <b>${sale['price']:.6g}</b>\n"
        f"Realized PnL: <b>{fmt_money(sale['realized'])}</b>\n"
        f"Fee: <b>{fmt_money(sale['fee'])}</b>\n\n"
        f"Cash: <b>{fmt_money(cash)}</b>\n"
        f"Equity: <b>{fmt_money(eq)}</b>\n"
        f"Unrealized: <b>{fmt_money(unreal)}</b>\n"
        f"Total Realized: <b>{fmt_money(realized)}</b>\n"
        f"Fees paid: <b>{fmt_money(fees)}</b>"
    )
    telegram_send(msg)

# ======================
# MAIN SCAN / TRADE LOOP
# ======================

def scan_and_trade_once(state: Dict) -> None:
    ts = now_ts()
    state["last_scan_ts"] = ts

    # 72h candles window
    end_ts = ts
    start_ts = ts - 72 * 3600

    # 1) Load products
    products = list_usd_products(MAX_PRODUCTS)
    print(f"[scan] products={len(products)} open_positions={open_positions_count(state)} cash={paper_cash(state):.2f}")

    # 2) Build prices dict for equity calc & management
    current_prices: Dict[str, float] = {}

    # 3) Manage existing positions first (sell logic)
    # We need updated prices for held positions; we’ll get them from candles.
    held_ids = list(get_open_positions(state).keys())
    for pid in held_ids:
        try:
            candles = get_candles(pid, start_ts, end_ts)
            if candles:
                current_prices[pid] = candles[-1].close
        except Exception as e:
            print(f"[warn] price fetch failed {pid}: {type(e).__name__}")

    manage_positions(state, current_prices)

    # Refresh held ids after possible sells
    held_ids = set(get_open_positions(state).keys())

    # 4) If we have capacity, find best new accumulation candidates to buy
    if TRADE_MODE not in ("paper", "live"):
        print(f"[warn] TRADE_MODE={TRADE_MODE} not recognized, forcing paper")
    if open_positions_count(state) >= MAX_OPEN_POSITIONS:
        save_state(state)
        return

    candidates: List[Metrics] = []
    for p in products:
        pid = p.get("product_id")
        if not pid or pid in held_ids:
            continue

        try:
            candles = get_candles(pid, start_ts, end_ts)
            m = compute_metrics(p, candles)
            if not m:
                continue

            # Order book filter (optional huge)
            if USE_SLIPPAGE_FILTER:
                spread, slip = calc_spread_and_slippage(pid, SIMULATED_ORDER_SIZE_USD)
                m.spread_pct = spread
                m.slippage_pct = slip

            m.score = score_metrics(m)
            ok, _reasons = passes_accumulation_gates(m)

            # track price for equity reporting
            current_prices[pid] = m.price

            if not ok:
                continue
            if m.score < MIN_SCORE_TO_BUY:
                continue

            candidates.append(m)

        except Exception as e:
            # noisy pair failures are normal
            continue

    candidates.sort(key=lambda x: (x.score, x.quote_vol_24h_usd), reverse=True)

    # 5) Execute one buy per scan max (keeps it sane)
    if candidates and open_positions_count(state) < MAX_OPEN_POSITIONS and paper_cash(state) >= QUOTE_PER_TRADE_USD:
        best = candidates[0]

        if TRADE_MODE == "paper":
            err = paper_buy(state, best)
            if err:
                print(f"[paper_buy] blocked: {err}")
            else:
                pos = state["positions"][best.product_id]
                fill = float(pos["entry_price"])
                print(f"[paper_buy] {best.product_id} score={best.score} fill={fill:.6g}")
                send_buy_receipt(state, best, fill)

        elif TRADE_MODE == "live":
            # Live trading not enabled in this file on purpose (safety).
            telegram_send("<b>LIVE MODE</b> requested but not implemented in this build. Keep TRADE_MODE=paper.")
            print("[live] not implemented")

    # 6) Save state
    save_state(state)

    # Optional: periodic account summary
    eq = paper_equity(state, current_prices)
    start_bal = float(state["paper"].get("start_balance", PAPER_START_BALANCE))
    realized = float(state["paper"].get("realized_pnl", 0.0))
    unreal = paper_unrealized_pnl(state, current_prices)
    growth = ((eq - start_bal) / start_bal * 100.0) if start_bal > 0 else 0.0

    print(f"[acct] equity={eq:.2f} growth={growth:.2f}% realized={realized:.2f} unreal={unreal:.2f} positions={open_positions_count(state)}")

def main():
    state = load_state()

    # Ensure paper account initialized properly
    if "paper" not in state:
        state["paper"] = {
            "start_balance": PAPER_START_BALANCE,
            "cash": PAPER_START_BALANCE,
            "realized_pnl": 0.0,
            "fees_paid": 0.0,
            "trades": 0
        }

    telegram_send(
        f"<b>BOT STARTED</b>\n"
        f"Mode: <b>{TRADE_MODE.upper()}</b>\n"
        f"Start balance: <b>{fmt_money(float(state['paper'].get('start_balance', PAPER_START_BALANCE)))}</b>\n"
        f"Cash: <b>{fmt_money(paper_cash(state))}</b>\n"
        f"Per trade: <b>{fmt_money(QUOTE_PER_TRADE_USD)}</b>\n"
        f"Max positions: <b>{MAX_OPEN_POSITIONS}</b>\n"
        f"Min score to buy: <b>{MIN_SCORE_TO_BUY}</b>\n"
        f"Scan every: <b>{SCAN_INTERVAL_SECONDS}s</b>\n"
        f"Fee: <b>{PAPER_FEE_PCT*100:.2f}%</b>"
    )

    print("[start] accumulation trading bot running (paper ledger enabled)")
    while True:
        try:
            scan_and_trade_once(state)
        except Exception as e:
            print(f"[fatal] scan failed: {type(e).__name__}: {e}")
        time.sleep(SCAN_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
