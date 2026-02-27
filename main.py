import os
import time
import json
import uuid
import math
import statistics
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import requests

# Coinbase official JWT helper (recommended)
# pip package: coinbase-advanced-py (imports as "coinbase")
from coinbase import jwt_generator  # uses ES256 and endpoint-bound JWTs per docs

# ======================
# ENV / CONFIG
# ======================

SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "300"))
MAX_PRODUCTS = int(os.getenv("MAX_PRODUCTS", "250"))
STATE_FILE = os.getenv("STATE_FILE", "state.json")

TRADE_MODE = os.getenv("TRADE_MODE", "paper").lower()  # paper | live
QUOTE_PER_TRADE_USD = float(os.getenv("QUOTE_PER_TRADE_USD", "50"))
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "1"))
MIN_SCORE_TO_BUY = int(os.getenv("MIN_SCORE_TO_BUY", "7"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

COINBASE_API_KEY = os.getenv("COINBASE_API_KEY", "").strip()
COINBASE_API_SECRET = os.getenv("COINBASE_API_SECRET", "").strip()

HTTP_TIMEOUT = 20
USER_AGENT = "accumulation-trading-bot/1.0"

COINBASE_BASE = "https://api.coinbase.com"
# Public market data
CB_PRODUCTS_URL = COINBASE_BASE + "/api/v3/brokerage/market/products"
CB_CANDLES_URL = COINBASE_BASE + "/api/v3/brokerage/market/products/{product_id}/candles"
CB_BOOK_URL = COINBASE_BASE + "/api/v3/brokerage/market/product_book"
# Private trading
CB_CREATE_ORDER_URL = COINBASE_BASE + "/api/v3/brokerage/orders"
CB_LIST_ACCOUNTS_URL = COINBASE_BASE + "/api/v3/brokerage/accounts"
CB_GET_PRODUCT_URL = COINBASE_BASE + "/api/v3/brokerage/products/{product_id}"

GRANULARITY = "ONE_HOUR"

# ======================
# ACCUMULATION GATES
# ======================
MAX_RANGE_PCT_72H = 8.0
MIN_VOLUME_RATIO_6H_OVER_PRIOR24H = 1.3
MAX_VOLATILITY_PCT_24H = 3.0
MAX_CHANGE_PCT_24H = 5.0
MIN_QUOTE_VOL_24H_USD = 250_000.0  # degen min liquidity
MIN_MARKET_CAP_USD = 10_000_000.0  # if available

# Optional huge filter
USE_SLIPPAGE_FILTER = True
SIMULATED_ORDER_SIZE_USD = 200.0
MAX_SPREAD_PCT = 0.6
MAX_SLIPPAGE_PCT = 0.8
BOOK_LIMIT_LEVELS = 50

# Exit logic
TP1_PCT = 12.0
TP2_PCT = 20.0
TRAIL_PCT = 6.0          # trailing stop for runner portion
HARD_STOP_BELOW_RANGE = 7.0  # % below 72h low triggers emergency exit

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

@dataclass
class Position:
    product_id: str
    entry_price: float
    entry_ts: int
    base_size: float
    quote_spent: float
    tp1_done: bool = False
    tp2_done: bool = False
    peak_price: float = 0.0
    accum_low: float = 0.0  # the 72h low at entry time, used for hard stop

# ======================
# HELPERS
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

def load_state() -> Dict:
    if not os.path.exists(STATE_FILE):
        return {"positions": {}, "last_buy_ts": 0}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"positions": {}, "last_buy_ts": 0}

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

def http_get_json(url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None) -> Dict:
    h = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if headers:
        h.update(headers)
    r = requests.get(url, params=params or {}, headers=h, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()

def http_post_json(url: str, body: Dict, headers: Optional[Dict] = None) -> Dict:
    h = {"User-Agent": USER_AGENT, "Accept": "application/json", "Content-Type": "application/json"}
    if headers:
        h.update(headers)
    r = requests.post(url, json=body, headers=h, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()

# ======================
# COINBASE AUTH (JWT)
# ======================

def coinbase_auth_headers(method: str, path: str) -> Dict:
    """
    Coinbase Advanced Trade private endpoints: Authorization: Bearer <JWT>
    JWT must be built for (method, path). See Coinbase SDK examples.
    """
    if not COINBASE_API_KEY or not COINBASE_API_SECRET:
        raise RuntimeError("Missing COINBASE_API_KEY or COINBASE_API_SECRET")

    jwt_uri = jwt_generator.format_jwt_uri(method.upper(), path)
    token = jwt_generator.build_rest_jwt(jwt_uri, COINBASE_API_KEY, COINBASE_API_SECRET)
    return {"Authorization": "Bearer " + token}

# ======================
# MARKET DATA
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

def compute_quote_volumes(candles: List[Candle]) -> List[float]:
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
    if m.range_pct_72h <= 5.0: score += 2
    elif m.range_pct_72h <= 8.0: score += 1
    # Volume ratio
    if m.volume_ratio >= 1.5: score += 2
    elif m.volume_ratio >= 1.3: score += 1
    # Volatility
    if m.volatility_pct_24h <= 2.0: score += 2
    elif m.volatility_pct_24h <= 3.0: score += 1
    # Liquidity
    if m.quote_vol_24h_usd >= 5_000_000: score += 2
    elif m.quote_vol_24h_usd >= 1_000_000: score += 1
    # Market cap (if available)
    if m.market_cap_usd is not None:
        if m.market_cap_usd >= 100_000_000: score += 2
        elif m.market_cap_usd >= 25_000_000: score += 1
    # Tight execution bonus
    if m.spread_pct is not None and m.spread_pct <= 0.3: score += 1
    if m.slippage_pct is not None and m.slippage_pct <= 0.4: score += 1

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
# TRADING (SPOT)
# ======================

def place_market_buy(product_id: str, quote_usd: float) -> Dict:
    """
    Advanced Trade Create Order endpoint uses Authorization: Bearer <JWT>
    Body uses order_configuration.market_market_ioc with quote_size or base_size.
    """
    body = {
        "client_order_id": str(uuid.uuid4()),
       
