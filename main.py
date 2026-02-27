import os
import time
import json
import uuid
import math
from typing import Dict, List, Optional, Tuple

import requests

# =========================
# CONFIG / ENV
# =========================

STATE_FILE = os.getenv("STATE_FILE", "state.json")

SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "60"))   # active bot = faster scans
MAX_PRODUCTS = int(os.getenv("MAX_PRODUCTS", "250"))

TRADE_MODE = os.getenv("TRADE_MODE", "paper").lower()  # paper only in this build

# Paper account
PAPER_START_BALANCE = float(os.getenv("PAPER_START_BALANCE", "1000"))
PAPER_FEE_PCT = float(os.getenv("PAPER_FEE_PCT", "0.006"))  # 0.6% taker-ish simulation

# Risk controls
QUOTE_PER_TRADE_USD = float(os.getenv("QUOTE_PER_TRADE_USD", "100"))
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "2"))
MAX_BUYS_PER_HOUR = int(os.getenv("MAX_BUYS_PER_HOUR", "4"))
COOLDOWN_SECONDS_AFTER_BUY = int(os.getenv("COOLDOWN_SECONDS_AFTER_BUY", "300"))  # 5 min

# Strategy (micro breakout)
GRANULARITY = os.getenv("GRANULARITY", "FIVE_MINUTE")  # 5m candles
LOOKBACK_CANDLES = int(os.getenv("LOOKBACK_CANDLES", "72"))  # 6 hours of 5m candles
BREAKOUT_WINDOW = int(os.getenv("BREAKOUT_WINDOW", "12"))     # 1 hour window (12 x 5m)
BREAKOUT_BUFFER_PCT = float(os.getenv("BREAKOUT_BUFFER_PCT", "0.12"))  # 0.12% above range high

MIN_VOL_SPIKE_MULT = float(os.getenv("MIN_VOL_SPIKE_MULT", "1.8"))    # last vol > avg * mult
MIN_ATR_PCT = float(os.getenv("MIN_ATR_PCT", "0.20"))                 # avoid dead coins
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "0.35"))           # execution quality
USE_SPREAD_FILTER = os.getenv("USE_SPREAD_FILTER", "true").lower() in ("1", "true", "yes", "y")

# Exits (fast)
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "1.6"))          # +1.6%
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.9"))              # -0.9%
TRAIL_START_PCT = float(os.getenv("TRAIL_START_PCT", "1.0"))          # start trailing after +1.0%
TRAIL_PCT = float(os.getenv("TRAIL_PCT", "0.7"))                      # trail by 0.7%
MAX_HOLD_SECONDS = int(os.getenv("MAX_HOLD_SECONDS", str(2 * 60 * 60)))  # 2 hours

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# Coinbase public API
BASE_URL = "https://api.coinbase.com"
PRODUCTS_URL = BASE_URL + "/api/v3/brokerage/market/products"
CANDLES_URL = BASE_URL + "/api/v3/brokerage/market/products/{product_id}/candles"
BOOK_URL = BASE_URL + "/api/v3/brokerage/market/product_book"

HTTP_TIMEOUT = 20
USER_AGENT = "active-momentum-paper-bot/1.0"

# =========================
# UTIL
# =========================

def now_ts() -> int:
    return int(time.time())

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def fmt_money(x: float) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    return f"{sign}${x:,.2f}"

def fmt_pct(x: float) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.2f}%"

def http_get_json(url: str, params: Optional[Dict] = None) -> Dict:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    r = requests.get(url, params=params or {}, headers=headers, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()

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
    try:
        requests.post(url, json=payload, timeout=HTTP_TIMEOUT).raise_for_status()
    except Exception:
        # don't crash trading loop on telegram failures
        print("[warn] telegram failed")

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
            "trades": 0,
            "wins": 0,
            "losses": 0,
        },
        "positions": {},  # product_id -> position dict
        "last_buy_ts": 0,
        "buy_timestamps": [],  # for hourly throttle
        "last_summary_ts": 0,
    }

def load_state() -> Dict:
    if not os.path.exists(STATE_FILE):
        return fresh_state()
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            s = json.load(f)
        # backfill
        if "paper" not in s or "positions" not in s:
            return fresh_state()
        s["paper"].setdefault("start_balance", PAPER_START_BALANCE)
        s["paper"].setdefault("cash", s["paper"]["start_balance"])
        s["paper"].setdefault("realized_pnl", 0.0)
        s["paper"].setdefault("fees_paid", 0.0)
        s["paper"].setdefault("trades", 0)
        s["paper"].setdefault("wins", 0)
        s["paper"].setdefault("losses", 0)
        s.setdefault("positions", {})
        s.setdefault("last_buy_ts", 0)
        s.setdefault("buy_timestamps", [])
        s.setdefault("last_summary_ts", 0)
        return s
    except Exception:
        return fresh_state()

def save_state(state: Dict) -> None:
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, STATE_FILE)

# =========================
# PAPER ACCOUNTING
# =========================

def cash(state: Dict) -> float:
    return float(state["paper"].get("cash", 0.0))

def apply_fee(amount: float) -> Tuple[float, float]:
    fee = abs(amount) * PAPER_FEE_PCT
    return amount - fee, fee

def equity(state: Dict, prices: Dict[str, float]) -> float:
    eq = cash(state)
    for pid, pos in state["positions"].items():
        base = float(pos.get("base_size", 0.0))
        px = prices.get(pid, float(pos.get("last_price", 0.0)))
        if base > 0 and px > 0:
            eq += base * px
    return eq

# =========================
# COINBASE DATA
# =========================

def list_usd_products(max_products: int) -> List[Dict]:
    """
    Pull Coinbase USD products. Uses pagination cursor if present.
    """
    results: List[Dict] = []
    cursor = None

    while len(results) < max_products:
        params = {}
        if cursor:
            params["cursor"] = cursor

        data = http_get_json(PRODUCTS_URL, params=params)
        products = data.get("products") or []
        for p in products:
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
            results.append(p)
            if len(results) >= max_products:
                break

        pagination = data.get("pagination") or {}
        cursor = pagination.get("next_cursor")
        if not cursor or not pagination.get("has_next", False):
            break

    return results

def get_candles(product_id: str, lookback_candles: int) -> Optional[Dict[str, List[float]]]:
    """
    Returns dict with lists: open, high, low, close, volume (base volume)
    Uses proper start/end parameters.
    """
    end_ts = now_ts()
    # Candle length for FIVE_MINUTE: 300 sec; ONE_MINUTE: 60 sec; etc.
    # We'll estimate seconds-per-candle based on granularity name.
    gran = GRANULARITY.upper()
    sec_per = 300 if gran == "FIVE_MINUTE" else 3600 if gran == "ONE_HOUR" else 900 if gran == "FIFTEEN_MINUTE" else 60
    start_ts = end_ts - (lookback_candles + 10) * sec_per  # buffer

    url = CANDLES_URL.format(product_id=product_id)
    params = {"start": str(start_ts), "end": str(end_ts), "granularity": gran}

    data = http_get_json(url, params=params)
    raw = data.get("candles") or []
    if len(raw) < lookback_candles:
        return None

    # Coinbase returns newest-first typically; we sort by start
    raw_sorted = sorted(raw, key=lambda c: int(c["start"]))
    raw_sorted = raw_sorted[-lookback_candles:]

    o, h, l, c, v = [], [], [], [], []
    for x in raw_sorted:
        o.append(safe_float(x.get("open"), 0.0))
        h.append(safe_float(x.get("high"), 0.0))
        l.append(safe_float(x.get("low"), 0.0))
        c.append(safe_float(x.get("close"), 0.0))
        v.append(safe_float(x.get("volume"), 0.0))
    if not c or c[-1] <= 0:
        return None
    return {"open": o, "high": h, "low": l, "close": c, "volume": v}

def get_spread_pct(product_id: str) -> Optional[float]:
    try:
        data = http_get_json(BOOK_URL, params={"product_id": product_id, "limit": "1"})
        pb = data.get("pricebook") or {}
        bids = pb.get("bids") or []
        asks = pb.get("asks") or []
        if not bids or not asks:
            return None
        best_bid = safe_float(bids[0].get("price"), 0.0)
        best_ask = safe_float(asks[0].get("price"), 0.0)
        if best_bid <= 0 or best_ask <= 0:
            return None
        return ((best_ask - best_bid) / best_ask) * 100.0
    except Exception:
        return None

# =========================
# INDICATORS
# =========================

def atr_pct(high: List[float], low: List[float], close: List[float], period: int = 14) -> float:
    if len(close) < period + 1:
        return 0.0
    trs = []
    for i in range(-period, 0):
        prev_close = close[i - 1]
        tr = max(high[i] - low[i], abs(high[i] - prev_close), abs(low[i] - prev_close))
        trs.append(tr)
    atr = sum(trs) / len(trs)
    px = close[-1]
    return (atr / px) * 100.0 if px > 0 else 0.0

def rsi(close: List[float], period: int = 14) -> float:
    if len(close) < period + 1:
        return 50.0
    gains = 0.0
    losses = 0.0
    for i in range(-period, 0):
        delta = close[i] - close[i - 1]
        if delta >= 0:
            gains += delta
        else:
            losses += abs(delta)
    if losses == 0:
        return 100.0
    rs = (gains / period) / (losses / period)
    return 100.0 - (100.0 / (1.0 + rs))

def pct_change(a: float, b: float) -> float:
    if a <= 0:
        return 0.0
    return (b - a) / a * 100.0

# =========================
# ENTRY SIGNAL (ACTIVE TRADING)
# =========================

def is_breakout_signal(cndl: Dict[str, List[float]]) -> Tuple[bool, Dict]:
    """
    Active trading signal:
    - Break above the previous 1 hour range high by BREAKOUT_BUFFER_PCT
    - Volume spike vs avg vol of that window
    - ATR not too tiny (avoid dead moves)
    - RSI supports momentum (avoid weak drift)
    """
    o = cndl["open"]
    h = cndl["high"]
    l = cndl["low"]
    c = cndl["close"]
    v = cndl["volume"]

    if len(c) < max(LOOKBACK_CANDLES, BREAKOUT_WINDOW + 5):
        return False, {}

    # Previous window excludes the latest candle
    window_high = max(h[-(BREAKOUT_WINDOW + 1):-1])
    window_low = min(l[-(BREAKOUT_WINDOW + 1):-1])

    last_close = c[-1]
    last_vol = v[-1]
    avg_vol = sum(v[-(BREAKOUT_WINDOW + 1):-1]) / float(BREAKOUT_WINDOW) if BREAKOUT_WINDOW > 0 else 0.0

    # Conditions
    buffer = 1.0 + (BREAKOUT_BUFFER_PCT / 100.0)
    breakout = last_close > (window_high * buffer)

    vol_spike = (avg_vol > 0) and (last_vol >= avg_vol * MIN_VOL_SPIKE_MULT)

    ap = atr_pct(h, l, c, period=14)
    atr_ok = ap >= MIN_ATR_PCT

    r = rsi(c, period=14)
    rsi_ok = r >= 56.0

    # Avoid already-extended candles (optional sanity)
    last_up = pct_change(o[-1], c[-1])
    not_too_extended = last_up <= 3.5  # don't buy crazy 8% 5m candles

    meta = {
        "window_high": window_high,
        "window_low": window_low,
        "last_close": last_close,
        "avg_vol": avg_vol,
        "last_vol": last_vol,
        "atr_pct": ap,
        "rsi": r,
        "last_candle_pct": last_up,
    }

    ok = breakout and vol_spike and atr_ok and rsi_ok and not_too_extended
    return ok, meta

# =========================
# TRADING ACTIONS (PAPER)
# =========================

def can_buy(state: Dict) -> Tuple[bool, str]:
    if cash(state) < QUOTE_PER_TRADE_USD:
        return False, "insufficient_cash"
    if len(state["positions"]) >= MAX_OPEN_POSITIONS:
        return False, "max_positions"
    if (now_ts() - int(state.get("last_buy_ts", 0))) < COOLDOWN_SECONDS_AFTER_BUY:
        return False, "cooldown"

    # hourly throttle
    cutoff = now_ts() - 3600
    state["buy_timestamps"] = [t for t in state.get("buy_timestamps", []) if t >= cutoff]
    if len(state["buy_timestamps"]) >= MAX_BUYS_PER_HOUR:
        return False, "hourly_throttle"

    return True, "ok"

def paper_buy(state: Dict, product_id: str, price: float, signal_meta: Dict) -> None:
    # Prevent duplicates
    if product_id in state["positions"]:
        return

    spend = QUOTE_PER_TRADE_USD
    net_spend, fee = apply_fee(spend)

    base = net_spend / price if price > 0 else 0.0
    if base <= 0:
        return

    state["paper"]["cash"] = cash(state) - spend
    state["paper"]["fees_paid"] = float(state["paper"].get("fees_paid", 0.0)) + fee
    state["paper"]["trades"] = int(state["paper"].get("trades", 0)) + 1

    state["positions"][product_id] = {
        "id": str(uuid.uuid4()),
        "product_id": product_id,
        "entry_price": price,
        "entry_ts": now_ts(),
        "base_size": base,
        "peak_price": price,
        "last_price": price,
        "meta": {
            "atr_pct": signal_meta.get("atr_pct"),
            "rsi": signal_meta.get("rsi"),
            "window_high": signal_meta.get("window_high"),
        }
    }

    state["last_buy_ts"] = now_ts()
    state.setdefault("buy_timestamps", []).append(now_ts())

    save_state(state)

    telegram_send(
        f"<b>PAPER BUY</b> — <b>{product_id}</b>\n"
        f"Entry: <b>${price:.6g}</b> | Size: <b>{fmt_money(spend)}</b>\n"
        f"ATR14: <b>{signal_meta.get('atr_pct', 0):.2f}%</b> | RSI14: <b>{signal_meta.get('rsi', 0):.1f}</b>\n"
        f"VolSpike: <b>{(signal_meta.get('last_vol',0)/(signal_meta.get('avg_vol',1) or 1)):.2f}x</b>\n"
        f"Cash: <b>{fmt_money(cash(state))}</b>"
    )

def paper_sell_all(state: Dict, product_id: str, price: float, reason: str) -> None:
    pos = state["positions"].get(product_id)
    if not pos:
        return

    base = float(pos.get("base_size", 0.0))
    entry = float(pos.get("entry_price", 0.0))
    if base <= 0 or entry <= 0 or price <= 0:
        state["positions"].pop(product_id, None)
        save_state(state)
        return

    gross = base * price
    net, fee = apply_fee(gross)

    cost_basis = base * entry
    pnl = net - cost_basis
    pnl_pct = (pnl / cost_basis) * 100.0 if cost_basis > 0 else 0.0

    state["paper"]["cash"] = cash(state) + net
    state["paper"]["fees_paid"] = float(state["paper"].get("fees_paid", 0.0)) + fee
    state["paper"]["realized_pnl"] = float(state["paper"].get("realized_pnl", 0.0)) + pnl
    state["paper"]["trades"] = int(state["paper"].get("trades", 0)) + 1

    if pnl >= 0:
        state["paper"]["wins"] = int(state["paper"].get("wins", 0)) + 1
    else:
        state["paper"]["losses"] = int(state["paper"].get("losses", 0)) + 1

    state["positions"].pop(product_id, None)
    save_state(state)

    telegram_send(
        f"<b>PAPER SELL</b> — <b>{product_id}</b>\n"
        f"Reason: <b>{reason}</b>\n"
        f"Exit: <b>${price:.6g}</b>\n"
        f"PnL: <b>{fmt_money(pnl)}</b> ({fmt_pct(pnl_pct)})\n"
        f"Cash: <b>{fmt_money(cash(state))}</b>"
    )

# =========================
# POSITION MANAGEMENT (FAST)
# =========================

def manage_positions(state: Dict, prices: Dict[str, float]) -> None:
    for pid in list(state["positions"].keys()):
        pos = state["positions"].get(pid)
        if not pos:
            continue

        px = prices.get(pid)
        if px is None or px <= 0:
            continue

        entry = float(pos.get("entry_price", 0.0))
        if entry <= 0:
            continue

        # update peak
        peak = float(pos.get("peak_price", entry))
        if px > peak:
            peak = px
            pos["peak_price"] = peak
        pos["last_price"] = px

        # hard exits
        change_pct = pct_change(entry, px)

        # Take profit
        if change_pct >= TAKE_PROFIT_PCT:
            paper_sell_all(state, pid, px, reason=f"TP {TAKE_PROFIT_PCT:.2f}%")
            continue

        # Stop loss
        if change_pct <= -STOP_LOSS_PCT:
            paper_sell_all(state, pid, px, reason=f"SL {STOP_LOSS_PCT:.2f}%")
            continue

        # Trailing stop after we are up a bit
        peak_gain = pct_change(entry, peak)
        if peak_gain >= TRAIL_START_PCT:
            trail_price = peak * (1.0 - (TRAIL_PCT / 100.0))
            if px <= trail_price:
                paper_sell_all(state, pid, px, reason=f"TRAIL {TRAIL_PCT:.2f}%")
                continue

        # Max hold time (avoid capital getting stuck)
        age = now_ts() - int(pos.get("entry_ts", now_ts()))
        if age >= MAX_HOLD_SECONDS:
            paper_sell_all(state, pid, px, reason=f"MAX_HOLD {MAX_HOLD_SECONDS//60}m")
            continue

# =========================
# SUMMARY REPORTING
# =========================

def maybe_send_summary(state: Dict, prices: Dict[str, float]) -> None:
    # send every 30 minutes
    interval = 1800
    last = int(state.get("last_summary_ts", 0))
    if (now_ts() - last) < interval:
        return

    start_bal = float(state["paper"].get("start_balance", PAPER_START_BALANCE))
    eq = equity(state, prices)
    realized = float(state["paper"].get("realized_pnl", 0.0))
    fees = float(state["paper"].get("fees_paid", 0.0))
    wins = int(state["paper"].get("wins", 0))
    losses = int(state["paper"].get("losses", 0))
    trades = int(state["paper"].get("trades", 0))
    growth_pct = ((eq - start_bal) / start_bal * 100.0) if start_bal > 0 else 0.0
    open_pos = len(state["positions"])

    msg = (
        f"<b>PAPER SUMMARY</b>\n"
        f"Equity: <b>{fmt_money(eq)}</b> ({fmt_pct(growth_pct)})\n"
        f"Cash: <b>{fmt_money(cash(state))}</b>\n"
        f"Realized: <b>{fmt_money(realized)}</b>\n"
        f"Fees: <b>{fmt_money(fees)}</b>\n"
        f"Trades: <b>{trades}</b> | W/L: <b>{wins}/{losses}</b>\n"
        f"Open positions: <b>{open_pos}</b>"
    )
    telegram_send(msg)
    state["last_summary_ts"] = now_ts()
    save_state(state)

# =========================
# MAIN LOOP
# =========================

def main():
    state = load_state()

    telegram_send(
        f"<b>ACTIVE BOT STARTED</b>\n"
        f"Mode: <b>{TRADE_MODE.upper()}</b>\n"
        f"State: <b>{STATE_FILE}</b>\n"
        f"Start: <b>{fmt_money(float(state['paper'].get('start_balance', PAPER_START_BALANCE)))}</b>\n"
        f"Cash: <b>{fmt_money(cash(state))}</b>\n"
        f"Per trade: <b>{fmt_money(QUOTE_PER_TRADE_USD)}</b>\n"
        f"Max pos: <b>{MAX_OPEN_POSITIONS}</b> | Scan: <b>{SCAN_INTERVAL_SECONDS}s</b>\n"
        f"TP/SL: <b>+{TAKE_PROFIT_PCT:.2f}%</b> / <b>-{STOP_LOSS_PCT:.2f}%</b>"
    )

    # cache products (refresh every ~30 min)
    products_cache: List[Dict] = []
    products_cache_ts = 0

    while True:
        cycle_start = now_ts()
        prices: Dict[str, float] = {}

        try:
            # refresh products periodically
            if not products_cache or (cycle_start - products_cache_ts) > 1800:
                products_cache = list_usd_products(MAX_PRODUCTS)
                products_cache_ts = cycle_start
                print(f"[info] loaded products={len(products_cache)}")

            # 1) Get prices for current positions + manage them first
            for pid in list(state["positions"].keys()):
                cndl = get_candles(pid, lookback_candles=LOOKBACK_CANDLES)
                if cndl and cndl["close"]:
                    prices[pid] = cndl["close"][-1]
            manage_positions(state, prices)

            # 2) If we can buy, scan for new fast breakouts
            ok_buy, reason = can_buy(state)
            if ok_buy:
                for p in products_cache:
                    pid = p.get("product_id")
                    if not pid or pid in state["positions"]:
                        continue

                    # Quick skip if we already filled capacity mid-loop
                    if len(state["positions"]) >= MAX_OPEN_POSITIONS:
                        break

                    cndl = get_candles(pid, lookback_candles=LOOKBACK_CANDLES)
                    if not cndl:
                        continue
                    last_px = cndl["close"][-1]
                    prices[pid] = last_px

                    # Spread filter (execution quality)
                    if USE_SPREAD_FILTER:
                        sp = get_spread_pct(pid)
                        if sp is None or sp > MAX_SPREAD_PCT:
                            continue

                    sig, meta = is_breakout_signal(cndl)
                    if not sig:
                        continue

                    # Buy immediately on signal
                    paper_buy(state, pid, last_px, meta)

                    # after a buy, stop scanning this cycle to avoid overtrading
                    break

            # 3) Periodic summary
            maybe_send_summary(state, prices)

            save_state(state)

        except Exception as e:
            print(f"[error] {type(e).__name__}: {e}")

        # sleep remainder
        elapsed = now_ts() - cycle_start
        sleep_for = max(1, SCAN_INTERVAL_SECONDS - elapsed)
        time.sleep(sleep_for)

if __name__ == "__main__":
    main()
