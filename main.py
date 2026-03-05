import os
import time
import requests
from collections import deque

# --- Configuration ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 15)) # Faster scans for faster entries

# Tightened Filtering for High-Conviction Runners
MIN_LIQUIDITY = 30000 
MAX_LIQUIDITY = 750000
MIN_VOLUME_5M = 35000 # Increased to filter out "ghost" tokens
MIN_PRICE_CHANGE_1M = 2.5 # Looking for sharper vertical moves
MIN_PRICE_CHANGE_5M = 8.0 
MIN_TRADES_5M = 40 # High activity is key for micro-caps
MIN_BUY_RATIO = 0.65 # Want strong buy pressure
MIN_FDV = 150000
MAX_FDV = 15000000

alerted_tokens = {} 
watchlist = deque(maxlen=5)
queries = ["sol", "eth", "pepe", "pump", "ai", "base", "meme", "wiz", "inu"]

def send_telegram(msg):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"Tele-Log: {msg[:60]}...")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID, 
            "text": msg,
            "disable_web_page_preview": True,
            "parse_mode": "Markdown" # Allows for bolding/formatting
        }, timeout=10)
    except Exception as e:
        print(f"Telegram Error: {e}")

def get_pairs():
    all_pairs = []
    seen = set()
    for q in queries:
        try:
            url = f"https://api.dexscreener.com/latest/dex/search/?q={q}"
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                data = r.json()
                for p in data.get("pairs", []):
                    addr = p.get("pairAddress")
                    # Filter for only main chains to avoid "junk" chains
                    chain = p.get("chainId")
                    if addr and addr not in seen and chain in ['solana', 'base', 'ethereum']:
                        seen.add(addr)
                        all_pairs.append(p)
            time.sleep(0.2) 
        except:
            continue
    return all_pairs

def passes_filters(pair):
    liq = pair.get("liquidity", {}).get("usd", 0)
    vol5 = pair.get("volume", {}).get("m5", 0)
    m1 = pair.get("priceChange", {}).get("m1", 0)
    m5 = pair.get("priceChange", {}).get("m5", 0)
    fdv = pair.get("fdv", 0)
    
    txns = pair.get("txns", {}).get("m5", {})
    buys, sells = txns.get("buys", 0), txns.get("sells", 0)
    total = buys + sells

    # --- THE PROFIT IMPROVERS ---
    # 1. Filter out wash-trading (Volume shouldn't be 10x Liquidity in 5 mins)
    if vol5 > (liq * 5): return False 
    
    # 2. Basic filter set
    if not (MIN_LIQUIDITY <= liq <= MAX_LIQUIDITY): return False
    if vol5 < MIN_VOLUME_5M or total < MIN_TRADES_5M: return False
    if m1 < MIN_PRICE_CHANGE_1M or m5 < MIN_PRICE_CHANGE_5M: return False
    
    # 3. Buy Ratio Logic
    if total > 0 and (buys / total) < MIN_BUY_RATIO: return False
    if not (MIN_FDV <= fdv <= MAX_FDV): return False
    
    return True

def score_pair(pair):
    score = 0
    m1 = pair.get("priceChange", {}).get("m1", 0)
    m5 = pair.get("priceChange", {}).get("m5", 0)
    vol5 = pair.get("volume", {}).get("m5", 0)
    liq = pair.get("liquidity", {}).get("usd", 0)
    
    # Velocity Score: Is the 1m move making up the bulk of the 5m move? (Bullish)
    if m1 >= (m5 * 0.4): score += 4 
    # Liquidity Health: Better score if volume is 50%-100% of liquidity
    if (liq * 0.5) < vol5 < liq: score += 3
    # Absolute Strength
    if m1 > 5: score += 3
    
    return score

def run():
    print("Optimization active. Scanning for high-velocity runners...")
    while True:
        start_time = time.time()
        pairs = get_pairs()
        heads_scanned = len(pairs)
        runners = []

        for pair in pairs:
            if not passes_filters(pair): continue

            addr = pair.get("pairAddress")
            token_addr = pair.get("baseToken", {}).get("address")
            symbol = pair.get("baseToken", {}).get("symbol", "UNK")
            price = float(pair.get("priceUsd", 0))
            
            # Anti-Spam: Only re-alert if price jumps another 15%
            if addr in alerted_tokens and price < (alerted_tokens[addr] * 1.15):
                continue

            score = score_pair(pair)
            if score < 6: continue # Only alert on high-conviction scores

            alerted_tokens[addr] = price
            m5 = pair.get("priceChange", {}).get("m5", 0)
            vol = pair.get("volume", {}).get("m5", 0)
            
            alert = (f"🚀 *EXPLOSIVE RUNNER DETECTED*\n\n"
                     f"**Token:** {symbol}\n"
                     f"**Score:** {score}/10\n"
                     f"**5m Change:** {m5}%\n"
                     f"**5m Vol:** ${int(vol):,}\n\n"
                     f"[DexScreener](https://dexscreener.com/search?q={addr}) | "
                     f"[RugCheck](https://rugcheck.xyz/tokens/{token_addr})")
            runners.append(alert)

        # Heartbeat
        report = f"🔍 *Scan Complete*\nHeads: {heads_scanned} | Runners: {len(runners)}"
        send_telegram(report)

        for r in runners:
            send_telegram(r)

        elapsed = time.time() - start_time
        time.sleep(max(1, SCAN_INTERVAL - elapsed))

if __name__ == "__main__":
    run()
