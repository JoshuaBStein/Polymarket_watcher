import requests
import pandas as pd
import polars as pl
import os
import time
import glob
from datetime import datetime

# ==========================================
# 🔴 CONFIGURATION
# ==========================================
GRAPH_API_KEY = "add-api-key-here"  # <--- PASTE KEY HERE

# API Endpoints
GRAPH_URL = "add-url-here"
GAMMA_MARKETS_URL = "add-url-here"

# Directories (UPDATED FOR NEW STRUCTURE)
DIR_INPUT = "data/10_raw_trades"
DIR_OUTPUT_ROOT = "data/11_wallet_profiles"
DIR_HISTORY = f"{DIR_OUTPUT_ROOT}/history"
DIR_PROFILES = f"{DIR_OUTPUT_ROOT}/profiles"

# Settings
BATCH_SIZE = 10           # Number of wallets to query at once
HISTORY_LIMIT = 1000      # Max trades to fetch per batch query

# ==========================================
# HELPER: FIND LATEST INPUT FILE
# ==========================================
def get_latest_trades_file():
    """Finds the most recent csv file in the input directory."""
    if not os.path.exists(DIR_INPUT):
        print(f"❌ Error: Input directory '{DIR_INPUT}' does not exist.")
        return None

    # Get list of all csv files in the folder
    files = glob.glob(f"{DIR_INPUT}/trades_*.csv")
    
    if not files:
        print(f"❌ Error: No trade files found in '{DIR_INPUT}'. Run script 10 first.")
        return None

    # Sort by modification time (newest first)
    latest_file = max(files, key=os.path.getmtime)
    print(f"📂 Using latest input file: {latest_file}")
    return latest_file

# ==========================================
# STEP 1: INDEX MARKETS (For "Domain Focus")
# ==========================================
def fetch_all_markets():
    print("\n--- 1. Indexing All Markets (Active & Closed) ---")
    
    # We save this metadata file once in the root data folder or cache it
    meta_dir = "data/metadata"
    os.makedirs(meta_dir, exist_ok=True)
    markets_file = f"{meta_dir}/markets_meta.csv"
    
    # If we already have it and it's recent (optional logic), skip. 
    # For now, we fetch to be safe.
    
    all_markets = []
    
    def fetch_batch(active):
        params = {"limit": 2000, "active": str(active).lower(), "closed": str(not active).lower()}
        try:
            r = requests.get(GAMMA_MARKETS_URL, params=params)
            if r.status_code == 200:
                data = r.json()
                print(f"   > Fetched {len(data)} {'active' if active else 'closed'} markets.")
                for m in data:
                    all_markets.append({
                        "id": m.get("conditionId", "").lower(), 
                        "question": m.get("question", ""),
                        "market_slug": m.get("slug", "")
                    })
            else:
                print(f"   ❌ Error: {r.status_code}")
        except Exception as e:
            print(f"   ❌ Exception: {e}")

    fetch_batch(True)  # Active
    fetch_batch(False) # Closed

    if all_markets:
        df = pd.DataFrame(all_markets)
        df = df.drop_duplicates(subset=["id"])
        df.to_csv(markets_file, index=False)
        print(f"✅ Saved Market Metadata: {markets_file}")
        return markets_file
    else:
        print("❌ Failed to fetch markets.")
        return None

# ==========================================
# STEP 2: FETCH WALLET HISTORIES
# ==========================================
def fetch_wallet_histories(input_path):
    print("\n--- 2. Fetching Historical Data for Wallets ---")
    
    # Setup Output Directories
    os.makedirs(DIR_HISTORY, exist_ok=True)
    
    # Load Input
    df_input = pd.read_csv(input_path)
    if "maker" not in df_input.columns:
        print("❌ Input CSV missing 'maker' column.")
        return None
        
    unique_wallets = df_input["maker"].dropna().unique().tolist()
    print(f"   > Identified {len(unique_wallets)} unique wallets to profile.")
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GRAPH_API_KEY}"}
    all_history = []

    # Process in batches
    for i in range(0, len(unique_wallets), BATCH_SIZE):
        batch = unique_wallets[i : i + BATCH_SIZE]
        print(f"   > Fetching Batch {i+1}-{min(i+BATCH_SIZE, len(unique_wallets))}...")
        
        query = """
        query($users: [String!]) {
          splits(first: %d, orderBy: timestamp, orderDirection: desc, where: { stakeholder_in: $users, amount_gt: "1000000" }) {
            timestamp, stakeholder, amount, condition { id }
          }
          merges(first: %d, orderBy: timestamp, orderDirection: desc, where: { stakeholder_in: $users, amount_gt: "1000000" }) {
            timestamp, stakeholder, amount, condition { id }
          }
        }
        """ % (HISTORY_LIMIT, HISTORY_LIMIT)
        
        try:
            response = requests.post(GRAPH_URL, json={'query': query, 'variables': {'users': batch}}, headers=headers)
            data = response.json()
            
            if "errors" in data:
                print(f"     ⚠️ Graph Error: {data['errors'][0]['message']}")
                continue
                
            splits = data.get("data", {}).get("splits", [])
            merges = data.get("data", {}).get("merges", [])
            
            def add_rows(rows, action):
                for r in rows:
                    raw_amt = float(r.get("amount", 0))
                    cond = r.get("condition")
                    m_id = (cond.get("id") if isinstance(cond, dict) else cond) or ""
                    
                    all_history.append({
                        "timestamp": datetime.fromtimestamp(int(r["timestamp"])),
                        "maker": r.get("stakeholder"),
                        "action": action,
                        "market_id": m_id.lower(),
                        "usd_amount": raw_amt / 1e6
                    })

            add_rows(splits, "BUY")
            add_rows(merges, "SELL")
            
        except Exception as e:
            print(f"     ❌ Critical Error: {e}")
            time.sleep(1)

    # Output File Name (Matches Input Date)
    date_suffix = datetime.now().strftime("%Y-%m-%d")
    output_history_file = f"{DIR_HISTORY}/history_{date_suffix}.csv"

    if all_history:
        df_hist = pd.DataFrame(all_history)
        df_hist.to_csv(output_history_file, index=False)
        print(f"✅ Saved Raw History: {output_history_file} ({len(df_hist)} rows)")
        return output_history_file
    else:
        print("⚠️ No history found.")
        return None

# ==========================================
# STEP 3: ANALYZE BEHAVIOR (POLARS)
# ==========================================
def analyze_behavior(history_file, markets_file):
    print("\n--- 3. Running Behavioral Analysis (Polars) ---")
    
    os.makedirs(DIR_PROFILES, exist_ok=True)
    
    if not history_file or not markets_file:
        print("❌ Missing data files. Cannot analyze.")
        return

    # 1. Lazy Load
    trades = pl.scan_csv(history_file)
    markets = pl.scan_csv(markets_file)

    # 2. Preprocess & Join
    trades = trades.with_columns(
        pl.col("timestamp").str.to_datetime().cast(pl.Datetime),
        pl.col("usd_amount").cast(pl.Float64)
    )

    data = trades.join(markets, left_on="market_id", right_on="id", how="left")

    # 3. Define Patterns
    politics_pattern = r"(?i)election|president|senate|congress|democrat|republican|trump|biden|harris|nominee|poll|vote|cabinet|war|geopolitics|ukraine|israel|policy"
    sports_pattern = r"(?i)nfl|nba|mlb|nhl|soccer|ufc|f1|formula 1|premier league|champions league|super bowl|finals|playoff|vs|tournament|cup|medal|tennis|golf"
    crypto_pattern = r"(?i)bitcoin|btc|ethereum|eth|solana|sol|crypto|nft|airdrop|defi|token|stablecoin|usdc|usdt|tether|binance|coinbase|etf|memecoin|halving"

    # 4. Global Reference
    try:
        max_date = trades.select(pl.col("timestamp").max()).collect().item()
    except:
        max_date = datetime.now()

    # 5. Feature Engineering
    wallet_profile_query = data.group_by("maker").agg([
        ((max_date - pl.col("timestamp").min()).dt.total_days()).alias("tenure_days"),
        pl.col("usd_amount").mean().alias("aggression_avg_usd"),
        pl.col("usd_amount").max().alias("aggression_max_usd"),
        (pl.len() / ((max_date - pl.col("timestamp").min()).dt.total_days().clip(lower_bound=1))).alias("frequency_trades_daily"),
        ((pl.col("usd_amount").filter(pl.col("action") == "SELL").sum().fill_null(0) - 
          pl.col("usd_amount").filter(pl.col("action") == "BUY").sum().fill_null(0))).alias("net_usdc_flow"),
        ((pl.col("market_slug").str.contains(sports_pattern) | pl.col("question").str.contains(sports_pattern)).sum() / pl.len() * 100).alias("domain_focus_sports_pct"),
        ((pl.col("market_slug").str.contains(politics_pattern) | pl.col("question").str.contains(politics_pattern)).sum() / pl.len() * 100).alias("domain_focus_politics_pct"),
        ((pl.col("market_slug").str.contains(crypto_pattern) | pl.col("question").str.contains(crypto_pattern)).sum() / pl.len() * 100).alias("domain_focus_crypto_pct"),
        pl.col("market_id").n_unique().alias("diversification_markets_count"),
        pl.len().alias("total_trades_count")
    ])

    # 6. Execute
    print("   > Calculating metrics...")
    df_result = wallet_profile_query.collect()
    
    # Output File Name
    date_suffix = datetime.now().strftime("%Y-%m-%d")
    output_profile = f"{DIR_PROFILES}/profiles_{date_suffix}.csv"

    df_result.write_csv(output_profile)
    print(f"✅ DONE! Behavioral Profiles saved to: {output_profile}")
    print(df_result.head())

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    if "PASTE_YOUR" in GRAPH_API_KEY:
        print("❌ STOP: Paste your Graph API Key.")
    else:
        # 1. Find the latest input from Step 10
        target_input = get_latest_trades_file()
        
        if target_input:
            # 2. Update Market Metadata
            markets_csv = fetch_all_markets()
            
            # 3. Fetch History for these wallets
            history_csv = fetch_wallet_histories(target_input)
            
            # 4. Generate Profiles
            if history_csv and markets_csv:
                analyze_behavior(history_csv, markets_csv)
