import pandas as pd
import numpy as np
import joblib              # For Loading K-Means
import requests
import json
import os
import subprocess
import sys
import time
import glob
import ast
import warnings
from datetime import datetime, timedelta

# Suppress warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# 🛠️ CONFIGURATION
# ==============================================================================
BASE_DIR = '/Users/joshuastein/Documents/Python/Polymarket/whale_hunter_project'

# --- MODELS ---
CLUSTERING_MODEL_PATH = os.path.join(BASE_DIR, 'Exploratory', 'poly_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'Exploratory', 'poly_scaler.pkl')

# --- DIRECTORIES ---
DIR_TRADES = os.path.join(BASE_DIR, 'data', '10_raw_trades')
DIR_PROFILES = os.path.join(BASE_DIR, 'data', '11_wallet_profiles', 'profiles')
DIR_MARKETS = os.path.join(BASE_DIR, 'data', '00_markets') 
DIR_CLASSIFIED = os.path.join(BASE_DIR, 'data', '12_classified')

# --- DISCORD ---
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1466974651858288722/IYRm2oBxT-r0B-o1gnw6EiE3qvbUZNQ27CAbs_LDiVeVhQNaioV_-Yu2oxhFZdc9N8zk"

# --- SETTINGS ---
TARGET_CLUSTERS = [3]  
ANOMALY_THRESHOLD = 4.0 
MIN_BET_USD = 500.0       
LEADERBOARD_SIZE = 5      

# ==============================================================================
# 🧩 MODULE 0: MARKET INDEXER
# ==============================================================================
def index_all_markets():
    print("\n[0/5] 📚 Indexing Markets...")
    if not os.path.exists(DIR_MARKETS): os.makedirs(DIR_MARKETS)

    url = "https://gamma-api.polymarket.com/events"
    all_markets = []
    seen_ids = set()
    offset = 0
    limit = 100 
    
    while True:
        try:
            params = {"closed": "false", "limit": limit, "offset": offset}
            resp = requests.get(url, params=params)
            data = resp.json()
            
            if not data: break
            
            new_tokens = 0
            for item in data:
                if item['id'] in seen_ids: continue
                seen_ids.add(item['id'])

                if 'markets' in item:
                    for m in item['markets']:
                        question = m.get('question')
                        slug = item.get('slug')
                        raw_ids = m.get('clobTokenIds', [])
                        
                        token_list = []
                        if isinstance(raw_ids, list): token_list = raw_ids
                        elif isinstance(raw_ids, str):
                            try: token_list = ast.literal_eval(raw_ids)
                            except: token_list = [raw_ids]
                        
                        if token_list:
                            for token_id in token_list:
                                if len(str(token_id)) < 5: continue 
                                all_markets.append({'asset_id': str(token_id), 'question': question, 'market_slug': slug})
                                new_tokens += 1

            if new_tokens == 0 and len(data) > 0: break
            if offset % 500 == 0: print(f"   -> Offset {offset}: Found {new_tokens} tokens...")
            
            offset += limit
            time.sleep(0.1)
        except Exception: break
            
    df = pd.DataFrame(all_markets)
    if not df.empty and 'asset_id' in df.columns:
        df.dropna(subset=['asset_id'], inplace=True)
        df.drop_duplicates(subset=['asset_id'], inplace=True)
    
    save_path = os.path.join(DIR_MARKETS, "active_markets_lookup.csv")
    df.to_csv(save_path, index=False)
    print(f"   ✅ Indexed {len(df)} markets.")
    return df

# ==============================================================================
# 🧩 MODULE 1: DEEP TRADE FETCHING
# ==============================================================================
def fetch_24h_trades():
    print("\n[1/5] 🌐 Fetching 24h Trades...")
    if not os.path.exists(DIR_TRADES): os.makedirs(DIR_TRADES)
    
    all_trades = []
    cutoff_time = datetime.now() - timedelta(hours=24)
    cutoff_ts = int(cutoff_time.timestamp())
    
    url = "https://data-api.polymarket.com/trades"
    last_timestamp = int(time.time())
    page = 0
    
    while last_timestamp > cutoff_ts:
        print(f"   -> Page {page+1} (Depth: {datetime.fromtimestamp(last_timestamp).strftime('%H:%M')})")
        params = {"limit": 500, "ts_lt": last_timestamp}
        try:
            response = requests.get(url, params=params)
            data = response.json()
            if not data: break
            all_trades.extend(data)
            last_timestamp = int(data[-1]['timestamp']) 
            page += 1
            time.sleep(0.1)
            if page > 50: break 
        except Exception: break

    if not all_trades: return None
    df = pd.DataFrame(all_trades)
    
    col_map = {'proxyWallet': 'maker', 'maker_address': 'maker', 'asset': 'asset_id', 'side': 'maker_direction'}
    df.rename(columns=col_map, inplace=True)
    
    if 'usd_amount' not in df.columns and 'size' in df.columns and 'price' in df.columns:
        df['usd_amount'] = df['size'].astype(float) * df['price'].astype(float)
            
    df['dt'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df[df['dt'] > cutoff_time]
    
    filename = os.path.join(DIR_TRADES, f"trades_recent.csv")
    df.to_csv(filename, index=False)
    return filename

# ==============================================================================
# 🧩 MODULE 2: PROFILE GENERATION
# ==============================================================================
def update_wallet_profiles():
    print("\n[2/5] 👤 Updating Profiles...")
    script_path = os.path.join(BASE_DIR, "11_generat_wallet_profiles.py")
    try:
        subprocess.run(["python", script_path], check=True, cwd=BASE_DIR)
        list_of_files = glob.glob(os.path.join(DIR_PROFILES, '*.csv'))
        if not list_of_files: return None
        return max(list_of_files, key=os.path.getctime)
    except Exception as e:
        print(f"   ⚠️ Profile Gen Failed: {e}")
        return None

# ==============================================================================
# 🧩 MODULE 3: PREDICTION (CLUSTERING + ANOMALY)
# ==============================================================================
def predict_clusters(profiles_file, trades_file, market_df):
    print("\n[3/5] 🧠 Applying K-Means Model (FIXED MAPPING)...")
    
    df_profiles = pd.read_csv(profiles_file)
    df_trades = pd.read_csv(trades_file)
    
    # --- 1. FEATURE MAPPING (The Critical Fix) ---
    # We map 'total_invested' to 'aggression_max_usd' so the model has input
    # (The model expects 'total_invested', so we create it from what we have)
    if 'total_invested' not in df_profiles.columns:
        if 'aggression_max_usd' in df_profiles.columns:
            print("   ⚠️ Mapping 'aggression_max_usd' -> 'total_invested'")
            df_profiles['total_invested'] = df_profiles['aggression_max_usd']
        else:
            # Fallback if even that is missing
            df_profiles['total_invested'] = 0
            
    # Also ensure other potentially missing cols exist
    if 'wins' not in df_profiles.columns: df_profiles['wins'] = 0
    if 'unique_markets' not in df_profiles.columns: df_profiles['unique_markets'] = df_profiles['diversification_markets_count']
    if 'avg_time_to_impact_hours' not in df_profiles.columns: df_profiles['avg_time_to_impact_hours'] = 0
    if 'conviction_avg_buy_usd' not in df_profiles.columns: df_profiles['conviction_avg_buy_usd'] = df_profiles['aggression_avg_usd']
    if 'win_rate_pct' not in df_profiles.columns: df_profiles['win_rate_pct'] = 0.5
    # ---------------------------------------------

    # 2. Load Model
    if not os.path.exists(CLUSTERING_MODEL_PATH):
        print("   ❌ Model file missing!")
        return [], []
    model = joblib.load(CLUSTERING_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # 3. Prepare Features
    features = [
        'tenure_days', 'aggression_avg_usd', 'frequency_trades_daily', 
        'domain_focus_sports_pct', 'domain_focus_politics_pct', 'domain_focus_crypto_pct',
        'diversification_markets_count', 'total_invested', 'wins', 
        'unique_markets', 'avg_time_to_impact_hours', 'conviction_avg_buy_usd', 'win_rate_pct'
    ]
    
    # Sanity check to ensure no NaN crashes
    X = df_profiles[features].fillna(0)
    
    # 4. Predict & Score
    print("   -> Scaling and Predicting...")
    X_scaled = scaler.transform(X)
    df_profiles['cluster'] = model.predict(X_scaled)
    distances = model.transform(X_scaled)
    df_profiles['anomaly_score'] = np.min(distances, axis=1)

    # --- 5. SMART FILTERING ---
    
    # Filter A: The "Real Money" Filter (Using the newly mapped column)
    # We set this to $500. Anyone whose biggest bet is <$500 is ignored.
    real_traders = df_profiles[df_profiles['total_invested'] > 500].copy()
    
    if real_traders.empty:
        print("   ⚠️ No traders found with >$500 Max Bet.")
        return [], []

    # Filter B: The Percentile Cutoff
    percentile_cutoff = 90 # Lowered to 90% to ensure you see results today
    dynamic_threshold = np.percentile(real_traders['anomaly_score'], percentile_cutoff)
    
    print(f"      📊 Stats for {len(real_traders)} Real Traders:")
    print(f"      - Anomaly Threshold (Top 10%): {dynamic_threshold:.2f}")
    
    anomalies = real_traders[real_traders['anomaly_score'] >= dynamic_threshold].copy()
    
    print(f"      -> Found {len(anomalies)} Qualified Anomalies.")

    # Select Whales (Cluster 3 + High Vol)
    whales = df_profiles[
        (df_profiles['cluster'].isin(TARGET_CLUSTERS)) & 
        (df_profiles['total_invested'] > 5000)
    ].copy()

    # 6. Enrich info
    results_whales = []
    results_anomalies = []
    
    def get_last_trade(wallet):
        user_trades = df_trades[df_trades['maker'] == wallet]
        if user_trades.empty: return None
        top = user_trades.sort_values('usd_amount', ascending=False).iloc[0]
        
        question = "Unknown"
        if 'asset_id' in top and not market_df.empty:
            m = market_df[market_df['asset_id'] == str(top['asset_id'])]
            if not m.empty: question = m.iloc[0]['question']
            
        return {
            'wallet': wallet,
            'amount': top['usd_amount'],
            'side': top['maker_direction'],
            'market': question,
            'cluster': int(df_profiles[df_profiles['maker']==wallet].iloc[0]['cluster']),
            'score': 0 
        }

    for w in whales['maker']:
        info = get_last_trade(w)
        if info: results_whales.append(info)
            
    for a in anomalies['maker']:
        info = get_last_trade(a)
        if info:
            info['score'] = float(anomalies[anomalies['maker']==a].iloc[0]['anomaly_score'])
            results_anomalies.append(info)
            
    results_whales.sort(key=lambda x: x['amount'], reverse=True)
    results_anomalies.sort(key=lambda x: x['score'], reverse=True)
    
    # Print the top hit to the console
    if results_anomalies:
        top = results_anomalies[0]
        print(f"\n      🏆 TOP ANOMALY: {top['wallet'][:8]}... Score: {top['score']:.2f} | Max Bet: ${top['amount']:.0f}")

    return results_whales[:LEADERBOARD_SIZE], results_anomalies[:LEADERBOARD_SIZE]
# ==============================================================================
# 🧩 MODULE 4: ALERTING
# ==============================================================================
def send_discord_alerts(whales, anomalies):
    print("\n[4/5] 📡 Sending Alerts...")
    if not whales and not anomalies: return

    embeds = []
    
    if whales:
        fields = []
        for w in whales:
            val = (f"**${w['amount']:,.0f}** on {w['side']}\n"
                   f"Market: `{str(w['market'])[:40]}...`\n"
                   f"[Profile](https://polymarket.com/profile/{w['wallet']})")
            fields.append({"name": f"🐋 CLUSTER {w['cluster']} DETECTED", "value": val, "inline": False})
        embeds.append({"title": "🚨 High-Conviction Cluster Move", "color": 3447003, "fields": fields})

    if anomalies:
        fields = []
        for a in anomalies:
            val = (f"**Anomaly Score: {a['score']:.2f}** (Very Rare)\n"
                   f"Last Bet: ${a['amount']:,.0f}\n"
                   f"[Profile](https://polymarket.com/profile/{a['wallet']})")
            fields.append({"name": f"👽 ANOMALY DETECTED", "value": val, "inline": False})
        embeds.append({"title": "⚠️ Unusual Market Activity", "color": 15158332, "fields": fields})

    requests.post(DISCORD_WEBHOOK_URL, json={"username": "Cluster Hunter", "embeds": embeds})
    print("   ✅ Sent to Discord.")

# ==============================================================================
# 🚀 MAIN
# ==============================================================================
if __name__ == "__main__":
    print("--- 🧠 CLUSTERING LIVE HUNTER (V1) ---")
    
    market_lookup = index_all_markets()
    trades_file = fetch_24h_trades()
    
    if trades_file:
        profiles_file = update_wallet_profiles()
        if profiles_file:
            top_whales, top_anomalies = predict_clusters(profiles_file, trades_file, market_lookup)
            send_discord_alerts(top_whales, top_anomalies)
            
    print("\n✅ Cycle Complete.")