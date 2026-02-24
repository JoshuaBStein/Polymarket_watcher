import pandas as pd
import numpy as np
import joblib              # For Stage 1 (Scikit-Learn Scout)
import xgboost as xgb      # For Stage 2 (XGBoost Auditor)
import requests
import json
import os
import subprocess
import sys
import time
import glob
import ast                 # For fixing the market token list bug
import warnings
from datetime import datetime, timedelta

# Suppress version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ==============================================================================
# 🛠️ CONFIGURATION
# ==============================================================================
BASE_DIR = '/Users/joshuastein/Documents/Python/Polymarket/whale_hunter_project'

# --- MODELS ---
SCOUT_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'scout_model.joblib')
AUDITOR_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model_B_profitability.json')

# --- DIRECTORIES ---
DIR_TRADES = os.path.join(BASE_DIR, 'data', '10_raw_trades')
DIR_PROFILES = os.path.join(BASE_DIR, 'data', '11_wallet_profiles', 'profiles')
DIR_MARKETS = os.path.join(BASE_DIR, 'data', '00_markets') 
DIR_CLASSIFIED = os.path.join(BASE_DIR, 'data', '12_classified')

# --- DISCORD ---
DISCORD_WEBHOOK_URL = "add-discord-key-here"

# --- SETTINGS ---
MIN_AUDITOR_SCORE = 0.65  
MIN_BET_USD = 500.0       
LEADERBOARD_SIZE = 5      

# ==============================================================================
# 🧩 MODULE 0: MARKET INDEXER (TYPE-SAFE)
# ==============================================================================
def index_all_markets():
    print("\n[0/5] 📚 Indexing Markets (Type-Safe Mode)...")
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
            
            if not data: 
                print("   ✅ API returned empty page. Done.")
                break
            
            new_real_tokens = 0
            for item in data:
                if item['id'] in seen_ids: continue
                seen_ids.add(item['id'])

                if 'markets' in item:
                    for m in item['markets']:
                        question = m.get('question')
                        slug = item.get('slug')
                        raw_ids = m.get('clobTokenIds', [])
                        
                        # Handle String vs List bug
                        token_list = []
                        if isinstance(raw_ids, list):
                            token_list = raw_ids
                        elif isinstance(raw_ids, str):
                            try:
                                token_list = ast.literal_eval(raw_ids)
                            except:
                                token_list = [raw_ids]
                        
                        if token_list:
                            for token_id in token_list:
                                if len(str(token_id)) < 5: continue 
                                all_markets.append({'asset_id': str(token_id), 'question': question, 'market_slug': slug})
                                new_real_tokens += 1
                        else:
                            all_markets.append({'asset_id': str(m.get('id')), 'question': question, 'market_slug': slug})
                            new_real_tokens += 1

            if new_real_tokens == 0 and len(data) > 0:
                print("   🛑 Infinite loop or empty page detected. Stopping.")
                break

            if offset % 500 == 0:
                print(f"   -> Offset {offset}: Added {new_real_tokens} valid tokens (Total: {len(all_markets)})")
            
            offset += limit
            time.sleep(0.1)
        except Exception as e:
            print(f"   ⚠️ Market Fetch Error: {e}")
            break
            
    df = pd.DataFrame(all_markets)
    if not df.empty and 'asset_id' in df.columns:
        df.dropna(subset=['asset_id'], inplace=True)
        df.drop_duplicates(subset=['asset_id'], inplace=True)
    
    save_path = os.path.join(DIR_MARKETS, "active_markets_lookup.csv")
    df.to_csv(save_path, index=False)
    print(f"   ✅ Final Index: {len(df)} unique market tokens.")
    return df

# ==============================================================================
# 🧩 MODULE 1: DEEP TRADE FETCHING
# ==============================================================================
def fetch_24h_trades():
    print("\n[1/5] 🌐 Initializing 24-hour Deep Fetch...")
    if not os.path.exists(DIR_TRADES): os.makedirs(DIR_TRADES)
    
    all_trades = []
    cutoff_time = datetime.now() - timedelta(hours=24)
    cutoff_ts = int(cutoff_time.timestamp())
    
    url = "https://data-api.polymarket.com/trades"
    last_timestamp = int(time.time())
    page = 0
    
    while last_timestamp > cutoff_ts:
        print(f"   -> Fetching page {page+1} (Depth: {datetime.fromtimestamp(last_timestamp).strftime('%H:%M')})")
        params = {"limit": 500, "ts_lt": last_timestamp}
        try:
            response = requests.get(url, params=params)
            data = response.json()
            if not data: break
            all_trades.extend(data)
            last_timestamp = int(data[-1]['timestamp']) 
            page += 1
            time.sleep(0.1)
            if page > 100: break 
        except Exception: break

    if not all_trades: return None
    df = pd.DataFrame(all_trades)
    
    # Normalize Columns
    if 'maker' not in df.columns:
        if 'proxyWallet' in df.columns: df.rename(columns={'proxyWallet': 'maker'}, inplace=True)
        elif 'maker_address' in df.columns: df.rename(columns={'maker_address': 'maker'}, inplace=True)
    if 'asset_id' not in df.columns and 'asset' in df.columns: df.rename(columns={'asset': 'asset_id'}, inplace=True)
    if 'side' in df.columns and 'maker_direction' not in df.columns: df.rename(columns={'side': 'maker_direction'}, inplace=True)
    if 'usd_amount' not in df.columns and 'size' in df.columns and 'price' in df.columns:
        df['usd_amount'] = df['size'].astype(float) * df['price'].astype(float)
            
    df['dt'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df[df['dt'] > cutoff_time]
    
    filename = os.path.join(DIR_TRADES, f"trades_master_{datetime.now().strftime('%Y-%m-%d')}.csv")
    df.to_csv(filename, index=False)
    print(f"   ✅ Saved {len(df)} trades.")
    return filename

# ==============================================================================
# 🧩 MODULE 2: PROFILE UPDATER
# ==============================================================================
def update_wallet_profiles():
    print("\n[2/5] 👤 Updating Wallet Histories...")
    script_path = os.path.join(BASE_DIR, "11_generat_wallet_profiles.py")
    try:
        subprocess.run(["python", script_path], check=True, cwd=BASE_DIR)
        list_of_files = glob.glob(os.path.join(DIR_PROFILES, '*.csv'))
        if not list_of_files: return None
        return max(list_of_files, key=os.path.getctime)
    except subprocess.CalledProcessError:
        print("   ⚠️ Profile script failed.")
        return None

# ==============================================================================
# 🧩 MODULE 3: CONTEXT ENRICHMENT
# ==============================================================================
def enrich_data(trades_file, profiles_file, market_df):
    print("\n[3/5] 🧪 Merging Data & Mapping Markets...")
    df_trades = pd.read_csv(trades_file)
    df_profiles = pd.read_csv(profiles_file)
    
    if 'maker' not in df_trades.columns:
        if 'proxyWallet' in df_trades.columns: df_trades.rename(columns={'proxyWallet': 'maker'}, inplace=True)
        else: return None, None
            
    if 'usd_amount' not in df_trades.columns and 'size' in df_trades.columns:
         df_trades['usd_amount'] = df_trades['size'] * df_trades['price']
    
    # Map Markets
    if 'title' in df_trades.columns:
        df_trades['question'] = df_trades['title']
    elif market_df is not None and not market_df.empty and 'asset_id' in df_trades.columns:
        df_trades['asset_id'] = df_trades['asset_id'].astype(str)
        market_df['asset_id'] = market_df['asset_id'].astype(str)
        df_trades = df_trades.merge(market_df[['asset_id', 'question']], on='asset_id', how='left')
        
    df_trades['question'] = df_trades.get('question', 'Unknown Market').fillna('Unknown Market')

    max_trades = df_trades.groupby('maker')['usd_amount'].max().reset_index().rename(columns={'usd_amount': 'max_trade_24h'})
    merged = df_profiles.merge(max_trades, on='maker', how='left')
    
    for col in ['frequency_trades_daily', 'diversification_markets_count', 'tenure_days']:
        if col not in merged.columns: merged[col] = 0
        
    return merged, df_trades

# ==============================================================================
# 🧩 MODULE 4: DUAL-STAGE PREDICTION (UPDATED SAVING FORMAT)
# ==============================================================================
def predict_and_rank(df_profiles, df_trades):
    print("\n[4/5] 🧠 Running Dual-Stage AI...")
    
    # --- STAGE 1: THE SCOUT ---
    print("   🔹 Stage 1: The Scout (Random Forest)")
    
    if not os.path.exists(SCOUT_MODEL_PATH):
        print(f"   ⚠️ SCOUT MODEL NOT FOUND: {SCOUT_MODEL_PATH}")
        df_profiles['scout_whale_prob'] = 0.9 
        df_profiles['scout_insider_prob'] = 0.9
    else:
        scout_model = joblib.load(SCOUT_MODEL_PATH)
        scout_features = ['frequency_trades_daily', 'diversification_markets_count', 'tenure_days']
        try:
            X_scout = df_profiles[scout_features].fillna(0)
            scout_probs = scout_model.predict_proba(X_scout)
            df_profiles['scout_whale_prob'] = scout_probs[:, 3] 
            df_profiles['scout_insider_prob'] = scout_probs[:, 2]
        except Exception as e:
            print(f"   ❌ Scout Error: {e}. Bypassing...")
            df_profiles['scout_whale_prob'] = 0.9
            df_profiles['scout_insider_prob'] = 0.9

    # Filter Stage 1 Candidates
    interesting = df_profiles[
        (df_profiles['scout_whale_prob'] > 0.5) | 
        (df_profiles['scout_insider_prob'] > 0.5)
    ].copy()
    
    # --- 💾 NEW SAVING LOGIC (RENAMED) ---
    if not interesting.empty:
        if not os.path.exists(DIR_CLASSIFIED): os.makedirs(DIR_CLASSIFIED)
        
        # Format: classified_whales_YYYY-MM-DD.csv
        timestamp = datetime.now().strftime('%Y-%m-%d')
        save_filename = f"classified_whales_{timestamp}.csv"
        
        save_path = os.path.join(DIR_CLASSIFIED, save_filename)
        interesting.to_csv(save_path, index=False)
        print(f"      💾 Saved {len(interesting)} Stage 1 candidates to: data/12_classified/{save_filename}")
    else:
        print("      ⚠️ No candidates passed Stage 1.")
        return [], []
    # ---------------------------

    print(f"      -> {len(interesting)} candidates passed to Stage 2.")

    # --- STAGE 2: THE AUDITOR ---
    print("   🔹 Stage 2: The Auditor (XGBoost)")
    if os.path.exists(AUDITOR_MODEL_PATH):
        auditor_model = xgb.Booster()
        auditor_model.load_model(AUDITOR_MODEL_PATH)
        
        expected_features = auditor_model.feature_names
        
        if expected_features:
            for f in expected_features:
                if f not in interesting.columns: interesting[f] = 0
            X_auditor = interesting[expected_features]
            dmatrix = xgb.DMatrix(X_auditor)
            try:
                interesting['auditor_score'] = auditor_model.predict(dmatrix)
            except Exception as e:
                print(f"   ⚠️ Prediction Error: {e}. Defaulting to 0.5.")
                interesting['auditor_score'] = 0.5
        else:
            cols_to_ignore = ['scout_whale_prob', 'scout_insider_prob', 'maker', 'wallet', 'max_trade_24h']
            X_auditor = interesting.select_dtypes(include=[np.number]).drop(columns=cols_to_ignore, errors='ignore')
            dmatrix = xgb.DMatrix(X_auditor)
            try:
                interesting['auditor_score'] = auditor_model.predict(dmatrix)
            except:
                interesting['auditor_score'] = 0.5
    else:
        print(f"   ⚠️ AUDITOR MODEL MISSING: {AUDITOR_MODEL_PATH}")
        interesting['auditor_score'] = 0.5

    # --- RANKING ---
    candidates_whale = []
    candidates_insider = []
    
    for idx, row in interesting.iterrows():
        wallet = row['maker']
        their_trades = df_trades[df_trades['maker'] == wallet]
        if their_trades.empty: continue
        
        top_trade = their_trades.sort_values('usd_amount', ascending=False).iloc[0]
        
        if top_trade['usd_amount'] < MIN_BET_USD: continue
        
        if os.path.exists(AUDITOR_MODEL_PATH) and row['auditor_score'] < MIN_AUDITOR_SCORE: 
            continue

        item = {
            'wallet': wallet,
            'amount': top_trade['usd_amount'],
            'market': top_trade.get('question', 'Unknown'),
            'side': top_trade.get('maker_direction', 'Trade'),
            'scout_score': max(row['scout_whale_prob'], row['scout_insider_prob']),
            'auditor_score': row['auditor_score']
        }
        
        if row['scout_whale_prob'] > row['scout_insider_prob']:
            candidates_whale.append(item)
        else:
            candidates_insider.append(item)

    candidates_whale.sort(key=lambda x: (x['auditor_score'], x['amount']), reverse=True)
    candidates_insider.sort(key=lambda x: (x['auditor_score'], x['amount']), reverse=True)
    
    return candidates_whale[:LEADERBOARD_SIZE], candidates_insider[:LEADERBOARD_SIZE]

# ==============================================================================
# 🧩 MODULE 5: ALERTING (INCLUDES INSIDERS)
# ==============================================================================
def send_discord_leaderboard(whales, insiders):
    print("\n[5/5] 📡 Sending Discord Leaderboard...")
    if not whales and not insiders:
        print("   💤 No high-confidence targets found.")
        return

    embeds = []
    
    if whales:
        fields = []
        for i, w in enumerate(whales):
            val = (f"**${w['amount']:,.0f}** on {w['side']}\n"
                   f"Profit Score: `{w['auditor_score']:.1%}`\n"
                   f"Market: `{str(w['market'])[:40]}...`\n"
                   f"[Profile](https://polymarket.com/profile/{w['wallet']})")
            fields.append({"name": f"🏆 #{i+1} WHALE", "value": val, "inline": False})
        embeds.append({"title": "🐋 Verified Whales", "color": 3066993, "fields": fields})

    # ✅ INSIDER ALERT LOGIC IS HERE
    if insiders:
        fields = []
        for i, w in enumerate(insiders):
            val = (f"**${w['amount']:,.0f}** on {w['side']}\n"
                   f"Profit Score: `{w['auditor_score']:.1%}`\n"
                   f"Market: `{str(w['market'])[:40]}...`\n"
                   f"[Profile](https://polymarket.com/profile/{w['wallet']})")
            fields.append({"name": f"🧠 Verified Insiders", "value": val, "inline": False})
        embeds.append({"title": "🕶️ Verified Insiders", "color": 15105570, "fields": fields})

    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"username": "Double-Check Hunter", "embeds": embeds})
        print("   ✅ Alerts sent.")
    except Exception as e:
        print(f"   ❌ Discord failed: {e}")

# ==============================================================================
# 🚀 MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("="*60)
    print("      🌊 MASTER HUNTER: DUAL-MODEL PIPELINE (V4) 🌊")
    print("="*60)
    
    market_df = index_all_markets()
    trades_csv = fetch_24h_trades()
    
    if trades_csv:
        profiles_csv = update_wallet_profiles()
        if profiles_csv:
            df_rich, df_raw_trades = enrich_data(trades_csv, profiles_csv, market_df)
            if df_rich is not None:
                top_whales, top_insiders = predict_and_rank(df_rich, df_raw_trades)
                send_discord_leaderboard(top_whales, top_insiders)
    
    print("\n✅ DONE.")
