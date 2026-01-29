import pandas as pd
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = '/scratch/jbs263/Fintechstuff2/'
TRADES_PATH = os.path.join(BASE_DIR, 'processed', 'trades.csv')
MARKETS_PATH = os.path.join(BASE_DIR, 'markets.csv')
INPUT_PROFILE_PATH = '/scratch/jbs263/Fintechstuff2/wallet_profiles.csv' 
OUTPUT_FILE = '/scratch/jbs263/Fintechstuff2/wallet_profiles_enriched.csv'

def run_analysis():
    print(f"--- Starting Analysis ---")
    
    # ---------------------------------------------------------
    # 1. Load Markets & Create Lookup Dictionary
    # ---------------------------------------------------------
    print(f"Loading markets from {MARKETS_PATH}...")
    markets = pd.read_csv(MARKETS_PATH)
    
    # FIX 1: Handle Date Formats
    markets['createdAt'] = pd.to_datetime(markets['createdAt'], format='mixed', utc=True)
    
    # FIX 2: Ensure ID is string to match Trades
    markets['id'] = markets['id'].astype(str)
    
    # OPTIMIZATION: Create a dictionary for fast lookup instead of merging
    # This maps market_id -> createdAt and avoids the merge memory crash
    market_date_map = dict(zip(markets['id'], markets['createdAt']))
    
    print(f"Loaded {len(markets):,} markets.")

    # ---------------------------------------------------------
    # 2. Load Wallet Profiles
    # ---------------------------------------------------------
    print(f"Loading profiles from {INPUT_PROFILE_PATH}...")
    if INPUT_PROFILE_PATH.endswith('.xlsx'):
        profiles = pd.read_excel(INPUT_PROFILE_PATH)
    else:
        profiles = pd.read_csv(INPUT_PROFILE_PATH)
        
    target_makers = set(profiles['maker'].unique())
    print(f"Loaded {len(profiles):,} wallet profiles.")

    # ---------------------------------------------------------
    # 3. Load Trades in CHUNKS
    # ---------------------------------------------------------
    print(f"Loading trades from {TRADES_PATH} in chunks...")
    
    chunk_list = []
    chunk_size = 5000000 
    total_loaded = 0
    
    for i, chunk in enumerate(pd.read_csv(TRADES_PATH, chunksize=chunk_size)):
        
        # Filter relevant wallets
        filtered_chunk = chunk[chunk['maker'].isin(target_makers)].copy()
        
        if not filtered_chunk.empty:
            # Type Optimization
            filtered_chunk['maker'] = filtered_chunk['maker'].astype('category')
            filtered_chunk['maker_direction'] = filtered_chunk['maker_direction'].astype('category')
            filtered_chunk['market_id'] = filtered_chunk['market_id'].astype(str) # Ensure string match
            
            # Date Conversion
            filtered_chunk['timestamp'] = pd.to_datetime(filtered_chunk['timestamp'], format='mixed', utc=True)

            # OPTIMIZATION: Map the date here instead of merging later
            # This looks up the market creation time using the dictionary
            filtered_chunk['market_created'] = filtered_chunk['market_id'].map(market_date_map)
            
            chunk_list.append(filtered_chunk)
            total_loaded += len(filtered_chunk)
            
        print(f"Processed chunk {i+1}: accumulated {total_loaded:,} relevant trades so far...")
    
    if not chunk_list:
        print("Error: No trades found for the requested profiles!")
        return

    print("Concatenating chunks...")
    trades_enriched = pd.concat(chunk_list, ignore_index=True)
    del chunk_list 
    
    # ---------------------------------------------------------
    # 4. Calculate Metrics
    # ---------------------------------------------------------
    print("Calculating wallet metrics...")

    # Calculate Time-to-Impact (Trade Time - Market Creation Time)
    trades_enriched['trade_delay_hours'] = (trades_enriched['timestamp'] - trades_enriched['market_created']).dt.total_seconds() / 3600

    # Ensure usd_amount is numeric
    trades_enriched['usd_amount'] = pd.to_numeric(trades_enriched['usd_amount'], errors='coerce').fillna(0)
    
    # Create Signed USD (Buy = Negative, Sell = Positive)
    trades_enriched['signed_usd'] = np.where(
        trades_enriched['maker_direction'] == 'BUY', 
        -trades_enriched['usd_amount'], 
        trades_enriched['usd_amount']
    )
    
    # A. Market-Level Win/Loss (Did they win on this specific market?)
    print(" - Aggregating market outcomes...")
    market_level = trades_enriched.groupby(['maker', 'market_id'], observed=True).agg({
        'signed_usd': 'sum',           # Net PnL for this market
        'trade_delay_hours': 'min'     # Earliest entry
    }).reset_index()
    
    market_level['is_win'] = market_level['signed_usd'] > 0
    
    # B. Wallet-Level Aggregation
    print(" - Aggregating wallet stats...")
    
    # 1. Totals (Invested & Net PnL)
    wallet_stats = trades_enriched.groupby('maker', observed=True).agg({
        'usd_amount': [
            lambda x: x[trades_enriched.loc[x.index, 'maker_direction'] == 'BUY'].sum(), # Total Invested
        ],
        'signed_usd': 'sum' # Total Net PnL
    })
    wallet_stats.columns = ['total_invested', 'total_net_pnl']
    
    # 2. Win Rate & Timings
    wallet_market_stats = market_level.groupby('maker', observed=True).agg({
        'is_win': ['sum', 'count'],     
        'trade_delay_hours': 'mean'     
    })
    wallet_market_stats.columns = ['wins', 'unique_markets', 'avg_time_to_impact_hours']
    
    # 3. Conviction (Average Buy Size)
    buy_trades = trades_enriched[trades_enriched['maker_direction'] == 'BUY']
    conviction_stats = buy_trades.groupby('maker', observed=True)['usd_amount'].mean()
    conviction_stats.name = 'conviction_avg_buy_usd'
    
    # Merge all stats
    final_stats = wallet_stats.join(wallet_market_stats).join(conviction_stats)
    
    # Final Metrics
    final_stats['roi_pct'] = (final_stats['total_net_pnl'] / final_stats['total_invested'].replace(0, np.nan)) * 100
    final_stats['roi_pct'] = final_stats['roi_pct'].fillna(0)
    final_stats['win_rate_pct'] = (final_stats['wins'] / final_stats['unique_markets']) * 100
    
    # Alpha (Relative to group average)
    avg_market_roi = final_stats['roi_pct'].mean()
    final_stats['alpha_score'] = final_stats['roi_pct'] - avg_market_roi
    
    # ---------------------------------------------------------
    # 5. Save
    # ---------------------------------------------------------
    print("Merging with original profiles...")
    merged_df = profiles.merge(final_stats, on='maker', how='left')
    
    # Fill NaN
    fill_cols = ['roi_pct', 'alpha_score', 'win_rate_pct', 'conviction_avg_buy_usd', 'avg_time_to_impact_hours']
    merged_df[fill_cols] = merged_df[fill_cols].fillna(0)
    
    print(f"Saving to {OUTPUT_FILE}...")
    merged_df.to_csv(OUTPUT_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    run_analysis()
