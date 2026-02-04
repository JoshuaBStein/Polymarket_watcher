import pandas as pd
import numpy as np
import os
import gc
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score

# --- CONFIG ---
BASE_DIR = '/scratch/jbs263/Fintechstuff2'
# Try 'processed' folder first, fall back to base
trades_path = os.path.join(BASE_DIR, 'processed', 'trades.csv')
if not os.path.exists(trades_path):
    trades_path = os.path.join(BASE_DIR, 'trades.csv')
markets_path = os.path.join(BASE_DIR, 'markets.csv')

print(f"--- FINAL PIPELINE STARTING ---")

# --- 1. LOAD TRADES ---
try:
    print("Loading Trades...")
    # We load 'market_id' specifically to merge
    cols_to_load = ['timestamp', 'market_id', 'maker', 'usd_amount', 'price']
    
    dtypes = {
        'usd_amount': 'float32', 
        'price': 'float32',
        'market_id': 'string' # Load as string to be safe
    }
    
    trades = pd.read_csv(trades_path, usecols=cols_to_load, dtype=dtypes)
    print(f"Loaded {len(trades)} trades.")
except Exception as e:
    print(f"CRITICAL ERROR loading trades: {e}")
    exit()

# --- 2. LOAD MARKETS (And Rename 'id') ---
try:
    print("Loading Markets...")
    # We load 'id' (to match market_id) and 'market_slug' (to track price history)
    cols_to_load = ['id', 'market_slug']
    
    # Check for volume again just in case
    market_header = pd.read_csv(markets_path, nrows=0).columns.tolist()
    if 'volume' in market_header:
        cols_to_load.append('volume')

    markets = pd.read_csv(markets_path, usecols=cols_to_load, dtype={'id': 'string'})
    
    # --- THE FIX: RENAME 'id' TO 'market_id' ---
    print("Renaming 'id' to 'market_id'...")
    markets = markets.rename(columns={'id': 'market_id'})
    
    print(f"Loaded {len(markets)} markets.")
except Exception as e:
    print(f"CRITICAL ERROR loading markets: {e}")
    exit()

# --- 3. MERGE ---
print("Merging on 'market_id'...")
# Now both have 'market_id', so this will work
df = trades.merge(markets, on='market_id', how='left')

# Drop orphan trades (trades with no market info)
df = df.dropna(subset=['market_slug'])

# Free memory
del trades
del markets
gc.collect()

# --- 4. FEATURE ENGINEERING ---
print("Calculating Wallet Statistics...")
# Calculate standard deviation of bet size for every wallet
whale_stats = df.groupby('maker')['usd_amount'].agg(['mean', 'std']).reset_index()
whale_stats.columns = ['maker', 'avg_bet_size', 'std_bet_size']

df = df.merge(whale_stats, on='maker', how='left')

# Z-Score: How big is this bet compared to their usual?
df['z_score_size'] = (df['usd_amount'] - df['avg_bet_size']) / (df['std_bet_size'] + 1e-5)

# --- 5. TARGET GENERATION (Future ROI) ---
print("Calculating Future ROI (Labels)...")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

# We rely on market_slug to track the specific asset's price history
price_history = df[['market_slug', 'timestamp', 'price']].copy()
df['future_lookup_time'] = df['timestamp'] + pd.Timedelta(hours=6)

print("Running Time-Travel Merge (this may take a minute)...")
df_labeled = pd.merge_asof(
    df,
    price_history,
    left_on='future_lookup_time',
    right_on='timestamp',
    by='market_slug', # Matches price history for the SAME asset
    suffixes=('', '_future'),
    direction='nearest',
    tolerance=pd.Timedelta(hours=1)
)

del df
del price_history
gc.collect()

# Calculate ROI
df_labeled['future_roi'] = (df_labeled['price_future'] - df_labeled['price']) / df_labeled['price']
df_labeled = df_labeled.dropna(subset=['future_roi'])

# Target: 1 if > 5% profit, 0 otherwise
df_labeled['is_profitable'] = (df_labeled['future_roi'] > 0.05).astype(int)

print(f"Training Data: {len(df_labeled)} rows")
print(f"Profitable Trades: {df_labeled['is_profitable'].sum()}")

# --- 6. TRAIN XGBOOST ---
print("Training Model...")
features = ['usd_amount', 'z_score_size', 'price']
if 'volume' in df_labeled.columns:
    features.append('volume')

X = df_labeled[features]
y = df_labeled['is_profitable']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(
    n_estimators=100, 
    max_depth=6, 
    learning_rate=0.1,
    eval_metric='logloss',
    use_label_encoder=False,
    n_jobs=4
)
model.fit(X_train, y_train)

# --- 7. RESULTS ---
preds = model.predict(X_test)
precision = precision_score(y_test, preds)
accuracy = accuracy_score(y_test, preds)

print("\n" + "="*40)
print(f" FINAL MODEL METRICS ")
print("="*40)
print(f"Precision: {precision:.4f}")
print(f"Accuracy:  {accuracy:.4f}")
print("="*40)

# Save
model.save_model(os.path.join(BASE_DIR, 'whale_filter_xgb.json'))
print("Model saved.")
