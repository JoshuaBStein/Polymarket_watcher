import pandas as pd
import numpy as np
import os
import gc
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, recall_score

# --- CONFIG ---
BASE_DIR = '/scratch/jbs263/Fintechstuff2'
MODEL_PATH = os.path.join(BASE_DIR, 'whale_filter_xgb.json')

# Data Paths
trades_path = os.path.join(BASE_DIR, 'processed', 'trades.csv')
if not os.path.exists(trades_path): trades_path = os.path.join(BASE_DIR, 'trades.csv')
markets_path = os.path.join(BASE_DIR, 'markets.csv')

print(f"--- STARTING THRESHOLD ANALYSIS ---")

# --- 1. RE-LOAD & PREPARE DATA (Same logic as training) ---
print("Loading Data...")
# Trades
cols_trades = ['timestamp', 'market_id', 'maker', 'usd_amount', 'price']
dtypes_trades = {'usd_amount': 'float32', 'price': 'float32', 'market_id': 'string'}
trades = pd.read_csv(trades_path, usecols=cols_trades, dtype=dtypes_trades)

# Markets
cols_mkts = ['id', 'market_slug', 'volume'] if 'volume' in pd.read_csv(markets_path, nrows=0).columns else ['id', 'market_slug']
markets = pd.read_csv(markets_path, usecols=cols_mkts, dtype={'id': 'string'})
markets = markets.rename(columns={'id': 'market_id'})

# Merge
df = trades.merge(markets, on='market_id', how='left')
df = df.dropna(subset=['market_slug'])
del trades, markets
gc.collect()

# Feature Engineering: Wallet Stats
print("Re-calculating Features...")
whale_stats = df.groupby('maker')['usd_amount'].agg(['mean', 'std']).reset_index()
whale_stats.columns = ['maker', 'avg_bet_size', 'std_bet_size']
df = df.merge(whale_stats, on='maker', how='left')
df['z_score_size'] = (df['usd_amount'] - df['avg_bet_size']) / (df['std_bet_size'] + 1e-5)

# Generate Labels (Ground Truth)
print("Generating Truth Labels...")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')
price_history = df[['market_slug', 'timestamp', 'price']].copy()
df['future_lookup_time'] = df['timestamp'] + pd.Timedelta(hours=6)

df_labeled = pd.merge_asof(
    df, price_history, left_on='future_lookup_time', right_on='timestamp',
    by='market_slug', suffixes=('', '_future'), direction='nearest', tolerance=pd.Timedelta(hours=1)
)
del df, price_history
gc.collect()

df_labeled['future_roi'] = (df_labeled['price_future'] - df_labeled['price']) / df_labeled['price']
df_labeled = df_labeled.dropna(subset=['future_roi'])
df_labeled['is_profitable'] = (df_labeled['future_roi'] > 0.05).astype(int)

# --- 2. LOAD MODEL & PREDICT ---
print("Loading Model...")
features = ['usd_amount', 'z_score_size', 'price']
if 'volume' in df_labeled.columns: features.append('volume')

X = df_labeled[features]
y_true = df_labeled['is_profitable']

# Use a test split to ensure fairness (same random_state as training script)
_, X_test, _, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42)

model = XGBClassifier()
model.load_model(MODEL_PATH)

# Get PROBABILITIES (The confidence score 0.0 to 1.0)
print("Running Predictions...")
probs = model.predict_proba(X_test)[:, 1]

# --- 3. THE ANALYSIS LOOP ---
print("\n" + "="*65)
print(f"{'THRESHOLD':<10} | {'PRECISION (Win Rate)':<20} | {'TRADES TAKEN':<15} | {'ACCURACY':<10}")
print("="*65)

# Test thresholds from 0.50 to 0.95
for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
    # Apply logic: If Prob > Threshold, predict 1 (Buy), else 0 (Pass)
    preds = (probs > thresh).astype(int)
    
    # Calculate Metrics
    # Precision: TP / (TP + FP) -> % of our bets that won
    prec = precision_score(y_test, preds, zero_division=0)
    
    # Count how many trades we actually took
    trades_taken = preds.sum()
    
    # Accuracy just for context
    acc = accuracy_score(y_test, preds)
    
    print(f"{thresh:.2f}       | {prec:.2%}              | {trades_taken:<15} | {acc:.2%}")

print("="*65)
print("\nINTERPRETATION:")
print("- PRECISION is your Edge. (Higher is better)")
print("- TRADES TAKEN is your Volume. (Too low = missed opportunities)")
print("- OPTIMAL: Pick the highest Precision where 'Trades Taken' is still significant.")
