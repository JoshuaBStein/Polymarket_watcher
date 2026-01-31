import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
import joblib

# --- CONFIG ---
INPUT_FILE = "balanced_training_data.csv"

def main():
    print(f"Loading data from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("Error: File not found. Run 'create_balanced_training_set.py' first.")
        return

    # 1. Prepare Data
    # We use the 4 key behavioral features
    features = ['tenure_days', 'aggression_avg_usd', 'frequency_trades_daily', 'diversification_markets_count']
    X = df[features]
    y = df['target_class']

    # 2. Split Data (90% Train, 10% Test)
    print("Splitting data 90/10...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)
    
    print(f"Training set: {len(X_train)} rows")
    print(f"Testing set:  {len(X_test)} rows")

    # --- CONTENDER 1: RANDOM FOREST ---
    print("\n--- Training Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        class_weight='balanced', 
        random_state=42
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    
    # --- CONTENDER 2: XGBOOST ---
    print("--- Training XGBoost ---")
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softmax',  # Multi-class classification
        num_class=4,                # We have 4 classes (0,1,2,3)
        eval_metric='mlogloss',
        random_state=42
    )
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)

    # --- SCORING ---
    print("\n" + "="*40)
    print("FINAL SCORECARD")
    print("="*40)

    models = [("Random Forest", rf_preds), ("XGBoost", xgb_preds)]

    for name, preds in models:
        acc = accuracy_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        
        print(f"\nMODEL: {name}")
        print(f" > Overall Accuracy: {acc:.4f} (Higher is better)")
        print(f" > MSE (Class Error): {mse:.4f} (Lower is better)")
        
        # Important: Check how well it caught the WHALES (Class 3)
        # We extract the metrics specifically for label '3'
        report = classification_report(y_test, preds, output_dict=True)
        whale_stats = report['3']
        
        print(f" > WHALE DETECTION (Class 3):")
        print(f"   - Precision: {whale_stats['precision']:.2f} (When it says 'Whale', is it right?)")
        print(f"   - Recall:    {whale_stats['recall']:.2f}    (Did it find all the Whales?)")
        
    print("\n" + "="*40)
    
    # --- RECOMMENDATION LOGIC ---
    # We prioritize Precision on Whales. Losing money on fake whales is worse than missing real ones.
    rf_whale_prec = classification_report(y_test, rf_preds, output_dict=True)['3']['precision']
    xgb_whale_prec = classification_report(y_test, xgb_preds, output_dict=True)['3']['precision']
    
    if rf_whale_prec >= xgb_whale_prec:
        winner = rf
        winner_name = "Random Forest"
    else:
        winner = xgb
        winner_name = "XGBoost"

    print(f"WINNER: {winner_name}")
    joblib.dump(winner, "final_whale_hunter_model.pkl")
    print(f"Saved best model to 'final_whale_hunter_model.pkl'")

if __name__ == "__main__":
    main()
