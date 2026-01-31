(base) [jbs263@amarel2 NewDat]$ cat predict_roi.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = "/scratch/jbs263/Fintechstuff2/wallet_profiles_enriched.csv"
OUTPUT_MODEL_ROI = "/scratch/jbs263/Fintechstuff2/model_roi_predictor.json"
OUTPUT_MODEL_ALPHA = "/scratch/jbs263/Fintechstuff2/model_alpha_predictor.json"

FEATURES = [
    'tenure_days',
    'aggression_avg_usd',
    'frequency_trades_daily',
    'diversification_markets_count',
    'domain_focus_sports_pct',
    'domain_focus_politics_pct',
    'domain_focus_crypto_pct'
]

def get_smart_sample(df):
    """
    Creates a high-quality dataset by keeping all 'interesting' wallets
    and downsampling the boring ones.
    """
    print(f"Filtering dataset (Original: {len(df)} rows)...")
    
    # 1. Keep ALL "Interesting" Wallets
    interesting_mask = (
        (df['aggression_avg_usd'] > 500) | 
        (df['roi_pct'] > 20) | 
        (df['frequency_trades_daily'] > 0.5) 
    )
    df_interesting = df[interesting_mask]
    
    # 2. Downsample the "Noise" (Casuals)
    df_noise = df[~interesting_mask]
    # Sample 10% of noise to keep the dataset manageable but balanced
    df_noise_sampled = df_noise.sample(frac=0.10, random_state=42)
    
    # 3. Combine & Shuffle
    df_final = pd.concat([df_interesting, df_noise_sampled])
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Smart Sample Created: {len(df_final)} rows.")
    return df_final

def train_regressor(df, target_col, model_name):
    print(f"\n--- Training Model for Target: {target_col} ---")
    
    # Clean NaN
    X = df[FEATURES].fillna(0)
    y = df[target_col].fillna(0)
    
    # --- 90/10 SPLIT APPLIED HERE ---
    # test_size=0.10 means 10% for testing, 90% for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    
    print(f"Training Set: {len(X_train)} rows (90%)")
    print(f"Testing Set:  {len(X_test)} rows (10%)")

    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
	max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        objective='reg:squarederror',
        random_state=42,
        early_Stopping_rounds=50
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"Results for {target_col}:")
    print(f" > RMSE: {rmse:.4f}")
    print(f" > MAE:  {mae:.4f}")
    print(f" > R2:   {r2:.4f}")
    
    model.save_model(model_name)
    return model

def plot_feature_importance(model, target_name):
    plt.figure(figsize=(10, 6))
    importance = model.feature_importances_
    imp_df = pd.DataFrame({'Feature': FEATURES, 'Importance': importance})
    imp_df = imp_df.sort_values(by='Importance', ascending=False)
    
    sns.barplot(x='Importance', y='Feature', data=imp_df, palette='viridis')
    plt.title(f"Feature Importance ({target_name})")
    plt.tight_layout()
    plt.savefig(f"feature_importance_{target_name}.png")
    print(f"Plot saved: feature_importance_{target_name}.png")

def main():
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    # 1. APPLY SMART SAMPLING
    df = get_smart_sample(df)

    # 2. MODEL ROI (Clipped for stability)
    df_roi = df[(df['roi_pct'] > -100) & (df['roi_pct'] < 500)]
    model_roi = train_regressor(df_roi, 'roi_pct', OUTPUT_MODEL_ROI)
    plot_feature_importance(model_roi, 'ROI')

    # 3. MODEL ALPHA (Clipped for stability)
    df_alpha = df[(df['alpha_score'] > -10) & (df['alpha_score'] < 10)]
    model_alpha = train_regressor(df_alpha, 'alpha_score', OUTPUT_MODEL_ALPHA)
    plot_feature_importance(model_alpha, 'Alpha')

if __name__ == "__main__":
    main()
