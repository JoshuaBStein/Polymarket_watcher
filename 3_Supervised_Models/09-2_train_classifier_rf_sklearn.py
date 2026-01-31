import pandas as pd
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- CONFIG ---
INPUT_FILE = "processed_training_data.csv"
SCOUT_MODEL_FILE = "scout_model.joblib"       # Standard Python Model Format
SCOUT_METADATA_FILE = "scout_metadata.json"

# BEHAVIOR ONLY FEATURES (No money variables)
BEHAVIOR_FEATURES = [
    'frequency_trades_daily', 
    'diversification_markets_count',
    'tenure_days'
]

def main():
    print("--- TRAINING SCOUT: ORIGINAL RANDOM FOREST (SCIKIT-LEARN) ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Ensure you are in the correct directory.")
        return
    
    X = df[BEHAVIOR_FEATURES]
    y = df['target_class']

    # 2. Train/Test Split
    # Uses stratify=y to ensure Whales are evenly distributed between Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Train the "Lazy" Model
    # CRITICAL: We do NOT use 'class_weight="balanced"'.
    # This forces the model to be selective (High Precision), avoiding False Positives.
    print(f"Training Standard Random Forest on {len(X_train)} rows...")
    
    rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10,        # The optimized depth that prevents overfitting
        random_state=42,
        n_jobs=-1            # Uses all available CPU cores
    )
    
    rf.fit(X_train, y_train)
    
    # 4. Evaluation
    preds = rf.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    whale_stats = report['3']
    
    print("\nSCOUT PERFORMANCE (Original RF):")
    print(f" > Precision: {whale_stats['precision']:.2%} (Trust)")
    print(f" > Recall:    {whale_stats['recall']:.2%}    (Coverage)")
    
    # 5. Save Logic
    # Only save if the model is trustworthy (Precision > 70%)
    if whale_stats['precision'] > 0.70:
        print("\n--- CONCLUSION: SUCCESS ---")
        
        # Save the model object using joblib
        joblib.dump(rf, SCOUT_MODEL_FILE)
        print(f" > Saved Model to {SCOUT_MODEL_FILE}")
        
        metadata = {
            "model_name": "Whale Scout (Scikit-Learn RF)",
            "model_file": SCOUT_MODEL_FILE,
            "format": "joblib",
            "features": BEHAVIOR_FEATURES,
            "classes": {0: "Casual", 1: "Grinder", 2: "Insider", 3: "Whale"},
            "metrics": {
                "precision": whale_stats['precision'],
                "recall": whale_stats['recall']
            },
            "note": "Standard Scikit-Learn model. Requires 'joblib' to load."
        }
        
        with open(SCOUT_METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f" > Saved Metadata to {SCOUT_METADATA_FILE}")
        
    else:
        print("\n--- CONCLUSION: FAILED ---")
        print("Model precision was too low. Check input data quality.")

if __name__ == "__main__":
    main()
