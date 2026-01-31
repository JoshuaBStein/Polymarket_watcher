import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- CONFIGURATION ---
INPUT_FILE = "balanced_training_data.csv"
OUTPUT_JSON = "whale_hunter_model.json"

def main():
    print(f"Loading data from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("Error: 'balanced_training_data.csv' not found.") 
        print("Please run the balancing script first.")
        return

    # 1. Prepare Features
    # Ensuring we stick to the 4 behavioral features defined in your logic
    features = ['tenure_days', 'aggression_avg_usd', 'frequency_trades_daily', 'diversification_markets_count']
    X = df[features]
    y = df['target_class']

    # 2. Split Data (90% Train, 10% Test)
    print("Splitting data 90/10...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)

    # 3. Train XGBoost
    # We use XGBoost specifically because it supports native JSON export
    print("\n--- Training XGBoost Classifier ---")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softmax',  # Multi-class classification
        num_class=4,                # Classes 0, 1, 2, 3
        eval_metric='mlogloss',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # 4. Evaluation
    acc = accuracy_score(y_test, preds)
    print(f"\nModel Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    # Target names map back to your integer classes for readability
    target_names = ['Casual (0)', 'Grinder (1)', 'Insider (2)', 'Fresh Whale (3)']
    print(classification_report(y_test, preds, target_names=target_names))

    # 5. Export to JSON
    print(f"\n--- Saving Model to JSON ---")
    # This saves the model architecture and weights in a portable JSON format
    model.save_model(OUTPUT_JSON)
    
    print(f"SUCCESS: Model saved to: {OUTPUT_JSON}")
    print("You can verify the file contents using a text editor or load it into another application.")

if __name__ == "__main__":
    main()
