import pandas as pd
import numpy as np

# --- CONFIG ---
INPUT_FILE = "labeled_training_data.csv"
OUTPUT_FILE = "balanced_training_data.csv"

# --- SAMPLING STRATEGY ---
# We use a dictionary to define exactly how many rows we want from each class.
# -1 means "Take everything you have"
TARGET_COUNTS = {
    3: -1,    # Fresh Whales: TAKE ALL (Expected ~906)
    2: 3000,  # Insiders: Sample 3,000
    1: 3000,  # Grinders: Sample 3,000
    0: 3000   # Casuals: Sample 3,000
}

def main():
    print(f"Loading massive dataset from {INPUT_FILE}...")
    # Read only necessary columns to save memory if needed
    df = pd.read_csv(INPUT_FILE)
    
    print(f"Original Total Rows: {len(df)}")
    
    balanced_chunks = []
    
    print("\n--- Processing Classes ---")
    for class_id, target_count in TARGET_COUNTS.items():
        # Filter for this specific class
        class_df = df[df['target_class'] == class_id]
        available_rows = len(class_df)
        
        if available_rows == 0:
            print(f"WARNING: Class {class_id} has 0 rows! Skipping.")
            continue
            
        # Determine how many to take
        if target_count == -1 or available_rows < target_count:
            # Take everything if we want all (-1) or if we don't have enough to meet the target
            n_to_take = available_rows
            sampled_df = class_df.copy()
            action = "Kept ALL"
        else:
            # Randomly sample the target amount
            n_to_take = target_count
            sampled_df = class_df.sample(n=target_count, random_state=42)
            action = f"Sampled {target_count}"
            
        print(f"Class {class_id}: Found {available_rows} -> {action} rows.")
        balanced_chunks.append(sampled_df)
    
    # Combine all chunks
    balanced_df = pd.concat(balanced_chunks)
    
    # SHUFFLE the dataset
    # This is critical so the model doesn't learn order (e.g., "all whales are at the top")
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("\n--- Final Balanced Dataset ---")
    print(balanced_df['target_class'].value_counts().sort_index())
    
    # Save
    balanced_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccess! Saved {len(balanced_df)} rows to {OUTPUT_FILE}")
    print("You can now train your Random Forest/XGBoost on this file.")

if __name__ == "__main__":
    main()
