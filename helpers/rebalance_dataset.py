##Previous scripts created dataset by keeping all the whales and then adding nosie at a 1:1 ratio
##this script reblances the training dataset by ensuring all classification classes are represented
##in proportions that the model would encoutner upon live inference
import pandas as pd
import numpy as np

# --- CONFIG ---
INPUT_FILE = "/home/jbs263/Fintech/DecisionTree/NewDat/labeled_training_data.csv"
OUTPUT_FILE = "representative_training_data.csv" # Renamed to reflect goal

# --- REVISED SAMPLING STRATEGY ---
# The previous 1:1:1:1 ratio made Whales look too common.
# We now use a "Representative" strategy.
# We keep ALL Whales, but we significantly increase the "Background Noise"
# (Casuals and Grinders) so the model learns to be conservative.

TARGET_COUNTS = {
    # 3 (Fresh Whales): KEEP ALL (~906). 
    # This is our signal.
    3: -1,    

    # 2 (Insiders): KEEP ALL or High Cap. 
    # These are distinct enough from Whales that they usually don't confuse the model.
    # If you have < 5000, keep all.
    2: -1,    

    # 1 (Grinders): INCREASED to 10,000 (was 3,000).
    # Grinders (high freq/low agg) are the most common False Positives for Whales.
    # We need the model to see LOTS of examples of "High Activity != Whale".
    1: 10000, 

    # 0 (Casuals): INCREASED to 30,000 (was 3,000).
    # We need to flood the dataset with Casuals so the model learns 
    # the "Prior Probability" of being a Whale is low.
    0: 30000  
}

def main():
    print(f"Loading massive dataset from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    print(f"Original Total Rows: {len(df)}")
    print("Original Distribution:")
    print(df['target_class'].value_counts().sort_index())
    
    balanced_chunks = []
    
    print("\n--- Processing Classes ---")
    for class_id, target_count in TARGET_COUNTS.items():
        # Filter for this specific class
        class_df = df[df['target_class'] == class_id]
        available_rows = len(class_df)
        
        if available_rows == 0:
            print(f"WARNING: Class {class_id} has 0 rows! Skipping.")
            continue
            
        # Determine sampling logic
        if target_count == -1 or available_rows < target_count:
            # Take everything
            sampled_df = class_df.copy()
            action = f"Kept ALL ({available_rows})"
            # Weight is 1.0 because we kept the original representation
            sampled_df['sample_weight'] = 1.0 
        else:
            # Randomly sample the target amount
            sampled_df = class_df.sample(n=target_count, random_state=42)
            action = f"Sampled {target_count}/{available_rows}"
            
            # --- CRITICAL ADDITION: SAMPLE WEIGHTS ---
            # Even though we are downsampling Casuals, we want the model to know
            # that these 30,000 rows represent a much larger reality.
            # Weight = (Original_Count / Sampled_Count)
            # This is optional for RF/XGBoost but highly recommended if using 'weight' col in training.
            weight = available_rows / target_count
            sampled_df['sample_weight'] = weight

        print(f"Class {class_id}: Found {available_rows} -> {action}")
        balanced_chunks.append(sampled_df)
    
    # Combine all chunks
    balanced_df = pd.concat(balanced_chunks)
    
    # SHUFFLE the dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("\n--- Final Training Dataset Distribution ---")
    print(balanced_df['target_class'].value_counts().sort_index())
    
    # Save
    balanced_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccess! Saved {len(balanced_df)} rows to {OUTPUT_FILE}")
    print("NOTE: Ensure you use 'sample_weight' in your XGBoost training if you want strictly calibrated probabilities,")
    print("otherwise, this dataset ratio (approx 1 Whale : 30 Noise) will naturally fix the trigger happiness.")

if __name__ == "__main__":
    main()
