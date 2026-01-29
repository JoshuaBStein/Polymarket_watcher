import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless mode for Amarel/Cluster
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# --- CONFIGURATION ---
DATA_FILE = "/scratch/jbs263/Fintechstuff2/wallet_profiles.csv"
OUTPUT_FILE = "final_archetypes_waterfall.csv"
PLOT_FILE = "waterfall_archetype_plot.png"

# --- STRATEGIC THRESHOLDS ---
# "Fresh" = The trade happened within 7 days of the wallet being created
FRESH_WHALE_MAX_TENURE = 7     
FRESH_WHALE_MIN_BET = 1000      # Minimum bet to be considered a Whale

# "Grinder" = Top 5% of activity (High Freq or High Diversity)
GRINDER_PERCENTILE = 0.95   

def load_data():
    try:
        print(f"Loading data from {DATA_FILE}...")
        # Use float32 to save memory if dataset is huge
        dtype_dict = {
            'tenure_days': 'float32',
            'aggression_avg_usd': 'float32',
            'frequency_trades_daily': 'float32',
            'diversification_markets_count': 'float32'
        }
        df = pd.read_csv(DATA_FILE, dtype=dtype_dict)
        print(f"Loaded {len(df)} rows.")
        return df
    except FileNotFoundError:
        print("WARNING: File not found. Generating DUMMY data for testing...")
        np.random.seed(42)
        n = 5000
        df = pd.DataFrame({
            'tenure_days': np.random.randint(0, 1000, n),
            'aggression_avg_usd': np.random.exponential(500, n),
            'frequency_trades_daily': np.random.poisson(3, n),
            'diversification_markets_count': np.random.poisson(5, n)
        })
        # Inject "Fresh Whales" (Young at time of trade, Rich)
        df.loc[0:50, 'tenure_days'] = np.random.randint(0, 7, 51) # 0 to 7 days old
        df.loc[0:50, 'aggression_avg_usd'] = np.random.randint(5000, 50000, 51)
        
        # Inject "Grinders" (High Activity)
        df.loc[51:150, 'frequency_trades_daily'] = np.random.randint(50, 100, 100)
        
        return df

def main():
    df = load_data()
    
    # Initialize 'Archetype' with a default value
    df['Archetype'] = 'Unclassified'

    # --- STEP 1: IDENTIFY FRESH WHALES (The "Time Delta" Rule) ---
    print("--- Step 1: Extracting Fresh Whales ---")
    # Logic: Wallet was created <= 7 days before this trade happened.
    mask_whale = (df['tenure_days'] <= FRESH_WHALE_MAX_TENURE) & \
                 (df['aggression_avg_usd'] >= FRESH_WHALE_MIN_BET)
    
    df.loc[mask_whale, 'Archetype'] = 'Fresh Whale (Alert)'
    print(f" > Found {sum(mask_whale)} Fresh Whales.")

    # --- STEP 2: IDENTIFY GRINDERS (The "Activity" Rule) ---
    print("--- Step 2: Extracting Grinders ---")
    
    # We calculate the top 5% threshold dynamically
    freq_threshold = df['frequency_trades_daily'].quantile(GRINDER_PERCENTILE)
    div_threshold = df['diversification_markets_count'].quantile(GRINDER_PERCENTILE)
    
    print(f" > Grinder Cutoffs: >{freq_threshold:.1f} trades/day OR >{div_threshold:.1f} markets")
    
    # Apply mask (Only to those not already marked as Whales)
    mask_grinder = (df['Archetype'] == 'Unclassified') & \
                   ((df['frequency_trades_daily'] > freq_threshold) | 
                    (df['diversification_markets_count'] > div_threshold))
    
    df.loc[mask_grinder, 'Archetype'] = 'The Grinder (Ignore)'
    print(f" > Found {sum(mask_grinder)} Grinders.")

    # --- STEP 3: CLUSTER THE REMAINDER (Insider vs Casual) ---
    print("--- Step 3: Clustering the Remaining Users ---")
    
    # Filter for unclassified rows
    mask_remaining = df['Archetype'] == 'Unclassified'
    subset = df.loc[mask_remaining].copy()
    
    if len(subset) > 10:
        # We use Log Aggression to separate "Rich Insiders" from "Poor Casuals"
        subset['log_aggression'] = np.log1p(subset['aggression_avg_usd'])
        
        # Simple K-Means with k=2 to find the split
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(subset[['log_aggression']])
        
        # Identify which cluster index (0 or 1) has the HIGHER average money
        center_0 = kmeans.cluster_centers_[0][0]
        center_1 = kmeans.cluster_centers_[1][0]
        
        if center_0 > center_1:
            mapping = {0: 'The Insider (Target)', 1: 'Casual (Noise)'}
        else:
            mapping = {1: 'The Insider (Target)', 0: 'Casual (Noise)'}
            
        # Apply labels back to main dataframe
        df.loc[mask_remaining, 'Archetype'] = pd.Series(clusters, index=subset.index).map(mapping)
        
    print("\nFinal Archetype Counts:")
    print(df['Archetype'].value_counts())

    # --- SAVE DATA ---
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Results saved to {OUTPUT_FILE}")

    # --- PLOTTING ---
    print("Generating Visualization...")
    plt.figure(figsize=(12, 8))
    
    # If dataset is massive, sample it for the plot to avoid memory crash
    if len(df) > 50000:
        print(f"Dataset is large ({len(df)}). Plotting sample of 50,000...")
        plot_df = df.sample(50000, random_state=42)
    else:
        plot_df = df

    sns.scatterplot(
        data=plot_df,
        x='tenure_days',
        y='aggression_avg_usd',
        hue='Archetype',
        style='Archetype',
        palette={
            'Fresh Whale (Alert)': '#FF0000',     # Red
            'The Insider (Target)': '#008000',    # Green
            'The Grinder (Ignore)': '#808080',    # Grey
            'Casual (Noise)': '#ADD8E6',          # Light Blue
            'Unclassified': 'black'
        },
        alpha=0.7,
        s=60
    )
    
    plt.yscale('log')
    plt.title("Trader Archetypes (Waterfall Strategy)")
    plt.xlabel("Tenure at Trade (Days)")
    plt.ylabel("Avg Bet Size ($) - Log Scale")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="--", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    print(f"Plot saved to {PLOT_FILE}")

if __name__ == "__main__":
    main()
