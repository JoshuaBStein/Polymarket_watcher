import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Crucial for running on Amarel (Headless Node)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest

# --- CONFIGURATION ---
DATA_FILE = "/scratch/jbs263/Fintechstuff2/wallet_profiles.csv"
OUTPUT_FILE = "final_archetypes_hybrid.csv"
PLOT_FILE = "hybrid_archetype_plot.png"

# --- THRESHOLDS ---
# Updated based on your feedback: strict 7-day window for "Freshness"
FRESH_WHALE_MAX_TENURE = 7     
FRESH_WHALE_MIN_BET = 1000     

def load_data():
    try:
        print(f"Loading data from {DATA_FILE}...")
        df = pd.read_csv(DATA_FILE)
        print(f"Loaded {len(df)} rows.")
        return df
    except FileNotFoundError:
        print("WARNING: File not found. Generating DUMMY data for testing...")
        np.random.seed(42)
        n = 1000
        data = pd.DataFrame({
            'tenure_days': np.random.randint(0, 1000, n),
            'aggression_avg_usd': np.random.exponential(500, n),
            'frequency_trades_daily': np.random.poisson(5, n),
            'domain_focus_sports_pct': np.random.rand(n),
            'diversification_markets_count': np.random.randint(1, 20, n)
        })
        # Inject "Fresh Whale" (New & Rich)
        data.loc[0:15, 'tenure_days'] = np.random.randint(0, 7, 16)
        data.loc[0:15, 'aggression_avg_usd'] = 25000
        # Inject "Insider" (Old & Focused & Rich)
        data.loc[16:40, 'tenure_days'] = np.random.randint(200, 800, 25)
        data.loc[16:40, 'aggression_avg_usd'] = 15000
        data.loc[16:40, 'diversification_markets_count'] = 1
        return data

def main():
    df = load_data()
    
    # --- STEP 1: FEATURE ENGINEERING ---
    print("--- Step 1: Feature Engineering ---")
    
    # Log-Transform Aggression to fix scaling issues
    # (So $100k doesn't dwarf the other features)
    df['log_aggression'] = np.log1p(df['aggression_avg_usd'])
    
    # Select features for clustering
    train_features = [
        'tenure_days', 
        'log_aggression', 
        'frequency_trades_daily', 
        'diversification_markets_count'
    ]
    
    # Scale features (0-1 range is often safer for mixed types)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df[train_features])
    
    # --- STRATEGY A: HYBRID (RULES + K-MEANS) ---
    print("--- Strategy A: Hybrid (Rules + K-Means) ---")
    
    # 1. Apply Rule: Extract Fresh Whales
    whale_mask = (df['tenure_days'] <= FRESH_WHALE_MAX_TENURE) & \
                 (df['aggression_avg_usd'] >= FRESH_WHALE_MIN_BET)
    
    df['Archetype_Hybrid'] = 'Unclassified' 
    df.loc[whale_mask, 'Archetype_Hybrid'] = 'The Fresh Whale (Alert)'
    
    print(f"  > Identified {sum(whale_mask)} Fresh Whales using strict 7-day rule.")
    
    # 2. Cluster the REST (The Established Wallets)
    established_mask = ~whale_mask
    X_established = X_scaled[established_mask]
    
    if len(X_established) > 0:
        # K-Means on the remaining data
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_established)
        
        # Analyze centroids to name the clusters dynamically
        centers = kmeans.cluster_centers_
        
        # Indices
        idx_agg = train_features.index('log_aggression')
        idx_freq = train_features.index('frequency_trades_daily')
        idx_div = train_features.index('diversification_markets_count')
        
        # Logic to map Cluster ID -> Name
        # Grinder = Max Freq + Max Diversification
        grinder_id = np.argmax(centers[:, idx_freq] + centers[:, idx_div])
        
        # Insider = Max Aggression (excluding Grinder)
        remaining = [i for i in [0,1,2] if i != grinder_id]
        insider_id = remaining[np.argmax(centers[remaining, idx_agg])]
        
        # Casual = The leftovers
        casual_id = [i for i in [0,1,2] if i not in [grinder_id, insider_id]][0]
        
        cluster_map = {
            grinder_id: 'The Grinder (Ignore)',
            insider_id: 'The Insider (Target)',
            casual_id: 'Casual/Noise'
        }
        
        df.loc[established_mask, 'Archetype_Hybrid'] = pd.Series(clusters).map(cluster_map).values

    # --- STRATEGY B: ISOLATION FOREST (Pure Anomaly) ---
    print("--- Strategy B: Isolation Forest ---")
    # Contamination = expected % of anomalies (e.g., 5%)
    iso = IsolationForest(contamination=0.05, random_state=42)
    df['IsoForest_Anomaly'] = iso.fit_predict(X_scaled) 
    # Result: -1 = Anomaly, 1 = Normal
    
#    # --- STRATEGY C: DBSCAN (Density Clustering) ---
#    print("--- Strategy C: DBSCAN ---")
#    # Eps: maximum distance between two samples for one to be considered as in the neighborhood of the other.
#    # Min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
#    dbscan = DBSCAN(eps=0.3, min_samples=10)
#    df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)
#    # Result: -1 = Noise (Outlier), 0,1,2... = Clusters

    # --- SAVE RESULTS ---
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Results saved to {OUTPUT_FILE}")
    
    # --- PLOTTING ---
    print("Generating Visualization...")
    plt.figure(figsize=(12, 8))
    
    # Plotting using the Hybrid Archetypes (Best for Business Logic)
    sns.scatterplot(
        data=df,
        x='tenure_days',
        y='aggression_avg_usd',
        hue='Archetype_Hybrid',
        style='IsoForest_Anomaly', # X shape means IsoForest thinks it's weird
        palette={
            'The Fresh Whale (Alert)': 'red',
            'The Insider (Target)': 'green',
            'The Grinder (Ignore)': 'gray',
            'Casual/Noise': 'lightblue',
            'Unclassified': 'black'
        },
        s=100,
        alpha=0.7
    )
    
    plt.yscale('log')
    plt.title("Archetype Analysis: Hybrid Strategy (Rules + KMeans)")
    plt.xlabel("Tenure (Days)")
    plt.ylabel("Avg Bet Size (USD) - Log Scale")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    print(f"Plot saved to {PLOT_FILE}")

if __name__ == "__main__":
    main()
