import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # No display driver for Amarel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- CONFIGURATION ---
DATA_FILE = "/scratch/jbs263/Fintechstuff2/wallet_profiles.csv"
OUTPUT_FILE = "wallet_archetypes.csv"
PLOT_FILE = "archetype_clusters.png"

def load_data():
    try:
        print(f"Loading data from {DATA_FILE}...")
        df = pd.read_csv(DATA_FILE)
        print(f"Loaded {len(df)} rows.")
        return df
    except FileNotFoundError:
        print(f"ERROR: Could not find file at {DATA_FILE}")
        exit(1)

def main():
    print("--- Starting K-Means Clustering ---")
    df = load_data()

    # 1. Feature Selection
    # These match your project description exactly
    features = [
        'tenure_days',
        'aggression_avg_usd',
        'frequency_trades_daily',
        'domain_focus_sports_pct',   # High for Insiders
        'diversification_markets_count' # High for Grinders
    ]
    
    # Check if columns exist
    missing = [c for c in features if c not in df.columns]
    if missing:
        print(f"CRITICAL ERROR: Missing columns: {missing}")
        exit(1)

    # 2. Preprocessing
    # K-Means is sensitive to scale. We must normalize features.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    # 3. K-Means Clustering
    # We force 3 clusters: Grinder, Insider, Whale
    print("Running K-Means (k=3)...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster_id'] = kmeans.fit_predict(X_scaled)

    # 4. Dynamic Archetype Labeling
    # We look at the average stats of each cluster to figure out which is which.
    centroids = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_), 
        columns=features
    )
    centroids['cluster_id'] = range(3)
    
    print("\nCluster Centroids (Averages):")
    print(centroids.to_string())

    # --- LOGIC MAPPING ---
    
    # Identify "The Grinder" -> Highest Frequency
    grinder_id = centroids.loc[centroids['frequency_trades_daily'].idxmax(), 'cluster_id']
    
    # Identify "The Fresh Whale" -> Lowest Tenure (and usually High Aggression)
    # We exclude the grinder_id from this search
    remaining = centroids[centroids['cluster_id'] != grinder_id]
    whale_id = remaining.loc[remaining['tenure_days'].idxmin(), 'cluster_id']
    
    # Identify "The Insider" -> The last one left (usually High Domain Focus)
    insider_id = [x for x in [0, 1, 2] if x not in [grinder_id, whale_id]][0]

    label_map = {
        grinder_id: "The Grinder (Ignore)",
        whale_id: "The Fresh Whale (Alert)",
        insider_id: "The Insider (Target)"
    }
    
    df['Archetype'] = df['cluster_id'].map(label_map)
    print("\nLabels assigned:")
    print(df['Archetype'].value_counts())

    # 5. Save Results
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Results saved to {OUTPUT_FILE}")

    # 6. Visualization
    print("Generating plot...")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x='tenure_days',
        y='aggression_avg_usd',
        hue='Archetype',
        style='Archetype',
        palette='viridis',
        s=60,
        alpha=0.8
    )
    plt.title("Identified Archetypes: Aggression vs Tenure")
    plt.yscale('log') # Log scale helps if whales are betting $10k and grinders $10
    plt.xlabel("Tenure (Days)")
    plt.ylabel("Avg Bet Size (USD) - Log Scale")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(PLOT_FILE)
    print(f"Plot saved to {PLOT_FILE}")

if __name__ == "__main__":
    main()
