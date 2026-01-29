import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# --- CONFIGURATION ---
# Set this to True to load your actual file, False to use dummy data for testing
USE_REAL_DATA = False 
DATA_FILE = "/scratch/jbs263/Fintechstuff2/wallet_profiles.csv" 

def load_data():
    if USE_REAL_DATA:
        # Ensure your CSV has the exact columns listed below
        return pd.read_csv(DATA_FILE)
    else:
        # Generate synthetic data for testing
        np.random.seed(42)
        n_samples = 1000
        data = pd.DataFrame({
            'tenure_days': np.random.randint(1, 1000, n_samples),
            'aggression_avg_usd': np.random.exponential(500, n_samples),
            'frequency_trades_daily': np.random.poisson(5, n_samples),
            'domain_focus_sports_pct': np.random.rand(n_samples),
            'domain_focus_politics_pct': np.random.rand(n_samples),
            'domain_focus_crypto_pct': np.random.rand(n_samples),
            'diversification_markets_count': np.random.randint(1, 20, n_samples)
        })
        # Inject specific archetypes for testing
        # 1. The Grinder (High Freq, Low Aggression)
        data.iloc[0:50, 2] = 100 # High frequency
        data.iloc[0:50, 1] = 50  # Low USD
        
        # 2. The Insider (Low Freq, High Aggression, Specific Focus)
        data.iloc[50:70, 2] = 1   # Low frequency
        data.iloc[50:70, 1] = 5000 # High USD
        data.iloc[50:70, 3] = 0.95 # High Sports focus
        
        # 3. The Fresh Whale (Zero Tenure, High Aggression)
        data.iloc[70:80, 0] = 0   # 0 Tenure
        data.iloc[70:80, 1] = 10000 # Massive bet
        
        return data

def main():
    print("--- Starting Analysis on Amarel Node ---")
    
    # 1. Load Data
    df = load_data()
    feature_cols = [
        'tenure_days', 'aggression_avg_usd', 'frequency_trades_daily', 
        'domain_focus_sports_pct', 'domain_focus_politics_pct', 
        'domain_focus_crypto_pct', 'diversification_markets_count'
    ]
    X = df[feature_cols]

    # 2. Preprocessing (Standard Scaling is crucial for K-Means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. K-Means Clustering
    # We use 3 clusters as per your design
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster_id'] = kmeans.fit_predict(X_scaled)

    # 4. Dynamic Archetype Naming
    # K-Means labels (0,1,2) are random. We must analyze centroids to name them.
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=feature_cols)
    centroids['cluster_id'] = range(3)
    
    # Heuristic Logic to map Cluster ID to Name
    # The Grinder: Max Frequency
    grinder_id = centroids['frequency_trades_daily'].idxmax()
    
    # The Fresh Whale: Min Tenure AND High Aggression (simplistic check)
    # We exclude the grinder ID from the search
    remaining_ids = [i for i in [0,1,2] if i != grinder_id]
    subset = centroids.loc[remaining_ids]
    # Find the one with lower tenure among the remaining
    whale_id = subset.sort_values('tenure_days').index[0]
    
    # The Insider: The last one remaining
    insider_id = [i for i in [0,1,2] if i not in [grinder_id, whale_id]][0]
    
    label_map = {
        grinder_id: "The Grinder (Bot)",
        whale_id: "The Fresh Whale",
        insider_id: "The Insider"
    }
    
    df['Archetype'] = df['cluster_id'].map(label_map)

    # 5. Isolation Forest (Anomaly Detection)
    # This specifically looks for 'outliers' regardless of the clusters
    iso = IsolationForest(contamination=0.05, random_state=42)
    df['is_anomaly'] = iso.fit_predict(X_scaled) # -1 is anomaly, 1 is normal
    
    # 6. Save Results
    output_filename = "archetype_results.csv"
    df.to_csv(output_filename, index=False)
    print(f"Analysis complete. Results saved to {output_filename}")

    # 7. Generate Visualization (Headless for Cluster)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, 
        x='tenure_days', 
        y='aggression_avg_usd', 
        hue='Archetype', 
        style='is_anomaly',
        palette='viridis'
    )
    plt.title("Trader Archetypes: Aggression vs Tenure")
    plt.xlabel("Tenure (Days)")
    plt.ylabel("Avg Bet Size ($)")
    plt.savefig("cluster_plot.png") # Save instead of show
    print("Plot saved to cluster_plot.png")

if __name__ == "__main__":
    main()
