import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- CONFIGURATION ---
FILE_NAME = 'wallet_profiles_enriched.csv'  # CHANGE THIS if your file name is different
n_clusters = 4
anomaly_threshold_percentile = 95

def main():
    print(f"--- Starting Analysis on {FILE_NAME} ---")
    
    # 1. LOAD DATA
    if not os.path.exists(FILE_NAME):
        print(f"ERROR: File '{FILE_NAME}' not found. Please check the file name.")
        return

    # Try loading with pyarrow for speed, fall back to default if not installed
    try:
        df = pd.read_csv(FILE_NAME, engine='pyarrow')
    except:
        print("PyArrow not found or error, falling back to standard loader...")
        df = pd.read_csv(FILE_NAME)

    print(f"Data Loaded: {df.shape[0]} rows found.")

    # 2. FEATURE SELECTION
    # We use the correct column names based on your dataset
    features = [
        'tenure_days',
        'aggression_avg_usd', 
        'frequency_trades_daily', 
        'domain_focus_sports_pct',
        'domain_focus_politics_pct',
        'domain_focus_crypto_pct',
        'diversification_markets_count',
        'total_invested',
        'wins',
        'unique_markets',
        'avg_time_to_impact_hours',
        'conviction_avg_buy_usd',
        'win_rate_pct'
    ]

    # Handle missing values (NaN) by filling with 0
    X = df[features].fillna(0)

    # 3. SCALING (Crucial for K-Means)
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. TRAIN MODEL
    print(f"Training K-Means Model with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # 5. ANOMALY DETECTION (Calculated Distance)
    print("Calculating Anomaly Scores...")
    # Get distance to ALL cluster centers
    distances = kmeans.transform(X_scaled)
    # Find distance to the NEAREST cluster center (this is the anomaly score)
    min_distances = np.min(distances, axis=1)
    df['anomaly_score'] = min_distances
    
    # Flag top 5% as anomalies
    threshold_val = np.percentile(min_distances, anomaly_threshold_percentile)
    df['is_anomaly'] = df['anomaly_score'] > threshold_val

    print(f"Anomaly Threshold set at score: {threshold_val:.2f}")
    print(f"Number of Anomalies Detected: {df['is_anomaly'].sum()}")

    # 6. VISUALIZATION

    # A. PCA Scatter Plot (The Map)
    print("Generating PCA Map...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['pca1'] = X_pca[:, 0]
    df['pca2'] = X_pca[:, 1]

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df, x='pca1', y='pca2', 
        hue='cluster', style='is_anomaly', 
        palette='viridis', s=100, alpha=0.7
    )
    plt.title('Polymarket Trader Map (with Anomalies)')
    plt.savefig('cluster_map_anomalies.png')
    plt.close()

    # B. Feature Breakdown
    print("Generating Feature Comparison...")
    # Compare "Aggression" vs "Frequency" (Log Scale)
    plt.figure(figsize=(10, 6))
    summary = df.groupby('cluster')[['aggression_avg_usd', 'frequency_trades_daily', 'total_invested']].mean()
    # Normalize for chart visibility
    summary_norm = (summary - summary.min()) / (summary.max() - summary.min())
    sns.heatmap(summary_norm, annot=True, cmap='Blues')
    plt.title('Cluster Archetypes (Normalized Heatmap)')
    plt.savefig('cluster_archetypes.png')
    plt.close()

    # 7. EXPORT RESULTS
    print("Saving Models and Data...")
    
    # Save the "Brains" for live prediction
    joblib.dump(kmeans, 'poly_model.pkl')
    joblib.dump(scaler, 'poly_scaler.pkl')
    
    # Save the labeled CSV so you can look at it in Excel
    output_file = 'wallet_profiles_labeled.csv'
    df.to_csv(output_file, index=False)
    
    print("\n--- SUCCESS ---")
    print(f"1. Analyzed Data saved to: {output_file}")
    print("2. Model saved as: poly_model.pkl")
    print("3. Scaler saved as: poly_scaler.pkl")
    print("4. Charts saved: cluster_map_anomalies.png, cluster_archetypes.png")
    print("\nTop 3 Anomalies found (Highest Distance Scores):")
    print(df.sort_values(by='anomaly_score', ascending=False)[['maker', 'cluster', 'anomaly_score']].head(3))

if __name__ == "__main__":
    main()
