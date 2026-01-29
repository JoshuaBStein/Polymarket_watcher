import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Essential for running on clusters/servers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
INPUT_FILE = "/scratch/jbs263/Fintechstuff2/wallet_profiles.csv"
OUTPUT_CSV = "labeled_training_data.csv"
OUTPUT_PLOT = "confirmation_blobs.png"

# --- THRESHOLDS (The "Genotype" Logic) ---
FRESH_WHALE_MAX_TENURE = 7      # Days
FRESH_WHALE_MIN_BET = 1000      # USD
GRINDER_PERCENTILE = 0.95       # Top 5% Activity

def load_data():
    try:
        print(f"Loading raw data from {INPUT_FILE}...")
        # Optimize types for memory efficiency
        dtype_dict = {
            'tenure_days': 'float32',
            'aggression_avg_usd': 'float32',
            'frequency_trades_daily': 'float32',
            'diversification_markets_count': 'float32'
        }
        df = pd.read_csv(INPUT_FILE, dtype=dtype_dict)
        df.fillna(0, inplace=True)
        return df
    except FileNotFoundError:
        print("Error: Input file not found.")
        exit(1)

def apply_classification(df):
    """
    Runs the Waterfall Logic to label every user.
    """
    print("Applying Waterfall Classification Logic...")
    df['Archetype'] = 'Unclassified'
    
    # 1. FRESH WHALES (Rule Based)
    mask_whale = (df['tenure_days'] <= FRESH_WHALE_MAX_TENURE) & \
                 (df['aggression_avg_usd'] >= FRESH_WHALE_MIN_BET)
    df.loc[mask_whale, 'Archetype'] = 'Fresh Whale (Alert)'
    
    # 2. GRINDERS (Rule Based - Top 5% Activity)
    freq_thresh = df['frequency_trades_daily'].quantile(GRINDER_PERCENTILE)
    div_thresh = df['diversification_markets_count'].quantile(GRINDER_PERCENTILE)
    
    # Only label unclassified users (Whales take priority)
    mask_grinder = (df['Archetype'] == 'Unclassified') & \
                   ((df['frequency_trades_daily'] > freq_thresh) | 
                    (df['diversification_markets_count'] > div_thresh))
    df.loc[mask_grinder, 'Archetype'] = 'The Grinder (Ignore)'

    # 3. INSIDERS vs CASUALS (K-Means on remaining users)
    mask_remaining = df['Archetype'] == 'Unclassified'
    subset = df.loc[mask_remaining].copy()
    
    if len(subset) > 10:
        # Use Log Aggression to separate Rich vs Poor
        subset['log_aggression'] = np.log1p(subset['aggression_avg_usd'])
        
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(subset[['log_aggression']])
        
        # Identify which cluster is the "Rich" one
        center_0 = kmeans.cluster_centers_[0][0]
        center_1 = kmeans.cluster_centers_[1][0]
        
        if center_0 > center_1:
            mapping = {0: 'The Insider (Target)', 1: 'Casual (Noise)'}
        else:
            mapping = {1: 'The Insider (Target)', 0: 'Casual (Noise)'}
            
        df.loc[mask_remaining, 'Archetype'] = pd.Series(clusters, index=subset.index).map(mapping)
    else:
        df.loc[mask_remaining, 'Archetype'] = 'Casual (Noise)'

    return df

def create_target_column(df):
    """
    Maps text labels to Integers for ML Training.
    """
    print("Creating Integer Target Column...")
    label_map = {
        'Casual (Noise)': 0,
        'The Grinder (Ignore)': 1,
        'The Insider (Target)': 2,
        'Fresh Whale (Alert)': 3
    }
    df['target_class'] = df['Archetype'].map(label_map)
    return df

def run_tsne_validation(df):
    """
    Runs t-SNE on a sample of the data to generate the verification plot.
    """
    print("\n--- Running t-SNE Verification ---")
    
    # Features to analyze
    features = ['tenure_days', 'aggression_avg_usd', 'frequency_trades_daily', 'diversification_markets_count']
    
    # Sample if data is too big for t-SNE (Plotting >10k points is messy/slow)
    if len(df) > 10000:
        print(f"Data large ({len(df)}). Sampling 10,000 points for visualization...")
        # Stratified sample to ensure we keep Whales in the plot
        plot_df = df.groupby('Archetype', group_keys=False).apply(lambda x: x.sample(min(len(x), 2500)))
    else:
        plot_df = df.copy()

    # Pre-processing (Log transform + Scale)
    X = plot_df[features].copy()
    X['aggression_avg_usd'] = np.log1p(X['aggression_avg_usd'])
    X['tenure_days'] = np.log1p(X['tenure_days'])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run t-SNE
    print("Computing t-SNE coordinates (creating the blobs)...")
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(X_scaled)
    
    plot_df['tsne_one'] = tsne_results[:,0]
    plot_df['tsne_two'] = tsne_results[:,1]
    
    # Generate Plot
    print("Generating Plot...")
    plt.figure(figsize=(16, 10))
    plt.style.use('dark_background')
    
    sns.scatterplot(
        x="tsne_one", y="tsne_two",
        hue="Archetype",
        palette={
            'Fresh Whale (Alert)': '#FF0055',     # Neon Red/Pink
            'The Insider (Target)': '#00FF00',    # Neon Green
            'The Grinder (Ignore)': '#444444',    # Dark Grey
            'Casual (Noise)': '#0088FF'           # Neon Blue
        },
        data=plot_df,
        legend="full",
        alpha=0.8,
        s=40
    )
    
    plt.title('Validation: Wallet Behavior Map (t-SNE)', fontsize=20, color='white')
    plt.xlabel('Dimension 1', color='gray')
    plt.ylabel('Dimension 2', color='gray')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', facecolor='#222')
    plt.grid(False)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_PLOT)
    print(f"Plot saved to: {OUTPUT_PLOT}")

def main():
    # 1. Load
    df = load_data()
    
    # 2. Classify
    df = apply_classification(df)
    
    # 3. Create Targets for ML
    df = create_target_column(df)
    
    # 4. Save Training Data (Full Dataset)
    keep_cols = [
        'tenure_days', 'aggression_avg_usd', 
        'frequency_trades_daily', 'diversification_markets_count',
        'Archetype', 'target_class'
    ]
    df[keep_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"\nSUCCESS: Labeled training data saved to {OUTPUT_CSV}")
    print("Class Distribution:")
    print(df['Archetype'].value_counts())
    
    # 5. Run Validation Plot (t-SNE)
    run_tsne_validation(df)
    
    print("\nNext Step: Run 'train_hunter.py' using the csv file generated above.")

if __name__ == "__main__":
    main()
