import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless mode for clusters
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
DATA_FILE = "/scratch/jbs263/Fintechstuff2/wallet_profiles.csv"
OUTPUT_FILE = "final_archetypes_tsne.csv"
PLOT_FILE = "rna_style_blobs.png"

# --- LOGIC THRESHOLDS ---
FRESH_WHALE_MAX_TENURE = 7
FRESH_WHALE_MIN_BET = 1000
GRINDER_PERCENTILE = 0.95

def load_data():
    try:
        print(f"Loading data from {DATA_FILE}...")
        df = pd.read_csv(DATA_FILE)
        # Handle NaN values by filling with 0 (safe for these metrics)
        df.fillna(0, inplace=True)
        return df
    except FileNotFoundError:
        print("Using Dummy Data for simulation...")
        np.random.seed(42)
        n = 2000
        df = pd.DataFrame({
            'tenure_days': np.random.randint(0, 1000, n),
            'aggression_avg_usd': np.random.exponential(500, n),
            'frequency_trades_daily': np.random.poisson(3, n),
            'diversification_markets_count': np.random.poisson(5, n)
        })
        # Inject explicit Fresh Whales
        df.loc[0:50, 'tenure_days'] = np.random.randint(0, 7, 51)
        df.loc[0:50, 'aggression_avg_usd'] = np.random.randint(5000, 25000, 51)
        return df

def classify_archetypes(df):
    """
    Applies the Waterfall Logic to create labels (The 'Genotype').
    """
    df['Archetype'] = 'Casual (Noise)' # Default
    
    # 1. Grinders (High Activity)
    # Note: We identify Grinders BEFORE Whales? No, usually Whales first to capture new rich users.
    # But let's stick to your request: Activity implies Grinder.
    freq_thresh = df['frequency_trades_daily'].quantile(GRINDER_PERCENTILE)
    div_thresh = df['diversification_markets_count'].quantile(GRINDER_PERCENTILE)
    
    mask_grinder = (df['frequency_trades_daily'] > freq_thresh) | \
                   (df['diversification_markets_count'] > div_thresh)
    df.loc[mask_grinder, 'Archetype'] = 'The Grinder (Ignore)'
    
    # 2. Fresh Whales (New & Rich) - Overwrites Grinder if they are brand new
    mask_whale = (df['tenure_days'] <= FRESH_WHALE_MAX_TENURE) & \
                 (df['aggression_avg_usd'] >= FRESH_WHALE_MIN_BET)
    df.loc[mask_whale, 'Archetype'] = 'Fresh Whale (Alert)'
    
    # 3. Insiders (Old & Rich)
    # Anyone left who isn't a Grinder or Fresh Whale, but bets big.
    # We define "Rich" as top 10% of remaining bet sizes for this demo
    mask_potential = df['Archetype'] == 'Casual (Noise)'
    if mask_potential.sum() > 0:
        rich_thresh = df.loc[mask_potential, 'aggression_avg_usd'].quantile(0.90)
        mask_insider = mask_potential & (df['aggression_avg_usd'] > rich_thresh)
        df.loc[mask_insider, 'Archetype'] = 'The Insider (Target)'
        
    return df

def main():
    df = load_data()
    
    # --- STEP 1: CLASSIFY (Create Labels) ---
    print("Classifying users...")
    df = classify_archetypes(df)
    print("Counts:\n", df['Archetype'].value_counts())
    
    # --- STEP 2: PREPARE FOR t-SNE (Create Features) ---
    print("Preparing features for t-SNE...")
    
    # Select features that define behavior
    features = ['tenure_days', 'aggression_avg_usd', 'frequency_trades_daily', 'diversification_markets_count']
    
    # IMPORTANT: t-SNE requires scaling or large numbers (USD) dominate small numbers (Freq)
    # We use Log transform on Money and Tenure to compress the massive range
    X = df[features].copy()
    X['aggression_avg_usd'] = np.log1p(X['aggression_avg_usd'])
    X['tenure_days'] = np.log1p(X['tenure_days'])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- STEP 3: RUN t-SNE ---
    # t-SNE is heavy. If > 10k rows, we must sample for the PLOT, but we save full data.
    # We will sample 10,000 points max for the visualization to get nice blobs without crashing.
    
    if len(df) > 10000:
        print(f"Data large ({len(df)}). Sampling 10,000 for t-SNE visualization...")
        # Stratified sample to ensure we keep rare Whales in the plot
        sample_df = df.groupby('Archetype', group_keys=False).apply(lambda x: x.sample(min(len(x), 2500)))
        # Rescale the sample
        X_sample = sample_df[features].copy()
        X_sample['aggression_avg_usd'] = np.log1p(X_sample['aggression_avg_usd'])
        X_sample['tenure_days'] = np.log1p(X_sample['tenure_days'])
        X_final = scaler.fit_transform(X_sample)
    else:
        sample_df = df
        X_final = X_scaled

    print("Running t-SNE (This creates the blobs)...")
    # Perplexity = related to number of nearest neighbors. 30-50 is standard.
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(X_final)
    
    sample_df['tsne_one'] = tsne_results[:,0]
    sample_df['tsne_two'] = tsne_results[:,1]

    # --- STEP 4: PLOT LIKE RNA-SEQ ---
    print("Generating RNA-Seq style plot...")
    plt.figure(figsize=(16, 10))
    
    # Use a dark style for that "scientific/cyber" look
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
        data=sample_df,
        legend="full",
        alpha=0.8,
        s=40 # Dot size
    )

    plt.title('Wallet Behavior Map (t-SNE Projection)', fontsize=20, color='white')
    plt.xlabel('Dimension 1 (Similarity)', color='gray')
    plt.ylabel('Dimension 2 (Similarity)', color='gray')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', facecolor='#222')
    
    # Remove grid for cleaner "blob" look
    plt.grid(False)
    
    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    print(f"Visualization saved to {PLOT_FILE}")
    
    # Save the classified data (Full set)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Full Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
