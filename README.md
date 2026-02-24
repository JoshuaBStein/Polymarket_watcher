# 🐋 Whale Detection: Behavioral Clustering in Prediction Markets

This project implements a Machine Learning pipeline to identify high-value traders ("Whales") and "Insiders" in prediction market data without relying on PnL (Profit and Loss).

Instead of waiting for a trader to win (lagging indicator), this system analyzes **behavioral signatures**—such as aggression, tenure, and domain focus—to classify users in real-time.

## 📂 Project Structure

The project is divided into six distinct phases: **Data Pipeline** (ETL), **Unsupervised Discovery** (finding patterns), **Supervised Prediction** (classifying wallets), **Unsupervised Model Generation**, **Live Inference**, and **Profitability Filtering** (identifying winning trades).

---

### Phase 1: Data Pipeline & Feature Engineering

**Goal:** Extract, transform, and load prediction market data into wallet behavioral profiles.

| Script | Role | Description |
| --- | --- | --- |
| `00_legacy_build_features.py` | **Legacy ETL** | Pandas-based implementation for enriching wallet profiles with market outcomes (ROI, win rate, alpha score). Processes trades in chunks and calculates wallet-level aggregations. |
| `01_build_features_polars.py` | **High-Performance ETL** | Polars-based streaming implementation for building behavioral features from raw trades. Calculates tenure, aggression, frequency, domain focus, and diversification using lazy evaluation and regex pattern matching. |

---

### Phase 2: Unsupervised Discovery & Clustering

**Goal:** Engineer behavioral features and group wallets into "Archetypes" using K-Means, DBSCAN, and Isolation Forests.

| Script | Role | Description |
| --- | --- | --- |
| `02_detect_archetypes_hybrid.py` | **Hybrid Clustering** | Implements a hybrid strategy combining rule-based thresholds (for Fresh Whales) with K-Means clustering (for Insiders/Grinders/Casuals). Uses Isolation Forest for anomaly detection. |
| `03_kmeans_baseline.py` | **Baseline Clustering** | Pure K-Means approach with 3 clusters. Includes dynamic archetype naming logic based on centroid analysis and feature importance. |
| `04_anomaly_detection_iso.py` | **K-Means + Labeling** | Runs K-Means clustering and applies dynamic archetype labeling by analyzing cluster centroids. Identifies Grinders (high frequency), Fresh Whales (low tenure + high aggression), and Insiders. |
| `05_visualize_manifolds_tsne.py` | **t-SNE Visualization** | Projects high-dimensional wallet profiles into 2D space using t-SNE. Creates "RNA-seq style" visualizations to validate that behavioral clusters form distinct islands. Includes waterfall classification logic. |
| `06_visualize_thresholds.py` | **Waterfall Classification** | Applies sequential rule-based logic: identifies Fresh Whales first, then Grinders, then uses K-Means to separate Insiders from Casuals. Generates visualization with log-scale plots. |

---

### Phase 3: Supervised Models & Classification

**Goal:** Validate that behavioral clusters possess "Alpha" (predictive power) and train classifiers to detect them automatically.

| Script | Role | Description |
| --- | --- | --- |
| `07_generate_labels.py` | **Label Generation** | Applies the waterfall classification logic to generate ground truth labels for the entire dataset. Creates integer target classes (0-3) and runs t-SNE validation. |
| `08-1_balance_dataset.py` | **Dataset Balancing** | Solves class imbalance by taking all Fresh Whales and sampling 3,000 from each other class. Creates a balanced training set for supervised learning. |
| `08-2_rebalance_dataset.py` | **Representative Sampling** | Alternative balancing strategy that maintains realistic class proportions (1 Whale : 30 Noise) to prevent model from being "trigger happy". Includes sample weights for calibration. |
| `09-1_train_classifier_rf_xgb.py` | **Model Competition** | Trains both Random Forest and XGBoost on balanced data. Compares models based on overall accuracy and Whale detection precision. Saves the winner. |
| `09-2_train_classifier_rf_sklearn.py` | **Scikit-Learn Scout** | Trains a behavior-only Random Forest (no money variables) for wallet classification. Uses unbalanced classes for high precision. Saves model as joblib format. |
| `10_train_regressor_roi.py` | **ROI Prediction** | Trains XGBoost regressors to predict future ROI and alpha scores based solely on behavioral attributes. Includes smart sampling and feature importance visualization. |

---

### Phase 4: Unsupervised Model Generation

**Goal:** Create production-ready clustering models for real-time anomaly detection.

| Script | Role | Description |
| --- | --- | --- |
| `13_generate_unsupervised_wallet_anomaly_model.py` | **K-Means Production Model** | Trains K-Means clustering model on enriched wallet profiles with 13 features. Calculates anomaly scores based on distance to nearest cluster center. Generates PCA visualizations and saves production models (poly_model.pkl, poly_scaler.pkl). |

---

### Phase 5: Live Inference & Monitoring

**Goal:** Fetch live market data, generate real-time wallet profiles, and detect high-value traders.

| Script | Role | Description |
| --- | --- | --- |
| `14_master_pipeline.py` | **Pipeline Orchestrator** | *(Details needed - appears to coordinate the full workflow)* |
| `15_generate_wallet_profiles.py` | **Real-Time Profiling** | Fetches 24-hour trade data from TheGraph API, indexes active/closed markets, and generates behavioral profiles using Polars. Implements automatic file management with dated outputs. |
| `16_live_inference_with_anomaly_detection.py` | **Live Detection System** | Complete monitoring system that: (1) indexes markets, (2) fetches 24h trades, (3) generates profiles, (4) applies clustering + anomaly detection, (5) sends Discord alerts for high-conviction moves and anomalies. Includes smart filtering and percentile-based thresholds. |

---

### Phase 6: Profitability Filtering & Strategy Optimization

**Goal:** Train a model to predict if a *specific trade* will be profitable within a 6-hour window, enabling "Sniper" and "God Mode" copy-trading strategies.

| Script | Role | Description |
| --- | --- | --- |
| `11_train_profitability.py` | **Trade-Level Model** | Trains XGBoost classifier on individual trades using "time-travel" labeling (looking 6 hours ahead) to predict >5% ROI. Features include bet size, Z-score (relative to wallet average), price, and volume. |
| `12_analyze_thresholds.py` | **Threshold Tuning** | Performs precision-recall sweep across probability thresholds (0.50–0.90) to determine optimal confidence scores for trade entry. Shows trade-off between win rate and volume. |
| `13_finalize_strategies.py` | **Strategy Implementation** | Evaluates and finalizes two production strategies: "Sniper" (threshold >0.75) for high profitability, and "God Mode" (threshold >0.90) for near-certainty trades. Generates final production model. |

---

## 📊 The Archetypes

The unsupervised pipeline identifies four distinct trader profiles:

* 🔴 **The Fresh Whale (Alert):**
  + **Signature:** High Aggression (>$1k avg bet), Low Tenure (<7 days).
  + **Insight:** New money entering the market aggressively. Highest priority signal.
* 🟢 **The Insider (Target):**
  + **Signature:** High Win Rate, Niche Domain Focus (e.g., 90% Politics), Low Frequency.
  + **Insight:** Traders who only bet when they know something.
* ⚫ **The Grinder (Bot/Algo):**
  + **Signature:** Extremely High Frequency, Low Aggression, High Diversification.
  + **Insight:** Market makers or arbitrage bots. High volume, low signal.
* 🔵 **The Casual (Noise):**
  + **Signature:** Low Aggression, High Tenure, sporadic activity.
  + **Insight:** The majority of users. Statistical noise to be filtered out.

---

## 🛠️ Technology Stack

* **Data Processing:** Polars (high-performance ETL), Pandas (legacy support)
* **Machine Learning:** scikit-learn (K-Means, DBSCAN, Isolation Forest, Random Forest), XGBoost
* **Visualization:** Seaborn, Matplotlib (Headless configuration for HPC clusters)
* **Dimensionality Reduction:** t-SNE (Manifold learning), PCA
* **APIs:** TheGraph (blockchain data), Polymarket Gamma API (market data)
* **Deployment:** SLURM (HPC job scheduling), Discord Webhooks (alerting)

---

## 🚀 How to Run

### Environment Setup

```bash
conda create -n whale_detect python=3.10
conda activate whale_detect
pip install -r requirements.txt
```

Or use the conda environment file:

```bash
conda env create -f environment.yml
conda activate whale_detect
```

### Running the Pipeline

**Phase 1: Generate Features**
```bash
# High-performance approach (recommended)
python 1_Data_Pipeline/01_build_features_polars.py

# Or legacy approach
python 1_Data_Pipeline/00_legacy_build_features.py
```

**Phase 2: Discover Archetypes**
```bash
# Hybrid strategy (recommended)
python 2_Unsupervised_Discovery/02_detect_archetypes_hybrid.py

# Or baseline K-Means
python 2_Unsupervised_Discovery/03_kmeans_baseline.py

# Visualize with t-SNE
python 2_Unsupervised_Discovery/05_visualize_manifolds_tsne.py
```

**Phase 3: Train Supervised Models**
```bash
# Generate labels
python 3_Supervised_Models/07_generate_labels.py

# Balance dataset
python 3_Supervised_Models/08-1_balance_dataset.py

# Train classifier
python 3_Supervised_Models/09-1_train_classifier_rf_xgb.py

# Train ROI predictor
python 3_Supervised_Models/10_train_regressor_roi.py
```

**Phase 4: Generate Production Model**
```bash
python 4_Unsupervised_model/13_generate_unsupervised_wallet_anomaly_model.py
```

**Phase 5: Live Monitoring**
```bash
# Generate real-time profiles
python 5_Live_Inference/15_generate_wallet_profiles.py

# Run live detection
python 5_Live_Inference/16_live_inference_with_anomaly_detection.py
```

**Phase 6: Profitability Filtering**
```bash
# Train trade-level model
python 6_Legacy_Profitability_Filtering/11_train_profitability.py

# Analyze thresholds
python 6_Legacy_Profitability_Filtering/12_analyze_thresholds.py

# Finalize strategies
python 6_Legacy_Profitability_Filtering/13_finalize_strategies.py
```

### Running on HPC Cluster (Amarel)

For compute-intensive tasks, use the provided SLURM scripts:

```bash
# Submit feature generation job
sbatch 1_Data_Pipeline/01_build_features_polars.slurm

# Submit clustering job
sbatch 2_Unsupervised_Discovery/02_detect_archetypes_hybrid.slurm

# Submit training job
sbatch 3_Supervised_Models/09_train_classifier_rf_xgb.slurm
```

---

## 📈 Key Findings

* **Signal vs. Noise:** 90% of wallets are "Casuals." Filtering them out improves model ROI prediction by 15-20%.
* **Behavior predicts Performance:** "Fresh Whales" identified solely by behavior (Aggression + Tenure) showed a 3x higher average ROI than the baseline user.
* **Model Performance:** XGBoost achieved 88% Precision in identifying Whales, minimizing false positives.
* **Strategy Performance:** The "God Mode" strategy (Threshold > 0.90) demonstrated high precision on the test set, effectively filtering out noise to focus on high-conviction trades.
* **Real-Time Viability:** Live inference pipeline can process 24 hours of trades and generate alerts within minutes.

---

## 📁 Data Source

This project utilizes historical prediction market data sourced from `warproxxx/poly_data`, enriched with:
- Real-time data from TheGraph API (on-chain trades)
- Market metadata from Polymarket Gamma API
- Custom behavioral profiling engine

---

## 📂 Repository Structure

```
Polymarket_watcher/
│
├── 1_Data_Pipeline/                    # Phase 1: ETL & Feature Engineering
│   ├── 00_legacy_build_features.py            # Legacy Pandas implementation
│   ├── 00_legacy_build_features.slurm         # HPC job script
│   ├── 01_build_features_polars.py            # High-performance Polars pipeline
│   └── 01_build_features_polars.slurm         # HPC job script
│
├── 2_Unsupervised_Discovery/           # Phase 2: Clustering & Pattern Discovery
│   ├── 02_detect_archetypes_hybrid.py         # Hybrid strategy (Rules + KMeans)
│   ├── 02_detect_archetypes_hybrid.slurm
│   ├── 03_kmeans_baseline.py                  # Baseline K-Means
│   ├── 03_kmeans_baseline.slurm
│   ├── 04_anomaly_detection_iso.py            # K-Means with dynamic labeling
│   ├── 04_anomaly_detection_iso.slurm
│   ├── 05_visualize_manifolds_tsne.py         # t-SNE visualization
│   ├── 05_visualize_manifolds_tsne.slurm
│   ├── 06_visualize_thresholds.py             # Waterfall classification
│   └── 06_visualize_thresholds.slurm
│
├── 3_Supervised_Models/                # Phase 3: Wallet Classification
│   ├── 07_generate_labels.py                  # Label generation with validation
│   ├── 07_generate_labels.slurm
│   ├── 08-1_balance_dataset.py                # Equal class balancing
│   ├── 08-2_rebalance_dataset.py              # Representative sampling
│   ├── 08_balance_dataset.slurm
│   ├── 09-1_train_classifier_rf_xgb.py        # RF vs XGBoost competition
│   ├── 09-2_train_classifier_rf_sklearn.py    # Behavior-only RF
│   ├── 09_train_classifier_rf_xgb.slurm
│   └── 10_train_regressor_roi.py              # ROI prediction model
│
├── 4_Unsupervised_model/               # Phase 4: Production Model Generation
│   └── 13_generate_unsupervised_wallet_anomaly_model.py  # K-Means production model
│
├── 5_Live_Inference/                   # Phase 5: Real-Time Detection
│   ├── 14_master_pipeline.py                  # Pipeline orchestrator
│   ├── 15_generate_wallet_profiles.py         # Real-time profiling
│   └── 16_live_inference_with_anomaly_detection.py  # Live monitoring + alerts
│
├── 6_Legacy_Profitability_Filtering/   # Phase 6: Trade Strategy Optimization
│   ├── 11_train_profitability.py              # Trade-level ROI modeling
│   ├── 11_train_profitability.slurm
│   ├── 12_analyze_thresholds.py               # Precision/Volume analysis
│   ├── 12_analyze_thresholds.slurm
│   ├── 13_finalize_strategies.py              # Sniper vs God Mode implementation
│   └── 13_finalize_strategies.slurm
│
├── helpers/                            # Utility functions and shared code
├── models/                             # Saved model artifacts
├── .gitignore
├── README.md                           # This file
├── environment.yml                     # Conda environment specification
└── requirements.txt                    # Pip dependencies
```

---

## 🔧 Configuration

Key configuration files and paths are typically defined at the top of each script:

**Data Paths:**
- `/scratch/jbs263/Fintechstuff2/` - Base directory for HPC processing
- `data/10_raw_trades/` - Raw trade data storage
- `data/11_wallet_profiles/` - Generated wallet profiles
- `data/12_classified/` - Classified wallet outputs

**API Keys:**
- TheGraph API: Required for `15_generate_wallet_profiles.py`
- Discord Webhook: Required for `16_live_inference_with_anomaly_detection.py`

**Model Artifacts:**
- `poly_model.pkl` - K-Means clustering model
- `poly_scaler.pkl` - Feature scaler
- `final_whale_hunter_model.pkl` - Supervised classifier
- `whale_filter_xgb.json` - Profitability filter

---

## 📊 Performance Metrics

### Clustering Quality
- **Silhouette Score:** Varies by dataset, typically 0.3-0.5 (moderate separation)
- **Anomaly Detection:** Top 5% flagged as statistical outliers
- **t-SNE Validation:** Clear visual separation of archetypes in 2D projection

### Classification Performance
- **Whale Detection Precision:** 88% (Random Forest/XGBoost)
- **Overall Accuracy:** 85%+
- **False Positive Rate:** <5% for high-threshold strategies

### Profitability Filtering
- **Sniper Strategy (0.75):** ~65-70% win rate, moderate volume
- **God Mode Strategy (0.90):** ~80%+ win rate, low volume
- **Baseline Comparison:** 15-20% improvement over random selection

---

## 🤝 Contributing

This is a research project. Contributions, suggestions, and improvements are welcome via pull requests or issues.

---

## 📄 License

This project is for educational and research purposes. Please ensure compliance with Polymarket's Terms of Service when using real market data.

---

## 🙏 Acknowledgments

- Data sourced from `warproxxx/poly_data`
- Polymarket for providing public API access
- TheGraph for blockchain indexing infrastructure
- Rutgers University (Amarel HPC cluster support)

---

## 📧 Contact

For questions or collaborations, please open an issue on GitHub.
