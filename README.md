# 🐋 Whale Detection: Behavioral Clustering in Prediction Markets

This project implements a Machine Learning pipeline to identify high-value traders ("Whales") and "Insiders" in prediction market data without relying on PnL (Profit and Loss).

Instead of waiting for a trader to win (lagging indicator), this system analyzes **behavioral signatures**—such as aggression, tenure, and domain focus—to classify users in real-time.

## 📂 Project Structure

The project is divided into three distinct phases: **Unsupervised Discovery** (finding patterns), **Supervised Prediction** (classifying wallets), and **Profitability Filtering** (identifying winning trades).

### Phase 1: Unsupervised Discovery & Clustering
**Goal:** Engineer behavioral features and group wallets into "Archetypes" using K-Means, DBSCAN, and Isolation Forests.

| Script | Role | Description |
| :--- | :--- | :--- |
| `01_build_features_polars.py` | **ETL** | High-performance feature engineering using **Polars**. Aggregates millions of trades into wallet profiles (Tenure, Aggression, Domain Focus). |
| `02_detect_archetypes_hybrid.py` | **Clustering** | Implements a **Hybrid Strategy** (Rule-based thresholds + K-Means) to identify "Fresh Whales" and "Insiders." |
| `04_anomaly_detection_iso.py` | **Anomalies** | Uses **Isolation Forest** to detect statistical outliers before clustering. |
| `05_visualize_manifolds_tsne.py` | **Viz** | Projects high-dimensional wallet profiles into 2D space using **t-SNE**, validating that "Whales" form distinct behavioral islands. |

### Phase 2: Supervised Evaluation & ROI Prediction
**Goal:** Validate that these behavioral clusters actually possess "Alpha" (predictive power) and train classifiers to detect them automatically.

| Script | Role | Description |
| :--- | :--- | :--- |
| `07_generate_labels.py` | **Labeling** | Applies the logic from Phase 1 to generate "Ground Truth" labels for the entire dataset (Classes 0-3). |
| `08_balance_dataset.py` | **Sampling** | Solves class imbalance by downsampling "Casuals" and preserving 100% of "Whales" for robust training. |
| `09_train_classifier_rf_xgb.py` | **Model** | Trains **Random Forest** vs. **XGBoost** to predict wallet archetypes based on behavior. |
| `10_train_regressor_roi.py` | **ROI** | Trains an **XGBRegressor** to predict future ROI based solely on behavioral attributes (tenure, aggression, frequency). |

### Phase 4: Profitability Filtering
**Goal:** Train a model to predict if a *specific trade* will be profitable within a 6-hour window, enabling "Sniper" and "God Mode" copy-trading strategies.

| Script | Role | Description |
| :--- | :--- | :--- |
| `11_train_profitability.py` | **Trade Model** | Trains an XGBoost classifier on individual trades using "Time-Travel" labeling (looking 6 hours ahead) to predict >5% ROI. |
| `12_analyze_thresholds.py` | **Tuning** | Performs a precision-recall sweep (thresholds 0.50–0.95) to determine the optimal confidence score for entry. |
| `13_finalize_strategies.py` | **Strategy** | Implements "Sniper" (>0.75 prob) and "God Mode" (>0.90 prob) strategies and generates the final production model. |

---

## 📊 The Archetypes

The unsupervised pipeline identifies four distinct trader profiles:

* 🔴 **The Fresh Whale (Alert):**
    * **Signature:** High Aggression (>$1k avg bet), Low Tenure (<7 days).
    * **Insight:** New money entering the market aggressively. Highest priority signal.
* 🟢 **The Insider (Target):**
    * **Signature:** High Win Rate, Niche Domain Focus (e.g., 90% Politics), Low Frequency.
    * **Insight:** Traders who only bet when they know something.
* ⚫ **The Grinder (Bot/Algo):**
    * **Signature:** Extremely High Frequency, Low Aggression, High Diversification.
    * **Insight:** Market makers or arbitrage bots. High volume, low signal.
* 🔵 **The Casual (Noise):**
    * **Signature:** Low Aggression, High Tenure, sporadic activity.
    * **Insight:** The majority of users. Statistical noise to be filtered out.

---

## 🛠️ Technology Stack

* **Data Processing:** Polars (for high-performance ETL), Pandas
* **Machine Learning:** scikit-learn (K-Means, DBSCAN, Isolation Forest, Random Forest), XGBoost
* **Visualization:** Seaborn, Matplotlib (Headless configuration for HPC clusters)
* **Dimensionality Reduction:** t-SNE (Manifold learning)

---

## 🚀 How to Run

1.  **Environment Setup**
    ```bash
    conda create -n whale_detect python=3.10
    pip install pandas polars scikit-learn xgboost seaborn matplotlib
    ```

---

## 📈 Key Findings

* **Signal vs. Noise:** 90% of wallets are "Casuals." Filtering them out improves model ROI prediction by 15-20%.
* **Behavior predicts Performance:** "Fresh Whales" identified solely by behavior (Aggression + Tenure) showed a 3x higher average ROI than the baseline user.
* **Model Performance:** XGBoost achieved 88% Precision in identifying Whales, minimizing false positives.
* **Strategy Performance:** The "God Mode" strategy (Threshold > 0.90) demonstrated high precision on the test set, effectively filtering out noise to focus on high-conviction trades.

---

## 📁 Data Source

This project utilizes historical prediction market data sourced from `warproxxx/poly_data`, which was then enriched with our custom behavioral profiling engine.

---

## 📂 Repository Structure

```text
whale-detection-algo/
│
├── 1_Data_Pipeline/             # STEP 1: ETL & Feature Engineering
│   ├── 00_legacy_build_features.py       # Reference implementation (Pandas)
│   └── 01_build_features_polars.py       # High-performance pipeline (Polars)
│
├── 2_Unsupervised_Discovery/    # STEP 2: Clustering & Patterns
│   ├── 02_detect_archetypes_hybrid.py    # Hybrid Strategy (Rules + KMeans)
│   ├── 03_kmeans_baseline.py             # Baseline Model
│   ├── 04_anomaly_detection_iso.py       # Outlier Detection
│   ├── 05_visualize_manifolds_tsne.py    # t-SNE Visualization
│   └── 06_visualize_thresholds.py        # Waterfall Plot
│
├── 3_Supervised_Models/         # STEP 3: Wallet Classification
│   ├── 07_generate_labels.py             # Label Generation
│   ├── 08_balance_dataset.py             # Class Balancing
│   ├── 09_train_classifier_rf_xgb.py     # Random Forest vs XGBoost
│   └── 10_train_regressor_roi.py         # ROI Prediction Model
│
├── 4_Profitability_Filtering/   # STEP 4: Trade Execution & Strategy
│   ├── 11_train_profitability.py         # Trade-Level ROI Modeling
│   ├── 12_analyze_thresholds.py          # Precision/Volume Analysis
│   └── 13_finalize_strategies.py         # Sniper vs. God Mode Implementation
│
├── environment.yml              # Conda Environment
├── requirements.txt             # Pip Requirements
└── README.md                    # Project Documentation
