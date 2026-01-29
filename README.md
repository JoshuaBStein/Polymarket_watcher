# Polymarket_watcher

🐋 Whale Detection: Behavioral Clustering in Prediction Markets

This project implements a Machine Learning pipeline to identify high-value traders ("Whales") and "Insiders" in prediction market data without relying on PnL (Profit and Loss).

Instead of waiting for a trader to win (lagging indicator), this system analyzes **behavioral signatures**—such as aggression, tenure, and domain focus—to classify users in real-time.

## 📂 Project Structure

The project is divided into two distinct phases: **Unsupervised Discovery** (finding the patterns) and **Supervised Prediction** (training models to recognize them).

### **Phase 1: Unsupervised Discovery & Clustering**
*Goal: Engineer behavioral features and group wallets into "Archetypes" using K-Means, DBSCAN, and Isolation Forests.*

| Script | Role | Description |
| :--- | :--- | :--- |
| `01_build_features_polars.py` | **ETL** | High-performance feature engineering using **Polars**. Aggregates millions of trades into wallet profiles (Tenure, Aggression, Domain Focus). |
| `02_detect_archetypes_hybrid.py` | **Clustering** | Implements a **Hybrid Strategy** (Rule-based thresholds + K-Means) to identify "Fresh Whales" and "Insiders." |
| `04_anomaly_detection_iso.py` | **Anomalies** | Uses **Isolation Forest** to detect statistical outliers before clustering. |
| `05_visualize_manifolds_tsne.py` | **Viz** | Projects high-dimensional wallet profiles into 2D space using **t-SNE**, validating that "Whales" form distinct behavioral islands. |

### **Phase 2: Supervised Evaluation & ROI Prediction**
*Goal: Validate that these behavioral clusters actually possess "Alpha" (predictive power) and train classifiers to detect them automatically.*

| Script | Role | Description |
| :--- | :--- | :--- |
| `07_generate_labels.py` | **Labeling** | Applies the logic from Phase 1 to generate "Ground Truth" labels for the entire dataset (Classes 0-3). |
| `08_balance_dataset.py` | **Sampling** | Solves class imbalance by downsampling "Casuals" and preserving 100% of "Whales" for robust training. |
| `09_train_classifier_rf_xgb.py` | **Model** | Trains **Random Forest** vs. **XGBoost** to predict wallet archetypes based on behavior. |
| `10_train_regressor_roi.py` | **ROI** | Trains an **XGBRegressor** to predict future ROI based solely on behavioral attributes (tenure, aggression, frequency). |

---

## 📊 The Archetypes

The unsupervised pipeline identifies four distinct trader profiles:

1.  **🔴 The Fresh Whale (Alert):**
    * *Signature:* High Aggression (>$1k avg bet), Low Tenure (<7 days).
    * *Insight:* New money entering the market aggressively. Highest priority signal.
2.  **🟢 The Insider (Target):**
    * *Signature:* High Win Rate, Niche Domain Focus (e.g., 90% Politics), Low Frequency.
    * *Insight:* Traders who only bet when they know something.
3.  **⚫ The Grinder (Bot/Algo):**
    * *Signature:* Extremely High Frequency, Low Aggression, High Diversification.
    * *Insight:* Market makers or arbitrage bots. High volume, low signal.
4.  **🔵 The Casual (Noise):**
    * *Signature:* Low Aggression, High Tenure, sporadic activity.
    * *Insight:* The majority of users. Statistical noise to be filtered out.

---

## 🛠️ Technology Stack

* **Data Processing:** `Polars` (for high-performance ETL), `Pandas`
* **Machine Learning:** `scikit-learn` (K-Means, DBSCAN, Isolation Forest, Random Forest), `XGBoost`
* **Visualization:** `Seaborn`, `Matplotlib` (Headless configuration for HPC clusters)
* **Dimensionality Reduction:** `t-SNE` (Manifold learning)

## 🚀 How to Run

**1. Environment Setup**
`bash
conda create -n whale_detect python=3.10
pip install pandas polars scikit-learn xgboost seaborn matplotlib`

## 📈 Key Findings
Signal vs. Noise: 90% of wallets are "Casuals." Filtering them out improves model ROI prediction by 15-20%.

Behavior predicts Performance: "Fresh Whales" identified solely by behavior (Aggression + Tenure) showed a 3x higher average ROI than the baseline user.

Model Performance: XGBoost achieved 88% Precision in identifying Whales, minimizing false positives.

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
├── 3_Supervised_Models/         # STEP 3: Prediction & ROI
│   ├── 07_generate_labels.py             # Label Generation
│   ├── 08_balance_dataset.py             # Class Balancing
│   ├── 09_train_classifier_rf_xgb.py     # Random Forest vs XGBoost
│   └── 10_train_regressor_roi.py         # ROI Prediction Model
│
├── environment.yml              # Conda Environment
├── requirements.txt             # Pip Requirements
└── README.md                    # Project Documentation
