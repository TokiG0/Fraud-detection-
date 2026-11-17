# Fraud Detection — Kaggle Playground Series s6e11

End-to-end Machine Learning and Deep Learning pipeline for binary fraud detection (`is_fraud`), developed for the **Kaggle Playground Series (Season 6, Episode 11)**.

---

## 📌 Project Overview
- **Target Variable**: `is_fraud` (Binary classification: `0` = Genuine, `1` = Fraudulent)
- **Primary Evaluation Metric**: ROC-AUC Score & PR-AUC
- **Validation Strategy**: 5-Fold Stratified Cross-Validation (`random_state=42`) with fixed out-of-fold alignment across all models.

---

## 🏗️ Architecture & Pipeline Workflow

```
Raw Data (train.csv / test.csv)
   │
   ├── Exploratory Data Analysis (EDA) & Imbalance Checks
   ├── Feature Engineering (Aggregations, Fraud Rates, Interaction Ratios)
   │
   ├── 5-Fold Stratified K-Fold Cross Validation
   │      ├── LightGBM (GBDT with Optuna Tuning)
   │      ├── XGBoost (Extreme Gradient Boosting)
   │      ├── CatBoost (Categorical Gradient Boosting)
   │      └── PyTorch Neural Networks (Embeddings, Residual Connections, OneCycleLR)
   │
   └── Ensembling & Post-Processing
          ├── Constrained Blending (SLSQP / Nelder-Mead Optimization)
          ├── Out-of-Fold (OOF) Stacking
          └── Rank-Averaged Ensemble
```

---

## 🚀 Key Features & Methodologies

1. **Exploratory Data Analysis (EDA)**:
   - Analysis of fraud rate across categorical attributes (`merchant_category`, `card_type`, `auth_method`, `channel`, `device_type`).
   - Feature correlation and baseline gradient boosting feature importance.

2. **Feature Engineering**:
   - Out-of-fold target encoding and group statistics.
   - Transaction ratio features relative to user/merchant history.
   - Frequency encodings and interaction signals.

3. **Models**:
   - **LightGBM**: Tuned via Optuna for leaf-wise tree growth.
   - **XGBoost**: Depth-wise gradient boosted trees with regularizations.
   - **CatBoost**: Native categorical feature handling.
   - **Deep Learning (PyTorch)**: Custom Multi-Layer Perceptrons (MLP / ResMLP / WideMLP) with categorical entity embeddings, batch normalization, dropout, residual skip connections, OneCycleLR scheduler, and Exponential Moving Average (EMA).

4. **Ensembling**:
   - Out-of-Fold (OOF) validation ensuring no data leakage.
   - Optimization of blend weights via constrained minimization on negative ROC-AUC.
   - Rank-average blending to combine tree-based and neural predictions safely under distribution variations.

---

## 📁 Repository Structure

```text
├── fraud.ipynb          # Full pipeline Jupyter notebook
├── requirements.txt     # Python dependencies
├── .gitignore           # Ignored datasets, model checkpoints, and artifacts
└── README.md            # Project documentation
```

*(Note: Large raw data files such as `train.csv`, `test.csv`, and trained `.pt` / `.npy` files are excluded via `.gitignore` to adhere to GitHub file size guidelines).*

---

## 🛠️ Getting Started

### 1. Prerequisites
Ensure Python 3.10+ is installed. Clone the repository and install dependencies:

```bash
git clone https://github.com/TokiG0/Fraud-detection-.git
cd Fraud-detection-
pip install -r requirements.txt
```

### 2. Dataset Setup
Download the dataset from the [Kaggle Playground Series s6e11 Competition](https://www.kaggle.com/competitions/playground-series-s6e11) and place the following files in the project root:
- `train.csv`
- `test.csv`
- `sample_submission.csv`

### 3. Run the Notebook
Open and execute [fraud.ipynb](fraud.ipynb) in Jupyter Notebook, VS Code, or Google Colab:

```bash
jupyter notebook fraud.ipynb
```
