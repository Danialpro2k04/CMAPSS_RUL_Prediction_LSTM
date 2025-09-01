# CMAPSS_RUL_Prediction

**Predict Remaining Useful Life (RUL) of aircraft engines using stacked LSTM networks on the NASA C-MAPSS benchmark.**  
This repository contains a single, well-documented Jupyter notebook that implements data preprocessing, sliding-window sequence creation, a robust LSTM pipeline, model saving, and evaluation with clear visual diagnostics.

---

## Project Snapshot

**What:** Time-series regression for RUL prediction using stacked LSTM  
**Dataset:** NASA C-MAPSS (FD001 → FD004)  
**Notebook:** `notebooks/CMAPSS_RUL_Prediction.ipynb`  
**Saved Models:** `models/FD00X_regressor_model.keras`

---

## Key Results

| Dataset | MAE   | MSE    | RMSE | R²     |
|:-------:|:-----:|:------:|:----:|:------:|
| FD001   | 10.69 | 234.23 | 15   | 0.8541 |
| FD002   | 16.39 | 476.20 | 21   | 0.7396 |
| FD003   | 9.93  | 201.02 | 14   | 0.8690 |
| FD004   | 16.07 | 461.97 | 21   | 0.7472 |

> These results are competitive for a straightforward LSTM pipeline on C-MAPSS. FD001 & FD003 show particularly low errors; FD002 & FD004 (more complex scenarios) are reasonable and good targets for further tuning.

---

✨ **Repository Structure:**  
- `data/` → contains the CMAPSS datasets:  
  - `RUL_FD001.txt`
  - `RUL_FD002.txt`
  - `RUL_FD003.txt`
  - `RUL_FD004.txt`
  - `test_FD001.txt`
  - `test_FD002.txt`
  - `test_FD003.txt`
  - `test_FD004.txt`   
  - `train_FD001.txt`
  - `train_FD002.txt`
  - `train_FD003.txt`
  - `train_FD004.txt` 

- `models/` → contains trained models:  
  - `FD001_regressor_model.keras`
  - `FD002_regressor_model.keras`
  - `FD003_regressor_model.keras`
  - `FD004_regressor_model.keras`  

- `CMAPSS_RUL_Prediction.ipynb` → main notebook for preprocessing, training, and evaluation.  


---

## Highlights & Strong Points

- **Industry-relevant dataset** — NASA C-MAPSS is a standard benchmark for turbofan RUL tasks.  
- **End-to-end pipeline** — data loading → preprocessing → sliding windows → group-aware train/val split → LSTM model → evaluation → saved models.  
- **Thoughtful preprocessing**
  - Drops uninformative sensors (`sensor 22-26`).
  - Scales operational settings with `StandardScaler` (handles negative values).  
  - Scales sensor measurements with `MinMaxScaler` (keeps data in 0–1 range).  
  - Clips RUL values to reduce noise (`upper=125`).
- **Sliding-window sequences** capture temporal patterns for better learning of engine degradation.  
- **Stacked LSTM architecture**
  - L2 regularization in LSTM layers.
  - Batch normalization and dropout for stable training and generalization.
  - Dense layers for fine-grained regression output.
- **Training optimizations**
  - Early stopping with `restore_best_weights=True`.
  - Learning rate scheduler for dynamic optimization.
  - Group-aware train/validation split to avoid data leakage.
- **Evaluation & Visualization**
  - True vs predicted RUL scatter plots.
  - Residual distribution histograms.
  - Error vs actual RUL scatter plots.
  - Unit-wise RUL predictions over time.

---

## Why This Notebook Matters

1. Provides **accurate RUL predictions** with low MAE/RMSE values on multiple C-MAPSS sub-datasets.  
2. Demonstrates **strong logic and engineering choices**, making it reusable for real-world predictive maintenance projects.  
3. Serves as a **learning resource** for understanding preprocessing, sequence modeling, and LSTM training strategies in time-series regression.  

---

