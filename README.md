# Car-Price-Prediction-Model 🚗
This project leverages multiple machine learning models to accurately predict a vehicle's selling price based on various attributes like mileage, brand, model, year, engine size, and fuel type.

The project evaluates **XGBoost**, **Multiple Linear Regression**, **K-Nearest Neighbors (KNN)**, and a **Multilayer Perceptron (MLP)**, comparing performance primarily via **RMSE**.


---

## 1) Project Overview

Models

### Key Objectives
- Prepared and explored a Kaggle-derived car dataset.
- Trained and evaluated Linear Regression, KNN, and MLP models.
- Compared models using RMSE and picked the most reliable approach.
- Translated results into practical, business-style recommendations.

---

## 2) Data### Data Splitting for Modeling
For modeling purposes, the dataset was separated into:
- **Full dataset** (`cardata.csv`) — used for exploratory data analysis and initial model trials.
- **Training set** (`cardata_Train.csv`) — used to fit models (e.g., KNN, MLP) in WEKA.
- **Test set** (`cardata_Test.csv`) — used to evaluate the final performance of models after training.

The split ensured that performance metrics like RMSE were calculated on unseen data, providing a realistic estimate of model generalization ability.


- **Source:** Kaggle (dataset credited to *P. Malekian, 2023*).  
- **Files used (place in `data/`):**
  - `cardata.csv` (original)
  - `cardata_Train.csv` (training subset)
  - `cardata_Test.csv` (test subset)
- **Size:** 301 rows with numerical and categorical fields.

### Features (attributes)
- `Car Name`
- `Year`
- `Owner`
- `Present Price` *(in thousands of USD)*
- `Kms Driven`
- `Fuel Type` *(Petrol / Diesel / CNG)*
- `Seller Type` *(Dealer / Individual)*
- `Transmission` *(Manual / Automatic)*
- `Selling Price` *(target, in thousands of USD)*

> In WEKA-based preprocessing, `Car Name` and `Owner` were dropped as low-signal predictors for price.

---

## 3) Methods

Analyses were performed primarily in **WEKA**:

1. Import dataset into WEKA Explorer.
2. Remove `Car Name` and `Owner` from features.
3. Normalize attributes (WEKA filters) for algorithms that benefit from scale.
4. Train/test as follows:
   - **Linear Regression:** 10-fold cross-validation on the full dataset.
   - **KNN:** Multiple K values evaluated; final K chosen by lowest RMSE. Training on first 251 rows; testing on last 50 rows.
   - **MLP:** Default/selected hyperparameters in WEKA; evaluated with RMSE.

### Linear Regression Model (LR)
- **Correlation coefficient:** `0.9202`
- **RMSE:** `2.0142`
- **Learned equation (using normalized/coded variables):**

\[
\hat{Y} = 6.0475\,x + 40.2159\,x_1 - 3.5097\,x_2 + 1.8827\,x_3 + 1.1635\,x_4 + 1.4344\,x_5 - 3.8721
\]

Where:
- \(Y\): Selling Price  
- \(x\): Year  
- \(x_1\): Present Price  
- \(x_2\): Kms Driven  
- \(x_3\): Fuel Type (encoded)  
- \(x_4\): Seller Type (encoded)  
- \(x_5\): Transmission (encoded)

> For categorical variables, only the relevant dummy coefficient applies (others are zero).

### K-Nearest Neighbors (KNN)
**K sweep (RMSE):**

| K | RMSE  |
|---|------:|
| 3 | 2.4100 |
| 5 | 2.4827 |
| 7 | 2.6395 |
| 10 | 2.7291 |
| 11 | 2.7308 |
| 12 | 2.7127 |
| 15 | 2.7741 |
| 17 | 2.8829 |
| 20 | 3.0209 |

- **Chosen K:** `3` (lowest RMSE in sweep)
- **Supplied test set (last 50 rows):** `RMSE = 0.6051`

### Multilayer Perceptron (MLP)
- **RMSE:** `1.4805`

---

## 4) Results & Conclusion

- **Best overall model:** **KNN (K=3)**, achieving **RMSE = 0.6051** on the held-out test set of 50 rows.
- Linear Regression achieved **RMSE = 2.0142** with a strong positive correlation (0.9202), but higher error than KNN.
- MLP achieved **RMSE = 1.4805**, better than LR but not as strong as KNN in this setup.

**Conclusion:** KNN provided the most accurate predictions for this dataset and split, likely due to local similarity patterns that LR and MLP didn’t capture as effectively under the given preprocessing and hyperparameters.

---

## 5) Practical Recommendations

- **Mileage matters:** The negative coefficient for `Kms Driven` in LR indicates price generally **decreases** as mileage increases.
- **Newer year, higher resale:** Newer `Year` tends to **increase** predicted price.
- **Present price correlates:** Higher `Present Price` tends to align with a higher `Selling Price` (unsurprising but confirmed).
- Use KNN for near-term predictions on similar data; consider feature scaling and careful train/test partitioning.

---

## 6) Reproducing the Study (WEKA)

1. Open WEKA → *Explorer* → *Preprocess*.
2. Load `data/cardata.csv`.
3. Remove attributes: `Car Name`, `Owner`.
4. Apply normalization filter to numeric attributes.
5. **Linear Regression:** *Classify* → choose *LinearRegression* → *Use training set* or *Cross-validation (10 folds)* → *Start*.
6. **KNN:** *IBk* classifier → try K = 3, 5, 7, 10, ... → *Supplied test set* with `cardata_Test.csv` (train on `cardata_Train.csv`).
7. **MLP:** *MultilayerPerceptron* → run with chosen params → record RMSE.

> Can also optionally mirror the analysis in Python (e.g., scikit-learn) using the same splits and encodings.

---

## 7) Citation

- Malekian, P. (2023). *Multiple linear regression on cars data*. Kaggle.

---

## 8) Notes

- Can also share the Python /R workflows if preferred over WEKA.
