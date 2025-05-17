# Housing Price Prediction with SOM and ANN

**Author:** Jonathan Hansen  
**Date:** March 1, 2025

---

## Project Overview
This repository demonstrates how to enhance a standard regression model—an Artificial Neural Network (ANN)—by first uncovering hidden structure in the data using a Self-Organizing Map (SOM). We work with the classic Boston Housing dataset to predict the median value of owner-occupied homes (`MEDV`).  
**Workflow:**
1. **Data Processing:** load, clean, split, and standardize features  
2. **SOM Implementation:** train a 2D grid of neurons to map neighborhoods  
3. **Clustering:** apply K-Means to SOM codebook vectors, assign each sample a cluster label  
4. **Feature Augmentation:** one-hot encode cluster labels and append to original features  
5. **ANN Modeling:** train a feed-forward neural network on the augmented feature set  
6. **Evaluation:** measure Test Loss (MSE), MAE, and R²; visualize training history and predictions  

---

## Dataset
- **Source:** Boston Housing dataset (506 samples, 14 attributes)  
- **Target:** `MEDV` — median value of owner-occupied homes in \$1000s  
- **Features:**
  - `CRIM`, `ZN`, `INDUS`, `CHAS`, `NOX`, `RM`, `AGE`, `DIS`, `RAD`, `TAX`, `PTRATIO`, `B`, `LSTAT`

---

## Tools & Libraries
- **Data wrangling & visualization:** `pandas`, `numpy`, `seaborn`, `matplotlib`  
- **SOM:** `minisom`  
- **Clustering & preprocessing:** `scikit-learn` (`KMeans`, `StandardScaler`, `OneHotEncoder`)  
- **ANN:** `tensorflow.keras` (`Sequential`, `Dense`)  

## Implementation Details
1. **Data Processing**
Checked for nulls; none found

Explored feature distributions via box plots—kept outliers (they improved performance)

Split into train/test (80/20) and applied StandardScaler

2. **Self-Organizing Map (SOM)**
Grid size: 7×7 (also tried 10×10, 15×15)

Trained for 1,000 iterations on scaled training data

Extracted 49 codebook vectors, flattened to shape (49, 13)

3. **Clustering**
Applied K-Means (n_clusters=4) on SOM codebook vectors

Mapped each sample to its Best Matching Unit (BMU) on the SOM → cluster label

One-hot encoded cluster labels and concatenated with original features

4. **Artificial Neural Network (ANN)**
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_final.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='linear'),
])
model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['mae'])
history = model.fit(
    X_train_final, y_train,
    epochs=100, batch_size=32,
    validation_split=0.2, verbose=0
)

- Tried optimizers: Adam, AdamW, SGD, Adafactor — Adam performed best.

### **Results**
| Configuration                         | Test MSE | Test MAE | Test R²  |
|---------------------------------------|---------:|---------:|---------:|
| Standard Scaling + Adam + 7×7 SOM     |    12.43 |     2.37 |   0.8305 |
| AdamW (same setup)                    |    12.78 |     2.39 |   0.8257 |
| No scaling (Adam)                     |    24.40 |     3.81 |   0.6673 |
| Removing outliers (Adam)              |    28.47 |     3.80 |   0.4204 |
| 10×10 SOM + Adam                      |    12.97 |     2.34 |   0.8232 |
| 15×15 SOM + Adam                      |    12.03 |     2.41 |   0.8359 |


Best performance: Standard scaling, Adam optimizer, 7×7 SOM (R² ≈ 0.83)

## Visualizations
- SOM U-Matrix showing neighborhood groupings

- Training history plots (MSE & MAE over epochs)

- True vs. Predicted scatter plot with 45° reference line

*See the /notebooks folder for all figures.*

## Conclusions
SOM + ANN: augmenting ANN inputs with SOM-derived cluster labels yields strong performance (R² > 0.83).

Outliers matter: retaining them helps capture real variability in housing prices.

Grid size: 7×7 strikes a good balance; larger grids give marginal gains.

## Usage
1. Clone this repo

2. Place housing.csv in the project root

3. Install dependencies (see Tools & Libraries)

4. Run the pipeline:

python som_ann_pipeline.py
or open notebooks/som_ann.ipynb for an interactive walkthrough.