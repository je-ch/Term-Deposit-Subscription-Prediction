# Term Deposit Subscription Prediction

A binary classification project predicting whether a bank customer will subscribe to a term deposit, based on demographic and campaign data from a Portuguese banking institution.

---

## Problem Statement

Banks run marketing campaigns to promote term deposit subscriptions, but reaching out to every customer is costly and inefficient. By predicting which customers are likely to subscribe, banks can target their campaigns more effectively — reducing costs while improving conversion rates.

---

## Dataset

- **Source:** [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) (`bank.csv`)
- **Size:** 45,211 records
- **Features:** 16 (demographic, financial, and campaign-related)
- **Target:** `y`, whether the client subscribed to a term deposit (yes/no)
- **Class distribution:** Imbalanced — majority "no" subscriptions

---

## Approach

1. **Exploratory Data Analysis (EDA)**
   - Target variable distribution
   - Subscription rate by education level
   - Balance distribution by subscription status
   - Correlation heatmap of numerical features
   - Most frequent job category analysis

2. **Preprocessing**
   - One-hot encoding for categorical features (`job`, `marital`, `education`, etc.) using `pd.get_dummies`
   - Standard scaling for numerical features using `StandardScaler`
   - Stratified 80/20 train-test split to preserve class ratio

3. **Models Trained**
   - Logistic Regression (`solver=liblinear`, `max_iter=1000`)
   - Random Forest Classifier (`n_estimators=100`)

4. **Threshold Tuning**
   - Tested prediction thresholds at 0.3, 0.4, and 0.5 on Random Forest probabilities to optimize precision-recall trade-off

5. **Evaluation**
   - Classification Report (Precision, Recall, F1-score)
   - Confusion Matrix
   - ROC-AUC Score
   - ROC Curve (Logistic Regression)
   - Side-by-side model performance comparison chart

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 89% | 0.55 | 0.28 | 0.37 |
| Random Forest | 89% | 0.53 | 0.25 | 0.34 |

> Both models achieve similar overall accuracy. Logistic Regression edges ahead on precision and F1, making it the slightly more reliable choice for this task. The low recall reflects the class imbalance, the majority "no" class dominates predictions.

---

## Tools & Libraries

- Python
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn

---

## How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/your-username/term-deposit-prediction.git
   cd term-deposit-prediction
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset
   - Get `bank.csv` from the [UCI Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) or Kaggle
   - Place it in the root directory

4. Run the notebook
   ```bash
   jupyter notebook Term_Deposit_Subscription_Prediction.ipynb
   ```

---

## Project Structure

```
term-deposit-prediction/
│
├── Term_Deposit_Subscription_Prediction.ipynb
├── requirements.txt
└── README.md
```

---

## Key Takeaway

Both models hit 89% accuracy, but that number is misleading on an imbalanced dataset — a model predicting "no" for every customer would score similarly. The more meaningful metrics here are precision and recall on the minority "yes" class, where threshold tuning on predicted probabilities offers a practical lever to shift the precision-recall balance depending on campaign strategy.
