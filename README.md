# Heart Disease Predictor

**Project 2 — Artificial Intelligence Industrial Training Kit**
DecodeLabs | Batch 2026

---

## Overview

This project is a **supervised machine learning classifier** built as the predictive-modeling milestone of the DecodeLabs AI Engineering track. Where Project 1 focused on deterministic, rule-based logic, this project shifts to **teaching a machine to recognize patterns in real clinical data** and classify new patients based on what it has learned.

The goal of this project is to demonstrate a solid understanding of:
- The supervised learning pipeline: loading, cleaning, and preparing real-world data
- Handling missing values and outliers in medical datasets
- Encoding categorical data and scaling numerical features
- Training and tuning a K-Nearest Neighbors (KNN) classifier
- Evaluating a model beyond raw accuracy (precision, recall, F1-score, confusion matrix)
- Deploying a trained model as an interactive web application

---

## Features

### Core Features (Base Requirement)
- **Real-world dataset** — the UCI Heart Disease dataset, combining patient records from four hospitals (Cleveland, Hungary, Switzerland, and VA Long Beach).
- **Data cleaning pipeline** — missing values are imputed (median for numeric fields, most-frequent for categorical fields), and medically unrealistic outliers (e.g. impossible blood pressure readings) are removed.
- **Categorical encoding** — text fields such as chest pain type and thalassemia status are converted into numeric form using label encoding.
- **Feature scaling** — all inputs are standardized before training, since KNN relies on distance calculations.
- **Correlation-based feature selection** — features are automatically ranked by their correlation with the target, and weak/noisy features are dropped rather than manually guessed.
- **Hyperparameter tuning** — multiple values of K are tested automatically, and the best-performing one is selected for the final model.

### Additional Features
- **Data visualization suite** — includes a class balance chart, an age-vs-disease distribution plot, a full feature correlation heatmap, and a confusion matrix for the final model.
- **Model persistence** — the trained model, scaler, and encoders are saved with `pickle`, so the Streamlit app can make instant predictions without retraining.
- **Interactive Streamlit app** — a clean, two-column interface where a user can input patient details (age, blood pressure, cholesterol, chest pain type, etc.) and receive a real-time risk prediction with a confidence score.

---

## How It Works

1. The dataset is loaded and irrelevant columns (row ID, hospital source) are dropped.
2. The multi-class severity label is simplified into a binary target: disease present or not.
3. Missing values are filled, categorical text is encoded into numbers, and unrealistic outlier rows are removed.
4. Features are ranked by their correlation with the target, and only meaningfully correlated features are kept.
5. The cleaned data is split into training and testing sets, then scaled.
6. A KNN classifier is trained across several values of K, and the best-performing model is kept.
7. The final model is evaluated with a classification report and confusion matrix, then saved for deployment.
8. The Streamlit app loads the saved model and lets a user generate live predictions from manually entered patient data.

---

## Tech Stack

- **Language:** Python 3
- **Data handling:** pandas
- **Visualization:** matplotlib
- **Machine learning:** scikit-learn (KNeighborsClassifier, StandardScaler, LabelEncoder, SimpleImputer)
- **Deployment:** Streamlit
- **Model persistence:** pickle

---

## Running the Project

### 1. Train the model
```bash
python train_heart_model.py
```
This generates the trained model, scaler, encoders, and evaluation graphs.

### 2. Launch the app
```bash
streamlit run app.py
```
Enter patient details in the sidebar fields and click **Predict Risk** to see the result.

---

## Model Performance

| Metric | Score |
|---|---|
| Accuracy | **86.67%** |
| Precision (Disease) | 0.88 |
| Recall (Disease) | 0.83 |
| F1-score (Disease) | 0.86 |

The strongest predictors of heart disease in this dataset were **exercise-induced angina**, **ST depression (oldpeak)**, and **chest pain type** — all consistent with established clinical understanding of cardiac risk factors.

---

## Known Limitations

- **Not a diagnostic tool.** This model identifies statistical patterns in historical data; it does not replace clinical judgment, ECGs, or physician evaluation.
- **Dataset size.** With under 1,000 patient records, the model has learned from a relatively small sample compared to production-grade medical AI systems.
- **Feature scope.** The dataset does not include genetic history, lifestyle factors, or advanced imaging data, which a real diagnosis would typically consider.
- **Distance-based limitations.** KNN's performance can degrade with unseen data that falls outside the distribution of the training set.

---

## Author

**Talha Gillani**
Intern, DecodeLabs — Batch 2026
Artificial Intelligence Industrial Training Kit, Project 2

---

## License

This project was developed for educational purposes as part of the DecodeLabs internship program.
