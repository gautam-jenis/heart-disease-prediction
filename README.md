# Heart Disease Prediction Using Machine Learning

## Project Overview

This project was developed for CSC 486 — Data Mining Methods at the University of South Dakota.

The goal is to predict heart disease status using machine learning and compare different classification algorithms. We focused mainly on recall because missing actual heart disease cases is an important concern.

We also explored dimensionality reduction, hyperparameter tuning, and feature importance to better understand model performance.

## Dataset

We used the [Heart Failure Prediction dataset from Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).

The dataset contains:
- 918 records
- 11 input features
- 1 target variable (`HeartDisease`)
- 508 positive cases and 410 negative cases

During exploration, we found 172 zero cholesterol values and one zero resting blood pressure value. We treated these suspicious zeros as missing values during preprocessing.

## Project Steps

1. Explored the dataset using pandas and visualizations.
2. Cleaned suspicious values and prepared numerical and categorical features.
3. Split the dataset into training and testing sets.
4. Compared eight machine-learning models using five-fold cross-validation.
5. Tested PCA and LDA for dimensionality reduction.
6. Used GridSearchCV to tune SVM and Random Forest.
7. Investigated feature importance using permutation importance.
8. Evaluated the selected SVM model on the held-out test set.
9. Saved the model using Joblib.
10. Built a Streamlit application for educational predictions.

## Machine Learning Models

We compared eight algorithms:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Gaussian Naive Bayes
- Gradient Boosting
- AdaBoost

We evaluated accuracy, precision, recall, F1-score, and ROC-AUC.

## Final Results

We selected SVM with PCA based on its performance during training-set experiments.

| Metric | Test Result |
|---|---|
| Accuracy | 89.1% |
| Precision | 89.4% |
| Recall | 91.2% |
| F1-score | 90.3% |
| ROC-AUC | 0.943 |

The confusion matrix showed:

- 93 true positives
- 71 true negatives
- 11 false positives
- 9 false negatives

These results come from 184 held-out test records.

## Feature Importance

We used permutation importance to examine the trained Random Forest model.

The most influential features in our validation experiment were:

1. ST_Slope
2. ChestPainType
3. ExerciseAngina

These findings describe how the model used the features, not their medical causes or importance.

## Streamlit Application

We developed a simple Streamlit interface where users can enter 11 feature values and generate a class prediction using the saved SVM model.

The application loads `heart_disease_model.joblib`, processes inputs through the saved pipeline, and displays Class 0 or Class 1.

### How to Run

Clone the repository:

```bash
git clone https://github.com/gautam-jenis/heart-disease-prediction.git
cd heart-disease-prediction
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app.py
```

The app should open in your browser.

## Limitations

The dataset includes historical records from multiple sources and may not represent current or broader patient populations.

We identified suspicious zero measurements, and there may be other dataset-related limitations.

Although the model achieved promising results on the test set, we have not performed external or clinical validation.

**This project is for educational and research demonstration purposes only. It should not be used to diagnose heart disease or make medical decisions.**

## Future Improvements

- Improve model explainability and data-quality analysis.
- Evaluate the model on an independent external dataset.
- Improve the Streamlit interface.
- Explore other model settings and calibration methods without tuning to the existing test set.