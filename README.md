# Titanic Survival Prediction System

## Overview
This project develops a machine learning application to predict whether a Titanic passenger survived based on passenger information. The system follows a complete machine learning workflow, including data cleaning, exploratory data analysis (EDA), feature engineering, model selection, evaluation, and interactive prediction through a Streamlit web interface.

The objective is to build a reliable prediction model using a real-world dataset and to provide a user-friendly interface for making survival predictions based on relevant passenger attributes.

## Problem Statement
The Titanic disaster is a classic classification problem in machine learning. The goal is to predict the survival outcome of passengers using features such as sex, age, passenger class, ticket fare, family size, and embarkation port. This project applies supervised learning techniques to build and evaluate a predictive model.

## Dataset
The project uses the public Titanic dataset, which contains passenger records with demographic and travel information. The dataset includes attributes such as:

- PassengerId
- Pclass
- Name
- Sex
- Age
- SibSp
- Parch
- Ticket
- Fare
- Cabin
- Embarked
- Survived

The target variable is:
- Survived: 0 = Did not survive, 1 = Survived

This dataset is stored in `data/titanic.csv` and is used for both training and evaluation.

## Project Structure
```text
Titanic-predictor/
├── app.py                  # Streamlit-based interactive prediction interface
├── train.py                # End-to-end ML training pipeline
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── data/
│   └── titanic.csv         # Titanic dataset
├── model/
│   ├── titanic_model.joblib
│   ├── metrics.json
│   ├── model_comparison.csv
│   └── sample_predictions.csv
├── figures/
│   ├── eda_overview.png
│   ├── evaluation.png
│   ├── feature_importance.png
│   ├── correlation_heatmap.png
│   └── missing_values.png
└── .gitignore
```

## Methodology

### 1. Data Cleaning
The dataset is cleaned to handle missing and inconsistent values. This includes:
- removing duplicate records
- filling missing embarkation values using the mode
- deriving a cabin indicator (`HasCabin`)
- clipping extreme fare values to reduce skew
- preserving missing age values for model-based imputation

### 2. Exploratory Data Analysis (EDA)
The project performs EDA to understand the relationships between features and the target variable. Visualizations are generated to analyze:
- survival distribution
- survival rate by sex, class, family size, and title
- age distribution by outcome
- correlation among variables
- missing value patterns

### 3. Feature Engineering
Relevant features are engineered to improve predictive performance, including:
- `Title` extracted from the passenger name
- `FamilySize`
- `IsAlone`
- `FarePerPerson`
- `HasCabin`

Irrelevant or redundant columns such as unique identifiers and raw cabin names are excluded from the final feature set.

### 4. Model Training and Comparison
Multiple classification models are trained and compared, including:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machine (SVM)
- Random Forest
- Gradient Boosting

The models are evaluated using stratified cross-validation and compared based on their performance.

### 5. Hyperparameter Tuning
Hyperparameter tuning is performed using `GridSearchCV` for the strongest candidate models to improve validation performance and select the best configuration.

### 6. Model Evaluation
The final model is evaluated on a held-out test set using metrics such as:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Confusion matrices and ROC curves are also generated to assess classification quality.

## Requirements Coverage
This project fulfills the key requirements for a machine learning prediction system:

- Real-world dataset used: Titanic public data set
- Data cleaning and preprocessing performed
- Exploratory data analysis completed
- Relevant features selected and engineered
- At least one machine learning model trained
- Multiple models compared
- Model evaluated using appropriate metrics
- Prediction results displayed
- Interactive prediction interface implemented using Streamlit
- Hyperparameter tuning included as a bonus feature

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

## Installation
To run the project locally, first install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

### Train the model
```bash
python train.py
```

This script:
- loads the dataset
- performs preprocessing and feature engineering
- analyzes the data
- trains and compares models
- tunes hyperparameters
- evaluates results
- saves the best model and generated artifacts

### Launch the Streamlit app
```bash
streamlit run app.py
```

The interactive interface allows users to input passenger characteristics and obtain a survival probability prediction.

## Model Outputs
The project stores generated artifacts in the `model/` and `figures/` directories, including:
- trained model (`titanic_model.joblib`)
- metrics (`metrics.json`)
- comparison tables
- sample predictions
- EDA and evaluation plots

## Results
The best-performing model is selected using cross-validation performance and then evaluated on the held-out test set. The final results include both classification metrics and visual performance summaries, allowing users to compare models and understand which features contribute most to survival prediction.

## Conclusion
This project demonstrates an end-to-end machine learning workflow for a real-world classification problem. It combines data preparation, feature engineering, model comparison, evaluation, and deployment into a practical and interactive prediction system.

## License
This project is intended for educational and demonstration purposes.
