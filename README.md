# Titanic Survival Prediction System

A portfolio-ready machine learning project that predicts whether a Titanic passenger survived using a public dataset and a full end-to-end ML workflow.

## Project Summary
This project applies supervised learning to the classic Titanic survival prediction problem. The system uses a real-world dataset, performs data cleaning and preprocessing, explores key patterns through EDA, engineers relevant features, compares multiple machine learning models, and evaluates the best-performing model using standard classification metrics.

The final result is an interactive Streamlit application that allows users to enter passenger details and receive a survival probability prediction in real time.

## Why This Project
This project demonstrates practical machine learning skills in a way that is easy to explain in interviews and portfolio reviews. It includes:

- real-world data handling
- preprocessing and feature engineering
- model comparison and evaluation
- hyperparameter tuning
- deployment of an interactive prediction interface
- clear documentation and reproducible workflow

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

## Problem Statement
Predict whether a Titanic passenger survived based on attributes such as age, sex, passenger class, fare, embarked location, family size, and title derived from the passenger name.

## Dataset
The project uses the public Titanic dataset (`data/titanic.csv`), a well-known benchmark dataset used in supervised learning and classification tasks.

## Project Structure
```text
Titanic-predictor/
├── app.py                       # Streamlit interactive prediction interface
├── train.py                     # End-to-end ML training pipeline
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── data/
│   └── titanic.csv              # Titanic dataset
├── model/
│   ├── titanic_model.joblib     # Trained model
│   ├── metrics.json             # Model metrics and metadata
│   ├── model_comparison.csv     # Model comparison results
│   └── sample_predictions.csv   # Sample predictions
├── figures/
│   ├── eda_overview.png         # EDA overview
│   ├── evaluation.png           # Confusion matrix and ROC curves
│   ├── feature_importance.png   # Permutation importance plot
│   ├── correlation_heatmap.png # Correlation plot
│   └── missing_values.png       # Missing values chart
└── .gitignore
```

## Workflow

### 1. Data Cleaning and Preprocessing
The dataset was cleaned to handle missing values and inconsistencies, including:
- duplicate removal
- missing embarkation handling
- cabin information converted into a binary indicator
- fare clipping to reduce skewness
- median-based imputation for missing age values during model training

### 2. Exploratory Data Analysis (EDA)
EDA was performed to understand how passenger attributes relate to survival outcome. The project visualizes:
- target distribution
- survival rates by sex, passenger class, and title
- age distribution by outcome
- correlation between variables
- missing value patterns

### 3. Feature Engineering
New features were added to improve model performance, including:
- `Title` extracted from passenger names
- `FamilySize`
- `IsAlone`
- `FarePerPerson`
- `HasCabin`

Unnecessary features such as unique identifiers and raw cabin text were excluded from the final model input.

### 4. Model Training and Comparison
Several classifiers were trained and compared:
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- SVM
- Random Forest
- Gradient Boosting

The project uses stratified cross-validation to compare model performance on the training set.

### 5. Hyperparameter Tuning
Hyperparameter tuning was performed using `GridSearchCV` for the most promising models to improve generalization and selection.

### 6. Model Evaluation
The final selected model is evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

A confusion matrix and ROC curves are also generated to visualize model quality.

## Model Performance
The best-performing model selected by the project is `Gradient Boosting`.

### Validation Performance
- CV ROC-AUC: 0.9034
- CV Accuracy: 0.8315

### Test Performance
- Accuracy: 0.7933
- Precision: 0.7667
- Recall: 0.6667
- F1-score: 0.7132
- ROC-AUC: 0.8399

These results indicate the model performs well on a public classification task while maintaining a simple and explainable ML pipeline.

## App Functionality
The Streamlit app in `app.py` allows users to enter passenger information and receive:
- survival probability
- predicted class label
- model input summary
- visualization of prediction output

This makes the project suitable for demo purposes and easy to present in portfolio or interview settings.

## Screenshots

### Streamlit Prediction Interface
![Streamlit App](https://github.com/Nareshaspire/Titanic-predictor/raw/main/figures/eda_overview.png)

### EDA Overview
![EDA Overview](https://github.com/Nareshaspire/Titanic-predictor/raw/main/figures/eda_overview.png)

### Model Evaluation
![Model Evaluation](https://github.com/Nareshaspire/Titanic-predictor/raw/main/figures/evaluation.png)

### Feature Importance
![Feature Importance](https://github.com/Nareshaspire/Titanic-predictor/raw/main/figures/feature_importance.png)

## Setup and Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```

### 3. Run the app
```bash
streamlit run app.py
```

## Requirements Coverage
This project satisfies the main requirements of a machine learning prediction system:

- real-world dataset used
- data cleaning and preprocessing performed
- EDA completed
- relevant features selected and engineered
- model trained and compared
- evaluation metrics used
- prediction results displayed
- interactive prediction interface built
- hyperparameter tuning included

## Portfolio Notes
This project is suitable for inclusion in a Python, data science, or machine learning portfolio because it demonstrates the full lifecycle of a predictive modeling workflow from data preparation to deployment.

## License
This project is intended for educational and portfolio use.
