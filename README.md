# Titanic Survival Prediction System

An end-to-end machine learning project that predicts whether a Titanic passenger survived, with an interactive Streamlit interface.

**Dataset:** public Titanic passenger data (891 rows, 12 columns), `data/titanic.csv`
(source: https://github.com/datasciencedojo/datasets). Target: `Survived` (0/1).

## Project structure

```
train.py            full pipeline: clean -> EDA -> features -> models -> tuning -> evaluation
app.py              Streamlit prediction interface
data/titanic.csv    dataset
model/              saved model (joblib), metrics.json, comparison + sample predictions
figures/            EDA and evaluation plots
requirements.txt
```

## Run it

```bash
pip install -r requirements.txt
python train.py          # trains, evaluates, writes model/ and figures/
streamlit run app.py     # opens the interactive predictor
```

## What the pipeline does

1. **Cleaning:** `Embarked` (2 missing) filled with the mode; `Cabin` (77% missing) reduced to a `HasCabin` flag; `Fare` clipped at the 99th percentile; `Age` (20% missing) imputed with the median *inside* the sklearn pipeline, so there is no leakage from the test set.
2. **EDA:** target balance, survival by sex/class/title/family size, age distribution, correlation heatmap, missing-value chart (`figures/`).
3. **Feature engineering:** `Title` extracted from `Name` (rare titles grouped), `FamilySize`, `IsAlone`, `FarePerPerson`, `HasCabin`.
4. **Feature selection:** dropped identifiers (`PassengerId`, `Ticket`), raw `Name`/`Cabin`, and `SibSp`/`Parch` (captured by `FamilySize`). Permutation importance is reported afterwards.
5. **Model comparison:** Logistic Regression, KNN, Decision Tree, SVM, Random Forest, Gradient Boosting, 5-fold stratified CV on an 80% training split.
6. **Hyperparameter tuning:** `GridSearchCV` (scored on ROC-AUC) for Random Forest, Gradient Boosting and Logistic Regression.
7. **Evaluation:** the model with the best CV ROC-AUC is selected, then scored once on the 20% held-out test set (accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curves).
8. **Prediction display:** sample predictions with probabilities (`model/sample_predictions.csv`) and the Streamlit app.

## Results

| Model (tuned) | CV ROC-AUC | Test Accuracy | Test F1 | Test ROC-AUC |
|---|---|---|---|---|
| Gradient Boosting (selected by CV) | 0.903 | 0.793 | 0.713 | 0.840 |
| Random Forest | 0.891 | 0.804 | 0.729 | 0.842 |
| Logistic Regression | 0.871 | 0.827 | 0.770 | 0.871 |

Notes:
- The test set is only 179 passengers, so differences of a few points between models are within noise. Logistic Regression scored best on the test set even though Gradient Boosting won in cross-validation; the selection rule uses CV only, to avoid choosing on the test set.
- The strongest predictors are `Title` (which encodes sex, age group and marital status), then `Pclass`, `FarePerPerson` and `FamilySize`.

## Ideas to extend

Stacking/ensembling, threshold tuning for recall, SHAP explanations, a FastAPI endpoint, deployment to Streamlit Community Cloud.
