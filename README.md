# California house prices: machine learning pipeline

Challenge of the Semillero de Automatización de Procesos en Machine Learning (Universidad EIA), 2026-1. Prediction of the median house value of the 20.640 districts of California.

## What I did
- Exploration, stratified train/test split by income, geographic plots and correlations
- New attributes (rooms per house, bedrooms ratio, people per house)
- Preprocessing in a `ColumnTransformer`: imputation, ratios, log, one-hot and scaling
- Linear regression, decision tree and random forest with cross-validation
- GridSearch of the random forest and final evaluation on the test set

## Main results
| Model | RMSE |
|---|---|
| Linear regression (CV) | 71.791 USD |
| Random forest, best of GridSearch (CV) | 49.381 USD |
| Random forest, test set | 48.861 USD (95% interval 46.548 to 51.070) |

The median income is the most important variable, then the location and the INLAND category.

## How to run

```
pip install scikit-learn pandas numpy matplotlib seaborn scipy joblib
jupyter notebook california_housing_ml.ipynb
```

---
Alejandro Lezcano Osorio | Universidad EIA (Colombia) and Rennes School of Business (France)
