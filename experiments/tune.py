import sys
sys.path.insert(0, "src")

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor

import train as T

df = pd.read_csv("data/cleaned.csv")
X = df[T.NUM_COLS + T.CAT_COLS]
y = df["selling_price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Tune Random Forest with the same budget as XGBoost (20 trials, 3-fold CV)
rf_search = RandomizedSearchCV(
    T.make_model(RandomForestRegressor(random_state=42, n_jobs=1)),
    param_distributions={
        "regressor__model__n_estimators": [200, 400, 600],
        "regressor__model__max_depth": [None, 10, 20, 30],
        "regressor__model__min_samples_leaf": [1, 2, 4],
        "regressor__model__min_samples_split": [2, 5, 10],
        "regressor__model__max_features": ["sqrt", 0.5, 1.0],
    },
    n_iter=20, cv=3, scoring="neg_mean_absolute_error",
    n_jobs=-1, random_state=42, verbose=1,
)
rf_search.fit(X_train, y_train)
print("Best RF params:", rf_search.best_params_)

# XGBoost with the best params you already found
xgb = T.make_model(XGBRegressor(
    subsample=0.7, n_estimators=800, max_depth=5,
    learning_rate=0.03, colsample_bytree=0.7, random_state=42))
xgb.fit(X_train, y_train)

results = [
    T.evaluate("Random Forest (tuned)", rf_search.best_estimator_, X_test, y_test),
    T.evaluate("XGBoost (tuned)", xgb, X_test, y_test),
]
print(pd.DataFrame(results).to_string(index=False))