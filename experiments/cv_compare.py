import sys
sys.path.insert(0, "src")
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, train_test_split
from xgboost import XGBRegressor
import train as T

df = pd.read_csv("data/cleaned.csv")
X = df[T.NUM_COLS + T.CAT_COLS]
y = df["selling_price"]
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Random Forest": RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42),
    "XGBoost (tuned)": XGBRegressor(subsample=0.7, n_estimators=800, max_depth=5,
                                    learning_rate=0.03, colsample_bytree=0.7, random_state=42),
}
rows = []
for name, reg in models.items():
    cv = cross_validate(T.make_model(reg), X_train, y_train, cv=5,
                        scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"])
    rows.append({
        "model": name,
        "MAE": round(-cv["test_neg_mean_absolute_error"].mean()),
        "MAE std": round(cv["test_neg_mean_absolute_error"].std()),
        "RMSE": round(-cv["test_neg_root_mean_squared_error"].mean()),
        "R2": round(cv["test_r2"].mean(), 4),
    })
print(pd.DataFrame(rows))
