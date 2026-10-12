import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
RANDOM_STATE = 42

NUM_COLS = ["km_driven", "mileage", "engine", "max_power",
            "torque_nm", "seats", "car_age"]
CAT_COLS = ["fuel", "seller_type", "transmission", "owner", "brand"]
TARGET = "selling_price"


def make_model(regressor):
    """Preprocessing + regressor. Trains on log(price) but predicts rupees."""
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # brands with fewer than 10 cars are grouped together
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, NUM_COLS),
        ("cat", cat_pipe, CAT_COLS),
    ])
    pipe = Pipeline([("pre", pre), ("model", regressor)])
    return TransformedTargetRegressor(
        regressor=pipe, func=np.log1p, inverse_func=np.expm1
    )


def evaluate(name, model, X, y):
    pred = model.predict(X)
    return {
        "model": name,
        "MAE (INR)": round(mean_absolute_error(y, pred)),
        "RMSE (INR)": round(float(np.sqrt(mean_squared_error(y, pred)))),
        "R2": round(r2_score(y, pred), 4),
        "Median error %": round(float(np.median(np.abs(pred - y) / y) * 100), 1),
    }


def main():
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)

    df = pd.read_csv(ROOT / "data" / "cleaned.csv")
    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    # 1. Compare models
    candidates = {
        "Ridge (baseline)": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=5,
            random_state=RANDOM_STATE),
    }
    results = []
    for name, reg in candidates.items():
        m = make_model(reg)
        m.fit(X_train, y_train)
        results.append(evaluate(name, m, X_test, y_test))
        print("done:", name)

    # 2. Tune XGBoost
    param_dist = {
        "regressor__model__n_estimators": [300, 500, 800],
        "regressor__model__max_depth": [3, 5, 7],
        "regressor__model__learning_rate": [0.03, 0.05, 0.1],
        "regressor__model__subsample": [0.7, 0.9, 1.0],
        "regressor__model__colsample_bytree": [0.7, 0.9, 1.0],
    }
    search = RandomizedSearchCV(
        make_model(XGBRegressor(random_state=RANDOM_STATE)),
        param_distributions=param_dist,
        n_iter=20, cv=3, scoring="neg_mean_absolute_error",
        n_jobs=-1, random_state=RANDOM_STATE, verbose=1,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    print("Best params:", search.best_params_)

    final = evaluate("XGBoost (tuned)", best, X_test, y_test)
    results.append(final)
    results_df = pd.DataFrame(results)
    print("\n=== Model comparison (test set) ===")
    print(results_df.to_string(index=False))
    results_df.to_csv(ROOT / "reports" / "model_comparison.csv", index=False)

    # 3. Feature importance
    imp = permutation_importance(
        best, X_test, y_test, n_repeats=5,
        random_state=RANDOM_STATE, scoring="r2")
    importance = (pd.Series(imp.importances_mean, index=X_test.columns)
                  .sort_values())
    print("\n=== Permutation importance (drop in R2 when shuffled) ===")
    print(importance.sort_values(ascending=False).round(4))
    plt.figure(figsize=(7, 5))
    importance.plot(kind="barh")
    plt.title("Feature importance (permutation, test set)")
    plt.tight_layout()
    plt.savefig(ROOT / "reports" / "feature_importance.png", dpi=130)
    plt.close()

    # 4. Predicted vs actual
    pred = best.predict(X_test)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, pred, alpha=0.3, s=10)
    lims = [y_test.min(), y_test.max()]
    plt.plot(lims, lims, "r--")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Actual price (INR)")
    plt.ylabel("Predicted price (INR)")
    plt.title("Predicted vs actual (test set)")
    plt.tight_layout()
    plt.savefig(ROOT / "reports" / "pred_vs_actual.png", dpi=130)
    plt.close()

    # 5. Where does it struggle?
    err = X_test.assign(abs_pct_err=np.abs(pred - y_test) / y_test * 100)
    print("\n=== Median % error by brand (brands with 20+ test cars) ===")
    g = err.groupby("brand")["abs_pct_err"].agg(["median", "count"])
    print(g[g["count"] >= 20].sort_values("median", ascending=False).head(8).round(1))

    # 6. Refit on ALL data for the final app model, save with metadata
    best.fit(X, y)
    joblib.dump(best, ROOT / "models" / "model.joblib", compress=3)
    meta = {
        "brands": sorted(df["brand"].unique().tolist()),
        "fuels": sorted(df["fuel"].unique().tolist()),
        "seller_types": sorted(df["seller_type"].unique().tolist()),
        "transmissions": sorted(df["transmission"].unique().tolist()),
        "owners": sorted(df["owner"].unique().tolist()),
        "median_error_pct": final["Median error %"],
        "r2": final["R2"],
        "mae": final["MAE (INR)"],
        "defaults": {
            "mileage": float(df["mileage"].median()),
            "engine": float(df["engine"].median()),
            "max_power": float(df["max_power"].median()),
            "torque_nm": float(df["torque_nm"].median()),
            "seats": int(df["seats"].median()),
        },
    }
    (ROOT / "models" / "meta.json").write_text(json.dumps(meta, indent=2))
    print("\nSaved models/model.joblib and models/meta.json")


if __name__ == "__main__":
    main()
