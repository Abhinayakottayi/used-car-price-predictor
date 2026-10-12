import sys
sys.path.insert(0, "src")

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import train as T

df = pd.read_csv("data/cleaned.csv")
params = dict(subsample=0.7, n_estimators=800, max_depth=5,
              learning_rate=0.03, colsample_bytree=0.7, random_state=42)


def run(label, cat_cols):
    T.CAT_COLS = cat_cols
    X = df[T.NUM_COLS + cat_cols]
    y = df["selling_price"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    m = T.make_model(XGBRegressor(**params))
    m.fit(Xtr, ytr)
    return T.evaluate(label, m, Xte, yte)


base = ["fuel", "seller_type", "transmission", "owner", "brand"]
print(pd.DataFrame([run("without model", base),
                    run("with model", base + ["model"])]).to_string(index=False))