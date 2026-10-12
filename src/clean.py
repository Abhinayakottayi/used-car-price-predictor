import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

REFERENCE_YEAR = 2021


TWO_WORD_MODELS = {
    ("Swift", "Dzire"), ("Wagon", "R"), ("Grand", "i10"), ("Alto", "K10"),
    ("Alto", "800"), ("Santro", "Xing"), ("Land", "Cruiser"), ("Vitara", "Brezza"),
    ("Innova", "Crysta"), ("Elite", "i20"), ("Figo", "Aspire"), ("Safari", "Storme"),
    ("Grand", "Vitara"), ("Corolla", "Altis"),
    ("3", "Series"), ("5", "Series"), ("7", "Series"),
}


def split_name(name):
    tokens = str(name).split()
    if len(tokens) >= 2 and tokens[0] == "Land" and tokens[1] == "Rover":
        brand, rest = "Land Rover", tokens[2:]
    else:
        brand, rest = tokens[0], tokens[1:]
    if not rest:
        return brand, brand
    model = rest[0]
    if len(rest) > 1 and (rest[0], rest[1]) in TWO_WORD_MODELS:
        model = f"{rest[0]} {rest[1]}"
    return brand,model


def first_number(x):
    if pd.isna(x):
        return np.nan
    match = re.search(r"\d+(?:\.\d+)?", str(x).replace(",", ""))
    return float(match.group()) if match else np.nan


def parse_torque_nm(x):
    value = first_number(x)
    if np.isnan(value):
        return np.nan
    if "kgm" in str(x).lower():
        value *= 9.80665
    return value


def clean_data(df):
    df = df.drop_duplicates().copy()

    df = df[df["owner"] != "Test Drive Car"].copy()

    parts = df["name"].apply(split_name)
    df["brand"] = parts.str[0]
    df["model"] = parts.str[1]

    for col in ["mileage", "engine", "max_power"]:
        df[col] = df[col].apply(first_number)
    df["torque_nm"] = df["torque"].apply(parse_torque_nm)

    df["car_age"] = REFERENCE_YEAR - df["year"]

    df = df[df["km_driven"] < 500_000]

    df = df.drop(columns=["name", "torque", "year"])
    return df.reset_index(drop=True)


if __name__ == "__main__":
    raw = pd.read_csv(ROOT / "data" / "Car details v3.csv")
    cleaned = clean_data(raw)
    cleaned.to_csv(ROOT / "data" / "cleaned.csv", index=False)
    print("Raw shape:    ", raw.shape)
    print("Cleaned shape:", cleaned.shape)
    print("\nMissing values:\n", cleaned.isna().sum())
    print("\nSaved data/cleaned.csv")
