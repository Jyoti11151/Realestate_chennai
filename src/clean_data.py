"""
src/clean_data.py
=================
Cleaning pipeline for the Chennai real-estate dataset.

Steps (in order):
  1. Duplicate removal (full-row hash)
  2. KNN imputation for numeric columns
  3. Mode imputation for categorical columns
  4. IQR outlier capping for prices and rents
  5. Save cleaned outputs

Run:
    python src/clean_data.py
Outputs:
    data/sale_clean.csv
    data/rent_clean.csv
    results/cleaning_report.txt
"""
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def remove_duplicates(df, name):
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    print(f"[{name}] removed {removed} duplicates ({before} -> {len(df)})")
    return df, removed


def impute_numeric_knn(df, cols, k=5):
    cols = [c for c in cols if c in df.columns]
    if not cols: return df, 0
    n_miss = int(df[cols].isna().sum().sum())
    imputer = KNNImputer(n_neighbors=k)
    df[cols] = imputer.fit_transform(df[cols])
    print(f"  KNN imputed {n_miss} numeric cells across {cols}")
    return df, n_miss


def impute_categorical_mode(df, cols):
    cols = [c for c in cols if c in df.columns]
    n_miss = 0
    for c in cols:
        miss = int(df[c].isna().sum())
        if miss > 0:
            mode = df[c].mode()
            if len(mode) > 0:
                df[c] = df[c].fillna(mode.iloc[0])
                n_miss += miss
    print(f"  Mode-imputed {n_miss} categorical cells across {cols}")
    return df, n_miss


def cap_iqr(df, col, factor=1.5):
    if col not in df.columns: return df, 0
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - factor * iqr, q3 + factor * iqr
    capped = ((df[col] < low) | (df[col] > high)).sum()
    df[col] = df[col].clip(low, high)
    print(f"  IQR-capped {capped} outliers in {col} -> [{low:.0f}, {high:.0f}]")
    return df, int(capped)


def clean_sale(df):
    print("\n--- Cleaning SALE data ---")
    df, n_dup = remove_duplicates(df, "sale")
    df, n_num = impute_numeric_knn(df, ["property_age", "balconies",
                                          "floor_number", "amenities_count"])
    df, n_cat = impute_categorical_mode(df, ["furnishing_status", "facing", "parking"])
    df, _ = cap_iqr(df, "price_per_sqft")
    df, _ = cap_iqr(df, "property_price")
    return df, {"duplicates": n_dup, "numeric_imputed": n_num, "cat_imputed": n_cat}


def clean_rent(df):
    print("\n--- Cleaning RENT data ---")
    df, n_dup = remove_duplicates(df, "rent")
    df, n_num = impute_numeric_knn(df, ["deposit", "floor_number", "amenities_count"])
    df, n_cat = impute_categorical_mode(df, ["furnishing_status", "preferred_tenant"])
    df, _ = cap_iqr(df, "monthly_rent")
    return df, {"duplicates": n_dup, "numeric_imputed": n_num, "cat_imputed": n_cat}


def main():
    sale = pd.read_csv(DATA / "sale_raw.csv")
    rent = pd.read_csv(DATA / "rent_raw.csv")
    print(f"Loaded sale: {len(sale):,} rows | rent: {len(rent):,} rows")

    sale_clean, sale_stats = clean_sale(sale)
    rent_clean, rent_stats = clean_rent(rent)

    sale_clean.to_csv(DATA / "sale_clean.csv", index=False)
    rent_clean.to_csv(DATA / "rent_clean.csv", index=False)

    total_dup = sale_stats["duplicates"] + rent_stats["duplicates"]
    total_imp = (sale_stats["numeric_imputed"] + sale_stats["cat_imputed"]
                  + rent_stats["numeric_imputed"] + rent_stats["cat_imputed"])

    report = (
        "CLEANING REPORT\n" + "=" * 50 + "\n"
        f"Sale duplicates removed:     {sale_stats['duplicates']}\n"
        f"Rent duplicates removed:     {rent_stats['duplicates']}\n"
        f"TOTAL duplicates removed:    {total_dup}\n\n"
        f"Sale numeric imputations:    {sale_stats['numeric_imputed']}\n"
        f"Sale categorical imputations:{sale_stats['cat_imputed']}\n"
        f"Rent numeric imputations:    {rent_stats['numeric_imputed']}\n"
        f"Rent categorical imputations:{rent_stats['cat_imputed']}\n"
        f"TOTAL imputations:           {total_imp}\n\n"
        f"Final sale rows: {len(sale_clean):,}\n"
        f"Final rent rows: {len(rent_clean):,}\n"
    )
    (RESULTS / "cleaning_report.txt").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
