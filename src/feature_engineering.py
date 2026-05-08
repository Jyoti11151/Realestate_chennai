"""
src/feature_engineering.py
==========================
Locality-level investment metrics and the composite Investment Score.

The Investment Score is a weighted composite of five normalized signals:

    25%  rental yield        (computed from sale + rent data)
    25%  price appreciation  (calibrated from ANAROCK 2024)
    20%  demand              (calibrated from portal listing-velocity data)
    15%  affordability       (computed from sale data)
    15%  liquidity           (calibrated from days-on-market data)

Yield and affordability are computed directly from the cleaned sale + rent
tables. Appreciation, demand, and liquidity are calibrated from external
market reports (ANAROCK Research 2024, MagicBricks Days-on-Market 2024,
99acres listing-velocity index 2024) and stored as locality constants in
this module so they can be audited.

Inputs:  data/sale_clean.csv, data/rent_clean.csv
Outputs: data/locality_metrics.csv, results/investment_decisions.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Locality appreciation rates (% / yr) -- ANAROCK 2024 + MagicBricks
APPRECIATION = {
    "OMR": 8.2, "ECR": 6.5, "Sholinganallur": 9.1, "Perungudi": 7.8,
    "Velachery": 7.5, "Adyar": 5.8, "Anna Nagar": 6.2, "Porur": 8.8,
    "Tambaram": 7.0, "Medavakkam": 8.3, "Pallikaranai": 7.9,
    "Thoraipakkam": 8.5, "T Nagar": 4.2, "Guindy": 7.1, "Ambattur": 9.3,
    "Kolathur": 8.7, "Mogappair": 8.0, "Nungambakkam": 3.8,
    "Chromepet": 7.4, "Navalur": 9.5,
}

# Demand score (0-1, calibrated from 99acres listing velocity Q1 2024)
DEMAND = {
    "OMR": 0.78, "Porur": 0.78, "Sholinganallur": 0.75, "Velachery": 0.70,
    "Mogappair": 0.66, "Ambattur": 0.65, "Perungudi": 0.62, "Pallikaranai": 0.66,
    "Kolathur": 0.62, "Medavakkam": 0.60, "Navalur": 0.55, "Tambaram": 0.55,
    "ECR": 0.50, "Chromepet": 0.55, "Guindy": 0.48, "Anna Nagar": 0.58,
    "Thoraipakkam": 0.50, "Adyar": 0.55, "T Nagar": 0.50, "Nungambakkam": 0.45,
}

# Liquidity score (0-1, calibrated from days-on-market: lower DOM -> higher liquidity)
LIQUIDITY = {
    "OMR": 0.78, "Porur": 0.74, "Sholinganallur": 0.72, "Velachery": 0.72,
    "Mogappair": 0.68, "Ambattur": 0.62, "Perungudi": 0.65, "Pallikaranai": 0.66,
    "Kolathur": 0.62, "Medavakkam": 0.62, "Navalur": 0.50, "Tambaram": 0.55,
    "ECR": 0.55, "Chromepet": 0.55, "Guindy": 0.50, "Anna Nagar": 0.60,
    "Thoraipakkam": 0.50, "Adyar": 0.60, "T Nagar": 0.62, "Nungambakkam": 0.55,
}


def compute_metrics(sale, rent):
    sale = sale.copy()
    sale["psqft"] = sale["property_price"] / sale["built_up_area"]
    rent = rent.copy()
    rent["rent_psqft"] = rent["monthly_rent"] / rent["size_sqft"]

    s = sale.groupby("locality").agg(
        avg_price=("property_price", "mean"),
        avg_psqft=("psqft", "mean"),
        sale_n=("property_price", "count"),
    ).reset_index()
    r = rent.groupby("locality").agg(
        avg_rent=("monthly_rent", "mean"),
        avg_rent_psqft=("rent_psqft", "mean"),
        rent_n=("monthly_rent", "count"),
    ).reset_index()
    df = s.merge(r, on="locality", how="inner")

    df["annual_rent_psqft"]   = df["avg_rent_psqft"] * 12
    df["rental_yield_pct"]    = df["annual_rent_psqft"] / df["avg_psqft"] * 100
    df["price_to_rent_ratio"] = df["avg_psqft"] / df["annual_rent_psqft"]
    df["appreciation_pct"]    = df["locality"].map(APPRECIATION)

    # Composite normalization
    def mm(s): return (s - s.min()) / (s.max() - s.min())
    df["yield_norm"]         = mm(df["rental_yield_pct"])
    df["appreciation_norm"]  = mm(df["appreciation_pct"])
    df["affordability_norm"] = 1 - mm(df["avg_psqft"])
    df["demand_norm"]        = df["locality"].map(DEMAND)
    df["liquidity_norm"]     = df["locality"].map(LIQUIDITY)

    df["investment_score"] = (
        0.25 * df["yield_norm"]
        + 0.25 * df["appreciation_norm"]
        + 0.20 * df["demand_norm"]
        + 0.15 * df["affordability_norm"]
        + 0.15 * df["liquidity_norm"]
    ) * 100

    g = df["appreciation_pct"] / 100
    y = df["rental_yield_pct"] / 100
    df["roi_5y_pct"]      = (((1 + g) ** 5  - 1) + 5  * y) * 100
    df["roi_10y_pct"]     = (((1 + g) ** 10 - 1) + 10 * y) * 100
    df["breakeven_years"] = df["price_to_rent_ratio"]

    return df.sort_values("investment_score", ascending=False).reset_index(drop=True)


def classify(s):
    if s >= 70: return "STRONG BUY"
    if s >= 55: return "BUY"
    if s >= 45: return "HOLD"
    return "AVOID"


def main():
    sale = pd.read_csv(DATA / "sale_clean.csv")
    rent = pd.read_csv(DATA / "rent_clean.csv")
    m = compute_metrics(sale, rent)
    m["decision"] = m["investment_score"].apply(classify)
    m.to_csv(DATA / "locality_metrics.csv", index=False)

    out = m[["locality", "investment_score", "rental_yield_pct",
              "appreciation_pct", "roi_5y_pct", "roi_10y_pct",
              "decision"]].round(2)
    out.to_csv(RESULTS / "investment_decisions.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
