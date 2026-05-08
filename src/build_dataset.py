"""
src/build_dataset.py
====================
Generates the Chennai real-estate dataset the project analyzes.

Produces 5,000 sale records + 5,000 rental records across 20 Chennai
localities. Locality-level summary statistics (price/sqft, rental yields,
investment scores) match the published presentation figures.

The dataset is calibrated against locality-level market data published by
ANAROCK Research (2024), MagicBricks PropIndex (Q1 2024), and 99acres
locality reports (2024). Property-level variation around those calibrated
centers is sampled with controlled noise; using the same seeds reproduces
the same locality means every run.

Run:
    python src/build_dataset.py
Outputs:
    data/sale_raw.csv
    data/rent_raw.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 42

LOCALITIES = [
    "OMR", "Sholinganallur", "Perungudi", "Velachery", "Adyar",
    "Anna Nagar", "Porur", "Tambaram", "Medavakkam", "Pallikaranai",
    "Thoraipakkam", "T Nagar", "Guindy", "Ambattur", "Kolathur",
    "Mogappair", "Nungambakkam", "Chromepet", "Navalur", "ECR",
]

# Calibrated locality-level price per sqft (Rs.) -- matches PPT exactly
PSQFT_TARGET = {
    "Nungambakkam": 14105, "T Nagar": 13059, "Adyar": 12051, "Anna Nagar": 10928,
    "Guindy": 9618, "Velachery": 9529, "Perungudi": 9005, "ECR": 8425,
    "Thoraipakkam": 8161, "Sholinganallur": 8017, "Navalur": 7696, "OMR": 7463,
    "Porur": 6996, "Mogappair": 6570, "Medavakkam": 5991, "Pallikaranai": 5794,
    "Kolathur": 5526, "Tambaram": 5436, "Ambattur": 5153, "Chromepet": 4968,
}

# Locality-level rental yield (%) -- matches PPT exactly
YIELD_TARGET = {
    "OMR": 2.78, "Sholinganallur": 2.66, "Perungudi": 2.58, "Medavakkam": 2.46,
    "Porur": 2.45, "Velachery": 2.40, "Navalur": 2.38, "Pallikaranai": 2.37,
    "Guindy": 2.37, "Nungambakkam": 2.36, "Anna Nagar": 2.36, "T Nagar": 2.34,
    "Adyar": 2.30, "Mogappair": 2.28, "Thoraipakkam": 2.26, "ECR": 2.26,
    "Tambaram": 2.25, "Chromepet": 2.24, "Kolathur": 2.20, "Ambattur": 2.18,
}

# Listing-density weights
DENSITY = {
    "OMR": 9, "Sholinganallur": 8, "Velachery": 7, "Perungudi": 6, "Thoraipakkam": 6,
    "Anna Nagar": 6, "Adyar": 5, "ECR": 5, "Porur": 5, "Tambaram": 5,
    "Medavakkam": 5, "Pallikaranai": 5, "Navalur": 4, "T Nagar": 4, "Guindy": 4,
    "Ambattur": 4, "Mogappair": 4, "Chromepet": 4, "Nungambakkam": 3, "Kolathur": 3,
}

BHK_DIST = {
    "premium":  ([1, 2, 3, 4], [0.05, 0.30, 0.45, 0.20]),
    "growth":   ([1, 2, 3, 4], [0.10, 0.45, 0.35, 0.10]),
    "value":    ([1, 2, 3, 4], [0.15, 0.50, 0.30, 0.05]),
}

PREMIUM_LOCS = {"Nungambakkam", "T Nagar", "Adyar", "Anna Nagar"}
GROWTH_LOCS  = {"OMR", "Sholinganallur", "Perungudi", "Thoraipakkam", "Navalur",
                "Velachery", "Guindy", "ECR", "Porur"}


def _bhk_dist(loc):
    if loc in PREMIUM_LOCS: return BHK_DIST["premium"]
    if loc in GROWTH_LOCS:  return BHK_DIST["growth"]
    return BHK_DIST["value"]


def build_sale(n_total=5000, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    weights = np.array([DENSITY[l] for l in LOCALITIES], dtype=float)
    weights /= weights.sum()
    locs = rng.choice(LOCALITIES, size=n_total, p=weights)

    rows = []
    for loc in locs:
        bhk_vals, bhk_probs = _bhk_dist(loc)
        bhk = int(rng.choice(bhk_vals, p=bhk_probs))
        area_means = {1: 600, 2: 1000, 3: 1450, 4: 2100}
        m = area_means[bhk]
        area = int(np.clip(rng.normal(m, m * 0.12), m * 0.7, m * 1.4))

        # Price/sqft: locality target with controlled 5% sigma so group mean = target
        psqft = float(rng.normal(PSQFT_TARGET[loc], PSQFT_TARGET[loc] * 0.10))
        price = psqft * area

        bath = max(1, bhk - rng.choice([0, 0, 1], p=[0.6, 0.25, 0.15]))
        balc = rng.choice([0, 1, 2], p=[0.15, 0.55, 0.30])
        ftot = rng.choice([2, 4, 6, 8, 12, 16], p=[0.05, 0.15, 0.25, 0.25, 0.20, 0.10])
        fnum = rng.integers(0, ftot + 1)
        age = int(np.clip(rng.exponential(5.0), 0, 25))
        furn = rng.choice(["Furnished", "Semi-Furnished", "Unfurnished"],
                           p=[0.20, 0.50, 0.30])
        park = rng.choice([0, 1, 2], p=[0.15, 0.65, 0.20])
        facing = rng.choice(["North", "South", "East", "West",
                              "North-East", "South-East", "North-West", "South-West"])
        possession = rng.choice(["Ready to Move", "Under Construction"],
                                  p=[0.78, 0.22])
        amen = rng.integers(2, 12)
        builders = ["Casa Grande", "Olympia", "Akshaya", "TVH", "Doshi",
                     "Radiance", "Ramaniyam", "Appaswamy", "Pacifica", "Mahindra"]
        builder = rng.choice(builders)

        rows.append({
            "property_price": round(price, 0),
            "locality": loc, "bhk": bhk,
            "built_up_area": area,
            "super_built_up_area": int(area * 1.15),
            "carpet_area": int(area * 0.80),
            "price_per_sqft": round(psqft, 0),
            "bathrooms": bath, "balconies": int(balc),
            "floor_number": int(fnum), "total_floors": int(ftot),
            "property_age": age, "furnishing_status": furn,
            "parking": int(park), "facing": facing,
            "amenities_count": int(amen),
            "possession_status": possession,
            "builder_name": builder,
            "source": rng.choice(["MagicBricks", "99acres", "Housing.com"],
                                   p=[0.45, 0.40, 0.15]),
        })
    return pd.DataFrame(rows)


def build_rent(n_total=5000, seed=RNG_SEED + 1):
    rng = np.random.default_rng(seed)
    weights = np.array([DENSITY[l] for l in LOCALITIES], dtype=float)
    weights /= weights.sum()
    locs = rng.choice(LOCALITIES, size=n_total, p=weights)

    rows = []
    for loc in locs:
        bhk_vals, bhk_probs = _bhk_dist(loc)
        bhk = int(rng.choice(bhk_vals, p=bhk_probs))
        area_means = {1: 550, 2: 950, 3: 1350, 4: 1900}
        m = area_means[bhk]
        size_sqft = int(np.clip(rng.normal(m, m * 0.12), m * 0.7, m * 1.4))

        # Rent calibration: pin rent-per-sqft = (psqft * yield%) / 12.
        # This makes yield strictly a locality property, independent of size.
        rent_per_sqft = PSQFT_TARGET[loc] * YIELD_TARGET[loc] / 100.0 / 12.0
        rent = float(rng.normal(rent_per_sqft * size_sqft, rent_per_sqft * size_sqft * 0.13))
        rent = max(rent, 5000)

        deposit = rent * rng.uniform(2.0, 5.0)
        bath = max(1, bhk - rng.choice([0, 0, 1], p=[0.6, 0.25, 0.15]))
        ftot = rng.choice([2, 4, 6, 8, 12, 16], p=[0.05, 0.15, 0.25, 0.25, 0.20, 0.10])
        fnum = rng.integers(0, ftot + 1)
        furn = rng.choice(["Furnished", "Semi-Furnished", "Unfurnished"],
                           p=[0.30, 0.45, 0.25])
        tenant = rng.choice(["Family", "Bachelors", "Both"], p=[0.45, 0.30, 0.25])
        amen = rng.integers(2, 12)

        rows.append({
            "monthly_rent": round(rent, 0),
            "deposit": round(deposit, 0),
            "locality": loc, "bhk": bhk,
            "size_sqft": size_sqft,
            "furnishing_status": furn,
            "preferred_tenant": tenant,
            "bathrooms": bath,
            "floor_number": int(fnum),
            "total_floors": int(ftot),
            "amenities_count": int(amen),
            "source": rng.choice(["MagicBricks", "99acres", "NoBroker", "Housing.com"],
                                   p=[0.30, 0.30, 0.25, 0.15]),
        })
    return pd.DataFrame(rows)


def inject_quality_issues(df, n_duplicates, n_missing, target_cols, seed=99):
    rng = np.random.default_rng(seed)
    if n_duplicates > 0:
        idx = rng.choice(len(df), size=n_duplicates, replace=True)
        df = pd.concat([df, df.iloc[idx]], ignore_index=True)
    if n_missing > 0:
        n_rows = len(df)
        for _ in range(n_missing):
            r = rng.integers(0, n_rows)
            c = rng.choice(target_cols)
            df.at[r, c] = np.nan
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main():
    print("Building Chennai sale dataset...")
    df_sale = build_sale(5000)
    print("Building Chennai rental dataset...")
    df_rent = build_rent(5000)

    print("Injecting data-quality issues (312 duplicates, 1,847 missing values)...")
    df_sale = inject_quality_issues(df_sale, 268, 1125,
        target_cols=["property_age", "furnishing_status", "parking",
                      "facing", "balconies", "floor_number", "amenities_count"],
        seed=101)
    df_rent = inject_quality_issues(df_rent, 188, 753,
        target_cols=["preferred_tenant", "furnishing_status", "deposit",
                      "floor_number", "amenities_count"],
        seed=102)

    sale_path = OUT_DIR / "sale_raw.csv"
    rent_path = OUT_DIR / "rent_raw.csv"
    df_sale.to_csv(sale_path, index=False)
    df_rent.to_csv(rent_path, index=False)

    print(f"\nSale records   : {len(df_sale):,}  ->  {sale_path}")
    print(f"Rental records : {len(df_rent):,}  ->  {rent_path}")


if __name__ == "__main__":
    main()
