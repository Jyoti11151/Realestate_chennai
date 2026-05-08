"""
src/ml_models.py
================
Trains and compares 9 regression models for monthly-rent prediction.

Models: Linear, Ridge, Lasso, Elastic Net, Random Forest, Gradient Boosting,
        XGBoost, LightGBM, CatBoost

Inputs:  data/rent_clean.csv
Outputs: results/model_performance.csv
         results/feature_importance.csv

Run:
    python src/ml_models.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100)
    return rmse, mae, r2, mape


def get_models():
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge":             Ridge(alpha=1.0),
        "Lasso":             Lasso(alpha=10.0, max_iter=10000),
        "Elastic Net":       ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000),
        "Random Forest":     RandomForestRegressor(n_estimators=300, max_depth=14,
                                                     min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=5,
                                                        learning_rate=0.05, random_state=42),
    }
    try:
        import xgboost as xgb
        models["XGBoost"] = xgb.XGBRegressor(n_estimators=500, max_depth=7, learning_rate=0.04,
                                              subsample=0.85, colsample_bytree=0.85,
                                              random_state=42, n_jobs=-1, verbosity=0)
    except ImportError: pass
    try:
        import lightgbm as lgb
        models["LightGBM"] = lgb.LGBMRegressor(n_estimators=500, max_depth=7, learning_rate=0.04,
                                                  num_leaves=63, random_state=42, n_jobs=-1, verbose=-1)
    except ImportError: pass
    try:
        from catboost import CatBoostRegressor
        models["CatBoost"] = CatBoostRegressor(iterations=500, depth=7, learning_rate=0.04,
                                                  random_state=42, verbose=0)
    except ImportError: pass
    return models


def main():
    rent = pd.read_csv(DATA / "rent_clean.csv")
    rent = rent.dropna(subset=["monthly_rent", "size_sqft", "bhk", "bathrooms",
                                  "furnishing_status", "locality"])

    num_features = ["size_sqft", "bhk", "bathrooms", "floor_number",
                     "total_floors", "amenities_count"]
    cat_features = ["locality", "furnishing_status", "preferred_tenant"]

    X = rent[num_features + cat_features]
    y = rent["monthly_rent"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)
    print(f"Train: {len(X_tr):,} | Test: {len(X_te):,}")

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
    ])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, model in get_models().items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(X_tr, y_tr)
        y_hat = pipe.predict(X_te)
        rmse, mae, r2, mape = metrics(y_te.values, y_hat)
        cv = cross_val_score(pipe, X_tr, y_tr, cv=kf, scoring="r2", n_jobs=-1).mean()
        rows.append({"Model": name, "R2": round(r2, 3),
                      "RMSE": round(rmse, 0), "MAE": round(mae, 0),
                      "MAPE_pct": round(mape, 1), "CV_R2": round(cv, 3)})
        print(f"  {name:<22s}  R2={r2:.3f}  RMSE={rmse:>7.0f}  MAPE={mape:.1f}%  CV={cv:.3f}")

    out = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    out.to_csv(RESULTS / "model_performance.csv", index=False)

    # Feature importance from best tree model
    print("\n--- Feature Importance (best tree model) ---")
    try:
        import xgboost as xgb
        best = xgb.XGBRegressor(n_estimators=500, max_depth=7, learning_rate=0.04,
                                 subsample=0.85, colsample_bytree=0.85,
                                 random_state=42, n_jobs=-1, verbosity=0)
    except ImportError:
        best = RandomForestRegressor(n_estimators=300, max_depth=14, random_state=42, n_jobs=-1)

    pipe = Pipeline([("pre", pre), ("model", best)])
    pipe.fit(X_tr, y_tr)
    feat_names = (num_features +
                   list(pipe.named_steps["pre"].named_transformers_["cat"]
                            .get_feature_names_out(cat_features)))
    imp = pipe.named_steps["model"].feature_importances_
    fi = pd.DataFrame({"feature": feat_names, "importance": imp}) \
            .sort_values("importance", ascending=False).reset_index(drop=True)
    fi.head(15).to_csv(RESULTS / "feature_importance.csv", index=False)
    print(fi.head(15).to_string(index=False))

    print("\n--- LEADERBOARD ---")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
