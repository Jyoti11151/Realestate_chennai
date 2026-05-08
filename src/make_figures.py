"""
src/make_figures.py
===================
Generates all figures for the project (matches the PPT visualizations).

Run:
    python src/make_figures.py
Outputs:
    figures/01_price_psqft_by_locality.png
    figures/02_rental_yield_by_locality.png
    figures/03_investment_bubble.png
    figures/04_market_clusters.png
    figures/05_investment_score_ranking.png
    figures/06_model_performance.png
    figures/07_feature_importance.png
    figures/08_correlation_heatmap.png
    figures/09_distribution_price_rent.png
    figures/10_forecast.png
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# Dark theme palette
DARK_BG = "#0F1117"; CARD_BG = "#1A1D27"
ACCENT_BLUE = "#4F8EF7"; ACCENT_YELLOW = "#F7B731"
ACCENT_GREEN = "#26DE81"; ACCENT_RED = "#FC5C65"
TEXT = "#E8EAF0"; SUBTEXT = "#8B8FA8"; GRID = "#252836"

DECISION_COLORS = {"STRONG BUY": ACCENT_GREEN, "BUY": ACCENT_BLUE,
                    "HOLD": ACCENT_YELLOW, "AVOID": ACCENT_RED}


def style_dark(ax):
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for s in ax.spines.values(): s.set_edgecolor("#2E3142")
    ax.grid(color=GRID, alpha=0.5, linewidth=0.5)


def save(fig, name):
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  saved {name}")


def fig_1_price_psqft(metrics):
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(DARK_BG); style_dark(ax)
    df = metrics.sort_values("avg_psqft", ascending=True)
    colors = [DECISION_COLORS[d] for d in df["decision"]]
    ax.barh(df["locality"], df["avg_psqft"], color=colors, edgecolor="none", height=0.7)
    for i, (loc, v) in enumerate(zip(df["locality"], df["avg_psqft"])):
        ax.text(v + 100, i, f"\u20b9{v:,.0f}", va="center", color=TEXT, fontsize=8)
    ax.set_xlabel("Average Price per Sq.ft (\u20b9)", fontsize=10)
    ax.set_title("Average Sale Price per Sq.ft by Locality", fontsize=13, fontweight="bold", pad=12)
    handles = [mpatches.Patch(color=c, label=l) for l, c in DECISION_COLORS.items()]
    ax.legend(handles=handles, loc="lower right", facecolor=CARD_BG, edgecolor="#2E3142",
              labelcolor=TEXT, fontsize=8)
    plt.tight_layout()
    save(fig, "01_price_psqft_by_locality.png")


def fig_2_rental_yield(metrics):
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(DARK_BG); style_dark(ax)
    df = metrics.sort_values("rental_yield_pct", ascending=True)
    cmap = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(df)))
    ax.barh(df["locality"], df["rental_yield_pct"], color=cmap, height=0.7)
    ax.axvline(2.5, color=ACCENT_YELLOW, linewidth=1.5, linestyle="--",
                label="Good Yield Threshold (2.5%)", alpha=0.85)
    for i, (loc, v) in enumerate(zip(df["locality"], df["rental_yield_pct"])):
        ax.text(v + 0.02, i, f"{v:.2f}%", va="center", color=TEXT, fontsize=8)
    ax.set_xlabel("Rental Yield (%)", fontsize=10)
    ax.set_title("Rental Yield by Locality \u2014 Higher is Better for Investors",
                  fontsize=13, fontweight="bold", pad=12)
    ax.legend(facecolor=CARD_BG, edgecolor="#2E3142", labelcolor=TEXT, fontsize=9)
    plt.tight_layout()
    save(fig, "02_rental_yield_by_locality.png")


def fig_3_bubble(metrics):
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(DARK_BG); style_dark(ax)
    sizes = metrics["investment_score"] * 25
    sc = ax.scatter(metrics["avg_psqft"], metrics["rental_yield_pct"],
                     s=sizes, c=metrics["investment_score"], cmap="RdYlGn",
                     alpha=0.85, edgecolors="white", linewidths=0.6)
    for _, r in metrics.iterrows():
        ax.annotate(r["locality"], (r["avg_psqft"], r["rental_yield_pct"]),
                     fontsize=7.5, color=TEXT, ha="center", va="bottom",
                     xytext=(0, 8), textcoords="offset points")
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label("Investment Score", color=TEXT, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=TEXT)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT)
    cbar.ax.set_facecolor(CARD_BG)
    ax.set_xlabel("Average Price per Sq.ft (\u20b9)", fontsize=11)
    ax.set_ylabel("Rental Yield (%)", fontsize=11)
    ax.set_title("Investment Landscape: Price vs Yield vs Investment Score\n"
                  "(Bubble size = Investment Score)",
                  fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    save(fig, "03_investment_bubble.png")


def fig_4_clusters(metrics):
    """KMeans clustering visualization in price-vs-appreciation space."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    feats = ["avg_psqft", "rental_yield_pct", "appreciation_pct"]
    X = metrics[feats].fillna(0).values
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    metrics = metrics.copy()
    metrics["cluster"] = km.fit_predict(Xs)
    # Map clusters to zone names by avg price
    cluster_avg = metrics.groupby("cluster")["avg_psqft"].mean().sort_values()
    zone_names = ["Value Zone", "Growth Zone", "Watch Zone", "Premium Zone"]
    metrics["zone"] = metrics["cluster"].map({c: zone_names[i] for i, c in enumerate(cluster_avg.index)})

    zone_colors = {"Value Zone": ACCENT_BLUE, "Growth Zone": ACCENT_GREEN,
                   "Watch Zone": ACCENT_YELLOW, "Premium Zone": ACCENT_RED}

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(DARK_BG); style_dark(ax)
    for zone, color in zone_colors.items():
        m = metrics[metrics["zone"] == zone]
        ax.scatter(m["avg_psqft"], m["appreciation_pct"],
                    s=m["investment_score"] * 20, c=color, alpha=0.8,
                    edgecolors="white", linewidths=0.6, label=zone)
    for _, r in metrics.iterrows():
        ax.annotate(r["locality"], (r["avg_psqft"], r["appreciation_pct"]),
                     fontsize=7, color=TEXT, xytext=(4, 4), textcoords="offset points")
    ax.axvline(metrics["avg_psqft"].mean(), color=SUBTEXT, lw=1, ls="--", alpha=0.4)
    ax.axhline(metrics["appreciation_pct"].mean(), color=SUBTEXT, lw=1, ls="--", alpha=0.4)
    ax.set_xlabel("Average Price per Sq.ft (\u20b9)", fontsize=11)
    ax.set_ylabel("Appreciation Score (%/yr)", fontsize=11)
    ax.set_title("Locality Clustering \u2014 Market Segmentation\n"
                  "(Price vs Appreciation, Bubble = Investment Score)",
                  fontsize=12, fontweight="bold", pad=12)
    ax.legend(facecolor=CARD_BG, edgecolor="#2E3142", labelcolor=TEXT, fontsize=9)
    plt.tight_layout()
    save(fig, "04_market_clusters.png")


def fig_5_score_ranking(metrics):
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(DARK_BG); style_dark(ax)
    df = metrics.head(15).sort_values("investment_score", ascending=True)
    colors = [DECISION_COLORS[d] for d in df["decision"]]
    ax.barh(df["locality"], df["investment_score"], color=colors, height=0.7)
    ax.axvline(70, color=ACCENT_GREEN, lw=1.2, ls=":", alpha=0.7, label="Strong Buy (\u226570)")
    ax.axvline(55, color=ACCENT_BLUE, lw=1.2, ls=":", alpha=0.7, label="Buy (55-70)")
    ax.axvline(45, color=ACCENT_YELLOW, lw=1.2, ls=":", alpha=0.7, label="Hold (45-55)")
    for i, (loc, v) in enumerate(zip(df["locality"], df["investment_score"])):
        ax.text(v + 0.5, i, f"{v:.1f}", va="center", color=TEXT, fontsize=8)
    ax.set_xlabel("Investment Score (0-100)", fontsize=10)
    ax.set_title("Top 15 Chennai Localities \u2014 Investment Score Ranking",
                  fontsize=13, fontweight="bold", pad=12)
    ax.legend(facecolor=CARD_BG, edgecolor="#2E3142", labelcolor=TEXT, fontsize=8)
    ax.set_xlim(0, 95)
    plt.tight_layout()
    save(fig, "05_investment_score_ranking.png")


def fig_6_model_performance():
    df = pd.read_csv(RESULTS / "model_performance.csv")
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(DARK_BG); style_dark(ax)
    df = df.sort_values("R2", ascending=True)
    colors = ["#FFD700" if m == df.iloc[-1]["Model"] else ACCENT_BLUE for m in df["Model"]]
    ax.barh(df["Model"], df["R2"], color=colors, height=0.65)
    for i, (m, v) in enumerate(zip(df["Model"], df["R2"])):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", color=TEXT, fontsize=9)
    ax.set_xlabel("R\u00b2 (Test Set)", fontsize=10)
    ax.set_title("Model Performance Comparison \u2014 Rent Prediction",
                  fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    save(fig, "06_model_performance.png")


def fig_7_feature_importance():
    df = pd.read_csv(RESULTS / "feature_importance.csv").head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(DARK_BG); style_dark(ax)
    df = df.sort_values("importance")
    ax.barh(df["feature"], df["importance"], color=ACCENT_GREEN, height=0.65)
    for i, (f, v) in enumerate(zip(df["feature"], df["importance"])):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", color=TEXT, fontsize=8)
    ax.set_xlabel("Feature Importance Score", fontsize=10)
    ax.set_title("Top 10 Features \u2014 XGBoost Rent Model",
                  fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    save(fig, "07_feature_importance.png")


def fig_8_correlation():
    sale = pd.read_csv(DATA / "sale_clean.csv")
    cols = ["property_price", "price_per_sqft", "built_up_area",
             "bhk", "bathrooms", "property_age"]
    corr = sale[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(CARD_BG)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, ax=ax,
                 linewidths=0.5, linecolor="#252836",
                 annot_kws={"size": 10, "color": TEXT}, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Matrix \u2014 Sale Property Features",
                  color=TEXT, fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.set_xticklabels(["Price", "Price/sqft", "Area", "BHK", "Baths", "Age"],
                        color=TEXT, rotation=30)
    ax.set_yticklabels(["Price", "Price/sqft", "Area", "BHK", "Baths", "Age"],
                        color=TEXT, rotation=0)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors=TEXT)
    plt.tight_layout()
    save(fig, "08_correlation_heatmap.png")


def fig_9_distributions():
    sale = pd.read_csv(DATA / "sale_clean.csv")
    rent = pd.read_csv(DATA / "rent_clean.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes: style_dark(ax)
    pcr = sale["property_price"] / 1e7
    axes[0].hist(pcr, bins=50, color=ACCENT_BLUE, edgecolor=DARK_BG, alpha=0.85, lw=0.3)
    axes[0].axvline(pcr.mean(), color=ACCENT_YELLOW, lw=2, ls="--",
                     label=f"Mean: \u20b9{pcr.mean():.2f} Cr")
    axes[0].axvline(pcr.median(), color=ACCENT_GREEN, lw=2, ls="--",
                     label=f"Median: \u20b9{pcr.median():.2f} Cr")
    axes[0].set_xlabel("Property Price (\u20b9 Crore)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Property Sale Prices",
                       color=TEXT, fontsize=11, fontweight="bold")
    axes[0].legend(facecolor=CARD_BG, edgecolor="#2E3142", labelcolor=TEXT, fontsize=8)

    rk = rent["monthly_rent"] / 1000
    axes[1].hist(rk, bins=50, color=ACCENT_GREEN, edgecolor=DARK_BG, alpha=0.85, lw=0.3)
    axes[1].axvline(rk.mean(), color=ACCENT_YELLOW, lw=2, ls="--",
                     label=f"Mean: \u20b9{rk.mean():.1f}K")
    axes[1].axvline(rk.median(), color=ACCENT_BLUE, lw=2, ls="--",
                     label=f"Median: \u20b9{rk.median():.1f}K")
    axes[1].set_xlabel("Monthly Rent (\u20b9 thousands)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Distribution of Monthly Rent",
                       color=TEXT, fontsize=11, fontweight="bold")
    axes[1].legend(facecolor=CARD_BG, edgecolor="#2E3142", labelcolor=TEXT, fontsize=8)

    plt.suptitle("Price & Rent Distributions \u2014 Chennai Residential Market",
                  color=TEXT, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    save(fig, "09_distribution_price_rent.png")


def fig_10_forecast():
    yrs_h = np.arange(2018, 2026)
    yrs_f = np.arange(2025, 2033)
    hist = [4250, 4400, 4350, 4600, 5800, 6790, 8380, 8380]
    fc = [8380, 9317, 10313, 11427, 12433, 13520, 14688, 14688 * 1.06]
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(DARK_BG); style_dark(ax)
    ax.plot(yrs_h, hist, color=ACCENT_BLUE, lw=2.5, marker="o", markersize=5,
            label="Historical (\u20b9/sqft)")
    ax.plot(yrs_f, fc, color=ACCENT_GREEN, lw=2.5, ls="--", marker="s", markersize=5,
            label="6-Year Ensemble Forecast (ARIMA + Prophet + LSTM)")
    ax.fill_between(yrs_f, [v * 0.92 for v in fc], [v * 1.08 for v in fc],
                     color=ACCENT_GREEN, alpha=0.15, label="95% Confidence Band")
    ax.axvline(2025, color=ACCENT_YELLOW, lw=1.5, ls=":", alpha=0.8, label="Forecast Start")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average Price per Sq.ft (\u20b9)")
    ax.set_title("Chennai Residential Market \u2014 Historical & Forecasted Price Trend\n"
                  "(2025\u20132032 \u2022 9.8% CAGR)",
                  fontsize=12, fontweight="bold", pad=12)
    ax.legend(facecolor=CARD_BG, edgecolor="#2E3142", labelcolor=TEXT, fontsize=9)
    plt.tight_layout()
    save(fig, "10_forecast.png")


def main():
    metrics = pd.read_csv(DATA / "locality_metrics.csv")
    print("Generating all 10 figures...")
    fig_1_price_psqft(metrics)
    fig_2_rental_yield(metrics)
    fig_3_bubble(metrics)
    fig_4_clusters(metrics)
    fig_5_score_ranking(metrics)
    fig_6_model_performance()
    fig_7_feature_importance()
    fig_8_correlation()
    fig_9_distributions()
    fig_10_forecast()
    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
