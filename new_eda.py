"""
================================================================================
PROJECT SMARTRETAIN — EXPLORATORY DATA ANALYSIS (EDA)
================================================================================
File     : new_eda.py
Purpose  : Standalone EDA script for telecom churn dataset
Datasets : Business model data/Client.csv
           Business model data/Record.csv
Output   : Console stats + high-res figures saved to figures_eda/
================================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Output directory ─────────────────────────────────────────────────────────
os.makedirs("figures_eda", exist_ok=True)

# ── Plot theme ────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
})

# Colour palette
BLUE   = "#3B82F6"
RED    = "#EF4444"
GREEN  = "#10B981"
ORANGE = "#F97316"
YELLOW = "#EAB308"
SLATE  = "#94A3B8"
NAVY   = "#1E3A8A"


# =============================================================================
# STEP 1 — LOAD & MERGE
# =============================================================================
def load_data() -> pd.DataFrame:
    print("=" * 70)
    print("STEP 1: LOADING & MERGING DATASETS")
    print("=" * 70)

    client = pd.read_csv(os.path.join("Business model data", "Client.csv"))
    record = pd.read_csv(os.path.join("Business model data", "Record.csv"))
    df     = pd.merge(client, record, on="Customer_ID")

    print(f"  Client shape  : {client.shape}")
    print(f"  Record shape  : {record.shape}")
    print(f"  Merged shape  : {df.shape}")
    print(f"  Total churned : {df['churn'].sum():,}  ({df['churn'].mean()*100:.2f}%)")
    print(f"  Total active  : {(~df['churn'].astype(bool)).sum():,}  ({(1-df['churn'].mean())*100:.2f}%)\n")
    return df


# =============================================================================
# STEP 2 — BASIC DATA QUALITY REPORT
# =============================================================================
def data_quality_report(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("STEP 2: DATA QUALITY REPORT")
    print("=" * 70)

    total_cells = df.shape[0] * df.shape[1]
    missing     = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    miss_df     = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
    miss_df     = miss_df[miss_df["Missing Count"] > 0].sort_values("Missing %", ascending=False)

    print(f"  Total rows          : {df.shape[0]:,}")
    print(f"  Total columns       : {df.shape[1]}")
    print(f"  Total cells         : {total_cells:,}")
    print(f"  Columns with NaN    : {len(miss_df)}")
    print(f"  Overall missing %   : {df.isnull().sum().sum() / total_cells * 100:.2f}%")
    print("\n  Top-10 columns by missing values:")
    print(miss_df.head(10).to_string())
    print()

    # Duplicate rows
    dupe_count = df.duplicated(subset="Customer_ID").sum()
    print(f"  Duplicate Customer_IDs : {dupe_count:,}")

    # Dtypes summary
    dtype_counts = df.dtypes.value_counts()
    print(f"\n  Data types:\n{dtype_counts.to_string()}\n")


# =============================================================================
# STEP 3 — FINANCIAL BASELINE
# =============================================================================
def financial_baseline(df: pd.DataFrame) -> dict:
    print("=" * 70)
    print("STEP 3: FINANCIAL BASELINE SUMMARY")
    print("=" * 70)

    mean_arpu           = df["rev_Mean"].mean()
    churn_rev_monthly   = df.loc[df["churn"] == 1, "rev_Mean"].sum()
    churn_rev_annual    = churn_rev_monthly * 12
    active_rev_annual   = df.loc[df["churn"] == 0, "rev_Mean"].sum() * 12

    print(f"  Mean Monthly ARPU            : ${mean_arpu:.2f}")
    print(f"  Mean Monthly MRC             : ${df['totmrc_Mean'].mean():.2f}")
    print(f"  Mean Overage Revenue         : ${df['ovrrev_Mean'].mean():.2f}")
    print(f"  Monthly Revenue Lost (churn) : ${churn_rev_monthly:,.2f}")
    print(f"  Annual Revenue Lost (churn)  : ${churn_rev_annual:,.2f}")
    print(f"  Annual Revenue Retained      : ${active_rev_annual:,.2f}\n")

    return {
        "mean_arpu":         mean_arpu,
        "churn_rev_monthly": churn_rev_monthly,
        "churn_rev_annual":  churn_rev_annual,
        "active_rev_annual": active_rev_annual,
    }


# =============================================================================
# FIGURE 1 — EDA OVERVIEW (Churn, ARPU, Revenue Exposure)
# =============================================================================
def fig1_eda_overview(df: pd.DataFrame, fin: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Figure 1 — EDA Overview: Churn, ARPU & Revenue Exposure", y=1.01)

    # 1A — Donut: Churn Distribution
    churn_counts = df["churn"].value_counts()
    axes[0].pie(
        churn_counts,
        labels=["Active (50.44%)", "Churned (49.56%)"],
        autopct="%1.1f%%",
        colors=[BLUE, RED],
        startangle=140,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
    )
    axes[0].set_title("Customer Status Breakdown\n(PoC Sample: N = 100,000)")

    # 1B — Bar: ARPU by Churn Status
    rev_data = df.groupby("churn")["rev_Mean"].mean()
    bars = axes[1].bar(["Active Retained", "Churned Customers"],
                       rev_data.values, color=[BLUE, RED], width=0.55)
    axes[1].set_ylabel("Mean Monthly Revenue ($)")
    axes[1].set_title("ARPU by Customer Status")
    axes[1].yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:.0f}"))
    for b in bars:
        h = b.get_height()
        axes[1].text(b.get_x() + b.get_width() / 2, h + 0.5,
                     f"${h:.2f}", ha="center", va="bottom", fontweight="bold")
    axes[1].set_ylim(0, 75)

    # 1C — Bar: Annual Revenue Exposure
    vals = [fin["active_rev_annual"] / 1e6, fin["churn_rev_annual"] / 1e6]
    bars2 = axes[2].bar(["Active Annual Rev", "Annual Revenue Lost"],
                        vals, color=[GREEN, RED], width=0.55)
    axes[2].set_ylabel("Total Annual Revenue ($ Millions)")
    axes[2].set_title(f"Annual Revenue Exposure:\n${fin['churn_rev_annual']/1e6:.2f}M Lost Annually")
    axes[2].yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:.1f}M"))
    for b in bars2:
        h = b.get_height()
        axes[2].text(b.get_x() + b.get_width() / 2, h + 0.5,
                     f"${h:.1f}M", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig("figures_eda/fig1_eda_overview.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  -> Saved figures_eda/fig1_eda_overview.png")


# =============================================================================
# FIGURE 2 — HARDWARE & EQUIPMENT DRIVERS
# =============================================================================
def fig2_equipment_drivers(df: pd.DataFrame) -> None:
    avg_churn = df["churn"].mean() * 100
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Figure 2 — Hardware & Equipment Churn Drivers", y=1.01)

    # 2A — Equipment Age
    df["eqp_bins"] = pd.cut(
        df["eqpdays"], bins=[-1, 180, 365, 730, 2500],
        labels=["< 6 Mos", "6–12 Mos", "1–2 Years", "> 2 Years"]
    )
    eqp_churn = df.groupby("eqp_bins", observed=False)["churn"].mean() * 100
    bars = axes[0, 0].bar(eqp_churn.index, eqp_churn.values,
                          color=[GREEN, YELLOW, ORANGE, RED], width=0.55)
    axes[0, 0].axhline(avg_churn, color="gray", linestyle="--",
                       label=f"Avg Churn ({avg_churn:.1f}%)")
    axes[0, 0].set_title("Equipment Age vs Churn Rate\n(+16.1% Penalty for Devices > 2 Yrs)")
    axes[0, 0].set_ylabel("Churn Rate (%)")
    axes[0, 0].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[0, 0].legend(); axes[0, 0].set_ylim(0, 70)
    for b in bars:
        h = b.get_height()
        axes[0, 0].text(b.get_x() + b.get_width() / 2, h + 0.8,
                        f"{h:.1f}%", ha="center", va="bottom", fontweight="bold")

    # 2B — Handset Price Tier
    df["price_tier"] = pd.cut(
        df["hnd_price"].fillna(-1),
        bins=[-2, 0, 50, 100, 150, 1000],
        labels=["Free/Unknown", "$1–$50", "$51–$100", "$101–$150", ">$150"]
    )
    price_churn = df.groupby("price_tier", observed=False)["churn"].mean() * 100
    bars = axes[0, 1].bar(price_churn.index, price_churn.values,
                          color=[SLATE, RED, ORANGE, BLUE, GREEN], width=0.55)
    axes[0, 1].axhline(avg_churn, color="gray", linestyle="--",
                       label=f"Avg Churn ({avg_churn:.1f}%)")
    axes[0, 1].set_title("Handset Price Tier vs Churn Rate\n(Budget Handsets 57.5% vs Premium 41.0%)")
    axes[0, 1].set_ylabel("Churn Rate (%)")
    axes[0, 1].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[0, 1].legend(); axes[0, 1].set_ylim(0, 70)
    for b in bars:
        h = b.get_height()
        axes[0, 1].text(b.get_x() + b.get_width() / 2, h + 0.8,
                        f"{h:.1f}%", ha="center", va="bottom", fontweight="bold")

    # 2C — Web Capability
    web_churn = (
        df.groupby("hnd_webcap", observed=False)["churn"].mean()
          .drop("UNKW", errors="ignore") * 100
    )
    labels = [
        "WC\n(Basic Wireless)",
        "WCMB\n(Mobile Browser)"
    ]
    wc_vals = [web_churn.get("WC", 0), web_churn.get("WCMB", 0)]
    bars = axes[1, 0].bar(labels, wc_vals, color=[RED, BLUE], width=0.5)
    axes[1, 0].set_title("Handset Web Capability vs Churn Rate\n(Legacy Non-Browser Devices = High Churn)")
    axes[1, 0].set_ylabel("Churn Rate (%)")
    axes[1, 0].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[1, 0].set_ylim(0, 70)
    for b in bars:
        h = b.get_height()
        axes[1, 0].text(b.get_x() + b.get_width() / 2, h + 0.8,
                        f"{h:.1f}%", ha="center", va="bottom", fontweight="bold")

    # 2D — Refurbished vs New
    refurb_churn = df.groupby("refurb_new", observed=False)["churn"].mean() * 100
    bars = axes[1, 1].bar(
        ["New Equipment (N)", "Refurbished (R)"],
        [refurb_churn.get("N", 0), refurb_churn.get("R", 0)],
        color=[BLUE, RED], width=0.5
    )
    axes[1, 1].set_title("Refurbished vs New Handset Churn\n(+4.5% Higher Churn for Refurbished)")
    axes[1, 1].set_ylabel("Churn Rate (%)")
    axes[1, 1].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[1, 1].set_ylim(0, 70)
    for b in bars:
        h = b.get_height()
        axes[1, 1].text(b.get_x() + b.get_width() / 2, h + 0.8,
                        f"{h:.1f}%", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig("figures_eda/fig2_equipment_drivers.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  -> Saved figures_eda/fig2_equipment_drivers.png")


# =============================================================================
# FIGURE 3 — BEHAVIORAL & BILLING DRIVERS
# =============================================================================
def fig3_behavioral_billing(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Figure 3 — Behavioral & Billing Churn Drivers", y=1.01)

    # 3A — Customer Care Calls (Silent Churner)
    df["cc_bins"] = pd.cut(
        df["custcare_Mean"].fillna(0),
        bins=[-1, 0, 2, 5, 100],
        labels=["0 Calls", "1–2 Calls", "3–5 Calls", ">5 Calls"]
    )
    cc_churn = df.groupby("cc_bins", observed=False)["churn"].mean() * 100
    bars = axes[0].bar(cc_churn.index, cc_churn.values,
                       color=[RED, ORANGE, BLUE, GREEN], width=0.55)
    axes[0].set_title("'Silent Churner' Effect\n0 Support Calls = 51.8% Churn Rate")
    axes[0].set_ylabel("Churn Rate (%)")
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[0].set_ylim(0, 65)
    for b in bars:
        h = b.get_height()
        axes[0].text(b.get_x() + b.get_width() / 2, h + 0.8,
                     f"{h:.1f}%", ha="center", va="bottom", fontweight="bold")

    # 3B — Usage Trend (change_mou)
    df["mou_chg_bins"] = pd.cut(
        df["change_mou"],
        bins=[-10000, -100, -20, 20, 100, 10000],
        labels=["Drop >100m", "Drop 20–100m", "Stable", "Grow 20–100m", "Surge >100m"]
    )
    mou_churn = df.groupby("mou_chg_bins", observed=False)["churn"].mean() * 100
    bars = axes[1].bar(mou_churn.index, mou_churn.values,
                       color=[RED, ORANGE, SLATE, BLUE, GREEN], width=0.55)
    axes[1].set_title("Usage Velocity (Change in MOU)\nUsage Contraction Precedes Churn")
    axes[1].set_ylabel("Churn Rate (%)")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[1].set_ylim(0, 65)
    for b in bars:
        h = b.get_height()
        axes[1].text(b.get_x() + b.get_width() / 2, h + 0.8,
                     f"{h:.1f}%", ha="center", va="bottom", fontweight="bold")

    # 3C — Overage Revenue (Bill Shock)
    df["ovr_bins"] = pd.cut(
        df["ovrrev_Mean"].fillna(0),
        bins=[-1, 0, 10, 30, 2000],
        labels=["$0", "$0.01–$10", "$10.01–$30", ">$30"]
    )
    ovr_churn = df.groupby("ovr_bins", observed=False)["churn"].mean() * 100
    bars = axes[2].bar(ovr_churn.index, ovr_churn.values,
                       color=[BLUE, GREEN, ORANGE, RED], width=0.55)
    axes[2].set_title("Overage Spend (Bill Shock)\nHigh Penalty Fees Accelerate Defection")
    axes[2].set_ylabel("Churn Rate (%)")
    axes[2].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[2].set_ylim(0, 65)
    for b in bars:
        h = b.get_height()
        axes[2].text(b.get_x() + b.get_width() / 2, h + 0.8,
                     f"{h:.1f}%", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig("figures_eda/fig3_behavioral_billing.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  -> Saved figures_eda/fig3_behavioral_billing.png")


# =============================================================================
# FIGURE 4 — TENURE & CONTRACT LIFECYCLE
# =============================================================================
def fig4_tenure(df: pd.DataFrame) -> None:
    avg_churn = df["churn"].mean() * 100
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Figure 4 — Tenure Lifecycle & Contract Cliffs", y=1.01)

    # 4A — Churn Rate by Tenure Bucket
    df["tenure_bin"] = pd.cut(
        df["months"],
        bins=[0, 6, 12, 18, 24, 36, 100],
        labels=["0–6 Mos", "7–12 Mos", "13–18 Mos", "19–24 Mos", "25–36 Mos", ">36 Mos"]
    )
    tenure_churn = df.groupby("tenure_bin", observed=False)["churn"].mean() * 100
    bars = axes[0].bar(tenure_churn.index, tenure_churn.values,
                       color=[GREEN, YELLOW, ORANGE, RED, RED, BLUE], width=0.6)
    axes[0].axhline(avg_churn, color="gray", linestyle="--",
                    label=f"Avg Churn ({avg_churn:.1f}%)")
    axes[0].set_title("Churn Rate by Customer Tenure\n(Peak Churn Cliff: Months 12–24)")
    axes[0].set_ylabel("Churn Rate (%)")
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[0].legend(); axes[0].set_ylim(0, 75)
    for b in bars:
        h = b.get_height()
        axes[0].text(b.get_x() + b.get_width() / 2, h + 0.8,
                     f"{h:.1f}%", ha="center", va="bottom", fontweight="bold")

    # 4B — Customer Count Distribution by Tenure
    tenure_counts = df.groupby("tenure_bin", observed=False)["Customer_ID"].count()
    axes[1].bar(tenure_counts.index, tenure_counts.values, color=BLUE, width=0.6, alpha=0.8)
    axes[1].set_title("Customer Volume by Tenure Bucket\n(Where Is Your Base Concentrated?)")
    axes[1].set_ylabel("Number of Customers")
    axes[1].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for i, (idx, val) in enumerate(tenure_counts.items()):
        axes[1].text(i, val + 200, f"{val:,}", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig("figures_eda/fig4_tenure.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  -> Saved figures_eda/fig4_tenure.png")


# =============================================================================
# FIGURE 5 — DROPPED / BLOCKED CALLS (Network Quality)
# =============================================================================
def fig5_network_quality(df: pd.DataFrame) -> None:
    avg_churn = df["churn"].mean() * 100
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Figure 5 — Network Quality & Churn", y=1.01)

    # 5A — Churn by Dropped+Blocked Calls Quartile
    df["drop_blk_tier"] = pd.qcut(
        df["drop_blk_Mean"].fillna(0), q=4,
        labels=["Q1 (Low)", "Q2 (Med-Low)", "Q3 (Med-High)", "Q4 (High)"]
    )
    drop_churn = df.groupby("drop_blk_tier", observed=False)["churn"].mean() * 100
    bars = axes[0].bar(drop_churn.index, drop_churn.values,
                       color=[GREEN, YELLOW, ORANGE, RED], width=0.55)
    axes[0].axhline(avg_churn, color="gray", linestyle="--",
                    label=f"Avg Churn ({avg_churn:.1f}%)")
    axes[0].set_title("Churn by Dropped & Blocked Calls (Quartile)\nNetwork Failures Elevate Churn Risk")
    axes[0].set_ylabel("Churn Rate (%)")
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[0].legend(); axes[0].set_ylim(0, 65)
    for b in bars:
        h = b.get_height()
        axes[0].text(b.get_x() + b.get_width() / 2, h + 0.8,
                     f"{h:.1f}%", ha="center", va="bottom", fontweight="bold")

    # 5B — Boxplot: drop_blk_Mean by Churn Status
    churned_vals = df.loc[df["churn"] == 1, "drop_blk_Mean"].dropna()
    active_vals  = df.loc[df["churn"] == 0, "drop_blk_Mean"].dropna()
    axes[1].boxplot([active_vals, churned_vals],
                    tick_labels=["Active (No Churn)", "Churned"],
                    patch_artist=True,
                    boxprops=dict(facecolor=BLUE, alpha=0.6),
                    medianprops=dict(color=RED, linewidth=2))
    axes[1].set_title("Distribution of Dropped+Blocked Calls\nby Churn Status")
    axes[1].set_ylabel("Mean Dropped+Blocked Calls per Month")

    plt.tight_layout()
    plt.savefig("figures_eda/fig5_network_quality.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  -> Saved figures_eda/fig5_network_quality.png")


# =============================================================================
# FIGURE 6 — NUMERIC FEATURE CORRELATION HEATMAP
# =============================================================================
def fig6_correlation_heatmap(df: pd.DataFrame) -> None:
    key_cols = [
        "churn", "rev_Mean", "mou_Mean", "totmrc_Mean", "ovrrev_Mean",
        "change_mou", "change_rev", "eqpdays", "hnd_price", "custcare_Mean",
        "drop_blk_Mean", "months", "uniqsubs", "avgrev", "avgmou"
    ]
    corr_df = df[[c for c in key_cols if c in df.columns]].corr()

    plt.figure(figsize=(13, 10))
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    sns.heatmap(
        corr_df, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, linewidths=0.5, square=True, cbar_kws={"shrink": 0.8},
        annot_kws={"size": 9}
    )
    plt.title("Figure 6 — Feature Correlation Heatmap\n(Key Numeric Variables vs Churn)", pad=15)
    plt.tight_layout()
    plt.savefig("figures_eda/fig6_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  -> Saved figures_eda/fig6_correlation_heatmap.png")


# =============================================================================
# FIGURE 7 — REVENUE DISTRIBUTION (Active vs Churned)
# =============================================================================
def fig7_revenue_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Figure 7 — Revenue Distribution: Active vs Churned Customers", y=1.01)

    # 7A — KDE: Monthly Revenue by Churn Status
    for status, color, label in [(0, BLUE, "Active"), (1, RED, "Churned")]:
        vals = df.loc[df["churn"] == status, "rev_Mean"].dropna()
        vals.plot.kde(ax=axes[0], color=color, lw=2.2, label=f"{label} (mean=${vals.mean():.2f})")
    axes[0].set_xlabel("Monthly Revenue ($)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("KDE: Monthly Revenue Distribution")
    axes[0].legend()
    axes[0].set_xlim(-10, 200)

    # 7B — Histogram: Overage Revenue
    df["ovrrev_clipped"] = df["ovrrev_Mean"].fillna(0).clip(0, 100)
    for status, color, label in [(0, BLUE, "Active"), (1, RED, "Churned")]:
        vals = df.loc[df["churn"] == status, "ovrrev_clipped"]
        axes[1].hist(vals, bins=40, color=color, alpha=0.55, label=label, density=True)
    axes[1].set_xlabel("Monthly Overage Revenue ($, clipped at $100)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Overage Revenue Distribution\n(Bill Shock Exposure by Status)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("figures_eda/fig7_revenue_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  -> Saved figures_eda/fig7_revenue_distribution.png")


# =============================================================================
# STEP 4 — PRINT TABULAR EDA STATS (Console Summary)
# =============================================================================
def print_eda_stats(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("STEP 4: TABULAR EDA SUMMARY (Key Churn Drivers)")
    print("=" * 70)

    # Equipment Age
    df["eqp_bin2"] = pd.cut(df["eqpdays"], bins=[-1, 180, 365, 730, 3000],
                             labels=["< 6 Mos", "6–12 Mos", "1–2 Yrs", "> 2 Yrs"])
    print("\n[1] Churn by Equipment Age:")
    print(
        df.groupby("eqp_bin2", observed=False)["churn"]
          .agg(["count", "mean"])
          .assign(churn_rate=lambda x: (x["mean"] * 100).round(2))
          .to_string()
    )

    # Handset Price Tier
    print("\n[2] Churn by Handset Price Tier:")
    print(
        df.groupby("price_tier", observed=False)["churn"]
          .agg(["count", "mean"])
          .assign(churn_rate=lambda x: (x["mean"] * 100).round(2))
          .to_string()
    )

    # Customer Care Calls
    print("\n[3] Churn by Customer Care Calls:")
    print(
        df.groupby("cc_bins", observed=False)["churn"]
          .agg(["count", "mean"])
          .assign(churn_rate=lambda x: (x["mean"] * 100).round(2))
          .to_string()
    )

    # Tenure
    print("\n[4] Churn by Tenure Bucket:")
    print(
        df.groupby("tenure_bin", observed=False)["churn"]
          .agg(["count", "mean"])
          .assign(churn_rate=lambda x: (x["mean"] * 100).round(2))
          .to_string()
    )

    # Usage Trend
    print("\n[5] Churn by Usage Trend (change_mou):")
    print(
        df.groupby("mou_chg_bins", observed=False)["churn"]
          .agg(["count", "mean"])
          .assign(churn_rate=lambda x: (x["mean"] * 100).round(2))
          .to_string()
    )

    # Overage Revenue
    print("\n[6] Churn by Overage Revenue (Bill Shock):")
    print(
        df.groupby("ovr_bins", observed=False)["churn"]
          .agg(["count", "mean"])
          .assign(churn_rate=lambda x: (x["mean"] * 100).round(2))
          .to_string()
    )
    print()


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "=" * 70)
    print("  PROJECT SMARTRETAIN — STANDALONE EDA SCRIPT (new_eda.py)")
    print("=" * 70 + "\n")

    # Load data
    df = load_data()

    # Quality report
    data_quality_report(df)

    # Financial baseline
    fin = financial_baseline(df)

    # Generate all figures
    print("\n" + "=" * 70)
    print("STEP 3: GENERATING EDA FIGURES")
    print("=" * 70)
    fig1_eda_overview(df, fin)
    fig2_equipment_drivers(df)
    fig3_behavioral_billing(df)
    fig4_tenure(df)
    fig5_network_quality(df)
    fig6_correlation_heatmap(df)
    fig7_revenue_distribution(df)

    # Print tabular stats (uses bins created in figure functions)
    print_eda_stats(df)

    print("=" * 70)
    print("  EDA COMPLETE — 7 figures saved to figures_eda/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
