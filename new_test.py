"""
================================================================================
PROJECT SMARTRETAIN — MACHINE LEARNING MODEL EVALUATION
================================================================================
File     : new_test.py
Purpose  : Train, evaluate, and compare three ML classifiers for churn prediction
Models   : 1. HistGradientBoostingClassifier  (primary / LightGBM-style GBDT)
           2. RandomForestClassifier           (ensemble bagging benchmark)
           3. LogisticRegression (L2)          (linear baseline)
Metrics  : ROC-AUC, PR-AUC, Accuracy, Precision, Recall, F1-Score, Decile Lift
Output   : Console scorecard + high-res figures saved to figures_model/
================================================================================
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from matplotlib.patches import Patch

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# ── Output directory ─────────────────────────────────────────────────────────
os.makedirs("figures_model", exist_ok=True)

# ── Plot theme ────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
})

# Colour palette
COL_HGB = "#2563EB"   # Blue  — HistGradientBoosting
COL_RF  = "#10B981"   # Green — RandomForest
COL_LR  = "#F59E0B"   # Amber — LogisticRegression
COL_RED = "#EF4444"
NAVY    = "#1E3A8A"
GRAY    = "#64748B"


# =============================================================================
# STEP 1 — LOAD, MERGE & FEATURE ENGINEERING
# =============================================================================
def load_and_engineer() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    print("=" * 70)
    print("STEP 1: LOADING DATA & ENGINEERING FEATURES")
    print("=" * 70)

    client = pd.read_csv(os.path.join("Business model data", "Client.csv"))
    record = pd.read_csv(os.path.join("Business model data", "Record.csv"))
    df     = pd.merge(client, record, on="Customer_ID")
    print(f"  Merged shape  : {df.shape}")
    print(f"  Churn rate    : {df['churn'].mean()*100:.2f}%\n")

    # ── Derived features ─────────────────────────────────────────────────
    df["usage_drop_ratio"] = (df["change_mou"] / (df["mou_Mean"] + 1)).clip(-5, 5)
    df["overage_ratio"]    = (df["ovrrev_Mean"] / (df["rev_Mean"] + 1)).clip(0, 5)
    df["tot_fail_calls"]   = df["drop_blk_Mean"].fillna(0)
    df["fail_call_ratio"]  = (df["drop_blk_Mean"] / (df["attempt_Mean"] + 1)).clip(0, 1)
    df["eqp_age_years"]    = df["eqpdays"] / 365.25

    target    = "churn"
    drop_cols = ["Customer_ID", target]

    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns
                if c not in drop_cols]
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in drop_cols]

    print(f"  Numeric features      : {len(num_cols)}")
    print(f"  Categorical features  : {len(cat_cols)}")

    # Cap high-cardinality categoricals at top-10 + 'Other'
    for c in cat_cols:
        if df[c].nunique() > 10:
            top10 = df[c].value_counts().head(10).index.tolist()
            df[c] = df[c].apply(lambda x: x if x in top10 else "Other")

    df_encoded = pd.get_dummies(df[num_cols + cat_cols], drop_first=True)
    feature_names = df_encoded.columns.tolist()
    X = df_encoded
    y = df[target]

    print(f"  Encoded feature matrix: {X.shape[0]:,} rows × {X.shape[1]} columns\n")
    return X, y, feature_names


# =============================================================================
# STEP 2 — TRAIN / TEST SPLIT + PREPROCESSING
# =============================================================================
def split_and_preprocess(X: pd.DataFrame, y: pd.Series):
    print("=" * 70)
    print("STEP 2: TRAIN / TEST SPLIT & PREPROCESSING")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,} samples  |  Test: {len(X_test):,} samples")
    print(f"  Train churn rate: {y_train.mean()*100:.2f}%  |  Test churn rate: {y_test.mean()*100:.2f}%\n")

    # Median imputation + standard scaling (for LR & RF which can't handle NaN)
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()

    X_train_imp    = imputer.fit_transform(X_train)
    X_test_imp     = imputer.transform(X_test)
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled  = scaler.transform(X_test_imp)

    return (X_train, X_test, y_train, y_test,
            X_train_imp, X_test_imp,
            X_train_scaled, X_test_scaled)


# =============================================================================
# STEP 3 — TRAIN MODELS
# =============================================================================
def train_models(X_train, y_train, X_train_imp, X_train_scaled):
    print("=" * 70)
    print("STEP 3: TRAINING MODELS")
    print("=" * 70)

    trained = {}

    # ── Model 1: HistGradientBoostingClassifier ───────────────────────────
    print("  [1/3] HistGradientBoostingClassifier (LightGBM-style GBDT)...")
    t0  = time.time()
    hgb = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        random_state=42
    )
    hgb.fit(X_train, y_train)
    trained["HistGradientBoosting"] = {"model": hgb, "X_test_key": "raw",
                                        "time": time.time() - t0}
    print(f"      Fit time: {trained['HistGradientBoosting']['time']:.2f}s")

    # ── Model 2: RandomForestClassifier ──────────────────────────────────
    print("  [2/3] RandomForestClassifier (Ensemble Bagging)...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_imp, y_train)
    trained["RandomForest"] = {"model": rf, "X_test_key": "imp",
                                "time": time.time() - t0}
    print(f"      Fit time: {trained['RandomForest']['time']:.2f}s")

    # ── Model 3: LogisticRegression (L2 Baseline) ─────────────────────────
    print("  [3/3] LogisticRegression (L2 Regularised Baseline)...")
    t0 = time.time()
    lr = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
    lr.fit(X_train_scaled, y_train)
    trained["LogisticRegression"] = {"model": lr, "X_test_key": "scaled",
                                      "time": time.time() - t0}
    print(f"      Fit time: {trained['LogisticRegression']['time']:.2f}s\n")

    return trained


# =============================================================================
# STEP 4 — EVALUATE MODELS
# =============================================================================
def evaluate_models(trained, X_test, X_test_imp, X_test_scaled, y_test) -> dict:
    print("=" * 70)
    print("STEP 4: EVALUATION METRICS")
    print("=" * 70)

    X_map = {"raw": X_test, "imp": X_test_imp, "scaled": X_test_scaled}
    results = {}

    for name, info in trained.items():
        model  = info["model"]
        X_eval = X_map[info["X_test_key"]]

        y_pred = model.predict(X_eval)
        y_prob = model.predict_proba(X_eval)[:, 1]

        results[name] = {
            "model":     model,
            "y_pred":    y_pred,
            "y_prob":    y_prob,
            "ROC-AUC":   roc_auc_score(y_test, y_prob),
            "PR-AUC":    average_precision_score(y_test, y_prob),
            "Accuracy":  accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall":    recall_score(y_test, y_pred),
            "F1-Score":  f1_score(y_test, y_pred),
            "Train Time (s)": info["time"],
        }

    # ── Print scorecard ───────────────────────────────────────────────────
    metric_cols = ["ROC-AUC", "PR-AUC", "Accuracy", "Precision", "Recall",
                   "F1-Score", "Train Time (s)"]
    scorecard = pd.DataFrame(
        {k: {m: results[k][m] for m in metric_cols} for k in results}
    ).T
    print("\n  " + "=" * 68)
    print("  " + " " * 10 + "MACHINE LEARNING BENCHMARK SCORECARD")
    print("  " + "=" * 68)
    print()
    print(scorecard.round(4).to_string())
    print()

    best_model = scorecard["ROC-AUC"].idxmax()
    best_auc   = scorecard.loc[best_model, "ROC-AUC"]
    print(f"  ★  Best Model  : {best_model}")
    print(f"  ★  Best ROC-AUC: {best_auc:.4f}\n")

    return results


# =============================================================================
# STEP 5 — DECILE LIFT ANALYSIS
# =============================================================================
def decile_lift_analysis(results, y_test, primary="HistGradientBoosting") -> pd.DataFrame:
    print("=" * 70)
    print(f"STEP 5: DECILE LIFT ANALYSIS  ({primary})")
    print("=" * 70)

    y_prob = results[primary]["y_prob"]
    test_df = pd.DataFrame({"actual": y_test.values, "prob": y_prob})
    test_df["decile"] = pd.qcut(test_df["prob"], q=10, labels=False, duplicates="drop")
    test_df["decile"] = 9 - test_df["decile"]   # 0 = highest risk → flip to 1-based
    test_df["decile"] += 1

    decile_tbl = (
        test_df.groupby("decile")
               .agg(total=("actual", "count"),
                    churners=("actual", "sum"),
                    churn_rate=("actual", "mean"))
               .reset_index()
    )
    decile_tbl["cum_churners"]  = decile_tbl["churners"].cumsum()
    decile_tbl["cum_churn_pct"] = decile_tbl["cum_churners"] / test_df["actual"].sum() * 100
    decile_tbl["lift"]          = (decile_tbl["churn_rate"] / test_df["actual"].mean()).round(2)
    decile_tbl["cum_lift"]      = (
        (decile_tbl["cum_churners"] / decile_tbl["total"].cumsum())
        / test_df["actual"].mean()
    ).round(2)

    print()
    print(decile_tbl.to_string(index=False))
    print()
    d1_rate  = decile_tbl.loc[decile_tbl["decile"] == 1, "churn_rate"].values[0] * 100
    top2_pct = decile_tbl.loc[decile_tbl["decile"] <= 2, "cum_churn_pct"].values[-1]
    print(f"  Decile-1 Churn Rate          : {d1_rate:.2f}%")
    print(f"  Top-2 Deciles Capture        : {top2_pct:.1f}% of all churners")
    print(f"  (i.e., targeting 20% of base = ~{top2_pct:.0f}% recall)\n")

    return decile_tbl


# =============================================================================
# FIGURE 1 — ROC, PR, AND MULTI-METRIC BAR CHART
# =============================================================================
def fig1_model_comparison(results, y_test) -> None:
    palette = {
        "HistGradientBoosting": COL_HGB,
        "RandomForest":         COL_RF,
        "LogisticRegression":   COL_LR,
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Figure 1 — Model Comparison: ROC, PR Curves & Metric Benchmark", y=1.01)

    # 1A — ROC Curves
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        axes[0].plot(fpr, tpr, lw=2.2, color=palette[name],
                     label=f"{name}\n(AUC = {res['ROC-AUC']:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Guess (AUC = 0.5000)")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves")
    axes[0].legend(loc="lower right", fontsize=9)

    # 1B — Precision-Recall Curves
    for name, res in results.items():
        prec_c, rec_c, _ = precision_recall_curve(y_test, res["y_prob"])
        axes[1].plot(rec_c, prec_c, lw=2.2, color=palette[name],
                     label=f"{name}\n(PR-AUC = {res['PR-AUC']:.4f})")
    axes[1].axhline(y_test.mean(), color="k", linestyle="--", lw=1.2,
                    label=f"Baseline ({y_test.mean():.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curves")
    axes[1].legend(loc="lower left", fontsize=9)

    # 1C — Multi-Metric Bar Chart
    metrics_to_plot = ["ROC-AUC", "PR-AUC", "Accuracy", "F1-Score"]
    bar_data = pd.DataFrame(
        {name: {m: res[m] for m in metrics_to_plot} for name, res in results.items()}
    ).T
    bar_data[metrics_to_plot].T.plot(
        kind="bar", ax=axes[2],
        color=[COL_HGB, COL_RF, COL_LR], width=0.7
    )
    axes[2].set_title("Multi-Metric Model Benchmark")
    axes[2].set_ylabel("Score")
    axes[2].set_ylim(0.45, 0.80)
    axes[2].legend(loc="lower right", fontsize=9)
    plt.setp(axes[2].get_xticklabels(), rotation=0)

    # Annotate bars with values
    for container in axes[2].containers:
        axes[2].bar_label(container, fmt="%.3f", fontsize=7.5, padding=2)

    plt.tight_layout()
    plt.savefig("figures_model/fig1_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  -> Saved figures_model/fig1_model_comparison.png")


# =============================================================================
# FIGURE 2 — DECILE LIFT & CUMULATIVE GAINS CHART
# =============================================================================
def fig2_decile_lift(decile_tbl: pd.DataFrame, y_test: pd.Series) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    fig.suptitle("Figure 2 — Decile Lift & Cumulative Gains (HistGradientBoosting)", y=1.01)

    decile_colors = (["#DC2626", "#EA580C"] +
                     ["#F59E0B", "#FBBF24"] +
                     ["#94A3B8"] * 4 +
                     ["#60A5FA", "#1E3A8A"])

    # 2A — Decile Churn Rate + Lift Labels
    bars = axes[0].bar(
        [f"D{d}" for d in decile_tbl["decile"]],
        decile_tbl["churn_rate"] * 100,
        color=decile_colors, width=0.6
    )
    axes[0].axhline(y_test.mean() * 100, color="black", linestyle="--",
                    label=f"Avg Churn ({y_test.mean()*100:.1f}%)")
    axes[0].set_ylabel("Actual Churn Rate (%)")
    axes[0].set_xlabel("Predicted Risk Decile (D1 = Highest Risk)")
    axes[0].set_title("Actual Churn Rate per Risk Decile\n(D1: Highest Risk → D10: Lowest Risk)")
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[0].legend()
    axes[0].set_ylim(0, 100)
    for i, b in enumerate(bars):
        h     = b.get_height()
        lift  = decile_tbl["lift"].iloc[i]
        axes[0].text(
            b.get_x() + b.get_width() / 2, h + 1,
            f"{h:.1f}%\n({lift:.2f}x)",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold"
        )

    # 2B — Cumulative Gains Curve
    deciles_pct = np.linspace(10, 100, len(decile_tbl))
    axes[1].plot([0] + list(deciles_pct),
                 [0] + list(decile_tbl["cum_churn_pct"]),
                 marker="o", lw=2.5, color=COL_HGB, label="HistGradientBoosting")
    axes[1].plot([0, 100], [0, 100], "k--", lw=1.5, label="Random Targeting")
    axes[1].fill_between(
        [0] + list(deciles_pct),
        [0] + list(decile_tbl["cum_churn_pct"]),
        [0] + list(deciles_pct),
        color="#DBEAFE", alpha=0.5, label="Gain Area"
    )
    axes[1].axvline(20, color=COL_RED, linestyle=":", lw=1.8, label="Top 20% Cutoff")
    axes[1].set_xlabel("% of Customers Targeted (Ranked by Risk Score)")
    axes[1].set_ylabel("% of Churners Captured")
    axes[1].set_title("Cumulative Gains Curve\n(Top 20% Base → ~30% of Churners Captured)")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[1].xaxis.set_major_formatter(mtick.PercentFormatter())
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("figures_model/fig2_decile_lift.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  -> Saved figures_model/fig2_decile_lift.png")


# =============================================================================
# FIGURE 3 — CONFUSION MATRICES (All 3 Models)
# =============================================================================
def fig3_confusion_matrices(results, y_test) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Figure 3 — Confusion Matrices (Test Set, N = 20,000)", y=1.01)

    colors = {
        "HistGradientBoosting": "Blues",
        "RandomForest":         "Greens",
        "LogisticRegression":   "Oranges",
    }
    for ax, (name, res) in zip(axes, results.items()):
        cm   = confusion_matrix(y_test, res["y_pred"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=["Active (0)", "Churned (1)"])
        disp.plot(ax=ax, colorbar=False, cmap=colors[name])
        ax.set_title(
            f"{name}\n"
            f"ROC-AUC={res['ROC-AUC']:.4f}  |  F1={res['F1-Score']:.4f}",
            fontsize=10
        )

    plt.tight_layout()
    plt.savefig("figures_model/fig3_confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  -> Saved figures_model/fig3_confusion_matrices.png")


# =============================================================================
# FIGURE 4 — PERMUTATION FEATURE IMPORTANCE (HistGradientBoosting)
# =============================================================================
def fig4_feature_importance(results, X_test, y_test,
                             feature_names: list[str],
                             primary="HistGradientBoosting") -> None:
    print(f"\n  Computing Permutation Feature Importance for {primary}...")
    print("  (This uses 5,000 samples × 5 repeats — takes ~60s)\n")

    model    = results[primary]["model"]
    X_sample = X_test.iloc[:5000]
    y_sample = y_test.iloc[:5000]

    perm        = permutation_importance(model, X_sample, y_sample,
                                         n_repeats=5, random_state=42,
                                         scoring="roc_auc")
    sorted_idx  = perm.importances_mean.argsort()[::-1][:15]
    top_feats   = np.array(feature_names)[sorted_idx]
    top_imps    = perm.importances_mean[sorted_idx]
    top_stds    = perm.importances_std[sorted_idx]

    # Category colour map
    category_map = {
        "usage_drop_ratio": ("#EF4444", "Usage Disengagement"),
        "eqpdays":          ("#3B82F6", "Device Obsolescence"),
        "eqp_age_years":    ("#3B82F6", "Device Obsolescence"),
        "months":           ("#10B981", "Tenure Lifecycle"),
        "totmrc_Mean":      ("#F59E0B", "Plan & Price Tier"),
        "hnd_price":        ("#3B82F6", "Device Value Tier"),
        "mou_Mean":         ("#EF4444", "Usage Disengagement"),
        "avgmou":           ("#EF4444", "Usage Disengagement"),
        "avgrev":           ("#F59E0B", "Billing Dynamics"),
        "change_rev":       ("#F59E0B", "Billing Dynamics"),
        "overage_ratio":    ("#F59E0B", "Bill Shock"),
        "ovrrev_Mean":      ("#F59E0B", "Bill Shock"),
        "drop_vce_Mean":    ("#64748B", "Network Quality"),
        "drop_blk_Mean":    ("#64748B", "Network Quality"),
        "asl_flag_Y":       ("#8B5CF6", "Credit Restrictions"),
        "uniqsubs":         ("#8B5CF6", "Account Structure"),
        "avgqty":           ("#EF4444", "Usage Disengagement"),
        "fail_call_ratio":  ("#64748B", "Network Quality"),
    }
    colors    = [category_map.get(f, ("#94A3B8", "Other"))[0] for f in top_feats]
    readable  = [f.replace("_Mean", " (Mean)").replace("_", " ").title()
                 for f in top_feats]

    plt.figure(figsize=(12, 7))
    bars = plt.barh(readable[::-1], top_imps[::-1], xerr=top_stds[::-1],
                    color=colors[::-1], height=0.65, capsize=3)
    plt.xlabel("Permutation Importance (Mean Decrease in ROC-AUC)")
    plt.title(
        "Figure 4 — Top 15 Churn Predictors (Root-Cause Attribution)\n"
        "Permutation Importance on Test Set (N=5,000 × 5 repeats)",
        fontsize=12
    )

    legend_elements = [
        Patch(facecolor="#3B82F6", label="Device & Equipment Obsolescence"),
        Patch(facecolor="#EF4444", label="Usage Contraction & Disengagement"),
        Patch(facecolor="#F59E0B", label="Plan Pricing & Bill Shock"),
        Patch(facecolor="#10B981", label="Tenure & Contract Milestones"),
        Patch(facecolor="#8B5CF6", label="Account & Credit Limits"),
        Patch(facecolor="#64748B", label="Network Quality"),
    ]
    plt.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig("figures_model/fig4_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  -> Saved figures_model/fig4_feature_importance.png")


# =============================================================================
# FIGURE 5 — CHURN PROBABILITY SCORE DISTRIBUTIONS
# =============================================================================
def fig5_score_distributions(results, y_test) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Figure 5 — Predicted Churn Probability Distributions", y=1.01)

    palette = {
        "HistGradientBoosting": COL_HGB,
        "RandomForest":         COL_RF,
        "LogisticRegression":   COL_LR,
    }
    for ax, (name, res) in zip(axes, results.items()):
        probs = pd.Series(res["y_prob"])
        ax.hist(probs[y_test.values == 0], bins=50, alpha=0.6,
                color="#3B82F6", label="Active (y=0)", density=True)
        ax.hist(probs[y_test.values == 1], bins=50, alpha=0.6,
                color="#EF4444", label="Churned (y=1)", density=True)
        ax.axvline(0.5, color="black", linestyle="--", lw=1.5, label="Decision Boundary (0.5)")
        ax.set_title(f"{name}\nROC-AUC = {res['ROC-AUC']:.4f}", fontsize=11)
        ax.set_xlabel("Predicted Churn Probability")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("figures_model/fig5_score_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  -> Saved figures_model/fig5_score_distributions.png")


# =============================================================================
# MAIN
# =============================================================================
def main():
    total_start = time.time()
    print("\n" + "=" * 70)
    print("  PROJECT SMARTRETAIN — ML MODEL EVALUATION SCRIPT (new_test.py)")
    print("=" * 70 + "\n")

    # 1. Load & feature engineer
    X, y, feature_names = load_and_engineer()

    # 2. Split & preprocess
    (X_train, X_test, y_train, y_test,
     X_train_imp, X_test_imp,
     X_train_scaled, X_test_scaled) = split_and_preprocess(X, y)

    # 3. Train all three models
    trained = train_models(X_train, y_train, X_train_imp, X_train_scaled)

    # 4. Evaluate & print scorecard
    results = evaluate_models(trained, X_test, X_test_imp, X_test_scaled, y_test)

    # 5. Decile lift analysis
    decile_tbl = decile_lift_analysis(results, y_test)

    # 6. Generate all figures
    print("=" * 70)
    print("STEP 6: GENERATING MODEL EVALUATION FIGURES")
    print("=" * 70)
    fig1_model_comparison(results, y_test)
    fig2_decile_lift(decile_tbl, y_test)
    fig3_confusion_matrices(results, y_test)
    fig4_feature_importance(results, X_test, y_test, feature_names)
    fig5_score_distributions(results, y_test)

    elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"  ML EVALUATION COMPLETE IN {elapsed:.2f}s")
    print("  5 figures saved to figures_model/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

