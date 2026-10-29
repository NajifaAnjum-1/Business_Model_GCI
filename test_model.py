import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score, classification_report,
                             confusion_matrix, roc_curve, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import time

print("Loading data...")
client = pd.read_csv('Business model data/Client.csv')
record = pd.read_csv('Business model data/Record.csv')
df = pd.merge(client, record, on='Customer_ID')

# Identify target and features
target = 'churn'
drop_cols = ['Customer_ID', target]

# Categorical vs numeric
cat_cols = df.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
# Filter out target if in cat_cols
cat_cols = [c for c in cat_cols if c not in drop_cols]
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop_cols]

print(f"Total features: {len(num_cols) + len(cat_cols)} (Numeric: {len(num_cols)}, Categorical: {len(cat_cols)})")

# Feature Engineering
df['usage_drop_ratio'] = (df['change_mou'] / (df['mou_Mean'] + 1)).clip(-5, 5)
df['overage_ratio'] = (df['ovrrev_Mean'] / (df['rev_Mean'] + 1)).clip(0, 5)
df['tot_fail_calls'] = df['drop_blk_Mean'].fillna(0)
df['fail_call_ratio'] = (df['drop_blk_Mean'] / (df['attempt_Mean'] + 1)).clip(0, 1)
df['eqp_age_years'] = df['eqpdays'] / 365.25

new_num = ['usage_drop_ratio', 'overage_ratio', 'tot_fail_calls', 'fail_call_ratio', 'eqp_age_years']
num_cols.extend(new_num)

# One-hot encode categoricals with low cardinality, frequency encode high cardinality
# Check cardinailty
print("\nCategorical columns and unique values:")
for c in cat_cols:
    print(f"  {c}: {df[c].nunique()} unique")

# For area and crclscod, keep top categories or target encode / dummy encode top
top_cats = {}
for c in cat_cols:
    if df[c].nunique() > 10:
        top_cats[c] = df[c].value_counts().head(10).index.tolist()
        df[c] = df[c].apply(lambda x: x if x in top_cats[c] else 'Other')

df_encoded = pd.get_dummies(df[num_cols + cat_cols], drop_first=True)
print(f"Encoded dataset shape: {df_encoded.shape}")

# Train/Test Split (80/20) with stratification
X = df_encoded
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train size: {len(X_train):,}, Test size: {len(X_test):,}")

# 1. HistGradientBoostingClassifier (handles NaNs natively!)
print("\n--- Training HistGradientBoostingClassifier ---")
t0 = time.time()
hgb = HistGradientBoostingClassifier(
    max_iter=150,
    learning_rate=0.08,
    max_leaf_nodes=31,
    random_state=42
)
hgb.fit(X_train, y_train)
t_hgb = time.time() - t0

y_pred_hgb = hgb.predict(X_test)
y_prob_hgb = hgb.predict_proba(X_test)[:, 1]

auc_hgb = roc_auc_score(y_test, y_prob_hgb)
pr_auc_hgb = average_precision_score(y_test, y_prob_hgb)
acc_hgb = accuracy_score(y_test, y_pred_hgb)
prec_hgb = precision_score(y_test, y_pred_hgb)
rec_hgb = recall_score(y_test, y_pred_hgb)
f1_hgb = f1_score(y_test, y_pred_hgb)

print(f"HistGradientBoosting - ROC-AUC: {auc_hgb:.4f}, PR-AUC: {pr_auc_hgb:.4f}, Acc: {acc_hgb:.4f}, F1: {f1_hgb:.4f} (Fit Time: {t_hgb:.2f}s)")

# 2. Logistic Regression (Imputed & Scaled)
print("\n--- Training Logistic Regression Baseline ---")
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

t0 = time.time()
lr = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
lr.fit(X_train_scaled, y_train)
t_lr = time.time() - t0

y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

auc_lr = roc_auc_score(y_test, y_prob_lr)
pr_auc_lr = average_precision_score(y_test, y_prob_lr)
acc_lr = accuracy_score(y_test, y_pred_lr)
prec_lr = precision_score(y_test, y_pred_lr)
rec_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

print(f"Logistic Regression - ROC-AUC: {auc_lr:.4f}, PR-AUC: {pr_auc_lr:.4f}, Acc: {acc_lr:.4f}, F1: {f1_lr:.4f} (Fit Time: {t_lr:.2f}s)")

# 3. Random Forest Classifier
print("\n--- Training Random Forest Classifier ---")
t0 = time.time()
rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train_imp, y_train)
t_rf = time.time() - t0

y_pred_rf = rf.predict(X_test_imp)
y_prob_rf = rf.predict_proba(X_test_imp)[:, 1]

auc_rf = roc_auc_score(y_test, y_prob_rf)
pr_auc_rf = average_precision_score(y_test, y_prob_rf)
acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
rec_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print(f"Random Forest - ROC-AUC: {auc_rf:.4f}, PR-AUC: {pr_auc_rf:.4f}, Acc: {acc_rf:.4f}, F1: {f1_rf:.4f} (Fit Time: {t_rf:.2f}s)")

# Decile Lift analysis on test set for HistGradientBoosting
test_df = pd.DataFrame({'actual': y_test, 'prob': y_prob_hgb})
test_df['decile'] = pd.qcut(test_df['prob'], q=10, labels=False)
test_df['decile'] = 10 - test_df['decile'] # Decile 1 = highest risk

decile_analysis = test_df.groupby('decile').agg(
    total=('actual', 'count'),
    churners=('actual', 'sum'),
    churn_rate=('actual', 'mean')
).reset_index()
decile_analysis['cum_churners'] = decile_analysis['churners'].cumsum()
decile_analysis['cum_churn_pct'] = decile_analysis['cum_churners'] / test_df['actual'].sum()
decile_analysis['lift'] = decile_analysis['churn_rate'] / test_df['actual'].mean()
decile_analysis['cum_lift'] = (decile_analysis['cum_churners'] / decile_analysis['total'].cumsum()) / test_df['actual'].mean()

print("\n=== DECILE LIFT ANALYSIS (HistGradientBoosting) ===")
print(decile_analysis.to_string(index=False))
