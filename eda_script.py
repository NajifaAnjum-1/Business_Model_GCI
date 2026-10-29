import pandas as pd
import numpy as np

# Load datasets
client = pd.read_csv('Business model data/Client.csv')
record = pd.read_csv('Business model data/Record.csv')
df = pd.merge(client, record, on='Customer_ID')

print(f"Total customers: {len(df):,}")
print(f"Overall churn rate: {df['churn'].mean()*100:.2f}%\n")

# 1. Equipment Age
df['eqp_bin'] = pd.cut(df['eqpdays'], bins=[-1, 180, 365, 730, 3000], labels=['< 6 mos', '6-12 mos', '1-2 yrs', '> 2 yrs'])
eqp_stat = df.groupby('eqp_bin', observed=False)['churn'].agg(['count', 'mean']).assign(churn_rate=lambda x: (x['mean']*100).round(2))
print("=== 1. CHURN BY EQUIPMENT AGE ===")
print(eqp_stat)
print()

# 2. Handset Price
df['price_tier'] = pd.cut(df['hnd_price'].fillna(-1), bins=[-2, 0, 50, 100, 150, 1000], labels=['Unknown/Free', '$1-$50', '$51-$100', '$101-$150', '>$150'])
price_stat = df.groupby('price_tier', observed=False)['churn'].agg(['count', 'mean']).assign(churn_rate=lambda x: (x['mean']*100).round(2))
print("=== 2. CHURN BY HANDSET PRICE ===")
print(price_stat)
print()

# 3. Web Capability
web_stat = df.groupby('hnd_webcap', observed=False)['churn'].agg(['count', 'mean']).assign(churn_rate=lambda x: (x['mean']*100).round(2))
print("=== 3. CHURN BY HANDSET WEB CAPABILITY ===")
print(web_stat)
print()

# 4. Refurbished vs New
refurb_stat = df.groupby('refurb_new', observed=False)['churn'].agg(['count', 'mean']).assign(churn_rate=lambda x: (x['mean']*100).round(2))
print("=== 4. CHURN BY REFURBISHED PHONE ===")
print(refurb_stat)
print()

# 5. Usage Change (change_mou)
df['mou_change_bin'] = pd.cut(df['change_mou'], bins=[-10000, -100, -20, 20, 100, 10000], labels=['Severe Drop (<-100m)', 'Drop (-100 to -20m)', 'Stable (-20 to +20m)', 'Growth (+20 to +100m)', 'Surge (>100m)'])
mou_chg_stat = df.groupby('mou_change_bin', observed=False)['churn'].agg(['count', 'mean']).assign(churn_rate=lambda x: (x['mean']*100).round(2))
print("=== 5. CHURN BY USAGE TREND (change_mou) ===")
print(mou_chg_stat)
print()

# 6. Overage Revenue
df['ovr_tier'] = pd.cut(df['ovrrev_Mean'].fillna(0), bins=[-1, 0, 10, 30, 5000], labels=['No Overage ($0)', 'Low ($0.01-$10)', 'Medium ($10.01-$30)', 'High (>$30)'])
ovr_stat = df.groupby('ovr_tier', observed=False)['churn'].agg(['count', 'mean']).assign(churn_rate=lambda x: (x['mean']*100).round(2))
print("=== 6. CHURN BY OVERAGE REVENUE (BILL SHOCK) ===")
print(ovr_stat)
print()

# 7. Dropped/Blocked Calls
df['drop_blk_tier'] = pd.qcut(df['drop_blk_Mean'].fillna(0), q=4, labels=['Low Q1', 'Med-Low Q2', 'Med-High Q3', 'High Q4'])
drop_stat = df.groupby('drop_blk_tier', observed=False)['churn'].agg(['count', 'mean']).assign(churn_rate=lambda x: (x['mean']*100).round(2))
print("=== 7. CHURN BY DROPPED & BLOCKED CALLS ===")
print(drop_stat)
print()

# 8. Customer Care Calls
df['cc_tier'] = pd.cut(df['custcare_Mean'].fillna(0), bins=[-1, 0, 2, 5, 1000], labels=['0 calls', '1-2 calls', '3-5 calls', '>5 calls'])
cc_stat = df.groupby('cc_tier', observed=False)['churn'].agg(['count', 'mean']).assign(churn_rate=lambda x: (x['mean']*100).round(2))
print("=== 8. CHURN BY CUSTOMER CARE CALLS ===")
print(cc_stat)
print()

# 9. Tenure (months)
df['tenure_bin'] = pd.cut(df['months'], bins=[0, 12, 24, 36, 100], labels=['<=12 mos', '13-24 mos', '25-36 mos', '>36 mos'])
tenure_stat = df.groupby('tenure_bin', observed=False)['churn'].agg(['count', 'mean']).assign(churn_rate=lambda x: (x['mean']*100).round(2))
print("=== 9. CHURN BY TENURE (MONTHS) ===")
print(tenure_stat)
print()

# 10. Financial Summary
print("=== 10. FINANCIAL BASELINE ===")
print(f"Mean Monthly Recurring Charge: ${df['totmrc_Mean'].mean():.2f}")
print(f"Mean Monthly Revenue: ${df['rev_Mean'].mean():.2f}")
print(f"Mean Overage Revenue: ${df['ovrrev_Mean'].mean():.2f}")
print(f"Total Churned Customers: {df['churn'].sum():,}")
print(f"Total Active Customers: {(1 - df['churn']).sum():,}")
avg_rev = df['rev_Mean'].mean()
churn_rev_lost_monthly = df[df['churn'] == 1]['rev_Mean'].sum()
print(f"Monthly Revenue Lost from Churned Customers: ${churn_rev_lost_monthly:,.2f}")
print(f"Annualized Revenue Lost: ${churn_rev_lost_monthly * 12:,.2f}")
