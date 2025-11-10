import pandas as pd
import sys

p = 'sample_data/query_group1_output.csv'
try:
    df = pd.read_csv(p)
except Exception as e:
    print('Error reading', p, e)
    sys.exit(2)

print('Total rows (excluding header):', len(df))
cols = ['mdm_verified_company','mdm_verified_website','company','website','SOURCE_NAME']
for col in cols:
    if col in df.columns:
        non_empty = df[col].fillna('').astype(str).str.strip().astype(bool).sum()
        print(f"{col}: non-empty count = {non_empty}")
    else:
        print(f"{col}: (missing)")

print('\nFirst 30 columns in output:')
print(', '.join(df.columns[:30]))
