import pandas as pd

p = 'sample_data/query_group1_output.csv'
cols = ['SOURCE_NAME', 'mdm_verified_company', 'mdm_verified_website']

df = pd.read_csv(p)
# ensure columns exist
for c in cols:
    if c not in df.columns:
        df[c] = ''

print('Showing first 10 rows (SOURCE_NAME, mdm_verified_company, mdm_verified_website):\n')
print(df[cols].head(10).to_string(index=True))
