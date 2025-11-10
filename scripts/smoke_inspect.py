import pandas as pd
p='sample_data/output_smoke.csv'
try:
    df=pd.read_csv(p)
except Exception as e:
    print('ERROR reading',p,e)
    raise
cols=['SOURCE_NAME','mdm_verified_company','mdm_verified_website']
for c in cols:
    if c not in df.columns:
        df[c]=''
print('\nFirst 5 rows:')
print(df[cols].head(5).to_string(index=True))
print('\nCounts:')
for c in cols:
    print(f"{c}:", df[c].fillna('').astype(str).str.strip().astype(bool).sum())
