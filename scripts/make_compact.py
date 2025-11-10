import pandas as pd

in_path = 'sample_data/output_smoke.csv'
# fallback to default output.csv if smoke file not present
import os
if not os.path.exists(in_path):
    in_path = 'sample_data/output.csv'
out_path = 'sample_data/compact_smoke.csv'

df = pd.read_csv(in_path)

# prefer these input columns if present
key_inputs = ['id', 'name', 'SOURCE_NAME', 'SOURCE_KEY']
present_inputs = [c for c in key_inputs if c in df.columns]

# gather mdm_verified columns
mdm_cols = [c for c in df.columns if c.startswith('mdm_verified_')]

# If no mdm_verified columns present, fall back to common original fields
if mdm_cols:
    cols = present_inputs + mdm_cols
else:
    fallback = [c for c in ['company', 'website'] if c in df.columns]
    cols = present_inputs + fallback
    if not cols:
        cols = list(df.columns[:3])

compact = df[cols]
compact.to_csv(out_path, index=False)
print('Wrote', out_path)
print(compact.head(10).to_string(index=False))
