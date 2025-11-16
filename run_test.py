#!/usr/bin/env python3
"""Quick test of the address verification pipeline."""
import sys
import os

# Add project to path
sys.path.insert(0, '/home/doghouse/MDM')
os.chdir('/home/doghouse/MDM')

from dotenv import load_dotenv
load_dotenv()

from src.cli import run_pipeline

# Run pipeline
input_csv = "sample_data/input_address.csv"
output_csv = "sample_data/output_address.csv"
report_path = "sample_data/comparison_report.txt"

print(f"Running pipeline on {input_csv}...")
print(f"Output will be written to {output_csv}")
print(f"Report will be written to {report_path}")

run_pipeline(
    input_path=input_csv,
    output_path=output_csv,
    report_path=report_path,
    use_multi_query=True
)

print("\nPipeline complete!")
print(f"\nOutput CSV: {output_csv}")
print(f"Report: {report_path}")

# Show first few lines of output
import pandas as pd
df = pd.read_csv(output_csv)
print(f"\nOutput CSV shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst row:")
print(df.iloc[0].to_dict())
