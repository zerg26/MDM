import pytest
from src.mdm.utils import normalize_text, normalize_website, rows_from_df
import pandas as pd


def test_normalize_text():
    assert normalize_text('  Acme   Corp \n') == 'Acme Corp'
    assert normalize_text(None) == ''
    assert normalize_text(123) == '123'


def test_normalize_website():
    assert normalize_website('https://Example.COM/') == 'https://example.com'
    assert normalize_website('example.com/path/') == 'http://example.com/path'
    assert normalize_website('') == ''


def test_rows_from_df_handles_nans():
    df = pd.DataFrame({'id': [1], 'name': ['Acme'], 'company': [pd.NA], 'website': [pd.NA]})
    rows = rows_from_df(df)
    assert rows[0]['company'] == ''
    assert rows[0]['website'] == ''
