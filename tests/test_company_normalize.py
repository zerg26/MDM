from src.mdm.utils import normalize_company


def test_normalize_company_basic():
    assert normalize_company('  Acme Corp, Inc.  ') == 'Acme Corp'
    assert normalize_company('globex llc') == 'Globex'
    assert normalize_company('ACME COMPANY') == 'Acme'


def test_normalize_company_empty():
    assert normalize_company(None) == ''
    assert normalize_company('') == ''
