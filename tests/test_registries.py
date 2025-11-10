from src.mdm.registries import check_registry


def test_check_registry_website():
    r = check_registry("website", "https://example.com")
    assert r["match"] is True
    assert r["confidence"] >= 0.9


def test_check_registry_company():
    r = check_registry("company", "Acme Corporation")
    assert r["match"] is True
    assert 0.7 <= r["confidence"] <= 0.9


def test_check_registry_negative():
    r = check_registry("company", "X")
    assert r["match"] is False
    assert r["confidence"] == 0.0
