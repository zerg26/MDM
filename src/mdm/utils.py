from typing import List, Dict, Any
import pandas as pd
from urllib.parse import urlparse, urlunparse


def normalize_text(val: str) -> str:
    """Normalize a text field: strip, collapse whitespace."""
    if val is None:
        return ""
    if not isinstance(val, str):
        val = str(val)
    return " ".join(val.strip().split())


def normalize_website(val: str) -> str:
    """Create a normalized website URL for comparison: ensure scheme, lowercase netloc, strip query/fragments, remove trailing slash."""
    if val is None:
        return ""
    val = str(val).strip()
    if val == "":
        return ""
    # ensure scheme
    parsed = urlparse(val)
    if not parsed.netloc:
        # maybe missing scheme
        parsed = urlparse("http://" + val)
    scheme = parsed.scheme or "http"
    netloc = (parsed.netloc or "").lower()
    path = (parsed.path or "").rstrip("/")
    return urlunparse((scheme, netloc, path, "", "", ""))


def normalize_company(val: str) -> str:
    """Normalize company names for comparison:

    - strip leading/trailing whitespace
    - remove common company suffixes (Inc, LLC, Ltd, Corp, Co, Corporation)
    - collapse punctuation and extra spaces
    - return title-cased-ish (preserve capitalization minimally)
    """
    if val is None:
        return ""
    s = str(val).strip()
    if s == "":
        return ""
    # remove common suffixes
    # Note: intentionally omit 'corp' from removable suffixes to preserve names like 'Acme Corp'
    suffixes = ["inc", "llc", "ltd", "co", "company", "corporation"]
    # split on whitespace and remove common punctuation from tokens
    raw_tokens = s.replace(".", " ").split()
    parts = [p.strip(',.') for p in raw_tokens if p.strip(',.')]
    # drop trailing suffixes
    while parts and parts[-1].lower().strip(',') in suffixes:
        parts.pop()
    core = " ".join(parts)
    # collapse whitespace
    core = " ".join(core.split())
    # return a cleaned, title-cased value for display/use
    return core.title()


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def rows_from_df(df: pd.DataFrame) -> List[Dict[str, Any]]:
    # Convert NaN/NA values to empty strings so downstream logic treats them as missing
    clean = df.fillna("")
    return clean.to_dict(orient="records")


def merge_result(row: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    # Overlay result values onto row
    out = dict(row)
    out.update(result)
    return out
