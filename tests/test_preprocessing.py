import pandas as pd
from src.data.preprocess import preprocess

def test_preprocess_removes_duplicates():
    df = pd.DataFrame({"text": ["hello", "hello", "world"], "label": [0, 0, 1]})
    result = preprocess(df)
    assert len(result) == 2

def test_preprocess_removes_nulls():
    df = pd.DataFrame({"text": ["hello", None, "world"], "label": [0, 1, 2]})
    result = preprocess(df)
    assert result["text"].isna().sum() == 0

def test_preprocess_strips_whitespace():
    df = pd.DataFrame({"text": ["  hello  ", "world  "], "label": [0, 1]})
    result = preprocess(df)
    assert result["text"].iloc[0] == "hello"