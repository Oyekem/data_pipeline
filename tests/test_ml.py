from ml_model import create_features
import pandas as pd

def test_features():
    df = pd.DataFrame({
        "price": [1,2,3,4,5,6,7]
    })

    out = create_features(df)

    assert "lag1" in out.columns
    assert "ma5" in out.columns