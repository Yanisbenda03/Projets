import numpy as np
import pandas as pd

from workflow import _impute_and_scale


def test_impute_and_scale_mean_strategy_produces_zero_mean_unit_var():
    df = pd.DataFrame(
        {
            "a": [1.0, np.nan, 5.0, 7.0],
            "b": [2.0, 4.0, np.nan, 8.0],
        }
    )

    X_scaled, imputer, scaler = _impute_and_scale(df, strategy="mean")

    imputed = imputer.transform(df)
    expected_means = np.array([
    sum(x for x in df[col].to_numpy() if x == x) /
    sum(1 for x in df[col].to_numpy() if x == x)
    for col in df.columns
    ])

    assert np.allclose(imputer.statistics_, expected_means)
    assert np.allclose(imputed.mean(axis=0), expected_means)

    # Le StandardScaler doit produire des colonnes centrées réduites
    assert np.allclose(X_scaled.mean(axis=0), np.zeros(df.shape[1]), atol=1e-9)
    assert np.allclose(X_scaled.var(axis=0), np.ones(df.shape[1]), atol=1e-9)

