import numpy as np
import pytest

from src.ridge import ridge_fit, ridge_predict, rolling_ridge


def test_ridge_with_zero_penalty_matches_linear_solution():
    x = np.array(
        [
            [1.0, 2.0],
            [2.0, 0.5],
            [3.0, 4.0],
            [4.0, 1.0],
            [5.0, 3.0],
        ]
    )
    y = 0.7 + 1.4 * x[:, 0] - 0.8 * x[:, 1]

    result = ridge_fit(x, y, alpha=0.0)

    assert result.intercept == pytest.approx(0.7)
    assert result.coefficients[0] == pytest.approx(1.4)
    assert result.coefficients[1] == pytest.approx(-0.8)
    assert np.max(np.abs(result.residuals)) < 1e-10


def test_ridge_shrinks_coefficients_under_collinearity():
    x1 = np.linspace(1.0, 10.0, 50)
    x2 = x1 + 0.01 * np.sin(x1)
    x = np.column_stack((x1, x2))
    y = 0.2 + 2.0 * x1 - 1.5 * x2

    ols_like = ridge_fit(x, y, alpha=0.0)
    ridge = ridge_fit(x, y, alpha=20.0)

    assert np.linalg.norm(ridge.coefficients) < np.linalg.norm(ols_like.coefficients)


def test_ridge_predict_rejects_bad_dimensions():
    with pytest.raises(ValueError):
        ridge_predict(np.ones((4, 3)), 0.0, [1.0, 2.0])


def test_ridge_rejects_negative_alpha():
    with pytest.raises(ValueError):
        ridge_fit(np.ones((3, 2)), np.ones(3), alpha=-1.0)


def test_rolling_ridge_uses_only_prior_rows():
    import pandas as pd

    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    h1 = np.arange(1.0, 9.0)
    h2 = np.array([2.0, 1.5, 2.5, 3.0, 3.5, 3.7, 4.0, 4.4])
    target = 1.0 + 0.5 * h1 + 0.25 * h2
    frame = pd.DataFrame({"A": target, "B": h1, "C": h2}, index=idx)

    out = rolling_ridge(frame, "A", ["B", "C"], window=5, alpha=0.0)
    manual = ridge_fit(frame.iloc[:5][["B", "C"]], frame.iloc[:5]["A"], alpha=0.0)
    expected = ridge_predict(frame.iloc[[5]][["B", "C"]], manual.intercept, manual.coefficients)[0]

    assert out.iloc[0]["date"] == idx[5]
    assert out.iloc[0]["fitted_log_price"] == pytest.approx(expected)
    assert out.iloc[0]["train_end"] == idx[4]
