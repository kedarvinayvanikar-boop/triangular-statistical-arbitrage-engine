from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest

from src.c_bindings import (
    benchmark_rolling_kernels,
    build_shared_library,
    residuals_c,
    residuals_python,
    rolling_ols_c,
    rolling_ols_python,
    rolling_ols_with_fallback,
    validate_c_against_python,
)
from src.database import connect_database, store_c_kernel_benchmark, store_c_kernel_test_results, store_c_kernel_validation


def sample_series(n: int = 80) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.linspace(0.0, 1.0, n)
    x1 = 5.0 + 0.03 * np.arange(n) + 0.02 * np.sin(np.arange(n) / 4.0)
    x2 = 3.0 + 0.02 * np.arange(n) + 0.03 * np.cos(np.arange(n) / 5.0)
    y = 0.4 + 0.8 * x1 - 0.25 * x2 + 0.01 * np.sin(8.0 * grid)
    return y.astype(float), x1.astype(float), x2.astype(float)


def test_python_rolling_kernel_matches_manual_lstsq():
    y, x1, x2 = sample_series(24)
    out = rolling_ols_python(y, x1, x2, window=8)
    train_x = np.column_stack((np.ones(8), x1[:8], x2[:8]))
    beta = np.linalg.lstsq(train_x, y[:8], rcond=None)[0]
    expected = beta[0] + beta[1] * x1[8] + beta[2] * x2[8]

    assert out.iloc[8]["alpha"] == pytest.approx(beta[0])
    assert out.iloc[8]["beta_1"] == pytest.approx(beta[1])
    assert out.iloc[8]["beta_2"] == pytest.approx(beta[2])
    assert out.iloc[8]["fitted_log_price"] == pytest.approx(expected)
    assert np.isnan(out.iloc[0]["residual"])


def test_fallback_uses_python_when_library_is_missing(tmp_path):
    y, x1, x2 = sample_series(30)
    result = rolling_ols_with_fallback(y, x1, x2, window=10, library_path=tmp_path / "missing.so")

    assert result.backend == "python"
    assert result.library_path is None
    assert not result.table.dropna(subset=["residual"]).empty


def test_residual_python_kernel_matches_formula():
    y, x1, x2 = sample_series(12)
    alpha = np.full(12, 0.4)
    beta1 = np.full(12, 0.8)
    beta2 = np.full(12, -0.25)

    out = residuals_python(y, x1, x2, alpha, beta1, beta2)

    expected = y - (alpha + beta1 * x1 + beta2 * x2)
    assert np.allclose(out["residual"], expected)


@pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("clang") is None, reason="C compiler unavailable")
def test_compiled_c_rolling_kernel_matches_python(tmp_path):
    y, x1, x2 = sample_series(90)
    lib_path = build_shared_library(output_path=tmp_path / "librolling_regression.so")

    validation = validate_c_against_python(y, x1, x2, window=20, library_path=lib_path)

    assert set(validation["output_column"]) == {"alpha", "beta_1", "beta_2", "fitted_log_price", "residual"}
    assert validation["max_abs_diff"].max() < 1e-7


@pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("clang") is None, reason="C compiler unavailable")
def test_compiled_c_residual_kernel_matches_python(tmp_path):
    y, x1, x2 = sample_series(50)
    lib_path = build_shared_library(output_path=tmp_path / "librolling_regression.so")
    rolling = rolling_ols_c(y, x1, x2, window=15, library_path=lib_path)

    py = residuals_python(y, x1, x2, rolling["alpha"], rolling["beta_1"], rolling["beta_2"])
    c = residuals_c(y, x1, x2, rolling["alpha"], rolling["beta_1"], rolling["beta_2"], library_path=lib_path)

    mask = np.isfinite(py["residual"]) & np.isfinite(c["residual"])
    assert np.max(np.abs(py.loc[mask, "residual"] - c.loc[mask, "residual"])) < 1e-10


def test_benchmark_and_database_store_functions(tmp_path):
    y, x1, x2 = sample_series(40)
    benchmark = benchmark_rolling_kernels(y, x1, x2, window=10, repeats=1, library_path=tmp_path / "missing.so")
    validation = pd.DataFrame(
        {
            "output_column": ["residual"],
            "max_abs_diff": [0.0],
            "mean_abs_diff": [0.0],
            "n_compared": [30],
        }
    )
    test_results = pd.DataFrame({"test_name": ["fallback"], "status": ["passed"], "detail": ["python backend"]})

    with connect_database(tmp_path / "c_bindings_test.db") as conn:
        store_c_kernel_benchmark(conn, benchmark.assign(sample_rows=40, window=10))
        store_c_kernel_validation(conn, validation)
        store_c_kernel_test_results(conn, test_results)
        stored = pd.read_sql_query("SELECT COUNT(*) AS n FROM c_kernel_test_results", conn)

    assert int(stored.loc[0, "n"]) == 1


def test_invalid_c_kernel_inputs_raise_clear_errors():
    y, x1, x2 = sample_series(10)
    with pytest.raises(ValueError):
        rolling_ols_python(y, x1, x2, window=3)
    with pytest.raises(ValueError):
        rolling_ols_python(y, x1[:-1], x2, window=5)
    with pytest.raises(ValueError):
        benchmark_rolling_kernels(y, x1, x2, window=5, repeats=0)
