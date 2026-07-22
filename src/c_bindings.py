"""
Optional C kernels for the rolling regression computation, called through
ctypes, with a pure-Python (NumPy) fallback used automatically whenever
the compiled library isn't present.

Why this exists: rolling_ols in src/regression.py refits a fresh
regression at every single day across the whole price history -- for a
long history and many triplets that's a lot of repeated small linear
algebra solves. The C kernels do the identical math in a tight compiled
loop, purely as a performance path; `validate_c_against_python` exists to
prove the C and Python versions agree numerically before trusting the
faster one for anything.
"""
from __future__ import annotations

import ctypes
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class KernelRunResult:
    table: pd.DataFrame
    backend: str  # "c" or "python" -- which implementation actually produced this result
    library_path: Optional[Path]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_library_name() -> str:
    # Shared library file extensions differ by OS -- .dylib on macOS, .dll
    # on Windows, .so everywhere else (Linux and most other Unix-likes).
    system = platform.system().lower()
    if system == "darwin":
        return "librolling_regression.dylib"
    if system == "windows":
        return "rolling_regression.dll"
    return "librolling_regression.so"


def default_library_path() -> Path:
    return project_root() / "c_src" / default_library_name()


def build_shared_library(
    source_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    compiler: str | None = None,
) -> Path:
    # Compiles c_src/*.c into a shared library the Python side can load
    # with ctypes. -fPIC (position-independent code) is required for
    # anything destined to become a shared library on Linux; macOS uses
    # -dynamiclib instead of -shared as its linker flag for the same
    # purpose, hence the platform branch below.
    src_dir = Path(source_dir) if source_dir is not None else project_root() / "c_src"
    out_path = Path(output_path) if output_path is not None else default_library_path()
    compiler_name = compiler or shutil.which("gcc") or shutil.which("clang")
    if compiler_name is None:
        raise RuntimeError("no C compiler found")

    sources = [src_dir / "rolling_regression.c", src_dir / "residual_kernel.c"]
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing C sources: {missing}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    command = [compiler_name, "-O3", "-shared", "-fPIC", *map(str, sources), "-o", str(out_path), "-lm"]
    if platform.system().lower() == "darwin":
        command = [compiler_name, "-O3", "-dynamiclib", *map(str, sources), "-o", str(out_path), "-lm"]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return out_path


def load_library(library_path: str | Path | None = None) -> ctypes.CDLL:
    path = Path(library_path) if library_path is not None else default_library_path()
    if not path.exists():
        raise FileNotFoundError(f"compiled C library not found: {path}")
    lib = ctypes.CDLL(str(path))
    _configure_signatures(lib)
    return lib


def c_kernels_available(library_path: str | Path | None = None) -> bool:
    # A simple probe: try to load the library, and treat any failure as
    # "not available" rather than propagating the exception -- callers
    # that just want to know whether to attempt the C path can use this
    # without a try/except of their own.
    try:
        load_library(library_path)
        return True
    except (OSError, FileNotFoundError):
        return False


def rolling_ols_python(y: object, x1: object, x2: object, window: int) -> pd.DataFrame:
    # Reference implementation: identical math to src/regression.py's
    # rolling_ols, reimplemented here as a flat NumPy loop (rather than
    # calling into src/regression.py directly) so this module has a
    # self-contained "ground truth" to check the C kernel against,
    # independent of anything else in the codebase changing.
    y_arr, x1_arr, x2_arr = _prepare_three_vectors(y, x1, x2)
    _validate_window(y_arr.size, window)
    alpha = np.full(y_arr.size, np.nan, dtype=float)
    beta1 = np.full(y_arr.size, np.nan, dtype=float)
    beta2 = np.full(y_arr.size, np.nan, dtype=float)
    fitted = np.full(y_arr.size, np.nan, dtype=float)
    residual = np.full(y_arr.size, np.nan, dtype=float)

    for i in range(window, y_arr.size):
        train = np.column_stack((np.ones(window), x1_arr[i - window : i], x2_arr[i - window : i]))
        target = y_arr[i - window : i]
        mask = np.isfinite(train).all(axis=1) & np.isfinite(target)
        if int(mask.sum()) <= 3 or not np.isfinite([y_arr[i], x1_arr[i], x2_arr[i]]).all():
            continue
        beta = np.linalg.lstsq(train[mask], target[mask], rcond=None)[0]
        alpha[i] = beta[0]
        beta1[i] = beta[1]
        beta2[i] = beta[2]
        fitted[i] = beta[0] + beta[1] * x1_arr[i] + beta[2] * x2_arr[i]
        residual[i] = y_arr[i] - fitted[i]

    return _kernel_frame(alpha, beta1, beta2, fitted, residual, backend="python")


def rolling_ols_c(
    y: object,
    x1: object,
    x2: object,
    window: int,
    library_path: str | Path | None = None,
) -> pd.DataFrame:
    # Calls the compiled C function directly. Output arrays are allocated
    # here on the Python/NumPy side (pre-filled with NaN) and passed to C
    # as raw pointers for it to write into in place -- C never allocates
    # or returns memory back to Python here, which sidesteps having to
    # manage cross-language memory ownership.
    y_arr, x1_arr, x2_arr = _prepare_three_vectors(y, x1, x2)
    _validate_window(y_arr.size, window)
    lib = load_library(library_path)
    n = y_arr.size
    alpha = np.full(n, np.nan, dtype=np.float64)
    beta1 = np.full(n, np.nan, dtype=np.float64)
    beta2 = np.full(n, np.nan, dtype=np.float64)
    fitted = np.full(n, np.nan, dtype=np.float64)
    residual = np.full(n, np.nan, dtype=np.float64)

    lib.rolling_ols_2d(
        _ptr(y_arr),
        _ptr(x1_arr),
        _ptr(x2_arr),
        ctypes.c_int(n),
        ctypes.c_int(int(window)),
        _ptr(alpha),
        _ptr(beta1),
        _ptr(beta2),
        _ptr(fitted),
        _ptr(residual),
    )
    return _kernel_frame(alpha, beta1, beta2, fitted, residual, backend="c")


def rolling_ols_with_fallback(
    y: object,
    x1: object,
    x2: object,
    window: int,
    library_path: str | Path | None = None,
    prefer_c: bool = True,
) -> KernelRunResult:
    # The actual entry point most callers should use: try the compiled
    # library first (if requested), and silently drop back to the pure
    # -Python version if it's missing or fails to load -- so this project
    # runs correctly on a machine with no C compiler at all, just slower.
    if prefer_c:
        try:
            path = Path(library_path) if library_path is not None else default_library_path()
            table = rolling_ols_c(y, x1, x2, window, path)
            return KernelRunResult(table=table, backend="c", library_path=path)
        except (OSError, FileNotFoundError):
            pass
    return KernelRunResult(table=rolling_ols_python(y, x1, x2, window), backend="python", library_path=None)


def residuals_python(
    y: object,
    x1: object,
    x2: object,
    alpha: object,
    beta1: object,
    beta2: object,
) -> pd.DataFrame:
    # Given an already-computed coefficient path (from either backend
    # above), recomputes fitted values and residuals -- a separate,
    # cheaper kernel from the rolling fit itself, useful when the
    # coefficients are already known and only the residual needs
    # recalculating (e.g. against a different price series).
    y_arr, x1_arr, x2_arr = _prepare_three_vectors(y, x1, x2)
    a_arr, b1_arr, b2_arr = _prepare_three_vectors(alpha, beta1, beta2)
    if y_arr.size != a_arr.size:
        raise ValueError("coefficient arrays must match y length")
    fitted = a_arr + b1_arr * x1_arr + b2_arr * x2_arr
    mask = np.isfinite(y_arr) & np.isfinite(x1_arr) & np.isfinite(x2_arr) & np.isfinite(a_arr) & np.isfinite(b1_arr) & np.isfinite(b2_arr)
    fitted = np.where(mask, fitted, np.nan)
    residual = np.where(mask, y_arr - fitted, np.nan)
    return pd.DataFrame({"fitted_log_price": fitted, "residual": residual, "backend": "python"})


def residuals_c(
    y: object,
    x1: object,
    x2: object,
    alpha: object,
    beta1: object,
    beta2: object,
    library_path: str | Path | None = None,
) -> pd.DataFrame:
    y_arr, x1_arr, x2_arr = _prepare_three_vectors(y, x1, x2)
    a_arr, b1_arr, b2_arr = _prepare_three_vectors(alpha, beta1, beta2)
    if y_arr.size != a_arr.size:
        raise ValueError("coefficient arrays must match y length")
    lib = load_library(library_path)
    n = y_arr.size
    fitted = np.full(n, np.nan, dtype=np.float64)
    residual = np.full(n, np.nan, dtype=np.float64)
    lib.residuals_2d(
        _ptr(y_arr),
        _ptr(x1_arr),
        _ptr(x2_arr),
        _ptr(a_arr),
        _ptr(b1_arr),
        _ptr(b2_arr),
        ctypes.c_int(n),
        _ptr(fitted),
        _ptr(residual),
    )
    return pd.DataFrame({"fitted_log_price": fitted, "residual": residual, "backend": "c"})


def validate_c_against_python(
    y: object,
    x1: object,
    x2: object,
    window: int,
    library_path: str | Path | None = None,
) -> pd.DataFrame:
    # Runs both implementations on the same input and reports the
    # max/mean absolute difference per output column -- this is the
    # actual proof that the C kernel is correct, not just fast. Small
    # nonzero differences (floating-point rounding between C's and
    # NumPy's arithmetic) are expected; a large difference would indicate
    # a real bug in one implementation or the other.
    python_table = rolling_ols_python(y, x1, x2, window)
    c_table = rolling_ols_c(y, x1, x2, window, library_path)
    rows = []
    for column in ["alpha", "beta_1", "beta_2", "fitted_log_price", "residual"]:
        diff = np.abs(python_table[column].to_numpy(dtype=float) - c_table[column].to_numpy(dtype=float))
        rows.append(
            {
                "output_column": column,
                "max_abs_diff": float(np.nanmax(diff)),
                "mean_abs_diff": float(np.nanmean(diff)),
                "n_compared": int(np.isfinite(diff).sum()),
            }
        )
    return pd.DataFrame(rows)


def benchmark_rolling_kernels(
    y: object,
    x1: object,
    x2: object,
    window: int,
    repeats: int = 5,
    library_path: str | Path | None = None,
) -> pd.DataFrame:
    # Times both backends over several repeats, reporting mean/min/max --
    # the actual evidence for whether the C kernel is worth the added
    # build complexity for a given data size, rather than assuming it is.
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    rows = []
    for backend, runner in [("python", rolling_ols_python), ("c", lambda a, b, c, w: rolling_ols_c(a, b, c, w, library_path))]:
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            try:
                runner(y, x1, x2, window)
            except (OSError, FileNotFoundError):
                timings = []
                break
            timings.append(time.perf_counter() - start)
        if timings:
            rows.append(
                {
                    "backend": backend,
                    "repeats": int(repeats),
                    "mean_seconds": float(np.mean(timings)),
                    "min_seconds": float(np.min(timings)),
                    "max_seconds": float(np.max(timings)),
                }
            )
    return pd.DataFrame(rows)


def _configure_signatures(lib: ctypes.CDLL) -> None:
    # ctypes doesn't know a C function's parameter/return types unless
    # told explicitly -- without this, ctypes would default to treating
    # every argument as a plain C int, which is wrong for our double*
    # pointers and would corrupt memory or crash. argtypes/restype are
    # how each function's actual C signature gets declared to ctypes.
    double_ptr = ctypes.POINTER(ctypes.c_double)
    lib.rolling_ols_2d.argtypes = [
        double_ptr,
        double_ptr,
        double_ptr,
        ctypes.c_int,
        ctypes.c_int,
        double_ptr,
        double_ptr,
        double_ptr,
        double_ptr,
        double_ptr,
    ]
    lib.rolling_ols_2d.restype = ctypes.c_int
    lib.residuals_2d.argtypes = [
        double_ptr,
        double_ptr,
        double_ptr,
        double_ptr,
        double_ptr,
        double_ptr,
        ctypes.c_int,
        double_ptr,
        double_ptr,
    ]
    lib.residuals_2d.restype = ctypes.c_int


def _prepare_three_vectors(a: object, b: object, c: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # np.ascontiguousarray matters here specifically because these arrays
    # get passed to C as raw pointers (_ptr below) -- a non-contiguous
    # NumPy array (e.g. a slice/view with custom strides) would have C
    # read memory in the wrong order or out of bounds, since C has no
    # concept of NumPy's stride metadata.
    arrays = []
    for value in (a, b, c):
        arr = np.ascontiguousarray(np.asarray(value, dtype=np.float64).reshape(-1))
        arrays.append(arr)
    if len({arr.size for arr in arrays}) != 1:
        raise ValueError("arrays must have the same length")
    return arrays[0], arrays[1], arrays[2]


def _validate_window(n: int, window: int) -> None:
    if window <= 3:
        raise ValueError("window must exceed the three regression parameters")
    if window >= n:
        raise ValueError("window must be smaller than the sample length")


def _ptr(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_double):
    # Gets a raw C-compatible pointer to a NumPy array's underlying
    # memory buffer -- this is the actual handoff point where Python
    # data becomes directly visible to the C function as a double*.
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _kernel_frame(
    alpha: np.ndarray,
    beta1: np.ndarray,
    beta2: np.ndarray,
    fitted: np.ndarray,
    residual: np.ndarray,
    backend: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row": np.arange(alpha.size, dtype=int),
            "alpha": alpha,
            "beta_1": beta1,
            "beta_2": beta2,
            "fitted_log_price": fitted,
            "residual": residual,
            "backend": backend,
        }
    )
