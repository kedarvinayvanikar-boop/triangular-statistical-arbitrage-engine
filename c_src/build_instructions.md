# C kernel build instructions

The C kernels are optional. The Python code falls back to NumPy when the shared library is not available.

Build from the repository root on macOS or Linux:

```bash
gcc -O3 -shared -fPIC \
  c_src/rolling_regression.c \
  c_src/residual_kernel.c \
  -o c_src/librolling_regression.so \
  -lm
```

On macOS, `.dylib` can also be used:

```bash
gcc -O3 -dynamiclib \
  c_src/rolling_regression.c \
  c_src/residual_kernel.c \
  -o c_src/librolling_regression.dylib \
  -lm
```

The test suite builds a temporary shared object when a C compiler is available. If compilation is unavailable, the project uses the Python fallback path.
