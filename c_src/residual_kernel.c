#include "residual_kernel.h"

#include <math.h>
#include <stddef.h>

int residuals_2d(
    const double *y,
    const double *x1,
    const double *x2,
    const double *alpha,
    const double *beta1,
    const double *beta2,
    int n,
    double *out_fitted,
    double *out_residual
) {
    int i;

    if (y == NULL || x1 == NULL || x2 == NULL || alpha == NULL || beta1 == NULL ||
        beta2 == NULL || out_fitted == NULL || out_residual == NULL) {
        return 1;
    }
    if (n <= 0) {
        return 2;
    }

    for (i = 0; i < n; ++i) {
        if (!isfinite(y[i]) || !isfinite(x1[i]) || !isfinite(x2[i]) || !isfinite(alpha[i]) ||
            !isfinite(beta1[i]) || !isfinite(beta2[i])) {
            out_fitted[i] = NAN;
            out_residual[i] = NAN;
            continue;
        }
        out_fitted[i] = alpha[i] + beta1[i] * x1[i] + beta2[i] * x2[i];
        out_residual[i] = y[i] - out_fitted[i];
    }
    return 0;
}
