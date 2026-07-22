#include "rolling_regression.h"

#include <math.h>
#include <stddef.h>

static int finite_triplet(double a, double b, double c) {
    return isfinite(a) && isfinite(b) && isfinite(c);
}

static int solve_3x3(double a[3][3], double b[3], double out[3]) {
    int i, j, k, pivot;
    double max_abs, tmp, factor;

    for (i = 0; i < 3; ++i) {
        pivot = i;
        max_abs = fabs(a[i][i]);
        for (j = i + 1; j < 3; ++j) {
            tmp = fabs(a[j][i]);
            if (tmp > max_abs) {
                max_abs = tmp;
                pivot = j;
            }
        }
        if (max_abs < 1e-12) {
            return 1;
        }
        if (pivot != i) {
            for (k = i; k < 3; ++k) {
                tmp = a[i][k];
                a[i][k] = a[pivot][k];
                a[pivot][k] = tmp;
            }
            tmp = b[i];
            b[i] = b[pivot];
            b[pivot] = tmp;
        }
        for (j = i + 1; j < 3; ++j) {
            factor = a[j][i] / a[i][i];
            a[j][i] = 0.0;
            for (k = i + 1; k < 3; ++k) {
                a[j][k] -= factor * a[i][k];
            }
            b[j] -= factor * b[i];
        }
    }

    for (i = 2; i >= 0; --i) {
        tmp = b[i];
        for (j = i + 1; j < 3; ++j) {
            tmp -= a[i][j] * out[j];
        }
        out[i] = tmp / a[i][i];
    }
    return 0;
}

int rolling_ols_2d(
    const double *y,
    const double *x1,
    const double *x2,
    int n,
    int window,
    double *out_alpha,
    double *out_beta1,
    double *out_beta2,
    double *out_fitted,
    double *out_residual
) {
    int i, j, invalid_count;

    if (y == NULL || x1 == NULL || x2 == NULL || out_alpha == NULL || out_beta1 == NULL ||
        out_beta2 == NULL || out_fitted == NULL || out_residual == NULL) {
        return 1;
    }
    if (n <= 0 || window <= 3 || window >= n) {
        return 2;
    }

    for (i = 0; i < n; ++i) {
        out_alpha[i] = NAN;
        out_beta1[i] = NAN;
        out_beta2[i] = NAN;
        out_fitted[i] = NAN;
        out_residual[i] = NAN;
    }

    invalid_count = 0;
    for (i = window; i < n; ++i) {
        double xtx[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
        double xty[3] = {0.0, 0.0, 0.0};
        double beta[3] = {0.0, 0.0, 0.0};
        int valid_rows = 0;

        for (j = i - window; j < i; ++j) {
            double row[3];
            if (!finite_triplet(y[j], x1[j], x2[j])) {
                continue;
            }
            row[0] = 1.0;
            row[1] = x1[j];
            row[2] = x2[j];
            xtx[0][0] += row[0] * row[0];
            xtx[0][1] += row[0] * row[1];
            xtx[0][2] += row[0] * row[2];
            xtx[1][1] += row[1] * row[1];
            xtx[1][2] += row[1] * row[2];
            xtx[2][2] += row[2] * row[2];
            xty[0] += row[0] * y[j];
            xty[1] += row[1] * y[j];
            xty[2] += row[2] * y[j];
            valid_rows += 1;
        }

        xtx[1][0] = xtx[0][1];
        xtx[2][0] = xtx[0][2];
        xtx[2][1] = xtx[1][2];

        if (valid_rows <= 3 || solve_3x3(xtx, xty, beta) != 0 || !finite_triplet(y[i], x1[i], x2[i])) {
            invalid_count += 1;
            continue;
        }

        out_alpha[i] = beta[0];
        out_beta1[i] = beta[1];
        out_beta2[i] = beta[2];
        out_fitted[i] = beta[0] + beta[1] * x1[i] + beta[2] * x2[i];
        out_residual[i] = y[i] - out_fitted[i];
    }

    return invalid_count;
}
