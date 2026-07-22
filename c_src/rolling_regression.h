#ifndef ROLLING_REGRESSION_H
#define ROLLING_REGRESSION_H

#ifdef __cplusplus
extern "C" {
#endif

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
);

#ifdef __cplusplus
}
#endif

#endif
