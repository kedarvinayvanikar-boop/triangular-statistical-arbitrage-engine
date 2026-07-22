#ifndef RESIDUAL_KERNEL_H
#define RESIDUAL_KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif

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
);

#ifdef __cplusplus
}
#endif

#endif
