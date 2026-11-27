"""Standard normal CDF / inverse CDF with NumPy only (no SciPy dependency)."""
import math
import numpy as np

_erf = np.frompyfunc(math.erf, 1, 1)


def cdf(x):
    x = np.asarray(x, float)
    return 0.5 * (1.0 + _erf(x / math.sqrt(2.0)).astype(float))


# Acklam's rational approximation with one Newton refinement step (|error| < 1e-12)
_a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
      -3.066479806614716e+01, 2.506628277459239e+00]
_b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
      -1.328068155288572e+01]
_c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
      4.374664141464968e+00, 2.938163982698783e+00]
_d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]


def ppf(p):
    p = np.clip(np.asarray(p, float), 1e-300, 1 - 1e-16)
    x = np.empty_like(p)
    lo, hi = p < 0.02425, p > 1 - 0.02425
    mid = ~(lo | hi)
    q = np.sqrt(-2 * np.log(p[lo]))
    x[lo] = (((((_c[0]*q+_c[1])*q+_c[2])*q+_c[3])*q+_c[4])*q+_c[5]) / ((((_d[0]*q+_d[1])*q+_d[2])*q+_d[3])*q+1)
    q = np.sqrt(-2 * np.log(1 - p[hi]))
    x[hi] = -(((((_c[0]*q+_c[1])*q+_c[2])*q+_c[3])*q+_c[4])*q+_c[5]) / ((((_d[0]*q+_d[1])*q+_d[2])*q+_d[3])*q+1)
    q = p[mid] - 0.5; r = q * q
    x[mid] = (((((_a[0]*r+_a[1])*r+_a[2])*r+_a[3])*r+_a[4])*r+_a[5])*q / (((((_b[0]*r+_b[1])*r+_b[2])*r+_b[3])*r+_b[4])*r+1)
    e = cdf(x) - p
    u = e * math.sqrt(2 * math.pi) * np.exp(x * x / 2)
    return x - u / (1 + x * u / 2)
