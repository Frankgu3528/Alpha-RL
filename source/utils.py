# utils.py
"""
Utility functions, including safe math operations.
"""
import numpy as np

# Safe math functions
def safe_log(x):
    """Safe logarithm that handles zeros and negative numbers"""
    x_arr = np.asarray(x) # Ensure input is numpy array
    return np.log(np.maximum(x_arr, 1e-10))

def safe_exp(x):
    """Safe exponential that prevents overflow"""
    x_arr = np.asarray(x)
    return np.exp(np.clip(x_arr, -20, 20))

def safe_div(a, b):
    """Safe division that handles division by zero and NaNs"""
    a = np.asarray(a)
    b = np.asarray(b)
    out = np.full_like(a, np.nan, dtype=np.float64) # Initialize with NaN
    # Valid condition: b is not zero, not NaN, not Inf AND a is not NaN, not Inf
    valid_mask = (b != 0) & (~np.isnan(b)) & (~np.isinf(b)) & (~np.isnan(a)) & (~np.isinf(a))
    if np.any(valid_mask): # Check if there's anything to divide
        np.divide(a[valid_mask], b[valid_mask], out=out[valid_mask])
    return out

def safe_sqrt(x):
    """Safe square root that handles negative numbers"""
    x_arr = np.asarray(x)
    return np.sqrt(np.maximum(x_arr, 0))

def safe_power2(x):
    """Square with overflow protection"""
    x_arr = np.asarray(x)
    clipped_x = np.clip(x_arr, -1e4, 1e4) # Adjust clip range if necessary
    return np.square(clipped_x)

def normalize_series(x):
    """Z-score normalization with safeguards for stability"""
    if isinstance(x, np.ndarray) and x.size > 1:
        try:
            valid_x = x[np.isfinite(x)] # Use only finite values for stats
            if valid_x.size > 1:
                std = np.std(valid_x)
                if std > 1e-9: # Increased tolerance slightly
                    mean = np.mean(valid_x)
                    # Apply normalization only to finite values, keep others NaN
                    finite_mask = np.isfinite(x)
                    result = np.full_like(x, np.nan, dtype=np.float64)
                    result[finite_mask] = (x[finite_mask] - mean) / std
                    return result
        except Exception:
            pass # Return original array if any error occurs
    # Return original if not a suitable numpy array or normalization failed/not needed
    return x

def handle_inf_nan(arr):
    """Replaces inf and -inf with NaN in a numpy array."""
    if isinstance(arr, np.ndarray):
        arr[np.isinf(arr)] = np.nan
    return arr