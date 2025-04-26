# evaluation.py
"""
Function for evaluating factor performance on different data splits.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import warnings

# Import environment class and config for evaluation settings
from environment import FactorEnv # Need to instantiate env for evaluation
from config import EVAL_MIN_POINTS, MAX_DEPTH # Use same depth for consistency


def evaluate_factor_on_split(factor_expr, data_split):
    """
    Evaluates a given factor expression string on a specific data split.

    Args:
        factor_expr (str): The mathematical expression of the factor.
        data_split (pd.DataFrame): The data (train, val, or test) to evaluate on.

    Returns:
        dict: A dictionary containing evaluation statistics (IC, p-value, etc.)
              or None if evaluation fails.
    """
    if factor_expr is None or data_split is None or data_split.empty:
        return None

    # Create a temporary environment specifically for this evaluation split
    # This ensures evaluate_tree uses the correct data and settings
    try:
        eval_env = FactorEnv(data_split) # Uses config settings like MAX_DEPTH internally
        eval_env.tree = factor_expr # Set the tree to evaluate
    except ValueError as e:
        warnings.warn(f"Failed to create eval env for split: {e}")
        return None

    try:
        # Evaluate the tree using the environment's method
        factor_values = eval_env.evaluate_tree()

        if factor_values is None or not isinstance(factor_values, np.ndarray):
            # warnings.warn(f"Factor evaluation returned None or non-array for: {factor_expr}")
            return None

        # Get returns from the split
        returns = data_split['Return'].values

        # Align and clean (handle NaNs/Infs)
        valid_mask = np.isfinite(factor_values) & np.isfinite(returns)
        num_valid = valid_mask.sum()

        if num_valid < EVAL_MIN_POINTS:
            # warnings.warn(f"Insufficient valid points ({num_valid}) for reliable eval: {factor_expr}")
            return {'error': 'insufficient_points', 'valid_points': num_valid}

        valid_factors = factor_values[valid_mask]
        valid_returns = returns[valid_mask]

        # Winsorize factor values (on valid data only) before correlation
        try:
            lower = np.percentile(valid_factors, 1)
            upper = np.percentile(valid_factors, 99)
            # Avoid clipping if bounds are identical (can happen with low variance)
            if upper > lower:
                 clipped_factors = np.clip(valid_factors, lower, upper)
            else:
                 clipped_factors = valid_factors # No clipping needed/possible
        except IndexError: # Can happen if valid_factors is somehow empty despite check
            warnings.warn("IndexError during percentile calculation.")
            return {'error': 'percentile_error', 'valid_points': num_valid}


        # Check variability again after potential clipping
        factor_std = np.std(clipped_factors)
        if factor_std < 1e-7:
            # warnings.warn(f"Factor std dev too low ({factor_std:.2e}) after clipping: {factor_expr}")
            return {'error': 'low_variance', 'valid_points': num_valid, 'factor_std': factor_std}

        # Calculate correlations
        try:
             spearman_corr, p_spearman = spearmanr(clipped_factors, valid_returns)
             pearson_corr, p_pearson = pearsonr(clipped_factors, valid_returns)
        except ValueError as e: # Catches potential errors in correlation functions
             warnings.warn(f"Correlation calculation error: {e}")
             return {'error': 'corr_error', 'valid_points': num_valid}


        # Check if correlation results are valid numbers
        if np.isnan(spearman_corr) or np.isnan(pearson_corr):
             # warnings.warn(f"NaN correlation result for: {factor_expr}")
             return {'error': 'nan_correlation', 'valid_points': num_valid}

        # Return successful evaluation statistics
        return {
            'spearman_ic': spearman_corr,
            'p_spearman': p_spearman,
            'pearson_corr': pearson_corr,
            'p_pearson': p_pearson,
            'valid_points': num_valid,
            'factor_std': factor_std,
            'error': None # Explicitly state no error
        }

    except Exception as e:
        warnings.warn(f"Unexpected error during evaluation of '{factor_expr}': {e}")
        return {'error': f'unexpected: {str(e)}'} # Return error details