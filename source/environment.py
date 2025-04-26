# environment.py
"""
Defines the Reinforcement Learning environment for factor discovery.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import warnings

# Import necessary components from other modules
from utils import safe_log, safe_exp, safe_div, safe_sqrt, safe_power2, normalize_series, handle_inf_nan
from config import FEATURES, OPERATORS, CONSTANTS, MAX_DEPTH, MAX_OPERATIONS, EVAL_MIN_POINTS

class FactorEnv:
    """
    RL Environment for building and evaluating factor expressions.
    Operates on a specific subset of data (train, val, or test).
    """
    def __init__(self, data_subset):
        if data_subset is None or data_subset.empty:
             raise ValueError("FactorEnv received an empty or None data_subset.")
        self.data = data_subset.copy()
        self.max_depth = MAX_DEPTH
        self.max_operations = MAX_OPERATIONS
        self.features = FEATURES
        self.operators = OPERATORS
        self.constants = CONSTANTS
        self.action_dim = len(self.features) + len(self.constants) + len(self.operators)

        if 'Return' not in self.data.columns:
             raise ValueError("The data_subset provided to FactorEnv must contain a 'Return' column.")
        # Pre-calculate returns as numpy array for faster access in reward calculation
        self.returns_np = self.data['Return'].values

        self.reset()

    def reset(self):
        """Resets the environment to start building a new expression."""
        self.tree = None
        self.current_depth = 0
        self.done = False
        self.operations_count = 0
        return self.get_state()

    def get_state(self):
        """Returns the current state of the environment."""
        if self.tree is None:
            # depth, tree_exists (0=No), normalized_ops_count
            return [0, 0, 0]
        else:
            # State: normalized_depth, tree_exists (1=Yes), normalized_ops_count
            norm_depth = self.current_depth / self.max_depth
            norm_ops = self.operations_count / self.max_operations
            return [min(norm_depth, 1.0), 1, min(norm_ops, 1.0)] # Keep state normalized

    def step(self, action):
        """Performs one step in the environment based on the agent's action."""
        reward = 0
        op_applied = False

        total_features = len(self.features)
        total_constants = len(self.constants)
        total_operators = len(self.operators)

        if not (0 <= action < self.action_dim):
             warnings.warn(f"Invalid action received: {action}. Assigning penalty.")
             reward = -0.2
             self.done = True # Terminate if action is fundamentally wrong
             return self.get_state(), reward, self.done

        if self.tree is None:
            # First step: Must start with a feature or constant
            if action < total_features:
                self.tree = self.features[action]
                op_applied = True
            elif action < total_features + total_constants:
                self.tree = str(self.constants[action - total_features])
                op_applied = True
            else: # Invalid first action (operator)
                reward = -0.15 # Penalty for invalid first action
                # Fallback: randomly start with a feature
                self.tree = np.random.choice(self.features)
                op_applied = True # We forced a start
        else:
            # Subsequent steps: Combine or apply operator
            if action < total_features: # Combine with feature using binary op
                feature = self.features[action]
                bin_op = np.random.choice(['+', '-', '*', 'safe_div']) # Use safe_div directly
                if np.random.random() < 0.5:
                    self.tree = f"({self.tree} {bin_op} {feature})"
                else:
                    self.tree = f"({feature} {bin_op} {self.tree})"
                self.operations_count += 1
                op_applied = True
            elif action < total_features + total_constants: # Combine with constant using binary op
                constant_str = str(self.constants[action - total_features])
                bin_op = np.random.choice(['+', '-', '*', 'safe_div'])
                if np.random.random() < 0.5:
                    self.tree = f"({self.tree} {bin_op} {constant_str})"
                else:
                    self.tree = f"({constant_str} {bin_op} {self.tree})"
                self.operations_count += 1
                op_applied = True
            else: # Apply an operator
                op_idx = action - (total_features + total_constants)
                operator_symbol = self.operators[op_idx]

                # Map operator symbols to safe function names or numpy functions
                op_map = {
                    'log': 'safe_log', 'exp': 'safe_exp', 'sqrt': 'safe_sqrt',
                    'abs': 'np.abs', '**2': 'safe_power2',
                    '+': '+', '-': '-', '*': '*', '/': 'safe_div' # Binary ops handled below
                }

                if operator_symbol in ['log', 'exp', 'sqrt', 'abs', '**2']:
                    safe_op_name = op_map[operator_symbol]
                    self.tree = f"{safe_op_name}({self.tree})"
                    self.operations_count += 1
                    op_applied = True
                elif operator_symbol in ['+', '-', '*', '/']:
                    # Apply binary operator by adding a *new* random operand
                    safe_op_name = op_map[operator_symbol]
                    if np.random.random() < 0.5: # Add constant
                        operand = str(np.random.choice(self.constants))
                    else: # Add feature
                        operand = np.random.choice(self.features)

                    if np.random.random() < 0.5:
                        self.tree = f"({self.tree} {safe_op_name} {operand})"
                    else:
                        self.tree = f"({operand} {safe_op_name} {self.tree})"
                    self.operations_count += 2 # Binary op adds complexity
                    op_applied = True
                else:
                    # Should not happen if config lists are correct
                    warnings.warn(f"Unhandled operator symbol: {operator_symbol}")
                    reward = -0.1

        # Update depth if an operation was successfully applied
        if op_applied:
            self.current_depth += 1

        # Check termination conditions
        if self.current_depth >= self.max_depth or self.operations_count >= self.max_operations:
            self.done = True

        # --- Calculate final reward if done ---
        if self.done:
            final_reward = self._calculate_final_reward()
            reward = final_reward # Override intermediate reward

        # Ensure reward is a scalar float
        reward = float(reward) if isinstance(reward, (float, int, np.number)) else -1.0

        return self.get_state(), reward, self.done

    def _calculate_final_reward(self):
        """Calculates the reward based on the finished expression tree."""
        try:
            factor_values = self.evaluate_tree()

            if factor_values is None or not isinstance(factor_values, np.ndarray):
                return -1.0 # Penalty for evaluation failure

            # Use pre-calculated numpy returns for efficiency
            returns = self.returns_np

            # Align and clean data (handle NaNs/Infs)
            valid_mask = np.isfinite(factor_values) & np.isfinite(returns)
            if valid_mask.sum() < EVAL_MIN_POINTS: # Check if enough valid points
                return -0.7 # Penalty for insufficient valid data

            valid_factors = factor_values[valid_mask]
            valid_returns = returns[valid_mask]

            # Winsorize factor values (on valid data only)
            lower_bound = np.percentile(valid_factors, 1)
            upper_bound = np.percentile(valid_factors, 99)
            clipped_factors = np.clip(valid_factors, lower_bound, upper_bound)

            # Check for sufficient variability
            if np.std(clipped_factors) < 1e-7:
                return -0.6 # Penalize constant-like factors

            # Calculate Spearman correlation
            corr, p_value = spearmanr(clipped_factors, valid_returns)

            if np.isnan(corr):
                return -0.5 # Penalize if correlation is NaN

            # Base reward is absolute correlation
            reward = abs(corr)

            # Optional: Complexity Penalty (simple example)
            # complexity_penalty = 0.005 * self.operations_count
            # reward = max(0, reward - complexity_penalty)

            return reward

        except Exception as e:
            # warnings.warn(f"Reward calculation error: {e} for tree: {self.tree}")
            return -1.0 # Penalize any exception

    def evaluate_tree(self):
        """Evaluates the current expression tree using the environment's data."""
        if self.tree is None:
            return None
        try:
            # Prepare evaluation context (local dictionary)
            eval_context = {}
            # Add features from the environment's data
            for feature in self.features:
                if feature in self.data.columns:
                    eval_context[feature] = self.data[feature].values
                else:
                    warnings.warn(f"Feature '{feature}' not found in data for eval.")
                    return None
            # Add constants
            for const in self.constants:
                eval_context[str(const)] = const
            # Add safe functions and numpy
            eval_context.update({
                'np': np, 'safe_log': safe_log, 'safe_exp': safe_exp,
                'safe_div': safe_div, 'safe_sqrt': safe_sqrt,
                'safe_power2': safe_power2, 'normalize': normalize_series
            })

            # Evaluate the expression safely
            factor_values = eval(self.tree, {"__builtins__": {}}, eval_context)

            # Ensure result is a numpy array of correct size
            if np.isscalar(factor_values):
                 factor_values = np.full(len(self.data), factor_values, dtype=np.float64)
            elif not isinstance(factor_values, np.ndarray):
                 warnings.warn("Tree evaluation did not return array or scalar.")
                 return None
            elif len(factor_values) != len(self.data):
                 warnings.warn(f"Eval result length mismatch: {len(factor_values)} vs data {len(self.data)}")
                 # Attempt to align if possible, otherwise fail (simple fail here)
                 return None # Length mismatch is problematic

            # Handle potential infinities after evaluation
            return handle_inf_nan(factor_values.astype(np.float64)) # Ensure float64

        except (SyntaxError, NameError, TypeError, ZeroDivisionError, OverflowError, MemoryError, ValueError) as e:
            # warnings.warn(f"Tree evaluation error: {e} for tree: {self.tree}")
            return None
        except Exception as e: # Catch any other unexpected errors
            # warnings.warn(f"Unexpected evaluation error: {e} for tree: {self.tree}")
            return None