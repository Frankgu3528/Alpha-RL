# environment.py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import warnings

from utils import safe_log, safe_exp, safe_div, safe_sqrt, safe_power2, normalize_series, handle_inf_nan
# Import config variables needed within the class
from config import FEATURES, OPERATORS, CONSTANTS, MAX_DEPTH, MAX_OPERATIONS, EVAL_MIN_POINTS

class FactorEnv:
    """
    RL Environment for building and evaluating factor expressions.
    Includes a STOP action for dynamic termination.
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
        self.num_build_actions = len(self.features) + len(self.constants) + len(self.operators)
        self.stop_action_index = self.num_build_actions # Index for the new STOP action
        self.action_dim = self.num_build_actions + 1    # Total actions including STOP
        # 添加无效奖励常量
        self.INVALID_REWARD = -1.0

        if 'Return' not in self.data.columns:
             raise ValueError("The data_subset provided to FactorEnv must contain a 'Return' column.")
        self.returns_np = self.data['Return'].values
        self.reset()

    def reset(self):
        """Resets the environment."""
        self.tree = None
        self.current_depth = 0
        self.done = False
        self.operations_count = 0
        return self.get_state()

    def get_state(self):
        """Returns the current state."""
        # Keep the state representation simple for now
        if self.tree is None:
            return [0, 0, 0]
        else:
            norm_depth = self.current_depth / self.max_depth
            norm_ops = self.operations_count / self.max_operations
            # Ensure state values are within reasonable bounds for the network
            return [min(norm_depth, 1.0), 1, min(norm_ops, 1.0)]

    def step(self, action):
        """Performs one step, handling build actions and the STOP action."""
        reward = 0 # Intermediate rewards are 0
        op_applied = False # Track if a build operation occurred

        # --- Check for STOP action ---
        if action == self.stop_action_index:
            if self.tree is None:
                # Penalize stopping with an empty tree
                reward = -0.5
                self.tree = np.random.choice(self.features) # Create a minimal tree for eval? Or just penalize.
                warnings.warn("Agent chose STOP with an empty tree.")
            else:
                # Agent chose to stop, calculate final reward based on current tree
                reward = self._calculate_final_reward()
            self.done = True
            return self.get_state(), float(reward), self.done
        # --- End STOP action check ---

        # --- Handle Build Actions (if action is not STOP) ---
        if action < self.num_build_actions:
            total_features = len(self.features)
            total_constants = len(self.constants)

            if self.tree is None:
                # First step: Must start with a feature or constant
                if action < total_features:
                    self.tree = self.features[action]
                    op_applied = True
                elif action < total_features + total_constants:
                    self.tree = str(self.constants[action - total_features])
                    op_applied = True
                else: # Invalid first action (operator)
                    reward = -0.15 # Small penalty
                    self.tree = np.random.choice(self.features) # Fallback
                    op_applied = True
            else:
                # Subsequent build steps (logic simplified slightly from previous version for clarity)
                current_action_type = None
                if action < total_features: current_action_type = 'feature'
                elif action < total_features + total_constants: current_action_type = 'constant'
                else: current_action_type = 'operator'

                if current_action_type == 'feature':
                    feature = self.features[action]
                    bin_op = np.random.choice(['+', '-', '*', 'safe_div'])
                    self.tree = f"({self.tree} {bin_op} {feature})" # Simplified structure slightly
                    self.operations_count += 1
                    op_applied = True
                elif current_action_type == 'constant':
                    constant_str = str(self.constants[action - total_features])
                    bin_op = np.random.choice(['+', '-', '*', 'safe_div'])
                    self.tree = f"({self.tree} {bin_op} {constant_str})" # Simplified structure slightly
                    self.operations_count += 1
                    op_applied = True
                elif current_action_type == 'operator':
                    op_idx = action - (total_features + total_constants)
                    operator_symbol = self.operators[op_idx]
                    op_map = { # Re-define map for clarity
                        'log': 'safe_log', 'exp': 'safe_exp', 'sqrt': 'safe_sqrt',
                        'abs': 'np.abs', '**2': 'safe_power2',
                        '+': '+', '-': '-', '*': '*', '/': 'safe_div'
                    }
                    if operator_symbol in ['log', 'exp', 'sqrt', 'abs', '**2']:
                        safe_op_name = op_map[operator_symbol]
                        self.tree = f"{safe_op_name}({self.tree})"
                        self.operations_count += 1
                        op_applied = True
                    elif operator_symbol in ['+', '-', '*', '/']:
                        safe_op_name = op_map[operator_symbol]
                        operand = str(np.random.choice(self.constants)) if np.random.rand() < 0.5 else np.random.choice(self.features)
                        self.tree = f"({self.tree} {safe_op_name} {operand})" # Simplified structure slightly
                        self.operations_count += 2
                        op_applied = True
        else:
            # Invalid action index (shouldn't happen if action_dim is correct)
            warnings.warn(f"Invalid action index received in step: {action}")
            reward = -0.2
            self.done = True # Terminate on invalid action
            return self.get_state(), float(reward), self.done

        # Update depth if a build operation was applied
        if op_applied:
            self.current_depth += 1

        # --- Check Termination by Fixed Limits (Safety Net) ---
        if not self.done: # Only check if not already done by STOP action
            if self.current_depth >= self.max_depth or self.operations_count >= self.max_operations:
                reward = self._calculate_final_reward() # Calculate reward as limits are hit
                self.done = True

        # --- Return state, reward, done ---
        # Ensure reward is float for consistency
        reward = float(reward) if isinstance(reward, (float, int, np.number)) else -1.0
        return self.get_state(), reward, self.done

    def _calculate_final_reward(self):
        """计算最终奖励，综合考虑IC值和分组收益"""
        try:
            # 计算因子值
            factor_values = self.evaluate_tree()
            if factor_values is None:
                return self.INVALID_REWARD
            
            # 获取收益率数据
            returns = self.data['Return'].values
            
            # 计算IC
            valid_mask = np.isfinite(factor_values) & np.isfinite(returns)
            if valid_mask.sum() < 50:  # 样本太少
                return self.INVALID_REWARD
                
            valid_factors = factor_values[valid_mask]
            valid_returns = returns[valid_mask]
            
            ic, _ = spearmanr(valid_factors, valid_returns)
            
            # 计算分组收益
            df = pd.DataFrame({
                'factor': valid_factors,
                'return': valid_returns
            })
            
            # 5分组
            try:
                df['group'] = pd.qcut(df['factor'], 5, labels=False)
                group_returns = df.groupby('group')['return'].mean()
                # 计算多空组合收益
                long_short_return = float(group_returns.iloc[-1] - group_returns.iloc[0])
            except Exception:
                long_short_return = 0
            
            # 综合奖励计算
            # IC权重为0.7，收益率权重为0.3
            ic_reward = ic * 0.3
            # 将收益率标准化到[-1, 1]范围
            ret_reward = np.clip(long_short_return * 100, -1, 1) * 0.7
            
            final_reward = ic_reward + ret_reward
            
            # 对无效值进行惩罚
            if np.isnan(final_reward):
                return self.INVALID_REWARD
                
            return final_reward
            
        except Exception as e:
            print(f"奖励计算错误: {str(e)}")
            return self.INVALID_REWARD
        # Calculates reward based on self.tree and self.data
        # Returns a float reward value or penalty
        try:
            factor_values = self.evaluate_tree()

            if factor_values is None or not isinstance(factor_values, np.ndarray):
                return -1.0 # Penalty for evaluation failure

            returns = self.returns_np
            valid_mask = np.isfinite(factor_values) & np.isfinite(returns)
            if valid_mask.sum() < EVAL_MIN_POINTS: # Check if enough valid points
                return -0.7 # Penalty for insufficient valid data

            valid_factors = factor_values[valid_mask]
            valid_returns = returns[valid_mask]

            lower_bound = np.percentile(valid_factors, 1)
            upper_bound = np.percentile(valid_factors, 99)
            clipped_factors = np.clip(valid_factors, lower_bound, upper_bound)

            if np.std(clipped_factors) < 1e-7:
                return -0.6 # Penalize constant-like factors

            corr, p_value = spearmanr(clipped_factors, valid_returns)

            if np.isnan(corr):
                return -0.5 # Penalize if correlation is NaN

            # --- Incorporate Complexity Penalty Here (Optional) ---
            reward = abs(corr)
            # complexity_penalty = config.COMPLEXITY_COEF * self.operations_count # Assumes COMPLEXITY_COEF in config
            # reward = max(0, reward - complexity_penalty)
            # ---

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