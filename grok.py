import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import warnings


import matplotlib.pyplot as plt


# Filter warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Load data
data = pd.read_csv('/Users/frank/Desktop/code/binance/data/binance_BTCUSDT_1min_90days.csv')

# Try to parse dates properly
try:
    data['Open time'] = pd.to_datetime(data['Open time'])
    data['Close time'] = pd.to_datetime(data['Close time'])
except Exception as e:
    print(f"Date parsing warning (non-critical): {e}")

# Calculate future 1-minute returns
data['Return'] = data['Close'].shift(-1) / data['Close'] - 1

# Define features, operators, and constants
features = ['Open', 'High', 'Low', 'Close', 'Volume']
operators = ['+', '-', '*', '/', 'log', 'exp', 'sqrt', 'abs', '**2']
constants = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, -0.1, -0.5, -1.0, -2.0]  # Add various constants

# Safe math functions to prevent numerical issues
def safe_log(x):
    """Safe logarithm that handles zeros and negative numbers"""
    return np.log(np.maximum(x, 1e-10))

def safe_exp(x):
    """Safe exponential that prevents overflow"""
    return np.exp(np.clip(x, -20, 20))

def safe_div(a, b):
    """Safe division that handles division by zero"""
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)

def safe_sqrt(x):
    """Safe square root that handles negative numbers"""
    return np.sqrt(np.maximum(x, 0))

def safe_power2(x):
    """Square with overflow protection"""
    return np.square(np.clip(x, -1e5, 1e5))

def normalize_series(x):
    """Z-score normalization with safeguards"""
    if isinstance(x, np.ndarray) and x.size > 1:
        std = np.std(x)
        if std > 1e-10:
            return (x - np.mean(x)) / std
    return x

# Environment class
class FactorEnv:
    def __init__(self, data, max_depth=7):
        self.data = data
        self.max_depth = max_depth
        self.reset()

    def reset(self):
        """Reset environment"""
        self.tree = None
        self.current_depth = 0
        self.done = False
        self.operations_count = 0
        return self.get_state()

    def get_state(self):
        """Get current state with more information"""
        if self.tree is None:
            return [0, 0, 0]
        return [self.current_depth, 1, min(self.operations_count, 10)]

    def step(self, action):
        # Total number of possible actions: features + operators + constants
        total_features = len(features)
        total_operators = len(operators)
        total_constants = len(constants)
        
        if self.tree is None:
            if action < total_features:  # Add feature
                feature = features[action]
                self.tree = feature
            elif action < total_features + total_constants:  # Add constant
                constant_idx = action - total_features
                constant = constants[constant_idx]
                self.tree = str(constant)
            else:  # Try to add operator - not valid as first step
                feature = np.random.choice(features)
                self.tree = feature
                reward = -0.1  # Small penalty
                self.current_depth += 1
                if self.current_depth >= self.max_depth or self.operations_count >= 15:
                    self.done = True
                return self.get_state(), reward, self.done
        else:
            # Normal action processing
            if action < total_features:  # Add feature
                feature = features[action]
                # Binary operator selection
                binary_ops = ['+', '-', '*', '/']
                operator = np.random.choice(binary_ops)
                
                # Randomize operand order
                if np.random.random() < 0.5:
                    self.tree = f"({self.tree} {operator} {feature})"
                else:
                    self.tree = f"({feature} {operator} {self.tree})"
                
                self.operations_count += 1
            elif action < total_features + total_constants:  # Add constant
                constant_idx = action - total_features
                constant = constants[constant_idx]
                
                # Binary operator selection for constant
                binary_ops = ['+', '-', '*', '/']
                operator = np.random.choice(binary_ops)
                
                # Randomize operand order
                if np.random.random() < 0.5:
                    self.tree = f"({self.tree} {operator} {constant})"
                else:
                    self.tree = f"({constant} {operator} {self.tree})"
                
                self.operations_count += 1
            else:  # Add operator
                operator_idx = action - (total_features + total_constants)
                if operator_idx < total_operators:
                    operator = operators[operator_idx]
                    
                    # Handle binary operators
                    if operator in ['+', '-', '*', '/', '**2']:
                        if operator == '**2':
                            # Square operation special handling
                            self.tree = f"safe_power2({self.tree})"
                        else:
                            # Decide between adding a feature or a constant
                            if np.random.random() < 0.5:  # 50% chance to use a constant
                                operand = np.random.choice(constants)
                            else:  # 50% chance to use a feature
                                operand = np.random.choice(features)
                            
                            # Randomize operand order
                            if np.random.random() < 0.5:
                                self.tree = f"({self.tree} {operator} {operand})"
                            else:
                                self.tree = f"({operand} {operator} {self.tree})"
                    # Handle unary operators
                    elif operator == 'log':
                        self.tree = f"safe_log({self.tree})"
                    elif operator == 'exp':
                        self.tree = f"safe_exp({self.tree})"
                    elif operator == 'sqrt':
                        self.tree = f"safe_sqrt({self.tree})"
                    elif operator == 'abs':
                        self.tree = f"np.abs({self.tree})"
                    
                    self.operations_count += 1
                else:
                    # Handle invalid operation
                    reward = -0.1
                    return self.get_state(), reward, self.done
        
        self.current_depth += 1
        # Check if termination conditions are met
        if self.current_depth >= self.max_depth or self.operations_count >= 15:
            self.done = True
        
        # Calculate reward
        if self.done:
            try:
                factor_values = self.evaluate_tree()
                if factor_values is not None:
                    # Process factor values
                    factor_values = pd.Series(factor_values)
                    # Check if there are enough valid values
                    valid_values = factor_values.dropna()
                    
                    if len(valid_values) > len(self.data) * 0.5:  # Require at least 50% valid data
                        # Handle extreme values
                        lower_bound = factor_values.quantile(0.01)
                        upper_bound = factor_values.quantile(0.99)
                        factor_values = factor_values.clip(lower=lower_bound, upper=upper_bound)
                        
                        # Align data
                        valid_data = pd.DataFrame({
                            'factor': factor_values,
                            'return': self.data['Return']
                        }).dropna()
                        
                        if len(valid_data) > 100:  # Require sufficient sample size
                            # Check if factor has enough variability
                            if valid_data['factor'].std() > 1e-6:
                                # Calculate Spearman rank correlation
                                from scipy.stats import spearmanr
                                corr, p_value = spearmanr(valid_data['factor'], valid_data['return'])
                                
                                # If correlation is valid, use its absolute value as reward
                                if not np.isnan(corr):
                                    # Consider correlation statistical significance
                                    reward = abs(corr)
                                    if p_value < 0.05:  # If correlation is statistically significant, increase reward
                                        reward *= 1.2
                                    
                                    # # Penalize complexity
                                    # complexity_penalty = 0.01 * self.operations_count
                                    # reward = max(reward - complexity_penalty, 0)
                                else:
                                    reward = -0.5
                            else:
                                reward = -0.5  # Penalize constant factors
                        else:
                            reward = -0.5  # Penalize insufficient data
                    else:
                        reward = -0.7  # Heavily penalize too many invalid data points
                else:
                    reward = -1  # Unable to evaluate
            except Exception as e:
                print(f"Reward calculation error: {e}")
                reward = -1
        else:
            reward = 0  # No reward for non-terminal states
        
        return self.get_state(), reward, self.done
    
    def evaluate_tree(self):
        """Evaluate the expression tree with improved numerical stability"""
        try:
            expr = self.tree
            
            # Create dictionary for expression variables
            var_dict = {}
            for feature in features:
                var_dict[feature] = self.data[feature].values.copy()
                
            # Add constants to the evaluation context
            for constant in constants:
                var_dict[str(constant)] = constant
            
            # Add safe functions
            safe_dict = {
                'np': np,
                'safe_log': safe_log,
                'safe_exp': safe_exp,
                'safe_div': safe_div,
                'safe_sqrt': safe_sqrt,
                'safe_power2': safe_power2,
                'normalize': normalize_series
            }
            
            # Modify expression to make division safer
            modified_expr = expr.replace('/', 'safe_div')
            
            # Execute expression
            try:
                factor_values = eval(modified_expr, {"__builtins__": {}}, {**var_dict, **safe_dict})
            except:
                # If modified expression fails, try original expression
                factor_values = eval(expr, {"__builtins__": {}}, {**var_dict, **safe_dict})
            
            # Handle infinities and NaNs
            if isinstance(factor_values, np.ndarray):
                factor_values = np.nan_to_num(factor_values, nan=np.nan, posinf=np.nan, neginf=np.nan)
            
            return factor_values
        except Exception as e:
            print(f"Tree evaluation error: {e} for tree: {self.tree}")
            return None

# Policy network with expanded state dimensions
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

# Similarly expanded value network
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.fc(x)

# PPO class
class PPO:
    def __init__(self, state_dim, action_dim, lr=0.0002, gamma=0.99, eps_clip=0.2):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()), lr=lr
        )
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.best_reward = -np.inf
        self.best_tree = None
        self.top_factors = []  # Store top performing factors
        self.factor_memory = set()  # Remember factors we've seen to avoid duplicates

    def select_action(self, state, exploration=0.1):
        """Select action with exploration mechanism"""
        state = torch.FloatTensor(state).unsqueeze(0)
        
        # Random exploration
        if np.random.random() < exploration:
            action = np.random.randint(0, self.policy.fc[-2].out_features)
            # Get log probability for this action
            with torch.no_grad():
                probs = self.policy(state)
                dist = Categorical(probs)
                log_prob = dist.log_prob(torch.tensor(action))
            return action, log_prob.item()
        
        # Normal policy selection
        with torch.no_grad():
            probs = self.policy(state)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def update(self, states, actions, log_probs, rewards, next_state, done, tree):
        """Update networks"""
        if len(states) == 0:
            return False
            
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        log_probs = torch.FloatTensor(log_probs)
        rewards = torch.FloatTensor(rewards)
        next_state = torch.FloatTensor([next_state]).unsqueeze(0)  # Ensure correct shape

        final_reward = rewards[-1]
        
        # If it's a valid factor and not a duplicate, record it
        if final_reward > 0 and tree is not None:
            # Check if we've seen this factor before
            if tree not in self.factor_memory:
                self.factor_memory.add(tree)
                
                # Store in top factors list
                self.top_factors.append((final_reward, tree))
                self.top_factors.sort(key=lambda x: x[0], reverse=True)
                self.top_factors = self.top_factors[:20]  # Keep top 20
                
                if final_reward > self.best_reward:
                    self.best_reward = final_reward
                    self.best_tree = tree
                    return True
                
        # Learning step
        try:
            with torch.no_grad():
                value_next = self.value(next_state).squeeze()
                
                # Calculate TD targets and returns
                returns = []
                discounted_reward = 0
                for reward in reversed(rewards):
                    discounted_reward = reward + self.gamma * discounted_reward
                    returns.insert(0, discounted_reward)
                returns = torch.FloatTensor(returns)
                
                # Normalize returns to improve stability
                if len(returns) > 1 and returns.std() > 1e-8:
                    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Get current value predictions
            values = self.value(states).squeeze()
            if values.dim() == 0:
                values = values.unsqueeze(0)
                
            # Calculate advantage
            advantage = returns - values
            
            # Policy update
            probs = self.policy(states)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            
            # PPO objective
            ratio = torch.exp(new_log_probs - log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values, returns)
            
            # Add entropy reward to encourage exploration
            entropy = dist.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)  # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
            self.optimizer.step()
            
        except Exception as e:
            print(f"Policy update error: {e}")
            
        return False

# Training process
env = FactorEnv(data, max_depth=10)
state_dim = 3  # State dimensions: [current_depth, tree_exists, operations_count]
action_dim = len(features) + len(constants) + len(operators)  # Action space: features + constants + operators
ppo = PPO(state_dim, action_dim)

num_episodes = 1000
best_factors_unchanged = 0  # Track rounds where best factor hasn't changed
# 初始化记录列表
episode_rewards = []
policy_losses = []
value_losses = []
for episode in range(num_episodes):
    state = env.reset()
    done = False
    episode_reward = 0
    states, actions, log_probs, rewards = [], [], [], []

    # Gradually reduce exploration as training progresses
    exploration_rate = max(0.05, 0.5 - 0.45 * episode / (num_episodes / 2))
    
    # Build factor expression
    while not done:
        action, log_prob = ppo.select_action(state, exploration=exploration_rate)
        next_state, reward, done = env.step(action)
        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        state = next_state
        episode_reward += reward
    episode_rewards.append(episode_reward)
    improved = ppo.update(states, actions, log_probs, rewards, next_state, done, env.tree)

    # Print results every 50 episodes or when improved
    if episode % 50 == 0 or improved:
        print(f"Episode {episode}, Reward: {episode_reward:.4f}, Tree: {env.tree}")
        if improved:
            print(f"New best factor found! IC: {ppo.best_reward:.4f}, Expression: {ppo.best_tree}")
            best_factors_unchanged = 0
        else:
            best_factors_unchanged += 1

    # Early stopping condition: if we have enough good factors and best factor hasn't improved for a while
    if (episode > 5000 and 
        len(ppo.top_factors) >= 10 and 
        ppo.best_reward > 0.15 and 
        best_factors_unchanged > 1000):
        print("Found stable good factors, early stopping...")
        break

plt.figure(figsize=(12, 6))

# 绘制奖励图
plt.subplot(1, 2, 1)
plt.plot(episode_rewards, label='Episode Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward over Episodes')
plt.legend()

# 绘制损失图
plt.subplot(1, 2, 2)
plt.plot(policy_losses, label='Policy Loss')
plt.plot(value_losses, label='Value Loss')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.title('Loss over Episodes')
plt.legend()

# 保存图像
plt.savefig('training_plot.png')
print("\n=== Training Complete ===")
print(f"Best factor found: {ppo.best_tree}")
print(f"IC value: {ppo.best_reward:.4f}")

# Show top factors
print("\nTop performing factors:")
for i, (ic, tree) in enumerate(ppo.top_factors):
    print(f"{i+1}. IC: {ic:.4f}, Expression: {tree}")

# Batch evaluate top factors
def evaluate_factor(factor_expr, data):
    """Fully evaluate a factor expression"""
    env.tree = factor_expr
    factor_values = env.evaluate_tree()
    if factor_values is not None:
        factor_series = pd.Series(factor_values)
        
        # Clean data for analysis
        valid_data = pd.DataFrame({
            'factor': factor_series,
            'return': data['Return']
        }).dropna()
        
        # Remove outliers to improve analysis
        lower_bound = valid_data['factor'].quantile(0.01)
        upper_bound = valid_data['factor'].quantile(0.99)
        clean_data = valid_data[(valid_data['factor'] >= lower_bound) & 
                               (valid_data['factor'] <= upper_bound)]
        
        if len(clean_data) > 10:
            # Calculate correlations
            from scipy.stats import pearsonr, spearmanr
            try:
                pearson_corr, p_pearson = pearsonr(clean_data['factor'], clean_data['return'])
                spearman_corr, p_spearman = spearmanr(clean_data['factor'], clean_data['return'])
                
                return {
                    'valid_points': len(clean_data),
                    'pearson': pearson_corr,
                    'p_pearson': p_pearson,
                    'spearman': spearman_corr,
                    'p_spearman': p_spearman,
                    'min': clean_data['factor'].min(),
                    'max': clean_data['factor'].max(),
                    'mean': clean_data['factor'].mean(),
                    'std': clean_data['factor'].std()
                }
            except Exception:
                return None
    return None

# # Evaluate all top factors
# print("\n=== Detailed Factor Evaluation ===")
# for i, (_, tree) in enumerate(ppo.top_factors[:5]):  # Only evaluate top 5
#     print(f"\nFactor {i+1}: {tree}")
#     stats = evaluate_factor(tree, data)
#     if stats:
#         print(f"  Valid data points: {stats['valid_points']}")
#         print(f"  Pearson correlation: {stats['pearson']:.4f} (p={stats['p_pearson']:.4f})")
#         print(f"  Spearman correlation: {stats['spearman']:.4f} (p={stats['p_spearman']:.4f})")
#         print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
#         print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
#     else:
#         print("  Could not evaluate this factor")