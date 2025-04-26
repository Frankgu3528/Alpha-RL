import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import warnings
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import train_test_split # Although time series splits are often done manually

# Filter warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning) # Ignore potential future pandas warnings

# --- 1. Data Loading and Splitting ---
data_path = './data/binance_BTCUSDT_1min_90days.csv'
data = pd.read_csv(data_path)

# Try to parse dates properly
try:
    data['Open time'] = pd.to_datetime(data['Open time'])
    data['Close time'] = pd.to_datetime(data['Close time'])
    # Optional: Set index if useful, but keep original columns for features
    # data = data.set_index('Open time')
except Exception as e:
    print(f"Date parsing warning (non-critical): {e}")

# Calculate future 1-minute returns (aligning carefully)
# shift(-1) means the return from time t to t+1 is stored at row t
data['Return'] = data['Close'].shift(-1) / data['Close'] - 1

# Drop rows with NaN returns (the very last row)
data.dropna(subset=['Return'], inplace=True)

# Define split points (e.g., 60% train, 20% validation, 20% test)
n = len(data)
train_end_idx = int(n * 0.6)
val_end_idx = int(n * 0.8)

train_data = data.iloc[:train_end_idx].copy()
val_data = data.iloc[train_end_idx:val_end_idx].copy()
test_data = data.iloc[val_end_idx:].copy()

print(f"Data Split: Train={len(train_data)}, Validation={len(val_data)}, Test={len(test_data)}")

# --- 2. Definitions (Features, Operators, Constants, Safe Functions) ---
features = ['Open', 'High', 'Low', 'Close', 'Volume']
# Consider adding more technical indicators here later
operators = ['+', '-', '*', '/', 'log', 'exp', 'sqrt', 'abs', '**2']
constants = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, -0.1, -0.5, -1.0, -2.0]

# Safe math functions (unchanged)
def safe_log(x):
    return np.log(np.maximum(x, 1e-10))

def safe_exp(x):
    return np.exp(np.clip(x, -20, 20))

def safe_div(a, b):
    # Ensure a and b are numpy arrays for consistent broadcasting
    a = np.asarray(a)
    b = np.asarray(b)
    # Create output array initialized to zero
    out = np.zeros_like(a, dtype=np.float64)
    # Find where b is non-zero and not NaN
    valid_b = (b != 0) & (~np.isnan(b)) & (~np.isinf(b))
    # Perform division only where it's safe
    np.divide(a[valid_b], b[valid_b], out=out[valid_b])
    # Set output to NaN where division is unsafe (b is zero, NaN, or inf)
    out[~valid_b] = np.nan
    # Set output to NaN where a is NaN or inf
    out[np.isnan(a) | np.isinf(a)] = np.nan
    return out


def safe_sqrt(x):
    return np.sqrt(np.maximum(x, 0))

def safe_power2(x):
    # Clip before squaring to prevent massive numbers leading to inf
    clipped_x = np.clip(x, -1e4, 1e4) # Adjust clip range if necessary
    return np.square(clipped_x)

def normalize_series(x):
    if isinstance(x, np.ndarray) and x.size > 1:
        try:
            # Calculate std deviation only on finite values
            valid_x = x[np.isfinite(x)]
            if valid_x.size > 1:
                std = np.std(valid_x)
                if std > 1e-10:
                    mean = np.mean(valid_x)
                    # Apply normalization, keeping NaNs/Infs as NaNs
                    finite_mask = np.isfinite(x)
                    result = np.full_like(x, np.nan, dtype=np.float64)
                    result[finite_mask] = (x[finite_mask] - mean) / std
                    return result
        except Exception: # Catch potential numerical errors during std/mean calc
            pass # Return original array if normalization fails
    # Return original array if it's not a suitable numpy array or normalization failed
    return x


# --- 3. Environment Class (`FactorEnv`) ---
class FactorEnv:
    # Now accepts specific data slice
    def __init__(self, data_subset, max_depth=7):
        self.data = data_subset.copy() # Use the provided data slice
        self.max_depth = max_depth
        # Ensure 'Return' exists in the provided data_subset
        if 'Return' not in self.data.columns:
             raise ValueError("The data_subset provided to FactorEnv must contain a 'Return' column.")
        self.reset()

    def reset(self):
        self.tree = None
        self.current_depth = 0
        self.done = False
        self.operations_count = 0
        return self.get_state()

    def get_state(self):
        if self.tree is None:
            return [0, 0, 0] # depth, tree_exists (0=No), ops_count
        # State: depth, tree_exists (1=Yes), capped ops_count
        return [min(self.current_depth, self.max_depth + 1), 1, min(self.operations_count, 15)]


    def step(self, action):
        total_features = len(features)
        total_operators = len(operators)
        total_constants = len(constants)
        action_dim = total_features + total_constants + total_operators

        reward = 0 # Default reward for intermediate steps
        op_applied = False # Flag to track if an operation was successfully applied

        if self.tree is None:
            # First step: must be a feature or a constant
            if action < total_features:
                feature = features[action]
                self.tree = feature
                op_applied = True
            elif action < total_features + total_constants:
                constant_idx = action - total_features
                constant = constants[constant_idx]
                self.tree = str(constant) # Store constants as strings in the tree
                op_applied = True
            else:
                # Invalid action: trying to apply operator first
                reward = -0.2 # Penalty for invalid first action
                # As a fallback, randomly select a feature to start the tree
                feature = np.random.choice(features)
                self.tree = feature
                op_applied = True # We forced a start
        else:
            # Subsequent steps
            if action < total_features: # Combine with a feature
                feature = features[action]
                binary_ops = ['+', '-', '*', '/']
                operator = np.random.choice(binary_ops)
                # Replace / with safe_div for evaluation
                eval_operator = 'safe_div' if operator == '/' else operator
                if np.random.random() < 0.5:
                    self.tree = f"({self.tree} {eval_operator} {feature})"
                else:
                    self.tree = f"({feature} {eval_operator} {self.tree})"
                self.operations_count += 1
                op_applied = True
            elif action < total_features + total_constants: # Combine with a constant
                constant_idx = action - total_features
                constant = constants[constant_idx]
                binary_ops = ['+', '-', '*', '/']
                operator = np.random.choice(binary_ops)
                eval_operator = 'safe_div' if operator == '/' else operator
                if np.random.random() < 0.5:
                    self.tree = f"({self.tree} {eval_operator} {str(constant)})"
                else:
                    self.tree = f"({str(constant)} {eval_operator} {self.tree})"
                self.operations_count += 1
                op_applied = True
            else: # Apply an operator
                operator_idx = action - (total_features + total_constants)
                if operator_idx < total_operators:
                    operator = operators[operator_idx]
                    # Apply unary operators
                    if operator == 'log':
                        self.tree = f"safe_log({self.tree})"
                    elif operator == 'exp':
                        self.tree = f"safe_exp({self.tree})"
                    elif operator == 'sqrt':
                        self.tree = f"safe_sqrt({self.tree})"
                    elif operator == 'abs':
                        self.tree = f"np.abs({self.tree})"
                    elif operator == '**2':
                         self.tree = f"safe_power2({self.tree})"
                    # Apply binary operators (requires adding a new random operand)
                    elif operator in ['+', '-', '*', '/']:
                        eval_operator = 'safe_div' if operator == '/' else operator
                        # Decide between adding a feature or a constant as the second operand
                        if np.random.random() < 0.5: # Add constant
                            operand = str(np.random.choice(constants))
                        else: # Add feature
                            operand = np.random.choice(features)

                        if np.random.random() < 0.5:
                             self.tree = f"({self.tree} {eval_operator} {operand})"
                        else:
                             self.tree = f"({operand} {eval_operator} {self.tree})"
                    self.operations_count += 1
                    op_applied = True
                else:
                     # Invalid operator index - should not happen if action_dim is correct
                     reward = -0.2 # Penalty

        # Update depth and check termination *only if* an operation was applied
        if op_applied:
            self.current_depth += 1

        if self.current_depth >= self.max_depth or self.operations_count >= 15: # Max complexity limit
            self.done = True

        # Calculate final reward only when done
        if self.done:
            try:
                factor_values = self.evaluate_tree()

                if factor_values is not None and isinstance(factor_values, np.ndarray):
                    # Align factor values with returns from the *environment's specific data subset*
                    # Ensure index alignment if using pandas Series, or rely on positional alignment if numpy arrays
                    returns = self.data['Return'].values
                    
                    # Create DataFrame for cleaning and correlation
                    eval_df = pd.DataFrame({'factor': factor_values, 'return': returns})

                    # Drop rows where factor or return is NaN or Inf
                    eval_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                    valid_data = eval_df.dropna()

                    # Check if enough valid data points remain
                    if len(valid_data) > max(100, len(self.data) * 0.1): # Need min 100 points or 10% of data
                        # Winsorize factor values to handle extreme outliers (using percentiles on valid data)
                        lower_bound = valid_data['factor'].quantile(0.01)
                        upper_bound = valid_data['factor'].quantile(0.99)
                        # Clip only the 'factor' column in the valid_data DataFrame
                        valid_data['factor_clipped'] = valid_data['factor'].clip(lower=lower_bound, upper=upper_bound)

                        # Check for sufficient variability in the clipped factor
                        if valid_data['factor_clipped'].std() > 1e-7:
                            # Calculate Spearman rank correlation on clipped, valid data
                            corr, p_value = spearmanr(valid_data['factor_clipped'], valid_data['return'])

                            # Check if correlation is valid number
                            if not np.isnan(corr):
                                reward = abs(corr)
                                # Optional: Small bonus for significance (use cautiously)
                                # if p_value < 0.05:
                                #     reward *= 1.1

                                # Optional: Complexity Penalty (uncomment to use)
                                # complexity_penalty = 0.005 * self.operations_count
                                # reward = max(0, reward - complexity_penalty)

                            else:
                                reward = -0.5 # Penalize if correlation calculation fails
                        else:
                            reward = -0.6 # Penalize factors with near-zero variance (constants)
                    else:
                        reward = -0.7 # Penalize if not enough valid data points after cleaning/alignment
                else:
                    reward = -1.0 # Penalize if tree evaluation fails completely or returns non-array
            except Exception as e:
                # print(f"Reward calculation error: {e} for tree: {self.tree}") # Debugging
                reward = -1.0 # Penalize any exception during reward calculation

        # Ensure reward is a scalar float
        if not isinstance(reward, (float, int)):
             reward = -1.0 # Fallback penalty

        return self.get_state(), reward, self.done

    def evaluate_tree(self):
        """Evaluate the expression tree using the environment's data subset"""
        if self.tree is None:
            return None
        try:
            # Prepare evaluation dictionary using the environment's specific data
            var_dict = {}
            for feature in features:
                 # Ensure feature exists and get numpy array
                 if feature in self.data.columns:
                      var_dict[feature] = self.data[feature].values
                 else:
                      # print(f"Warning: Feature '{feature}' not found in data subset.") # Debugging
                      return None # Cannot evaluate if feature is missing

            # Add constants to the evaluation context
            for constant in constants:
                var_dict[str(constant)] = constant # Use string representation as key

            # Add safe functions and numpy
            safe_dict = {
                'np': np,
                'safe_log': safe_log,
                'safe_exp': safe_exp,
                'safe_div': safe_div,
                'safe_sqrt': safe_sqrt,
                'safe_power2': safe_power2,
                'normalize': normalize_series # If you plan to use normalization within factors
            }

            # Evaluate the expression
            # Important: Use the prepared dictionaries for globals and locals
            # Pass empty "__builtins__": {} to restrict access
            factor_values = eval(self.tree, {"__builtins__": {}}, {**var_dict, **safe_dict})

            # Ensure result is a numpy array
            if not isinstance(factor_values, np.ndarray):
                 # If eval returns a scalar (e.g., tree is just "1.0"), broadcast it
                 if np.isscalar(factor_values):
                      factor_values = np.full(len(self.data), factor_values, dtype=np.float64)
                 else:
                      # print(f"Warning: Tree evaluation did not return a NumPy array or scalar.") # Debugging
                      return None

            # Handle potential infinities resulting from operations (replace with NaN)
            factor_values = np.nan_to_num(factor_values, nan=np.nan, posinf=np.nan, neginf=np.nan)

            return factor_values

        except (SyntaxError, NameError, TypeError, ZeroDivisionError, OverflowError, MemoryError, ValueError) as e:
            # Catch specific, common evaluation errors
            # print(f"Tree evaluation error: {e} for tree: {self.tree}") # Debugging
            return None
        except Exception as e: # Catch any other unexpected errors
            # print(f"Unexpected evaluation error: {e} for tree: {self.tree}") # Debugging
            return None

# --- 4. PPO Agent (Policy Network, Value Network, PPO Class) ---
# Policy network (unchanged)
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
        # Ensure input is float
        return self.fc(x.float())

# Value network (unchanged)
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
         # Ensure input is float
        return self.fc(x.float())


# PPO class (mostly unchanged, added tracking for losses)
class PPO:
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.98, eps_clip=0.2): # Adjusted lr, gamma
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()), lr=lr
        )
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.best_reward = -np.inf # Tracks best reward *during training*
        self.best_tree = None
        self.top_factors = []  # Store top performing factors (reward, tree) from training
        self.factor_memory = set() # Remember factors seen during training

        # Store losses for plotting
        self.policy_losses = []
        self.value_losses = []

    def select_action(self, state, exploration=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)

        if np.random.random() < exploration:
            # Explore: choose a random valid action
            action = np.random.randint(0, self.policy.fc[-2].out_features)
            # Calculate log_prob for the randomly chosen action
            with torch.no_grad():
                probs = self.policy(state)
                dist = Categorical(probs)
                # Ensure action is tensor for log_prob
                log_prob = dist.log_prob(torch.tensor(action))
            return action, log_prob.item()
        else:
            # Exploit: choose action based on policy
            with torch.no_grad():
                probs = self.policy(state)
                dist = Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            return action.item(), log_prob.item()

    def update(self, states, actions, log_probs, rewards, next_state, done, tree):
        if not states: # If no states were collected (e.g., episode ended immediately)
            return False

        # Convert lists to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        log_probs_old = torch.FloatTensor(log_probs) # Log probs from the policy used for sampling
        rewards = torch.FloatTensor(rewards)
        next_state = torch.FloatTensor([next_state]) # Ensure next_state is 2D tensor [1, state_dim]

        # --- Calculate Returns and Advantages ---
        with torch.no_grad():
            # Get value of the final next_state
            # If done, the value of the next state is 0, otherwise estimate it
            # Note: In this setup, the reward is only given at the *end*.
            # We need to properly compute discounted returns.
            returns = []
            discounted_reward = 0
            # If the episode finished (done=True), the last reward is the terminal reward.
            # We start calculating returns from the end.
            # The value of the terminal state is considered 0.
            # If not done (e.g., hit max steps but not true termination), bootstrap using value net.
            # However, in this env, 'done' always means termination and final reward calculation.
            for reward in reversed(rewards):
                 discounted_reward = reward + self.gamma * discounted_reward
                 returns.insert(0, discounted_reward)

            returns = torch.tensor(returns, dtype=torch.float32)

            # Normalize returns for stability if more than one step
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Calculate advantages
            values = self.value(states).squeeze()
            advantages = returns - values
            # Normalize advantages too? Sometimes helpful.
            # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


        # --- Policy and Value Update ---
        # Evaluate policy for current states to get new log_probs and entropy
        probs = self.policy(states)
        dist = Categorical(probs)
        log_probs_new = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # Calculate PPO ratio
        ratio = torch.exp(log_probs_new - log_probs_old) # Use the log_probs from action selection time

        # Calculate PPO surrogate objectives
        surr1 = ratio * advantages.detach() # Detach advantages for policy loss
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages.detach()

        # Calculate losses
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = nn.MSELoss()(self.value(states).squeeze(), returns.detach()) # Use detach returns for value loss

        # Total loss
        # Entropy bonus encourages exploration
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        # --- Optimization Step ---
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (optional but often good practice)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
        self.optimizer.step()

        # --- Store losses for plotting ---
        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())


        # --- Track Best Factor Found During Training ---
        # Use the *final* reward achieved in the episode (based on train_data)
        # final_reward = rewards[-1].item() if rewards else -np.inf
        final_reward = rewards[-1].item() if rewards.numel() > 0 else -np.inf # Check if tensor has elements

        improved = False
        if done and tree is not None and final_reward > -np.inf:
            # Store unique factors based on the tree string
            if tree not in self.factor_memory:
                 self.factor_memory.add(tree)
                 # Keep top N factors based on their training reward
                 if len(self.top_factors) < 20 or final_reward > self.top_factors[-1][0]:
                     self.top_factors.append((final_reward, tree))
                     self.top_factors.sort(key=lambda x: x[0], reverse=True)
                     self.top_factors = self.top_factors[:20] # Keep top 20

            # Check if this is the best single factor found so far
            if final_reward > self.best_reward:
                 print(f"   -- New best training reward: {final_reward:.4f} > {self.best_reward:.4f} --")
                 self.best_reward = final_reward
                 self.best_tree = tree
                 improved = True

        return improved


# --- 5. Training Process ---
# Use train_data for the training environment
train_env = FactorEnv(train_data, max_depth=8) # Adjusted max_depth slightly

state_dim = 3  # State: [depth, tree_exists, ops_count]
action_dim = len(features) + len(constants) + len(operators)
ppo = PPO(state_dim, action_dim)

num_episodes = 5000 # Adjust number of episodes
best_factors_unchanged_streak = 0
max_unchanged_streak = 500 # Stop if best hasn't improved for this many episodes

episode_rewards_history = [] # Store cumulative reward per episode (sum of intermediate + final)

print("\n=== Starting Training ===")
for episode in range(num_episodes):
    state = train_env.reset()
    done = False
    episode_reward_sum = 0
    states, actions, log_probs, rewards = [], [], [], []

    # Exploration rate decay
    exploration_rate = max(0.05, 0.5 * (1 - episode / (num_episodes * 0.8))) # Decay faster initially

    # Generate one factor expression (episode)
    step_count = 0
    max_steps_per_episode = train_env.max_depth + 5 # Allow a few more steps than depth
    while not done and step_count < max_steps_per_episode:
        action, log_prob = ppo.select_action(state, exploration=exploration_rate)
        next_state, reward, done = train_env.step(action)

        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward) # Store intermediate (0) and final rewards

        state = next_state
        episode_reward_sum += reward # Cumulative reward (will be dominated by final reward)
        step_count += 1

    episode_rewards_history.append(episode_reward_sum)

    # Perform PPO update after the episode ends
    improved = ppo.update(states, actions, log_probs, rewards, next_state, done, train_env.tree)

    # Logging and Early Stopping Logic
    if improved:
        print(f"Episode {episode}: New best factor! Train Reward: {ppo.best_reward:.4f}, Tree: {ppo.best_tree}")
        best_factors_unchanged_streak = 0
    else:
        best_factors_unchanged_streak += 1

    if episode % 50 == 0:
         avg_reward = np.mean(episode_rewards_history[-50:]) if episode_rewards_history else 0
         avg_policy_loss = np.mean(ppo.policy_losses[-50:]) if ppo.policy_losses else 0
         avg_value_loss = np.mean(ppo.value_losses[-50:]) if ppo.value_losses else 0
         print(f"Episode {episode}/{num_episodes} | Avg Reward (last 50): {avg_reward:.4f} | "
               f"Avg P Loss: {avg_policy_loss:.4f} | Avg V Loss: {avg_value_loss:.4f} | "
               f"Exploration: {exploration_rate:.3f} | Best Train Reward: {ppo.best_reward:.4f}")

    # Early stopping condition
    if episode > 1000 and best_factors_unchanged_streak >= max_unchanged_streak:
        print(f"\nEarly stopping triggered: Best training reward hasn't improved for {max_unchanged_streak} episodes.")
        break

print("\n=== Training Complete ===")
print(f"Best factor found during training: {ppo.best_tree}")
print(f"Best training reward (IC on train set): {ppo.best_reward:.4f}")


# --- 6. Plotting Training Curves ---
plt.figure(figsize=(18, 5))

# Rewards
plt.subplot(1, 3, 1)
plt.plot(episode_rewards_history, label='Episode Reward Sum', alpha=0.6)
# Moving average for rewards
if len(episode_rewards_history) >= 50:
    moving_avg_rewards = pd.Series(episode_rewards_history).rolling(50).mean()
    plt.plot(moving_avg_rewards, label='Reward MA (50 episodes)', color='red')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Episode Rewards during Training')
plt.legend()

# Policy Loss
plt.subplot(1, 3, 2)
plt.plot(ppo.policy_losses, label='Policy Loss', alpha=0.6)
if len(ppo.policy_losses) >= 50:
    moving_avg_ploss = pd.Series(ppo.policy_losses).rolling(50).mean()
    plt.plot(moving_avg_ploss, label='Policy Loss MA (50 episodes)', color='blue')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.title('Policy Network Loss')
plt.legend()

# Value Loss
plt.subplot(1, 3, 3)
plt.plot(ppo.value_losses, label='Value Loss', alpha=0.6)
if len(ppo.value_losses) >= 50:
     moving_avg_vloss = pd.Series(ppo.value_losses).rolling(50).mean()
     plt.plot(moving_avg_vloss, label='Value Loss MA (50 episodes)', color='green')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.title('Value Network Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_plot_with_splits.png')
print("\nTraining plots saved to 'training_plot_with_splits.png'")
# plt.show()


# --- 7. Out-of-Sample Evaluation ---

def evaluate_factor_on_split(factor_expr, data_split, max_depth=8):
    """Evaluates a given factor expression on a specific data split."""
    if factor_expr is None:
        return None

    # Create a temporary environment specific to this data split
    eval_env = FactorEnv(data_split, max_depth=max_depth)
    eval_env.tree = factor_expr # Set the tree to evaluate

    try:
        factor_values = eval_env.evaluate_tree()

        if factor_values is not None and isinstance(factor_values, np.ndarray):
            returns = data_split['Return'].values
            eval_df = pd.DataFrame({'factor': factor_values, 'return': returns})
            eval_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            valid_data = eval_df.dropna()

            # Need enough data points for meaningful evaluation
            if len(valid_data) > 50: # Lower threshold for validation/test? Maybe 50-100.
                # Winsorize factor
                lower = valid_data['factor'].quantile(0.01)
                upper = valid_data['factor'].quantile(0.99)
                valid_data['factor_clipped'] = valid_data['factor'].clip(lower=lower, upper=upper)

                if valid_data['factor_clipped'].std() > 1e-7:
                    spearman_corr, p_spearman = spearmanr(valid_data['factor_clipped'], valid_data['return'])
                    pearson_corr, p_pearson = pearsonr(valid_data['factor_clipped'], valid_data['return'])

                    if not np.isnan(spearman_corr) and not np.isnan(pearson_corr):
                        return {
                            'spearman_ic': spearman_corr,
                            'p_spearman': p_spearman,
                            'pearson_corr': pearson_corr,
                            'p_pearson': p_pearson,
                            'valid_points': len(valid_data),
                            'factor_std': valid_data['factor_clipped'].std()
                        }
    except Exception as e:
        print(f"Error evaluating factor '{factor_expr}' on split: {e}") # Debugging info

    return None # Return None if evaluation failed


print("\n=== Out-of-Sample Factor Evaluation ===")

# Get the list of top factors found during training
factors_to_evaluate = sorted(ppo.top_factors, key=lambda x: x[0], reverse=True) # Already sorted, but ensure

evaluation_results = []

for i, (train_ic, tree) in enumerate(factors_to_evaluate):
    print(f"\nEvaluating Factor {i+1}: {tree}")
    print(f"  - Train IC (from training): {train_ic:.4f}") # Reward achieved during training

    # Evaluate on Training set
    train_stats = evaluate_factor_on_split(tree, train_data)
    print(f"  - Train Eval: ", end="")
    if train_stats:
         print(f"Spearman IC: {train_stats['spearman_ic']:.4f} (p={train_stats['p_spearman']:.3f}), "
               f"Pearson: {train_stats['pearson_corr']:.4f}, Valid Points: {train_stats['valid_points']}")
    else:
         print("Evaluation failed.")

    # Evaluate on Validation set
    val_stats = evaluate_factor_on_split(tree, val_data)
    print(f"  - Validation Eval: ", end="")
    if val_stats:
         print(f"Spearman IC: {val_stats['spearman_ic']:.4f} (p={val_stats['p_spearman']:.3f}), "
               f"Pearson: {val_stats['pearson_corr']:.4f}, Valid Points: {val_stats['valid_points']}")
    else:
         print("Evaluation failed.")

    # Evaluate on Test set
    test_stats = evaluate_factor_on_split(tree, test_data)
    print(f"  - Test Eval: ", end="")
    if test_stats:
         print(f"Spearman IC: {test_stats['spearman_ic']:.4f} (p={test_stats['p_spearman']:.3f}), "
               f"Pearson: {test_stats['pearson_corr']:.4f}, Valid Points: {test_stats['valid_points']}")
    else:
         print("Evaluation failed.")

    evaluation_results.append({
        'rank': i+1,
        'tree': tree,
        'train_ic_orig': train_ic,
        'train_eval': train_stats,
        'val_eval': val_stats,
        'test_eval': test_stats
    })


# --- Optional: Display summary table ---
print("\n--- Evaluation Summary Table ---")
print("Rank | Train IC | Val IC   | Test IC  | Factor Expression")
print("----------------------------------------------------------------------")
for result in evaluation_results:
     val_ic_str = f"{result['val_eval']['spearman_ic']:.4f}" if result['val_eval'] else " N/A  "
     test_ic_str = f"{result['test_eval']['spearman_ic']:.4f}" if result['test_eval'] else " N/A  "
     # Truncate long trees for display
     tree_str = result['tree']
     if len(tree_str) > 60:
         tree_str = tree_str[:57] + "..."
     print(f"{result['rank']:<4} | {result['train_ic_orig']:.4f}   | {val_ic_str} | {test_ic_str} | {tree_str}")

print("\n--- End of Process ---")