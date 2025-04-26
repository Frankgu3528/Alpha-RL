import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=RuntimeWarning)

# 加载数据
data = pd.read_csv('/Users/frank/Desktop/code/binance/data/binance_BTCUSDT_1min_90days.csv')
try:
    data['Open time'] = pd.to_datetime(data['Open time'])
    data['Close time'] = pd.to_datetime(data['Close time'])
except Exception as e:
    print(f"Date parsing warning (non-critical): {e}")
data['Return'] = data['Close'].shift(-1) / data['Close'] - 1

# 安全数学函数
def safe_log(x):
    return np.log(np.maximum(x, 1e-10))

def safe_exp(x):
    return np.exp(np.clip(x, -20, 20))

def safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)

def safe_sqrt(x):
    return np.sqrt(np.maximum(x, 0))

def safe_power2(x):
    return np.square(np.clip(x, -1e5, 1e5))

def normalize_series(x):
    if isinstance(x, np.ndarray) and x.size > 1:
        std = np.std(x)
        if std > 1e-10:
            return (x - np.mean(x)) / std
    return x

# 环境类
class FactorEnv:
    def __init__(self, data, max_depth=7):
        self.data = data
        self.max_depth = max_depth
        self.features = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.operators = ['+', '-', '*', '/', 'log', 'exp', 'sqrt', 'abs', '**2']
        self.constants = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, -0.1, -0.5, -1.0, -2.0]
        self.action_space = len(self.features) + len(self.constants) + len(self.operators) + 1
        self.stop_action = self.action_space - 1
        self.reset()

    def reset(self):
        self.tree = None
        self.current_depth = 0
        self.operations_count = 0
        self.done = False
        return self.get_state()

    def get_state(self):
        tree_exists = 1 if self.tree is not None else 0
        return [self.current_depth, tree_exists, min(self.operations_count, 10)]

    def step(self, action):
        if action == self.stop_action:
            self.done = True
            if self.tree is None:
                reward = -1
            else:
                reward = self.calculate_reward()
        else:
            total_features = len(self.features)
            total_constants = len(self.constants)
            
            if self.tree is None:
                if action < total_features:
                    self.tree = self.features[action]
                elif action < total_features + total_constants:
                    constant_idx = action - total_features
                    self.tree = str(self.constants[constant_idx])
                else:
                    self.tree = np.random.choice(self.features)
                    reward = -0.1
            else:
                if action < total_features:
                    feature = self.features[action]
                    operator = np.random.choice(['+', '-', '*', '/'])
                    if np.random.random() < 0.5:
                        self.tree = f"({self.tree} {operator} {feature})"
                    else:
                        self.tree = f"({feature} {operator} {self.tree})"
                    self.operations_count += 1
                elif action < total_features + total_constants:
                    constant_idx = action - total_features
                    constant = self.constants[constant_idx]
                    operator = np.random.choice(['+', '-', '*', '/'])
                    if np.random.random() < 0.5:
                        self.tree = f"({self.tree} {operator} {constant})"
                    else:
                        self.tree = f"({constant} {operator} {self.tree})"
                    self.operations_count += 1
                else:
                    operator_idx = action - (total_features + total_constants)
                    if operator_idx < len(self.operators):
                        operator = self.operators[operator_idx]
                        if operator in ['+', '-', '*', '/', '**2']:
                            if operator == '**2':
                                self.tree = f"safe_power2({self.tree})"
                            else:
                                operand = np.random.choice(self.features + [str(c) for c in self.constants])
                                if np.random.random() < 0.5:
                                    self.tree = f"({self.tree} {operator} {operand})"
                                else:
                                    self.tree = f"({operand} {operator} {self.tree})"
                        elif operator == 'log':
                            self.tree = f"safe_log({self.tree})"
                        elif operator == 'exp':
                            self.tree = f"safe_exp({self.tree})"
                        elif operator == 'sqrt':
                            self.tree = f"safe_sqrt({self.tree})"
                        elif operator == 'abs':
                            self.tree = f"np.abs({self.tree})"
                        self.operations_count += 1

            self.current_depth += 1
            if self.current_depth >= self.max_depth:
                self.done = True
                reward = self.calculate_reward()
            else:
                reward = 0

        return self.get_state(), reward, self.done

    def calculate_reward(self):
        if self.tree is None:
            return -1
        try:
            factor_values = self.evaluate_tree()
            if factor_values is not None:
                factor_values = pd.Series(factor_values)
                valid_values = factor_values.dropna()
                if len(valid_values) > len(self.data) * 0.5:
                    lower_bound = factor_values.quantile(0.01)
                    upper_bound = factor_values.quantile(0.99)
                    factor_values = factor_values.clip(lower=lower_bound, upper=upper_bound)
                    valid_data = pd.DataFrame({
                        'factor': factor_values,
                        'return': self.data['Return']
                    }).dropna()
                    if len(valid_data) > 100:
                        if valid_data['factor'].std() > 1e-6:
                            from scipy.stats import spearmanr
                            corr, p_value = spearmanr(valid_data['factor'], valid_data['return'])
                            if not np.isnan(corr):
                                reward = abs(corr)
                                if p_value < 0.05:
                                    reward *= 1.2
                                return reward
                            else:
                                return -0.5
                        else:
                            return -0.5
                    else:
                        return -0.5
                else:
                    return -0.7
            else:
                return -1
        except Exception as e:
            print(f"Reward calculation error: {e}")
            return -1

    def evaluate_tree(self):
        try:
            expr = self.tree
            var_dict = {feature: self.data[feature].values.copy() for feature in self.features}
            for constant in self.constants:
                var_dict[str(constant)] = constant
            safe_dict = {
                'np': np,
                'safe_log': safe_log,
                'safe_exp': safe_exp,
                'safe_div': safe_div,
                'safe_sqrt': safe_sqrt,
                'safe_power2': safe_power2,
                'normalize': normalize_series
            }
            modified_expr = expr.replace('/', 'safe_div')
            factor_values = eval(modified_expr, {"__builtins__": {}}, {**var_dict, **safe_dict})
            if isinstance(factor_values, np.ndarray):
                factor_values = np.nan_to_num(factor_values, nan=np.nan, posinf=np.nan, neginf=np.nan)
            return factor_values
        except Exception as e:
            print(f"Tree evaluation error: {e} for tree: {self.tree}")
            return None

# 策略和价值网络
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

# PPO类
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
        self.top_factors = []
        self.factor_memory = set()

    def select_action(self, state, exploration=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)
        if np.random.random() < exploration:
            action = np.random.randint(0, self.policy.fc[-2].out_features)
            with torch.no_grad():
                probs = self.policy(state)
                dist = Categorical(probs)
                log_prob = dist.log_prob(torch.tensor(action))
            return action, log_prob.item()
        with torch.no_grad():
            probs = self.policy(state)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def update(self, states, actions, log_probs, rewards, next_state, done, tree):
        if len(states) == 0:
            return False
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        log_probs = torch.FloatTensor(log_probs)
        rewards = torch.FloatTensor(rewards)
        next_state = torch.FloatTensor([next_state]).unsqueeze(0)
        final_reward = rewards[-1]
        if final_reward > 0 and tree is not None and tree not in self.factor_memory:
            self.factor_memory.add(tree)
            self.top_factors.append((final_reward, tree))
            self.top_factors.sort(key=lambda x: x[0], reverse=True)
            self.top_factors = self.top_factors[:20]
            if final_reward > self.best_reward:
                self.best_reward = final_reward
                self.best_tree = tree
                return True
        try:
            with torch.no_grad():
                value_next = self.value(next_state).squeeze()
                returns = []
                discounted_reward = 0
                for reward in reversed(rewards):
                    discounted_reward = reward + self.gamma * discounted_reward
                    returns.insert(0, discounted_reward)
                returns = torch.FloatTensor(returns)
                if len(returns) > 1 and returns.std() > 1e-8:
                    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            values = self.value(states).squeeze()
            if values.dim() == 0:
                values = values.unsqueeze(0)
            advantage = returns - values
            probs = self.policy(states)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values, returns)
            entropy = dist.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
            self.optimizer.step()
        except Exception as e:
            print(f"Policy update error: {e}")
        return False

# 训练过程
env = FactorEnv(data, max_depth=7)
state_dim = 3
action_dim = env.action_space
ppo = PPO(state_dim, action_dim)

num_episodes = 1000
episode_rewards = []
for episode in range(num_episodes):
    state = env.reset()
    done = False
    episode_reward = 0
    states, actions, log_probs, rewards = [], [], [], []
    exploration_rate = max(0.05, 0.5 - 0.45 * episode / (num_episodes / 2))
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
    if episode % 50 == 0 or improved:
        print(f"Episode {episode}, Reward: {episode_reward:.4f}, Tree: {env.tree}")
        if improved:
            print(f"New best factor found! IC: {ppo.best_reward:.4f}, Expression: {ppo.best_tree}")

plt.figure(figsize=(12, 6))
plt.plot(episode_rewards, label='Episode Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward over Episodes')
plt.legend()
plt.savefig('training_plot.png')

print("\n=== Training Complete ===")
print(f"Best factor found: {ppo.best_tree}")
print(f"IC value: {ppo.best_reward:.4f}")
print("\nTop performing factors:")
for i, (ic, tree) in enumerate(ppo.top_factors):
    print(f"{i+1}. IC: {ic:.4f}, Expression: {tree}")