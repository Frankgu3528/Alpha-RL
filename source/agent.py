# agent.py
"""
Defines the PPO agent, including Policy and Value networks.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import warnings

# Import hyperparameters from config
from config import LR, GAMMA, EPS_CLIP, ENTROPY_COEF, VALUE_LOSS_COEF, TOP_N_FACTORS

# Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            # Use LogSoftmax for numerical stability with Categorical distribution using logits
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x.float())

# Value Network
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
        return self.fc(x.float())


# PPO Agent Class
class PPO:
    def __init__(self, state_dim, action_dim):
        self.gamma = GAMMA
        self.eps_clip = EPS_CLIP
        self.entropy_coef = ENTROPY_COEF
        self.value_loss_coef = VALUE_LOSS_COEF

        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        # Consider separate optimizers if needed, but one often works fine
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()), lr=LR
        )

        # Ensure networks are in training mode initially
        self.policy.train()
        self.value.train()

        # Tracking
        self.best_reward = -np.inf # Tracks best reward during training
        self.best_tree = None
        self.top_factors = []  # Stores (reward, tree) tuples from training
        self.factor_memory = set() # Stores unique tree strings seen

        # Loss history
        self.policy_losses = []
        self.value_losses = []

    def select_action(self, state, exploration_prob=0.1):
        """Selects an action based on policy or explores randomly."""
        state = torch.FloatTensor(state).unsqueeze(0) # Add batch dimension

        if np.random.random() < exploration_prob:
            # Explore: Choose a random valid action
            action = np.random.randint(0, self.policy.fc[-2].out_features)
            # Calculate log_prob for the random action using the current policy
            with torch.no_grad():
                log_probs_all = self.policy(state) # shape [1, action_dim]
                # Use log_probs directly with NLLLoss or convert back if needed, here just need the specific log_prob
                # dist = Categorical(logits=log_probs_all) # Use logits
                # log_prob = dist.log_prob(torch.tensor(action))
                # Directly index log_probs_all (more efficient if just needing one value)
                log_prob = log_probs_all[0, action]

            return action, log_prob.item() # Return scalar log_prob
        else:
            # Exploit: Sample action from policy distribution
            with torch.no_grad():
                log_probs_all = self.policy(state) # shape [1, action_dim]
                # Create distribution from logits (output of LogSoftmax)
                dist = Categorical(logits=log_probs_all)
                action = dist.sample()
                log_prob = dist.log_prob(action) # Calculate log_prob for the sampled action

            return action.item(), log_prob.item() # Return scalar action and log_prob

    def update(self, states, actions, log_probs_old_list, rewards, next_state, done, tree):
        """Performs the PPO update step."""
        if not states: return False # Nothing to update

        # Convert collected trajectory data to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        # Log probs collected during action selection (from the policy at that time)
        log_probs_old = torch.FloatTensor(log_probs_old_list)
        rewards = torch.FloatTensor(rewards)
        next_state = torch.FloatTensor(next_state).unsqueeze(0) # Shape [1, state_dim]

        # --- Calculate Discounted Returns (Generalized Advantage Estimation - GAE could be added here) ---
        # Simple Monte Carlo returns calculation for now
        returns = []
        discounted_reward = 0
        # If done, the value of the terminal state is 0.
        # If not done (e.g., max steps reached), maybe bootstrap with value net?
        # In this env, 'done' always means episode finished and final reward calculated.
        # So, value of terminal state is always 0.
        for reward in reversed(rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns, dtype=torch.float32).detach() # Detach returns

        # Normalize returns for stability (optional but often helpful)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # --- Calculate Advantages ---
        self.value.train() # Ensure value net is in train mode for grads
        values = self.value(states).squeeze() # Shape [num_steps]
        advantages = returns - values.detach() # Detach values when calculating advantages for policy loss

        # Normalize advantages (optional but often helpful)
        if len(advantages) > 1:
           advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


        # --- Policy Loss (PPO-Clip Objective) ---
        self.policy.train() # Ensure policy net is in train mode
        log_probs_new_all = self.policy(states) # Re-evaluate policy on states, gets log_probs [num_steps, action_dim]
        dist_new = Categorical(logits=log_probs_new_all)
        log_probs_new = dist_new.log_prob(actions) # Get log_probs for the actions taken [num_steps]
        entropy = dist_new.entropy().mean() # Average entropy over the batch

        # Ratio: exp(log_prob_new - log_prob_old)
        ratio = torch.exp(log_probs_new - log_probs_old) # Use log_probs saved during action selection

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean() # Negative because we want to maximize


        # --- Value Loss ---
        # Predict values again (necessary if value net changed during policy update, safer)
        values_pred = self.value(states).squeeze()
        value_loss = nn.MSELoss()(values_pred, returns) # Use detach returns as target

        # --- Total Loss ---
        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy


        # --- Optimization Step ---
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (helps prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
        self.optimizer.step()

        # Store loss history
        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())


        # --- Track Best Factor Found During Training ---
        final_reward = rewards[-1].item() if rewards.numel() > 0 else -np.inf
        improved = False
        if done and tree is not None and final_reward > -np.inf:
            unique_tree_str = str(tree) # Ensure it's a comparable type
            if unique_tree_str not in self.factor_memory:
                self.factor_memory.add(unique_tree_str)
                # Add to top factors if list not full or reward is high enough
                if len(self.top_factors) < TOP_N_FACTORS or final_reward > self.top_factors[-1][0]:
                    self.top_factors.append((final_reward, unique_tree_str))
                    self.top_factors.sort(key=lambda x: x[0], reverse=True)
                    self.top_factors = self.top_factors[:TOP_N_FACTORS]

            # Update overall best if needed
            if final_reward > self.best_reward:
                self.best_reward = final_reward
                self.best_tree = unique_tree_str
                improved = True

        return improved

    def get_losses(self):
        """Returns the recorded policy and value losses."""
        return self.policy_losses, self.value_losses

    def get_best_factor(self):
        """Returns the best factor tree and its training reward."""
        return self.best_tree, self.best_reward

    def get_top_factors(self):
        """Returns the list of top factors found."""
        return self.top_factors