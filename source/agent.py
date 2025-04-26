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
from config import LR, GAMMA, EPS_CLIP, ENTROPY_COEF, VALUE_LOSS_COEF, TOP_N_FACTORS, GAE_LAMBDA

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
        self.gae_lambda = GAE_LAMBDA # <-- Store lambda

        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()), lr=LR
        )
        self.policy.train()
        self.value.train()
        self.best_reward = -np.inf
        self.best_tree = None
        self.top_factors = []
        self.factor_memory = set()
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
        """Performs the PPO update step using GAE."""
        if not states: return False

        # Convert collected trajectory data to tensors
        # Ensure states is correctly shaped [num_steps, state_dim]
        states = torch.FloatTensor(np.array(states)) # Explicitly convert list of lists/arrays
        actions = torch.LongTensor(actions)
        log_probs_old = torch.FloatTensor(log_probs_old_list) # Log probs from sampling time
        rewards = torch.FloatTensor(rewards) # Shape [num_steps]
        # Ensure next_state is correctly shaped [1, state_dim] for value prediction
        next_state = torch.FloatTensor(np.array(next_state)).unsqueeze(0)

        # Put networks in eval mode for inference if using dropout/batchnorm,
        # but for simple linear networks, train() mode is fine.
        # self.policy.eval()
        # self.value.eval()
        # We need gradients later, so keep in train() or switch back. Let's keep in train().

        # ======================================================================
        # START OF MODIFIED SECTION: GAE Calculation
        # ======================================================================

        with torch.no_grad(): # Value predictions used for targets/advantages shouldn't have gradients
            values = self.value(states).squeeze() # Values for states s_t: V(s_0), V(s_1), ..., V(s_{T-1})
            # Value for the final next state s_T
            # If episode was 'done', the value of the terminal state is 0.
            # In this env, done is always True at the end of an episode.
            last_value = torch.tensor(0.0) # V(s_T) = 0 for terminal state

            # Initialize advantages and returns tensors
            advantages = torch.zeros_like(rewards)
            returns = torch.zeros_like(rewards) # Target for value function (MC returns)

            # Calculate GAE advantages and MC returns backward
            gae_advantage = 0
            mc_return = last_value # Start calculation from the value of the state after the last action
            for t in reversed(range(len(rewards))):
                # Monte Carlo Return R_t = r_t + gamma * R_{t+1}
                mc_return = rewards[t] + self.gamma * mc_return
                returns[t] = mc_return

                # TD Error (delta_t) = r_t + gamma * V(s_{t+1}) - V(s_t)
                # Determine V(s_{t+1})
                value_next = values[t+1] if t < len(rewards) - 1 else last_value
                td_error = rewards[t] + self.gamma * value_next - values[t]

                # GAE Advantage A_t = delta_t + gamma * lambda * A_{t+1}
                gae_advantage = td_error + self.gamma * self.gae_lambda * gae_advantage
                advantages[t] = gae_advantage

        # Detach returns before using as targets for value loss
        returns = returns.detach()
        # Advantages are already calculated within no_grad block, but detaching explicitly is fine too.
        advantages = advantages.detach()

        # Normalize advantages (optional but recommended)
        if len(advantages) > 1:
           advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Normalize returns (optional) - Already normalized if using MC returns
        # if len(returns) > 1:
        #     returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # ======================================================================
        # END OF MODIFIED SECTION
        # ======================================================================


        # --- Policy Loss (PPO-Clip Objective) ---
        # Ensure networks are back in training mode if they were switched to eval()
        self.policy.train()
        self.value.train()

        log_probs_new_all = self.policy(states) # Re-evaluate policy on states
        dist_new = Categorical(logits=log_probs_new_all)
        log_probs_new = dist_new.log_prob(actions) # Log probs of actions taken under current policy
        entropy = dist_new.entropy().mean()

        # Calculate PPO ratio r_t(theta) = exp(log pi_theta(a_t|s_t) - log pi_theta_old(a_t|s_t))
        ratio = torch.exp(log_probs_new - log_probs_old) # Use detached log_probs_old

        # Clipped surrogate objective
        surr1 = ratio * advantages # Use GAE advantages here
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages # Use GAE advantages here
        policy_loss = -torch.min(surr1, surr2).mean() # Maximize objective -> Minimize negative objective


        # --- Value Loss ---
        # Predict values again to ensure gradients flow correctly for the value network update
        values_pred = self.value(states).squeeze()
        value_loss = nn.MSELoss()(values_pred, returns) # Target is MC returns


        # --- Total Loss ---
        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy


        # --- Optimization Step ---
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
        self.optimizer.step()

        # Store loss history
        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())


        # --- Track Best Factor Found During Training ---
        # (This part remains the same, using the actual final reward from the episode)
        final_reward = rewards[-1].item() if rewards.numel() > 0 else -np.inf
        improved = False
        if done and tree is not None and final_reward > -np.inf:
            unique_tree_str = str(tree)
            if unique_tree_str not in self.factor_memory:
                self.factor_memory.add(unique_tree_str)
                if len(self.top_factors) < TOP_N_FACTORS or final_reward > self.top_factors[-1][0]:
                    self.top_factors.append((final_reward, unique_tree_str))
                    self.top_factors.sort(key=lambda x: x[0], reverse=True)
                    self.top_factors = self.top_factors[:TOP_N_FACTORS]

            if final_reward > self.best_reward:
                # Don't print here, let main loop handle it based on 'improved' flag
                self.best_reward = final_reward
                self.best_tree = unique_tree_str
                improved = True

        return improved # Return whether the best factor was improved this step


    def get_losses(self):
        """Returns the recorded policy and value losses."""
        return self.policy_losses, self.value_losses

    def get_best_factor(self):
        """Returns the best factor tree and its training reward."""
        return self.best_tree, self.best_reward

    def get_top_factors(self):
        """Returns the list of top factors found."""
        return self.top_factors