# main.py
"""
Main script to run the RL factor finding process.
"""
import torch
import numpy as np
import pandas as pd
import time
import warnings
import sys # For checking exit status

# Import modules
import config
from data_handler import load_and_split_data
from environment import FactorEnv 
from agent import PPO
from evaluation import evaluate_factor_on_split
from plotting import plot_training_results

# Filter warnings for cleaner output during run
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning) # Suppress some matplotlib/pandas warnings


def run_training():
    """Orchestrates the entire training and evaluation process."""
    print("--- Starting Factor Finding Process ---")
    start_time = time.time()

    # --- 1. Load and Prepare Data ---
    print("\n[Step 1/5] Loading and splitting data...")
    train_data, val_data, test_data = load_and_split_data(
        config.DATA_PATH, config.TRAIN_RATIO, config.VAL_RATIO
    )
    if train_data is None:
        print("Exiting due to data loading/splitting error.")
        return

    # --- 2. Initialize Environment and Agent ---
    print("\n[Step 2/5] Initializing environment and agent...")
    try:
        # Environment uses only training data for learning rewards
        train_env = FactorEnv(train_data)
    except ValueError as e:
         print(f"Error initializing environment: {e}")
         return

    # Calculate action dimension based on config
    # action_dim = len(config.FEATURES) + len(config.CONSTANTS) + len(config.OPERATORS)
    # --- FIX: Use action_dim directly from the environment instance ---
    action_dim = train_env.action_dim # Get action dim (including STOP) from env
    print(f"Action Dimension (including STOP action): {action_dim}")
    # Initialize PPO Agent
    ppo = PPO(state_dim=config.STATE_DIM, action_dim=action_dim)

    # --- 3. Training Loop ---
    print("\n[Step 3/5] Starting training loop...")
    episode_rewards_history = []
    best_factors_unchanged_streak = 0
    training_start_time = time.time()

    for episode in range(config.NUM_EPISODES):
        state = train_env.reset()
        done = False
        episode_reward_sum = 0
        # Store trajectory data for this episode
        states, actions, log_probs_old, rewards = [], [], [], []
        step_count = 0

        # Calculate exploration rate for this episode
        # Linear decay from START to END over DECAY_RATE portion of episodes
        decay_episodes = int(config.NUM_EPISODES * config.EXPLORATION_DECAY_RATE)
        progress = min(1.0, episode / decay_episodes) if decay_episodes > 0 else 1.0
        exploration_rate = config.EXPLORATION_START + (config.EXPLORATION_END - config.EXPLORATION_START) * progress
        exploration_rate = max(config.EXPLORATION_END, exploration_rate) # Ensure it doesn't go below min

        # --- Run one episode ---
        while not done and step_count < config.MAX_STEPS_PER_EPISODE:
            action, log_prob = ppo.select_action(state, exploration_prob=exploration_rate)
            next_state, reward, done = train_env.step(action)

            # Store transition
            #states.append(state.tolist()) # Store as list
            states.append(state) 
            actions.append(action)
            log_probs_old.append(log_prob)
            rewards.append(reward)

            state = next_state
            episode_reward_sum += reward # Accumulate rewards (mostly 0 until end)
            step_count += 1
        # --- End of episode ---

        episode_rewards_history.append(episode_reward_sum) # Store final cumulative reward

        # Perform PPO update using the collected trajectory
        improved = ppo.update(states, actions, log_probs_old, rewards, next_state, done, train_env.tree)

        # --- Logging and Early Stopping ---
        if improved:
            current_best_tree, current_best_reward = ppo.get_best_factor()
            print(f"Episode {episode:>5}/{config.NUM_EPISODES}: New best factor! Train Reward: {current_best_reward:.4f}, Tree: {current_best_tree}")
            best_factors_unchanged_streak = 0
        else:
            best_factors_unchanged_streak += 1

        # Periodic status update
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards_history[-50:])
            policy_losses, value_losses = ppo.get_losses()
            avg_policy_loss = np.mean(policy_losses[-50:]) if policy_losses else 0
            avg_value_loss = np.mean(value_losses[-50:]) if value_losses else 0
            _, current_best_reward = ppo.get_best_factor() # Get current best reward
            print(f"Episode {episode+1:>5}/{config.NUM_EPISODES} | Avg Reward (last 50): {avg_reward:+.4f} | "
                  f"Avg P Loss: {avg_policy_loss:.4f} | Avg V Loss: {avg_value_loss:.4f} | "
                  f"Explore: {exploration_rate:.3f} | Best Train Reward: {current_best_reward:.4f}")

        # Early stopping check
        if episode > 500 and best_factors_unchanged_streak >= config.EARLY_STOPPING_STREAK:
            print(f"\nEarly stopping triggered: Best training reward hasn't improved for {config.EARLY_STOPPING_STREAK} episodes.")
            break
        # Optional: Stop if reward is consistently high
        if episode > 100 and avg_reward > config.TARGET_REWARD_THRESHOLD:
             if all(r > config.TARGET_REWARD_THRESHOLD for r in episode_rewards_history[-10:]): # Check last 10
                 print(f"\nStopping early: Average reward threshold {config.TARGET_REWARD_THRESHOLD} met consistently.")
                 break


    training_duration = time.time() - training_start_time
    print(f"\n--- Training Complete ({training_duration:.2f} seconds) ---")
    best_tree_final, best_reward_final = ppo.get_best_factor()
    if best_tree_final:
        print(f"Best factor found during training: {best_tree_final}")
        print(f"Best training reward (IC on train set): {best_reward_final:.4f}")
    else:
        print("No best factor found during training.")

    # --- 4. Plot Training Results ---
    print("\n[Step 4/5] Generating training plots...")
    policy_losses, value_losses = ppo.get_losses()
    plot_training_results(episode_rewards_history, policy_losses, value_losses, config.PLOT_FILENAME)

    # --- 5. Out-of-Sample Evaluation ---
    print("\n[Step 5/5] Performing Out-of-Sample Evaluation...")
    top_factors = ppo.get_top_factors() # Get list of (reward, tree) tuples

    if not top_factors:
        print("No factors found during training to evaluate.")
    else:
        print(f"\nEvaluating top {len(top_factors)} factors found during training:")
        evaluation_results = []
        for i, (train_ic, tree) in enumerate(top_factors):
            print(f"\n--- Evaluating Factor {i+1}/{len(top_factors)} ---")
            print(f"  Expression: {tree}")
            print(f"  Reported Train IC: {train_ic:.4f}") # Reward achieved during training run

            # Evaluate on Train, Validation, and Test sets
            results = {}
            for split_name, data_split in [('Train', train_data), ('Validation', val_data), ('Test', test_data)]:
                if data_split is not None and not data_split.empty:
                    stats = evaluate_factor_on_split(tree, data_split)
                    results[split_name] = stats
                    print(f"  - {split_name} Eval: ", end="")
                    if stats and stats.get('error') is None:
                         print(f"Spearman IC: {stats['spearman_ic']:.4f} (p={stats['p_spearman']:.3f}), "
                               f"Pearson: {stats['pearson_corr']:.4f}, Std: {stats['factor_std']:.3g}, "
                               f"Valid Points: {stats['valid_points']}")
                    elif stats:
                         print(f"Evaluation failed ({stats.get('error', 'Unknown error')}). Valid points: {stats.get('valid_points', 'N/A')}")
                    else:
                         print("Evaluation failed (returned None).")
                else:
                     print(f"  - {split_name} Eval: Skipped (No data)")
                     results[split_name] = None # Store None if split is empty

            evaluation_results.append({
                'rank': i+1, 'tree': tree, 'train_ic_orig': train_ic,
                'train_eval': results.get('Train'),
                'val_eval': results.get('Validation'),
                'test_eval': results.get('Test')
            })

        # --- Print Summary Table ---
        print("\n--- Evaluation Summary Table (Spearman IC) ---")
        print("-" * 80)
        print(f"{'Rank':<5} | {'Train IC':<10} | {'Val IC':<10} | {'Test IC':<10} | {'Factor Expression (truncated)'}")
        print("-" * 80)
        for result in evaluation_results:
             # Safely access nested dictionary keys
             train_ic_str = f"{result['train_eval']['spearman_ic']:.4f}" if result.get('train_eval') and result['train_eval'].get('error') is None else " N/A "
             val_ic_str = f"{result['val_eval']['spearman_ic']:.4f}" if result.get('val_eval') and result['val_eval'].get('error') is None else " N/A "
             test_ic_str = f"{result['test_eval']['spearman_ic']:.4f}" if result.get('test_eval') and result['test_eval'].get('error') is None else " N/A "
             tree_str = result['tree']
             tree_display = (tree_str[:45] + '...') if len(tree_str) > 48 else tree_str
             print(f"{result['rank']:<5} | {train_ic_str:<10} | {val_ic_str:<10} | {test_ic_str:<10} | {tree_display}")
        print("-" * 80)


    total_duration = time.time() - start_time
    print(f"\n--- Process Finished ({total_duration:.2f} seconds) ---")

# --- Main Execution Guard ---
if __name__ == "__main__":
    run_training()