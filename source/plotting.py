# plotting.py
"""
Functions for plotting training results.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings

def plot_training_results(episode_rewards_history, policy_losses, value_losses, filename='training_plot.png'):
    """
    Generates and saves plots for episode rewards, policy loss, and value loss.

    Args:
        episode_rewards_history (list): List of cumulative rewards per episode.
        policy_losses (list): List of policy losses per update step.
        value_losses (list): List of value losses per update step.
        filename (str): Path to save the plot image.
    """
    num_episodes = len(episode_rewards_history)
    num_losses = len(policy_losses) # Should match value_losses length

    if num_episodes == 0 and num_losses == 0:
        warnings.warn("No data available for plotting.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=False) # Don't share x if lengths differ
    fig.suptitle("Training Process Overview", fontsize=16)

    # Determine common x-axis limits based on available data
    max_len = max(num_episodes, num_losses)
    x_axis_episodes = np.arange(num_episodes)
    x_axis_losses = np.arange(num_losses)

    # --- Rewards Plot ---
    ax = axes[0]
    if num_episodes > 0:
        ax.plot(x_axis_episodes, episode_rewards_history, label='Episode Reward Sum', alpha=0.6, linewidth=1)
        # Moving average (window size adapts slightly if few episodes)
        window_size = min(50, max(1, num_episodes // 5))
        if num_episodes >= window_size:
            moving_avg_rewards = pd.Series(episode_rewards_history).rolling(window_size).mean()
            ax.plot(x_axis_episodes, moving_avg_rewards, label=f'Reward MA ({window_size} eps)', color='red', linewidth=1.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Reward')
        ax.set_title('Episode Rewards')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim(0, max_len) # Use common limit
    else:
        ax.set_title('Episode Rewards (No Data)')


    # --- Policy Loss Plot ---
    ax = axes[1]
    if num_losses > 0:
        ax.plot(x_axis_losses, policy_losses, label='Policy Loss', alpha=0.6, linewidth=1)
        window_size = min(50, max(1, num_losses // 5))
        if num_losses >= window_size:
            moving_avg_ploss = pd.Series(policy_losses).rolling(window_size).mean()
            ax.plot(x_axis_losses, moving_avg_ploss, label=f'Policy Loss MA ({window_size} steps)', color='blue', linewidth=1.5)
        ax.set_xlabel('Update Step (Episode)')
        ax.set_ylabel('Loss')
        ax.set_title('Policy Network Loss')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim(0, max_len) # Use common limit
    else:
         ax.set_title('Policy Loss (No Data)')


    # --- Value Loss Plot ---
    ax = axes[2]
    if num_losses > 0:
        ax.plot(x_axis_losses, value_losses, label='Value Loss', alpha=0.6, linewidth=1)
        window_size = min(50, max(1, num_losses // 5))
        if num_losses >= window_size:
            moving_avg_vloss = pd.Series(value_losses).rolling(window_size).mean()
            ax.plot(x_axis_losses, moving_avg_vloss, label=f'Value Loss MA ({window_size} steps)', color='green', linewidth=1.5)
        ax.set_xlabel('Update Step (Episode)')
        ax.set_ylabel('Loss')
        ax.set_title('Value Network Loss')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim(0, max_len) # Use common limit
    else:
        ax.set_title('Value Loss (No Data)')


    # --- Save Figure ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    try:
        plt.savefig(filename)
        print(f"\nTraining plots saved to '{filename}'")
    except Exception as e:
        warnings.warn(f"Could not save plot: {e}")
    # plt.show() # Optionally display the plot
    plt.close(fig) # Close the figure to free memory