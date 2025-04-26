# config.py
"""
Configuration variables and hyperparameters for the RL factor finder.
"""

# --- Data Configuration ---
# Use absolute path or make sure it's relative to where main.py is run
DATA_PATH = '/Users/frank/Desktop/code/AlphaRL/data/binance_BTCUSDT_1min_1year.csv'
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
# TEST_RATIO is implicitly 1.0 - TRAIN_RATIO - VAL_RATIO

# --- Feature Engineering ---
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']
OPERATORS = ['+', '-', '*', '/', 'log', 'exp', 'sqrt', 'abs', '**2']
CONSTANTS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, -0.1, -0.5, -1.0, -2.0]

# --- Environment Configuration ---
MAX_DEPTH = 8       # Max depth of the expression tree
MAX_OPERATIONS = 15 # Max number of operations allowed in an expression

# --- Agent Configuration (PPO) ---
STATE_DIM = 3       # State: [depth, tree_exists, ops_count]
# ACTION_DIM will be calculated in main.py based on features, operators, constants
LR = 0.0003         # Learning rate for Adam optimizer
GAMMA = 0.98        # Discount factor for rewards
EPS_CLIP = 0.2      # Clipping parameter for PPO
ENTROPY_COEF = 0.01 # Coefficient for entropy bonus
VALUE_LOSS_COEF = 0.5 # Coefficient for value loss

# --- Training Configuration ---
NUM_EPISODES = 5000
EXPLORATION_START = 0.5
EXPLORATION_END = 0.05
EXPLORATION_DECAY_RATE = 0.8 # Percentage of total episodes over which decay happens
MAX_STEPS_PER_EPISODE = MAX_DEPTH + 5 # Safety break for episode steps
EARLY_STOPPING_STREAK = 500 # Stop if best reward doesn't improve for this many episodes
TARGET_REWARD_THRESHOLD = 0.95 # Optional: Stop if a very high reward is consistently achieved

# --- Evaluation & Plotting ---
TOP_N_FACTORS = 20  # Number of top factors to store and evaluate
PLOT_FILENAME = 'training_plot_modular.png'
EVAL_MIN_POINTS = 50 # Minimum valid points required for evaluation on a split