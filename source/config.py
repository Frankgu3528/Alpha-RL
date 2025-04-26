# config.py
"""
Configuration variables and hyperparameters for the RL factor finder.
"""
import numpy as np # Need numpy for NaN representation later if needed

# --- Data Configuration ---
DATA_PATH = '/Users/frank/Desktop/code/AlphaRL/data/binance_BTCUSDT_1min_90days.csv'
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2

# --- Technical Indicator Parameters ---
# Define parameters for the indicators you want to add
SMA_PERIODS = [10, 30, 60]  # Periods for Simple Moving Averages
EMA_PERIODS = [10, 30, 60]  # Periods for Exponential Moving Averages
RSI_PERIOD = 14
ATR_PERIOD = 14
BBANDS_PERIOD = 20
BBANDS_STD_DEV = 2.0
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# --- Feature Engineering ---
# Base features
BASE_FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']

# Dynamically generate indicator column names based on parameters
# These names must match the output columns from pandas-ta
INDICATOR_FEATURES = []
INDICATOR_FEATURES.extend([f"SMA_{p}" for p in SMA_PERIODS])
INDICATOR_FEATURES.extend([f"EMA_{p}" for p in EMA_PERIODS])
INDICATOR_FEATURES.append(f"RSI_{RSI_PERIOD}")
INDICATOR_FEATURES.append(f"ATRr_{ATR_PERIOD}") # Changed ATR_14 to ATRr_14
# Bollinger Bands (pandas-ta creates BBL_period_std, BBM_period_std, BBU_period_std)
INDICATOR_FEATURES.extend([f"BBL_{BBANDS_PERIOD}_{BBANDS_STD_DEV}",
                           f"BBM_{BBANDS_PERIOD}_{BBANDS_STD_DEV}",
                           f"BBU_{BBANDS_PERIOD}_{BBANDS_STD_DEV}"])
# MACD (pandas-ta creates MACD_fast_slow_signal, MACDh_f_s_s, MACDs_f_s_s) - Let's use Histogram (MACDh)
INDICATOR_FEATURES.append(f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}")


# Combine base features and indicator features
FEATURES = BASE_FEATURES + INDICATOR_FEATURES

# Available operators and constants for the RL agent
OPERATORS = ['+', '-', '*', '/', 'log', 'exp', 'sqrt', 'abs', '**2']
CONSTANTS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, -0.1, -0.5, -1.0, -2.0] # Added NaN as a constant

# --- Environment Configuration ---
MAX_DEPTH = 10
MAX_OPERATIONS = 15

# --- Agent Configuration (PPO) ---
STATE_DIM = 3
LR = 0.0003
GAMMA = 0.98
GAE_LAMBDA = 0.95   # GAE parameter (Lambda) <-- ADD THIS
EPS_CLIP = 0.2
ENTROPY_COEF = 0.01
VALUE_LOSS_COEF = 0.5

# --- Training Configuration ---
NUM_EPISODES = 5000 # Or keep original value
EXPLORATION_START = 0.5
EXPLORATION_END = 0.05
EXPLORATION_DECAY_RATE = 0.8
MAX_STEPS_PER_EPISODE = MAX_DEPTH + 5
EARLY_STOPPING_STREAK = 500
TARGET_REWARD_THRESHOLD = 0.95

# --- Evaluation & Plotting ---
TOP_N_FACTORS = 20
PLOT_FILENAME = 'training_plot_modular_with_ta.png' # Updated filename
EVAL_MIN_POINTS = 50