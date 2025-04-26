import numpy as np
import pandas as pd
from typing import Tuple, Dict
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# 生成示例数据（10只股票，100个交易日，3个基础因子）
def generate_demo_data():
    stocks = 10
    days = 100
    features = ['momentum', 'volatility', 'volume']
    
    data = {
        'returns': np.random.randn(stocks, days)*0.02,  # 日收益率
        'features': np.random.randn(stocks, days, len(features))
    }
    return data

# 自定义Gym环境
class FactorGenerationEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, data: Dict):
        super(FactorGenerationEnv, self).__init__()
        
        # 示例数据加载
        self.data = data
        self.current_stock = 0  # 当前处理的股票索引
        self.current_day = 0     # 当前交易日
        
        # 动作空间：选择操作符或终止
        # 0: +, 1: -, 2: *, 3: /, 4: sqrt, 5: log, 6: exp, 7: abs, 8: sign, 9: stop
        self.action_space = spaces.Discrete(10)
        
        # 状态空间：当前表达式树信息（最近5个操作）
        self.observation_space = spaces.Box(
            low=-1e5, high=1e5, shape=(5,), dtype=np.float32
        )
        
        # 表达式构建相关
        self.expression = []
        self.max_length = 10  # 最大表达式长度
        
    def _get_obs(self):
        """获取状态观测值（最近5个操作的特征）"""
        print(self.expression)
        if len(self.expression) == 0:
            return np.zeros(5, dtype=np.float32)
        
        # 获取最近五个操作的特征（示例简化处理）
        recent_ops = self.expression[-5:]
        return np.pad(recent_ops, (0, 5-len(recent_ops)), mode='constant').astype(np.float32)
    
    def _calculate_reward(self):
        """计算奖励（使用模拟IC和因子收益）"""
        # 模拟IC值
        simulated_IC = np.clip(np.random.randn()*0.2, -0.3, 0.3)
        
        # 模拟因子收益
        factor_return = np.clip(np.random.randn()*0.1, -0.2, 0.2)
        
        # 长度惩罚
        length_penalty = -0.05 * len(self.expression)
        
        return simulated_IC + factor_return + length_penalty
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # 执行动作
        if action == 9:  # 终止动作
            terminated = True
            truncated = False  # 任务未因时间限制截断
        else:
            self.expression.append(action)
            terminated = len(self.expression) >= self.max_length
            truncated = False  # 任务未因时间限制截断
        
        # 计算奖励
        reward = self._calculate_reward() if terminated else 0
        
        # 更新状态
        obs = self._get_obs()
        
        # 返回五个值
        return obs, reward, terminated, truncated, {}
    
    def reset(self, seed=None, options=None):
        # 重置环境状态
        self.expression = []
        return self._get_obs(), {}
    
    def render(self, mode='human'):
        # 显示当前生成的表达式
        op_map = {
            0: '+', 1: '-', 2: '*', 3: '/', 
            4: 'sqrt', 5: 'log', 6: 'exp', 
            7: 'abs', 8: 'sign', 9: 'STOP'
        }
        expr_str = " -> ".join([op_map.get(a, '?') for a in self.expression])
        print(f"Current Expression: [{expr_str}]")

# 训练流程
def main():
    # 生成示例数据
    data = generate_demo_data()
    
    # 创建环境
    env = FactorGenerationEnv(data)
    check_env(env)  # 检查环境兼容性
    
    # 初始化PPO模型
    model = PPO(
        "MlpPolicy", 
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=10
    )
    
    # 训练模型
    model.learn(total_timesteps=100000)
    
    # 测试训练后的模型
    obs, _ = env.reset()
    for _ in range(env.max_length + 2):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        env.render()
        if terminated or truncated:
            break

if __name__ == "__main__":
    main()