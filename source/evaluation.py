# evaluation.py
"""
Function for evaluating factor performance on different data splits.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import warnings

# Import environment class and config for evaluation settings
from environment import FactorEnv # Need to instantiate env for evaluation
from config import EVAL_MIN_POINTS, MAX_DEPTH # Use same depth for consistency


def calculate_factor_returns(factor_values, returns, n_groups=5):
    """计算因子分组收益"""
    try:
        # 确保数据有效
        valid_mask = np.isfinite(factor_values) & np.isfinite(returns)
        if valid_mask.sum() < 50:  # 确保有足够的样本
            return None
            
        valid_factors = factor_values[valid_mask]
        valid_returns = returns[valid_mask]
        
        # 打印收益率统计信息用于调试
        print(f"\n收益率统计: \n均值={np.mean(valid_returns):.6f}, "
              f"标准差={np.std(valid_returns):.6f}, "
              f"最大值={np.max(valid_returns):.6f}, "
              f"最小值={np.min(valid_returns):.6f}")
        
        # 将收益率转换为百分比形式（如果是小数形式）
        if np.abs(np.mean(valid_returns)) < 0.1:  # 判断是否为小数形式
            valid_returns = valid_returns * 100  # 转换为百分比
        
        # 因子分组（使用pandas的qcut进行等分位数分组）
        df = pd.DataFrame({
            'factor': valid_factors,
            'return': valid_returns
        })
        
        # 按因子值分组（处理重复值）
        try:
            df['group'] = pd.qcut(df['factor'], n_groups, labels=False)
        except ValueError:  # 处理重复值情况
            df['group'] = pd.qcut(df['factor'], n_groups, labels=False, duplicates='drop')
        
        # 计算各组平均收益
        group_returns = df.groupby('group')['return'].mean()
        
        # 打印分组收益统计信息
        print("\n分组收益统计:")
        for group in range(n_groups):
            print(f"Group {group}: {group_returns[group]:.4f}%")
        
        # 计算多空组合收益（最高分组减最低分组）
        long_short_return = float(group_returns.iloc[-1] - group_returns.iloc[0])
        
        # 返回结果字典
        return {
            'group_returns': group_returns.values.tolist(),
            'long_short_return': long_short_return,
            'top_group_return': float(group_returns.iloc[-1]),
            'bottom_group_return': float(group_returns.iloc[0])
        }
        
    except Exception as e:
        print(f"计算收益率时出错: {str(e)}")
        return None

def evaluate_factor_on_split(factor_expr, data_split):
    """评估因子在特定数据集上的表现"""
    try:
        # 创建环境实例
        eval_env = FactorEnv(data_split)
        eval_env.tree = factor_expr
        
        # 计算因子值
        factor_values = eval_env.evaluate_tree()
        if factor_values is None:
            return {'error': 'factor_calculation_failed'}
            
        # 获取下期收益（确保使用前向收益）
        returns = data_split['Return'].values
        
        # 计算IC相关指标
        valid_mask = np.isfinite(factor_values) & np.isfinite(returns)
        valid_factors = factor_values[valid_mask]
        valid_returns = returns[valid_mask]
        
        if len(valid_factors) < 50:
            return {'error': 'insufficient_valid_points'}
            
        spearman_ic, p_spearman = spearmanr(valid_factors, valid_returns)
        pearson_corr, p_pearson = pearsonr(valid_factors, valid_returns)
        
        # 计算收益率相关指标
        returns_stats = calculate_factor_returns(factor_values, returns)
        
        # 合并所有统计指标
        stats = {
            'spearman_ic': spearman_ic,
            'p_spearman': p_spearman,
            'pearson_corr': pearson_corr,
            'p_pearson': p_pearson,
            'factor_std': np.std(valid_factors),
            'valid_points': len(valid_factors),
            'error': None
        }
        
        if returns_stats:
            stats.update(returns_stats)
        
        return stats
        
    except Exception as e:
        return {'error': str(e)}