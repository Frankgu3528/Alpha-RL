import pickle
from pathlib import Path
from plotly.offline import plot

def view_backtest(backtest_path: str):
    # 加载报告
    with open(f"{backtest_path}-report.pkl", "rb") as f:
        report = pickle.load(f)
    print("=== 回测报告 ===")
    print(report)
    
    # 加载并显示图表
    with open(f"{backtest_path}-graph.pkl", "rb") as f:
        graph = pickle.load(f)
    plot(graph, filename=f"{backtest_path}-graph.html")
    print(f"图表已保存到 {backtest_path}-graph.html")

if __name__ == "__main__":
    # 示例：查看特定回测结果
    view_backtest("out/backtests/50-5/rl/0")