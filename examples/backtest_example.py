"""回测示例"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from stockquant.backtest import BacktestEngine
from stockquant.strategy import ValueQualityStrategy


def create_mock_data(symbols: list, days: int = 252) -> dict:
    """创建模拟数据用于测试"""
    data = {}
    end_date = datetime.now()

    for symbol in symbols:
        dates = pd.date_range(end=end_date, periods=days, freq='B')

        # 生成随机价格走势
        returns = np.random.normal(0.0005, 0.02, days)
        prices = 10 * (1 + returns).cumprod()

        df = pd.DataFrame({
            'trade_date': dates,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'vol': np.random.randint(1000000, 10000000, days),
            'amount': np.random.randint(10000000, 100000000, days),
            'pe_ttm': np.random.uniform(5, 30, days),
            'pb': np.random.uniform(0.5, 3, days),
            'roe': np.random.uniform(10, 25, days),
            'debt_to_assets': np.random.uniform(30, 60, days),
            'total_mv': np.random.uniform(50e8, 500e8, days),
        })
        df.set_index('trade_date', inplace=True)
        data[symbol] = df

    return data


def run_backtest_example():
    """运行回测示例"""
    print("=" * 60)
    print("StockQuant 回测示例")
    print("=" * 60)

    # 1. 创建模拟数据
    print("\n[1] 创建模拟数据...")
    symbols = ["000001.SZ", "000002.SZ", "600036.SH", "000333.SZ", "000858.SZ"]
    data = create_mock_data(symbols, days=252)
    print(f"    创建了 {len(symbols)} 只股票的模拟数据")

    # 2. 创建策略
    print("\n[2] 创建策略...")
    strategy = ValueQualityStrategy(
        max_holdings=3,
    )
    print(f"    策略: {strategy.name}")
    print(f"    最大持仓: {strategy.max_holdings}")

    # 3. 创建回测引擎
    print("\n[3] 配置回测引擎...")
    engine = BacktestEngine(
        initial_cash=50000,
        commission_rate=0.0003,
        min_commission=5,
        tax_rate=0.001,
        max_position_pct=0.3,
        max_total_position=0.8,
    )
    print(f"    初始资金: ¥{engine.initial_cash:,.2f}")
    print(f"    佣金费率: {engine.commission_rate:.2%}")

    # 4. 运行回测
    print("\n[4] 运行回测...")
    result = engine.run(
        strategy=strategy,
        data=data,
        rebalance_freq="weekly",
    )

    # 5. 显示结果
    print("\n[5] 回测结果")
    print(result.summary())

    # 6. 保存报告
    report_path = "backtest_report.xlsx"
    result.save_report(report_path)
    print(f"\n[6] 报告已保存: {report_path}")

    # 7. 生成图表
    try:
        fig = result.plot()
        fig.write_html("backtest_chart.html")
        print(f"[7] 图表已保存: backtest_chart.html")
    except Exception as e:
        print(f"[7] 图表生成失败: {e}")

    print("\n" + "=" * 60)
    print("回测完成！")
    print("=" * 60)

    return result


if __name__ == "__main__":
    run_backtest_example()
