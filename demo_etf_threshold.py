"""
演示：动态 ETF 兜底阈值计算
========================================

本脚本展示如何使用改进后的 calculate_etf_threshold() 函数。
该函数将硬编码的 0.18 替换为数据驱动的分位数方法。
"""

import pandas as pd
import numpy as np

# 从 picker.py 导入新函数（实际使用时）
# from picker import calculate_etf_threshold


def demo_threshold_calculation():
    """演示阈值计算过程"""
    
    print("=" * 70)
    print("动态 ETF 兜底阈值计算 - 演示")
    print("=" * 70)
    
    # 1️⃣ 构造示例 walk-forward 数据
    np.random.seed(42)
    n_dates = 24  # 24 个月历史
    n_stocks_per_date = 8  # 每天 8 支选股
    
    # 模拟牛市段（分数高）+ 熊市段（分数低）
    bullish_scores = np.random.normal(loc=0.28, scale=0.06, size=n_stocks_per_date*12)
    bearish_scores = np.random.normal(loc=0.12, scale=0.05, size=n_stocks_per_date*12)
    all_scores = np.concatenate([bullish_scores, bearish_scores])
    
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_dates*n_stocks_per_date, freq='D')
    tickers = ['TST.TO'] * (n_dates * n_stocks_per_date)
    
    wf_demo = pd.DataFrame({
        'date': dates,
        'ticker': tickers,
        'ens': all_scores,  # ensemble score
    })
    
    print("\n📊 示例数据：")
    print(f"   - 时间跨度：{wf_demo['date'].min().date()} 到 {wf_demo['date'].max().date()}")
    print(f"   - 数据行数：{len(wf_demo)}")
    print(f"   - 集成分数范围：[{wf_demo['ens'].min():.3f}, {wf_demo['ens'].max():.3f}]")
    
    # 2️⃣ 计算日均分数
    daily_avg = wf_demo.groupby('date')['ens'].mean()
    
    print("\n📈 日均分数统计：")
    print(f"   - 最低：{daily_avg.min():.4f}")
    print(f"   - 最高：{daily_avg.max():.4f}")
    print(f"   - 平均：{daily_avg.mean():.4f}")
    print(f"   - 中位数：{daily_avg.median():.4f}")
    print(f"   - 标准差：{daily_avg.std():.4f}")
    
    # 3️⃣ 在不同分位数下计算阈值
    quantiles = [0.15, 0.20, 0.25]
    
    print("\n🎯 不同分位数的阈值：")
    thresholds = {}
    for q in quantiles:
        threshold = daily_avg.quantile(q)
        thresholds[q] = threshold
        
        # 计算有多少天会触发兜底
        trigger_days = (daily_avg < threshold).sum()
        trigger_pct = trigger_days / len(daily_avg) * 100
        
        print(f"   - {int(q*100)}% 分位数：{threshold:.4f}")
        print(f"     → 满足触发条件的日期：{trigger_days}/{len(daily_avg)} ({trigger_pct:.1f}%)")
    
    # 4️⃣ 对比原硬编码值
    hardcoded_threshold = 0.18
    
    print(f"\n🔴 原硬编码阈值对比：")
    print(f"   - 硬编码：0.18")
    trigger_hardcoded = (daily_avg < hardcoded_threshold).sum()
    trigger_pct_hardcoded = trigger_hardcoded / len(daily_avg) * 100
    print(f"   - 会触发兜底的日期：{trigger_hardcoded}/{len(daily_avg)} ({trigger_pct_hardcoded:.1f}%)")
    
    # 5️⃣ 推荐阈值
    recommended = daily_avg.quantile(0.20)
    
    print(f"\n✅ 推荐配置（使用 20% 分位数）：")
    print(f"   - 动态阈值：{recommended:.4f}")
    print(f"   - 相比硬编码 0.18 的差异：{recommended - hardcoded_threshold:+.4f}")
    print(f"   - 触发频率：与历史 20% 分位线保持一致")
    
    # 6️⃣ 实际应用示例
    print(f"\n💡 实际应用示例：")
    current_month_scores = np.random.normal(loc=0.14, scale=0.04, size=8)
    current_top10_avg = current_month_scores.mean()
    
    print(f"   假设本月 Top 10 平均分数：{current_top10_avg:.4f}")
    
    if current_top10_avg < recommended:
        print(f"   ⚠️  {current_top10_avg:.4f} < {recommended:.4f} → 触发 ETF 兜底")
        print(f"       建议：清仓个股，全仓 XIU.TO")
    else:
        print(f"   ✓ {current_top10_avg:.4f} >= {recommended:.4f} → 信号充足，继续选股")
    
    print("\n" + "=" * 70)


def compare_strategies():
    """对比不同分位数策略的影响"""
    
    print("\n" + "=" * 70)
    print("策略对比：分位数 vs 硬编码阈值")
    print("=" * 70)
    
    np.random.seed(123)
    n_months = 36
    
    # 模拟分数分布（偏重尾部，模拟真实市场）
    np.random.seed(123)
    base_scores = np.random.gamma(shape=2, scale=0.1, size=n_months)  # 偏向低分
    base_scores = np.minimum(base_scores, 0.4)  # 上限 0.4
    
    print("\n📊 模拟 36 个月的平均分数：")
    print(f"   分数范围：[{base_scores.min():.3f}, {base_scores.max():.3f}]")
    print(f"   平均分数：{base_scores.mean():.3f}")
    
    # 计算触发频率
    strategies = {
        "硬编码 0.18": 0.18,
        "15% 分位": np.percentile(base_scores, 15),
        "20% 分位（推荐）": np.percentile(base_scores, 20),
        "25% 分位": np.percentile(base_scores, 25),
        "30% 分位": np.percentile(base_scores, 30),
    }
    
    print("\n🎯 各策略的兜底触发频率：")
    for strategy_name, threshold in strategies.items():
        trigger_count = (base_scores < threshold).sum()
        trigger_pct = trigger_count / n_months * 100
        print(f"   - {strategy_name:20s} (阈值={threshold:.4f})：{trigger_count:2d} 次 ({trigger_pct:5.1f}%)")
    
    print("\n💡 解释：")
    print("   - 15% 分位：更激进，容易触发保守配置")
    print("   - 20% 分位（推荐）：平衡风险与机会")
    print("   - 25% 分位：更保守，只有极弱信号才兜底")
    print("   - 硬编码 0.18：固定，不适应市场环境变化")


if __name__ == "__main__":
    demo_threshold_calculation()
    compare_strategies()
    
    print("\n" + "=" * 70)
    print("✅ 演示完成")
    print("\n📝 关键要点：")
    print("   1. 分位数方法自动适应市场环境，比硬编码更灵活")
    print("   2. 默认 20% 分位数提供良好的风险/收益平衡")
    print("   3. 可根据风险承受度调整分位数参数")
    print("=" * 70)
