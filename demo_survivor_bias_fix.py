"""
演示：存活者偏差修复 - 时间序列约束 v1 vs v2
═════════════════════════════════════════════════════════════

本脚本展示动态约束（asof_date）与一次性约束的差异
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def simulate_stock_timeseries(ticker, start_date, end_date, ipo_date=None, delisting_date=None):
    """
    模拟一支股票的价格和成交量时间序列
    
    可选：指定 IPO 日期（上市前无数据）和 delisting 日期（下市）
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    if ipo_date:
        dates = dates[dates >= ipo_date]
    if delisting_date:
        dates = dates[dates <= delisting_date]
    
    n = len(dates)
    data = {
        'close': 10 + np.cumsum(np.random.normal(0.01, 0.5, n)),
        'volume': np.random.uniform(1e6, 5e6, n),
    }
    
    return pd.DataFrame(data, index=dates)


def test_constraints_comparison():
    """对比新旧约束逻辑"""
    
    print("=" * 80)
    print("存活者偏差演示：一次性约束 vs 动态约束")
    print("=" * 80)
    
    # 模拟回测时间：2020-01 到 2023-12
    backtest_start = pd.Timestamp("2020-01-01")
    backtest_end = pd.Timestamp("2023-12-31")
    sample_date = pd.Timestamp("2023-12-31")  # 演示最后一个月的约束
    
    # 创建模拟股票组合
    # 场景1：一直存活的股票
    stock_alive = {
        "RBC.TO": simulate_stock_timeseries("RBC.TO", backtest_start, backtest_end),
        "TD.TO":  simulate_stock_timeseries("TD.TO", backtest_start, backtest_end),
        # ...更多全期股票
    }
    
    # 场景2：中途上市的股票
    stock_ipo_2022 = {
        "NEW.TO": simulate_stock_timeseries("NEW.TO", backtest_start, backtest_end, 
                                            ipo_date=pd.Timestamp("2022-06-01")),
    }
    
    # 场景3：中途下市的股票
    stock_delisted = {
        "OLD.TO": simulate_stock_timeseries("OLD.TO", backtest_start, backtest_end,
                                            delisting_date=pd.Timestamp("2021-11-30")),
    }
    
    # 场景4：流动性激增的股票（早期很贵，后期便宜）
    stock_price_change = {
        "PRC.TO": simulate_stock_timeseries("PRC.TO", backtest_start, backtest_end),
    }
    stock_price_change["PRC.TO"]["close"] = 500 - (stock_price_change["PRC.TO"].index - backtest_start).days * 0.2  # 不断贬值
    
    all_stocks = {**stock_alive, **stock_ipo_2022, **stock_delisted, **stock_price_change}
    
    print("\n📊 模拟数据：")
    print(f"   回测期间：{backtest_start.date()} - {backtest_end.date()}")
    print(f"   股票数量：{len(all_stocks)} 支")
    print(f"   - 全期存活：{len(stock_alive)}")
    print(f"   - 新上市：{len(stock_ipo_2022)}")
    print(f"   - 已下市：{len(stock_delisted)}")
    print(f"   - 价格变化显著：{len(stock_price_change)}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # 【旧逻辑】一次性约束（用 2023-12-31 的数据过滤所有历史时期）
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "─" * 80)
    print("【旧逻辑】一次性约束 (apply_base_constraints)")
    print("─" * 80)
    
    min_price = 2.0
    max_price = 400.0
    min_adv = 1e6
    min_listing = 252
    
    passed_old = []
    for ticker, df in all_stocks.items():
        # ★ 用 2023-12-31 的最后一条数据（全局最后）
        price = df["close"].iloc[-1]
        adv = (df["close"].tail(20) * df["volume"].tail(20)).mean()
        listing_days = len(df)
        
        if min_price <= price <= max_price and adv >= min_adv and listing_days >= min_listing:
            passed_old.append(ticker)
    
    print(f"\n✅ 通过筛选的股票 ({len(passed_old)} 支)：")
    for t in sorted(passed_old):
        df = all_stocks[t]
        price = df["close"].iloc[-1]
        adv = (df["close"].tail(20) * df["volume"].tail(20)).mean()
        print(f"   {t:<10} price=${price:.2f}  ADV=${adv/1e6:.2f}M  days={len(df)}")
    
    print(f"\n❌ 被过滤的股票 ({len(all_stocks) - len(passed_old)} 支)：")
    for ticker in all_stocks:
        if ticker not in passed_old:
            df = all_stocks[ticker]
            price = df["close"].iloc[-1]
            adv = (df["close"].tail(20) * df["volume"].tail(20)).mean()
            listing_days = len(df)
            reason = []
            if price < min_price:
                reason.append(f"价格 ${price:.2f} < $2")
            if price > max_price:
                reason.append(f"价格 ${price:.2f} > $400")
            if adv < min_adv:
                reason.append(f"ADV ${adv/1e6:.2f}M < $1M")
            if listing_days < min_listing:
                reason.append(f"上市 {listing_days} 天 < 252 天")
            print(f"   {ticker:<10} {' | '.join(reason)}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # 【新逻辑】动态约束（每月只用 asof_date 前的数据）
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "─" * 80)
    print("【新逻辑】动态约束 (apply_constraints_asof) @ 2023-12-31")
    print("─" * 80)
    
    asof_date = sample_date
    passed_new = []
    
    for ticker, df in all_stocks.items():
        # ★ 用 asof_date 前的历史数据
        hist = df[df.index <= asof_date]
        
        if len(hist) == 0:
            continue  # asof_date 前无数据（例外）
        
        price = hist["close"].iloc[-1]
        adv = (hist["close"].tail(20) * hist["volume"].tail(20)).mean() if len(hist) >= 20 else np.nan
        listing_days = len(hist)
        
        if min_price <= price <= max_price and adv >= min_adv and listing_days >= min_listing:
            passed_new.append(ticker)
    
    print(f"\n✅ 通过筛选的股票 ({len(passed_new)} 支)：")
    for t in sorted(passed_new):
        df = all_stocks[t]
        hist = df[df.index <= asof_date]
        price = hist["close"].iloc[-1]
        adv = (hist["close"].tail(20) * hist["volume"].tail(20)).mean()
        print(f"   {t:<10} price=${price:.2f}  ADV=${adv/1e6:.2f}M  days={len(hist)}")
    
    print(f"\n❌ 被过滤的股票 ({len(all_stocks) - len(passed_new)} 支)：")
    for ticker in all_stocks:
        if ticker not in passed_new:
            df = all_stocks[ticker]
            hist = df[df.index <= asof_date]
            
            if len(hist) == 0:
                print(f"   {ticker:<10} asof_date 前无数据")
                continue
            
            price = hist["close"].iloc[-1]
            adv = (hist["close"].tail(20) * hist["volume"].tail(20)).mean() if len(hist) >= 20 else np.nan
            listing_days = len(hist)
            
            reason = []
            if price < min_price:
                reason.append(f"价格 ${price:.2f} < $2")
            if price > max_price:
                reason.append(f"价格 ${price:.2f} > $400")
            if pd.isna(adv) or adv < min_adv:
                reason.append(f"ADV 不足" if pd.isna(adv) else f"ADV ${adv/1e6:.2f}M < $1M")
            if listing_days < min_listing:
                reason.append(f"上市 {listing_days} 天 < 252 天")
            print(f"   {ticker:<10} {' | '.join(reason)}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # 对比总结
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("📊 对比总结")
    print("=" * 80)
    
    print(f"\n通过筛选的股票数量：")
    print(f"   旧逻辑（一次性）：{len(passed_old)} 支")
    print(f"   新逻辑（动态）  ：{len(passed_new)} 支")
    print(f"   差异            ：{len(passed_old) - len(passed_new)} 支")
    
    print(f"\n新逻辑相对旧逻辑的变化：")
    added = set(passed_new) - set(passed_old)
    removed = set(passed_old) - set(passed_new)
    
    if added:
        print(f"   + 新增通过：{added}")
    if removed:
        print(f"   - 移除通过：{removed}")
    if not added and not removed:
        print(f"   （在该示例中完全相同）")
    
    print(f"\n🎯 关键差异解释：")
    print(f"   1. OLD：全部用 2023-12-31 的 DataFrame（全局最新）")
    print(f"   2. NEW：每支股票用 asof_date 前的 history（本地历史）")
    print(f"   3. 下市股票：OLD 会过滤失败（找不到 2023 数据），NEW 用最后一天数据")
    print(f"   4. IPO 股票：OLD 按 2023 年流动性判断，NEW 按上市后实际流动性判断")
    print(f"   5. 价格差异：OLD 用当前价格，NEW 用 asof_date 前的价格趋势")


if __name__ == "__main__":
    test_constraints_comparison()
    
    print("\n" + "=" * 80)
    print("✅ 演示完成")
    print("\n💡 关键要点：")
    print("   • 动态约束确保每个历史时期都用"当时可知"的信息")
    print("   • 消除了用 2026 年数据评估 2020 年策略的前视偏差")
    print("   • 虽然结果可能略有不同，但更准确反映真实历史表现")
    print("=" * 80)
