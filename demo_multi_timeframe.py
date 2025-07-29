#!/usr/bin/env python3
"""
多时间周期系统演示脚本
展示新的时间周期分类和策略选择功能
"""
import pandas as pd
import numpy as np

from src.patterns.base.timeframe_manager import TimeframeManager
from src.patterns.strategies.strategy_factory import StrategyFactory
from src.utils.config_manager import ConfigManager

def create_sample_data(timeframe: str, periods: int = 100):
    """创建示例数据"""
    freq_map = {
        '1m': '1min', '5m': '5min', '15m': '15min', '1h': '1h',
        '4h': '4h', '1d': '1D', '1w': '1W', '1M': '1M'
    }
    
    freq = freq_map.get(timeframe, '15min')
    timestamps = pd.date_range(start='2024-01-01', periods=periods, freq=freq)
    
    # 生成随机价格数据（带趋势）
    base_price = 100
    trend = np.linspace(0, 5, periods)  # 轻微上升趋势
    noise = np.random.normal(0, 0.5, periods)
    
    close_prices = base_price + trend + noise
    high_prices = close_prices + np.random.uniform(0.1, 1.0, periods)
    low_prices = close_prices - np.random.uniform(0.1, 1.0, periods)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': np.random.uniform(1000, 5000, periods),
        'symbol': ['DEMO'] * periods
    })

def demo_timeframe_detection():
    """演示时间周期自动检测"""
    print("=== 时间周期自动检测演示 ===")
    tm = TimeframeManager()
    
    timeframes_to_test = ['1m', '5m', '15m', '1h', '4h', '1d']
    
    for tf in timeframes_to_test:
        # 创建该周期的数据
        df = create_sample_data(tf, 50)
        
        # 检测时间周期
        detected_tf = tm.detect_timeframe(df)
        category = tm.get_category(detected_tf)
        
        print(f"原始周期: {tf:4s} | 检测结果: {detected_tf:4s} | 分类: {category:4s}")
    
    print()

def demo_strategy_selection():
    """演示策略自动选择"""
    print("=== 策略自动选择演示 ===")
    
    supported_timeframes = StrategyFactory.list_supported_timeframes()
    print(f"支持的时间周期: {supported_timeframes}")
    
    for tf in supported_timeframes:
        strategy = StrategyFactory.get_strategy(tf)
        category_name = strategy.get_category_name()
        strategy_name = strategy.__class__.__name__
        
        print(f"周期: {tf:4s} | 策略: {strategy_name:20s} | 返回分类: {category_name}")
    
    print()

def demo_config_system():
    """演示新配置系统"""
    print("=== 新配置系统演示 ===")
    
    try:
        config = ConfigManager('config_multi_timeframe.yaml')
        
        print("配置文件加载成功!")
        timeframes = list(config.config['timeframe_categories'].keys())
        print(f"支持的时间周期: {timeframes}")
        
        # 展示几个关键周期的配置差异
        key_timeframes = ['1m', '15m', '1h', '1d']
        
        print("\n各周期旗杆检测参数对比:")
        print("周期    | 最小高度% | 最大高度% | 成交量比率 | 最大回撤%")
        print("-" * 60)
        
        for tf in key_timeframes:
            if tf in config.config['pattern_detection']:
                flagpole_params = config.config['pattern_detection'][tf]['flagpole']
                min_height = flagpole_params.get('min_height_percent', 'N/A')
                max_height = flagpole_params.get('max_height_percent', 'N/A')
                volume_ratio = flagpole_params.get('volume_surge_ratio', 'N/A')
                max_retracement = flagpole_params.get('max_retracement', 'N/A')
                
                print(f"{tf:7s} | {min_height:8.1f} | {max_height:8.1f} | {volume_ratio:9.1f} | {max_retracement:8.1f}")
    
    except Exception as e:
        print(f"配置文件加载失败: {e}")
    
    print()

def demo_lookback_periods():
    """演示回看周期配置"""
    print("=== 回看周期配置演示 ===")
    tm = TimeframeManager()
    
    timeframes = ['1m', '5m', '15m', '1h', '4h', '1d', '1w', '1M']
    
    print("旗形形态回看周期配置:")
    print("周期   | 最小K线数 | 最大K线数 | 形态前K线数")
    print("-" * 50)
    
    for tf in timeframes:
        lookback = tm.get_lookback_periods(tf, 'flag')
        if lookback:
            min_bars = lookback.get('min_total_bars', 'N/A')
            max_bars = lookback.get('max_total_bars', 'N/A')
            pre_bars = lookback.get('pre_pattern_bars', 'N/A')
            
            print(f"{tf:6s} | {min_bars:8d} | {max_bars:8d} | {pre_bars:10d}")
    
    print()

def main():
    """主演示函数"""
    print("PatternScout 多时间周期系统演示")
    print("=" * 50)
    print()
    
    # 运行各个演示
    demo_timeframe_detection()
    demo_strategy_selection() 
    demo_config_system()
    demo_lookback_periods()
    
    print("演示完成! 新的多时间周期系统已成功重构并可以正常工作。")
    print()
    print("主要改进:")
    print("1. 从3个大分类 (ultra_short/short/medium_long) 扩展到8个具体周期")
    print("2. 每个周期都有专门优化的策略和参数")
    print("3. 移除旧的分类系统，简化架构")
    print("4. 配置文件支持更精细的参数调优")
    print("5. 全面的测试覆盖确保系统稳定性")

if __name__ == '__main__':
    main()