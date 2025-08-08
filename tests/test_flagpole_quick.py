#!/usr/bin/env python3
"""
旗杆识别快速测试脚本
目标：快速分析旗杆识别问题，使用少量数据和放宽的阈值
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

# 导入项目模块
from src.data.connectors.csv_connector import CSVDataConnector
from src.patterns.detectors.flagpole_detector import FlagpoleDetector
from src.patterns.base.market_regime_detector import BaselineManager
from src.data.models.base_models import Flagpole, MarketRegime


def quick_flagpole_test():
    """快速旗杆检测测试"""
    logger.info("开始快速旗杆检测测试")
    
    # 1. 初始化数据连接器
    csv_connector = CSVDataConnector("data/csv")
    if not csv_connector.connect():
        logger.error("无法连接到CSV数据源")
        return
    
    # 2. 加载少量数据（只取最近1000条）
    symbol = "RBL8"
    all_data = csv_connector.get_data(symbol)
    
    if all_data.empty:
        logger.error(f"无法加载数据: {symbol}")
        return
    
    # 只取最后1000条数据进行快速测试
    data = all_data.tail(1000).reset_index(drop=True)
    logger.info(f"使用数据: {len(data)} 条记录")
    logger.info(f"数据时间范围: {data['timestamp'].min()} 到 {data['timestamp'].max()}")
    
    # 3. 初始化基线管理器和检测器
    baseline_manager = BaselineManager()
    flagpole_detector = FlagpoleDetector(baseline_manager)
    
    # 4. 修改检测器的阈值，放宽条件以便观察更多结果
    original_get_fallback_thresholds = flagpole_detector._get_fallback_thresholds
    def relaxed_get_fallback_thresholds():
        return {
            'slope_score_p90': 0.1,   # 大幅降低斜率阈值
            'volume_burst_p85': 1.2,  # 降低量能阈值  
            'retrace_depth_p75': 0.5,  # 放宽回撤阈值
        }
    flagpole_detector._get_fallback_thresholds = relaxed_get_fallback_thresholds
    
    # 5. 执行检测
    logger.info("执行旗杆检测（使用放宽的阈值）...")
    flagpoles = flagpole_detector.detect_flagpoles(
        df=data,
        current_regime=MarketRegime.UNKNOWN,
        timeframe="15m"
    )
    
    logger.info(f"检测到 {len(flagpoles)} 个旗杆")
    
    # 6. 分析结果
    if flagpoles:
        print("\\n发现的旗杆:")
        print(f"{'ID':<3} {'方向':<4} {'起始时间':<20} {'结束时间':<20} {'高度%':<8} {'斜率分':<8} {'量能':<8}")
        print("-" * 80)
        
        for i, flagpole in enumerate(flagpoles):
            print(f"{i+1:<3} "
                 f"{flagpole.direction:<4} "
                 f"{str(flagpole.start_time):<20} "
                 f"{str(flagpole.end_time):<20} "
                 f"{flagpole.height_percent:<8.2%} "
                 f"{flagpole.slope_score:<8.2f} "
                 f"{flagpole.volume_burst:<8.2f}")
    else:
        print("\\n即使使用放宽的阈值也未发现任何旗杆")
        
        # 分析可能的原因
        print("\\n分析可能的原因:")
        
        # 检查ATR计算
        high_low = data['high'] - data['low']
        high_close_prev = abs(data['high'] - data['close'].shift(1))
        low_close_prev = abs(data['low'] - data['close'].shift(1))
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr_14 = tr.rolling(window=14).mean()
        
        print(f"ATR14 平均值: {atr_14.mean():.4f}")
        print(f"ATR14 范围: {atr_14.min():.4f} - {atr_14.max():.4f}")
        
        # 检查价格变化
        price_changes = data['close'].pct_change().abs()
        print(f"价格变化率平均: {price_changes.mean():.4%}")
        print(f"价格变化率最大: {price_changes.max():.4%}")
        
        # 检查成交量
        if 'volume' in data.columns:
            volume_changes = data['volume'].pct_change()
            print(f"成交量变化平均: {volume_changes.mean():.2f}")
            print(f"成交量变化范围: {volume_changes.min():.2f} - {volume_changes.max():.2f}")
        
        # 手动检查一些候选区间
        print("\\n手动检查候选区间:")
        manual_check_candidates(data, atr_14)
    
    # 恢复原始方法
    flagpole_detector._get_fallback_thresholds = original_get_fallback_thresholds


def manual_check_candidates(data: pd.DataFrame, atr_14: pd.Series):
    """手动检查一些候选旗杆区间"""
    print("检查最大价格变化区间...")
    
    # 寻找最大价格变化的区间
    for bar_count in [4, 6, 8, 10]:
        max_change = 0
        max_change_start = 0
        
        for i in range(len(data) - bar_count):
            start_price = data.iloc[i]['close']
            end_price = data.iloc[i + bar_count]['close']
            change = abs(end_price - start_price) / start_price
            
            if change > max_change:
                max_change = change
                max_change_start = i
        
        if max_change > 0:
            # 分析这个区间
            start_idx = max_change_start
            end_idx = max_change_start + bar_count
            
            candidate = data.iloc[start_idx:end_idx + 1]
            start_price = candidate.iloc[0]['close']
            end_price = candidate.iloc[-1]['close']
            direction = 'up' if end_price > start_price else 'down'
            
            # 计算关键指标
            avg_atr = atr_14.iloc[start_idx:end_idx + 1].mean()
            if avg_atr > 0:
                slope_score = abs((end_price - start_price) / bar_count) / avg_atr
            else:
                slope_score = 0
            
            # 计算动量K线占比
            impulse_count = 0
            for j in range(len(candidate)):
                row = candidate.iloc[j]
                body_size = abs(row['close'] - row['open'])
                atr_val = atr_14.iloc[start_idx + j]
                if not pd.isna(atr_val) and body_size >= 0.8 * atr_val:
                    impulse_count += 1
            
            impulse_ratio = impulse_count / len(candidate)
            
            print(f"  {bar_count}K线区间: 高度={max_change:.2%}, 方向={direction}, "
                 f"斜率分={slope_score:.2f}, 动量占比={impulse_ratio:.1%}")


def main():
    """主函数"""
    # 配置日志（只显示INFO及以上级别，避免大量DEBUG信息）
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add("logs/flagpole_quick_test.log", level="DEBUG")
    
    try:
        quick_flagpole_test()
    except Exception as e:
        logger.error(f"测试执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()