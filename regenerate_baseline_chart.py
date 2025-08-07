#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重新生成baseline_summary图表
测试修复后的中文字体显示
"""
import sys
import os
import numpy as np
from datetime import datetime, timedelta

# 设置控制台编码（Windows）
if sys.platform.startswith('win'):
    os.system('chcp 65001 > nul')
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# 添加src到Python路径
sys.path.append(str(__file__).replace('regenerate_baseline_chart.py', ''))

from src.visualization.pattern_chart_generator import PatternChartGenerator
from loguru import logger

# 配置loguru日志格式
logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
)

def create_mock_baseline_data():
    """创建模拟的基线数据"""
    return {
        'total_data_points': 1000,
        'regime_transitions': 3,
        'current_regime': 'low_volatility',
        'baseline_stability': 0.85,
        'coverage_stats': {
            '斜率分': 95,
            '量能爆发': 88,
            '回撤深度': 92,
            '量能收缩': 90,
            '波动下降': 87,
            '通道宽度': 94
        }
    }

def create_mock_regime_history():
    """创建模拟的状态历史数据"""
    history = []
    
    # 模拟一些状态转换
    regimes = ['high_volatility', 'low_volatility', 'high_volatility', 'low_volatility']
    base_time = datetime.now() - timedelta(hours=100)
    
    for i, regime in enumerate(regimes * 25):  # 创建100条记录
        history.append({
            'regime': regime,
            'timestamp': base_time + timedelta(hours=i),
            'confidence': np.random.uniform(0.7, 0.95)
        })
    
    return history

def generate_baseline_summary():
    """生成基线汇总图表"""
    logger.info("开始生成基线汇总图表...")
    
    try:
        # 初始化图表生成器
        config = {
            'charts': {
                'format': 'png',
                'width': 1400,
                'height': 900,
                'dpi': 300
            },
            'charts_path': 'output/charts',
            'dynamic_baseline': {
                'regime_colors': {
                    'high_volatility': '#ff5722',
                    'low_volatility': '#4caf50',
                    'unknown': '#9e9e9e'
                }
            }
        }
        
        generator = PatternChartGenerator(config)
        logger.info("图表生成器初始化成功")
        
        # 创建模拟数据
        baseline_data = create_mock_baseline_data()
        regime_history = create_mock_regime_history()
        
        logger.info(f"创建模拟数据 - 基线数据点: {baseline_data['total_data_points']}, 状态历史: {len(regime_history)}条")
        
        # 生成基线汇总图表
        chart_path = generator.generate_baseline_summary_chart(
            baseline_data=baseline_data,
            regime_history=regime_history
        )
        
        if chart_path:
            logger.success(f"✅ 基线汇总图表生成成功: {chart_path}")
            
            # 检查文件是否存在
            if os.path.exists(chart_path):
                file_size = os.path.getsize(chart_path) / 1024  # KB
                logger.info(f"图表文件大小: {file_size:.1f} KB")
            else:
                logger.error("图表文件不存在")
        else:
            logger.error("❌ 基线汇总图表生成失败")
            
    except Exception as e:
        logger.error(f"生成图表时发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """主函数"""
    logger.info("重新生成baseline_summary图表")
    logger.info("=" * 50)
    
    # 确保输出目录存在
    os.makedirs('output/charts', exist_ok=True)
    
    # 生成图表
    generate_baseline_summary()
    
    logger.info("=" * 50)
    logger.info("任务完成")

if __name__ == "__main__":
    main()