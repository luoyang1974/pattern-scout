#!/usr/bin/env python3
"""
调试形态检测问题
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

# 添加src到Python路径
sys.path.append(str(Path(__file__).parent))

from src.utils.config_manager import ConfigManager
from src.data.connectors.csv_connector import CSVDataConnector
from src.patterns.detectors.pattern_scanner import PatternScanner
from loguru import logger

def main():
    # 配置日志
    logger.remove()
    logger.add(sys.stdout, level="DEBUG")
    
    # 加载配置
    config = ConfigManager("config.yaml").config
    
    # 创建数据连接器
    connector = CSVDataConnector(config['data_sources']['csv']['directory'])
    connector.connect()
    
    # 获取小范围测试数据
    df = connector.get_data(
        symbol="RBL8",
        start_date=datetime(2024, 6, 1),
        end_date=datetime(2024, 7, 1)
    )
    
    logger.info(f"Test data loaded: {len(df)} records")
    logger.info(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # 创建形态扫描器
    scanner = PatternScanner(config)
    
    # 执行扫描
    result = scanner.scan(
        df, 
        enable_outcome_analysis=False,
        enable_data_export=False,
        enable_chart_generation=False
    )
    
    # 打印详细结果
    logger.info("=== 扫描结果 ===")
    logger.info(f"Success: {result.get('success', 'Unknown')}")
    logger.info(f"Flagpoles detected: {result.get('flagpoles_detected', 0)}")
    logger.info(f"Patterns detected: {result.get('patterns_detected', 0)}")
    
    # 如果有错误，打印错误信息
    if 'error' in result:
        logger.error(f"Error: {result['error']}")
    
    # 打印旗杆信息
    if 'flagpoles' in result and result['flagpoles']:
        logger.info("=== 旗杆信息 ===")
        for i, fp in enumerate(result['flagpoles'][:5]):  # 只显示前5个
            logger.info(f"Flagpole {i+1}: {fp}")
    
    # 打印形态信息
    if 'patterns' in result and result['patterns']:
        logger.info("=== 形态信息 ===")
        for i, pattern in enumerate(result['patterns']):
            logger.info(f"Pattern {i+1}: {pattern}")
    else:
        logger.warning("没有检测到有效形态")
    
    # 打印统计信息
    if 'scan_statistics' in result:
        logger.info("=== 统计信息 ===")
        stats = result['scan_statistics']
        for key, value in stats.items():
            logger.info(f"{key}: {value}")

if __name__ == "__main__":
    main()