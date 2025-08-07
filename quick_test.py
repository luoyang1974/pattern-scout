#!/usr/bin/env python3

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

# 配置简单日志
logger.remove()
logger.add(sys.stdout, level="INFO")

def main():
    # 加载配置
    config = ConfigManager("config.yaml").config
    
    # 创建数据连接器
    connector = CSVDataConnector(config['data_sources']['csv']['directory'])
    connector.connect()
    
    # 获取测试数据
    df = connector.get_data(
        symbol="RBL8",
        start_date=datetime(2024, 6, 1),
        end_date=datetime(2024, 7, 1)
    )
    
    print(f"\n=== 测试数据 ===")
    print(f"数据量: {len(df)} 条记录")
    print(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    
    # 创建形态扫描器
    scanner = PatternScanner(config)
    
    # 执行扫描
    result = scanner.scan(
        df, 
        enable_outcome_analysis=False,
        enable_data_export=False,
        enable_chart_generation=False
    )
    
    print(f"\n=== 扫描结果 ===")
    print(f"扫描成功: {result.get('success', 'Unknown')}")
    print(f"检测到旗杆: {result.get('flagpoles_detected', 0)} 个")
    print(f"检测到形态: {result.get('patterns_detected', 0)} 个")
    print(f"扫描时间: {result.get('scan_time', 0):.2f}秒")
    
    if 'error' in result:
        print(f"错误: {result['error']}")
        return
    
    # 显示旗杆信息
    if result.get('flagpoles_detected', 0) > 0:
        print(f"\n=== 旗杆详情 ===")
        flagpoles = result.get('flagpoles', [])
        for i, fp in enumerate(flagpoles[:3]):  # 只显示前3个
            print(f"旗杆 {i+1}:")
            print(f"  方向: {fp.get('direction', 'unknown')}")
            print(f"  高度: {fp.get('height_percent', 0):.2%}")
            print(f"  时间: {fp.get('start_time', 'unknown')} 到 {fp.get('end_time', 'unknown')}")
    
    # 显示形态信息
    if result.get('patterns_detected', 0) > 0:
        print(f"\n=== 形态详情 ===")
        patterns = result.get('patterns', [])
        for i, pattern in enumerate(patterns[:3]):
            print(f"形态 {i+1}:")
            print(f"  类型: {pattern.get('sub_type', 'unknown')}")
            print(f"  置信度: {pattern.get('confidence_score', 0):.2f}")
            print(f"  质量: {pattern.get('pattern_quality', 'unknown')}")

if __name__ == "__main__":
    main()