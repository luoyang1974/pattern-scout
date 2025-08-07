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

# 配置详细日志
logger.remove()
logger.add(sys.stdout, level="DEBUG")

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
    
    print(f"=== 测试数据 ===")
    print(f"数据量: {len(df)} 条记录")
    
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
    print(f"检测到旗杆: {result.get('flagpoles_detected', 0)} 个")
    print(f"检测到形态: {result.get('patterns_detected', 0)} 个")
    
    if result.get('flagpoles_detected', 0) > 0 and result.get('patterns_detected', 0) == 0:
        print("\n❌ 有旗杆但没有形态，检查上述DEBUG日志中的过滤原因")

if __name__ == "__main__":
    main()