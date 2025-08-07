#!/usr/bin/env python3
"""
测试数据加载功能的简单脚本
"""
import sys
import os
from pathlib import Path
from datetime import datetime

# 添加src到Python路径
sys.path.append(str(Path(__file__).parent))

from src.data.connectors.csv_connector import CSVDataConnector
from loguru import logger

def test_data_loading():
    """测试数据加载功能"""
    print("Testing data loading functionality...")
    
    # 创建CSV连接器
    connector = CSVDataConnector("data/csv/")
    
    # 连接
    if not connector.connect():
        print("Failed to connect to CSV data source")
        return False
    
    # 获取品种列表
    symbols = connector.get_symbols()
    print(f"Available symbols: {symbols}")
    
    if not symbols:
        print("No symbols found")
        return False
    
    # 测试RBL8数据范围检测
    rbl8_symbol = 'RBL8-15min' if 'RBL8-15min' in symbols else symbols[0]
    print(f"\nTesting data range detection for: {rbl8_symbol}")
    
    start_date, end_date = connector.get_data_range(rbl8_symbol)
    if start_date and end_date:
        print(f"Data range: {start_date} to {end_date}")
        print(f"Total days: {(end_date - start_date).days}")
    else:
        print("Failed to get data range")
        return False
    
    # 测试数据加载（不指定日期范围）
    print(f"\nTesting data loading without date constraints...")
    df = connector.get_data(rbl8_symbol)
    print(f"Loaded {len(df)} records")
    print(f"Data columns: {list(df.columns)}")
    print(f"Data types: {df.dtypes.to_dict()}")
    
    if len(df) > 0:
        print(f"First record: {df.iloc[0]['timestamp']}")
        print(f"Last record: {df.iloc[-1]['timestamp']}")
        print("Sample data:")
        print(df.head())
    
    # 测试指定日期范围加载
    print(f"\nTesting data loading with specific date range...")
    test_start = datetime(2024, 1, 1)
    test_end = datetime(2024, 12, 31)
    
    df_filtered = connector.get_data(rbl8_symbol, test_start, test_end)
    print(f"Loaded {len(df_filtered)} records for 2024")
    
    if len(df_filtered) > 0:
        print(f"Filtered data range: {df_filtered['timestamp'].min()} to {df_filtered['timestamp'].max()}")
    else:
        print("No data found for 2024 range")
    
    connector.close()
    return True

if __name__ == "__main__":
    success = test_data_loading()
    if success:
        print("\nData loading test completed successfully!")
    else:
        print("\nData loading test failed!")
        sys.exit(1)