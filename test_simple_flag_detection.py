#!/usr/bin/env python3
"""
简化的旗形检测测试，避免依赖复杂的可视化模块
"""
import sys
import os
from pathlib import Path
from datetime import datetime

# 添加src到Python路径
sys.path.append(str(Path(__file__).parent))

from src.data.connectors.csv_connector import CSVDataConnector
from src.patterns.detectors.dynamic_pattern_scanner import DynamicPatternScanner
from src.utils.config_manager import ConfigManager
from loguru import logger

def test_simple_flag_detection():
    """测试简化的旗形检测"""
    print("Testing simplified flag detection...")
    
    # 加载配置
    config_manager = ConfigManager("config.yaml")
    
    # 创建CSV连接器
    connector = CSVDataConnector("data/csv/")
    if not connector.connect():
        print("Failed to connect to CSV data source")
        return False
    
    # 获取RBL8数据
    print("Loading RBL8 data...")
    df = connector.get_data("RBL8-15min")
    print(f"Loaded {len(df)} records from {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    if len(df) < 100:
        print("Insufficient data for testing")
        return False
    
    # 创建动态形态扫描器
    print("Initializing dynamic pattern scanner...")
    try:
        scanner = DynamicPatternScanner(config_manager.config)
        print("Dynamic pattern scanner initialized successfully")
    except Exception as e:
        print(f"Failed to initialize scanner: {e}")
        return False
    
    # 执行扫描（仅检测，不生成图表）
    print("Scanning for patterns...")
    try:
        result = scanner.scan(
            df, 
            enable_outcome_analysis=False,  # 禁用结局分析
            enable_data_export=False,       # 禁用数据输出
            enable_chart_generation=False   # 禁用图表生成
        )
        
        if result['success']:
            patterns_count = result['patterns_detected']
            flagpoles_count = result['flagpoles_detected']
            scan_time = result['scan_time']
            market_regime = result['market_regime']
            
            print(f"Scan completed successfully!")
            print(f"  Market regime: {market_regime}")
            print(f"  Flagpoles detected: {flagpoles_count}")
            print(f"  Patterns detected: {patterns_count}")
            print(f"  Scan time: {scan_time:.2f}s")
            
            if patterns_count > 0:
                print("\nPattern details:")
                for i, pattern in enumerate(result['patterns'][:5]):  # 显示前5个
                    print(f"  Pattern {i+1}: {pattern.get('sub_type', 'unknown')} - confidence: {pattern.get('confidence_score', 0):.3f}")
            
            return True
        else:
            print(f"Scan failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"Scan failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        connector.close()

if __name__ == "__main__":
    success = test_simple_flag_detection()
    if success:
        print("\nSimplified flag detection test completed successfully!")
    else:
        print("\nSimplified flag detection test failed!")
        sys.exit(1)