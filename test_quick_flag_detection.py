#!/usr/bin/env python3
"""
快速旗形检测测试，只使用部分数据
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

# 设置日志级别为INFO，减少调试输出
logger.remove()
logger.add(sys.stdout, level="INFO", 
          format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

def test_quick_flag_detection():
    """测试快速旗形检测，使用部分数据"""
    print("Testing quick flag detection with subset of data...")
    
    # 加载配置
    config_manager = ConfigManager("config.yaml")
    
    # 创建CSV连接器
    connector = CSVDataConnector("data/csv/")
    if not connector.connect():
        print("Failed to connect to CSV data source")
        return False
    
    # 获取RBL8数据的一个子集（最近1000条记录）
    print("Loading RBL8 data subset...")
    df_full = connector.get_data("RBL8-15min")
    
    # 使用最近的1000条记录进行测试
    df = df_full.tail(1000).copy().reset_index(drop=True)
    print(f"Using {len(df)} records for testing: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
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
                print("\nPattern summary:")
                patterns = result['patterns']
                flag_count = sum(1 for p in patterns if p.get('sub_type') == 'flag')
                pennant_count = sum(1 for p in patterns if p.get('sub_type') == 'pennant')
                print(f"  Flags: {flag_count}")
                print(f"  Pennants: {pennant_count}")
                
                # 显示前3个最高置信度的形态
                sorted_patterns = sorted(patterns, key=lambda x: x.get('confidence_score', 0), reverse=True)
                print("\nTop patterns:")
                for i, pattern in enumerate(sorted_patterns[:3]):
                    conf = pattern.get('confidence_score', 0)
                    sub_type = pattern.get('sub_type', 'unknown')
                    print(f"  {i+1}. {sub_type.upper()}: confidence={conf:.3f}")
            
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
    success = test_quick_flag_detection()
    if success:
        print("\nQuick flag detection test completed successfully!")
    else:
        print("\nQuick flag detection test failed!")
        sys.exit(1)