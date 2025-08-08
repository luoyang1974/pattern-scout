#!/usr/bin/env python3
"""
旗杆识别完整时间范围测试脚本
测试完整的RBL8数据集（2019-2025）
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
import json
import time

# 导入项目模块
from src.data.connectors.csv_connector import CSVDataConnector
from src.patterns.detectors.flagpole_detector import FlagpoleDetector
from src.patterns.base.market_regime_detector import BaselineManager
from src.data.models.base_models import Flagpole, MarketRegime


class FullRangeFlagpoleTest:
    """完整时间范围旗杆检测测试"""
    
    def __init__(self):
        self.csv_connector = CSVDataConnector("data/csv")
        if not self.csv_connector.connect():
            raise ConnectionError("无法连接到CSV数据源")
        
        # 初始化基线管理器和检测器
        self.baseline_manager = BaselineManager()
        self.flagpole_detector = FlagpoleDetector(self.baseline_manager)
        
        # 测试结果
        self.test_results = {}
        
    def run_full_test(self, symbol: str = "RBL8", 
                     use_relaxed_thresholds: bool = True,
                     save_results: bool = True) -> dict:
        """
        运行完整时间范围的旗杆检测测试
        
        Args:
            symbol: 交易品种
            use_relaxed_thresholds: 是否使用放宽的阈值
            save_results: 是否保存结果
        """
        logger.info(f"开始完整时间范围旗杆检测测试: {symbol}")
        start_time = time.time()
        
        # 1. 加载完整数据
        logger.info("加载完整数据集...")
        full_data = self.csv_connector.get_data(symbol)
        
        if full_data.empty:
            raise ValueError(f"无法加载数据: {symbol}")
        
        logger.info(f"数据加载完成:")
        logger.info(f"  总记录数: {len(full_data):,}")
        logger.info(f"  时间范围: {full_data['timestamp'].min()} 到 {full_data['timestamp'].max()}")
        
        # 2. 配置检测器阈值
        if use_relaxed_thresholds:
            logger.info("使用放宽的阈值进行检测...")
            self._apply_relaxed_thresholds()
        else:
            logger.info("使用原始阈值进行检测...")
        
        # 3. 分批处理大数据集（避免内存问题）
        batch_size = 10000  # 每批处理10000条记录
        all_flagpoles = []
        batch_count = 0
        total_batches = (len(full_data) + batch_size - 1) // batch_size
        
        logger.info(f"开始分批处理，共{total_batches}批，每批{batch_size}条记录")
        
        for start_idx in range(0, len(full_data), batch_size):
            end_idx = min(start_idx + batch_size, len(full_data))
            batch_count += 1
            
            logger.info(f"处理第{batch_count}/{total_batches}批 "
                       f"({start_idx+1}-{end_idx}, {end_idx-start_idx}条记录)")
            
            # 提取批次数据（确保有足够的上下文）
            context_start = max(0, start_idx - 100)  # 前100条作为上下文
            batch_data = full_data.iloc[context_start:end_idx].copy()
            batch_data = batch_data.reset_index(drop=True)
            
            # 检测当前批次的旗杆
            try:
                batch_flagpoles = self._detect_batch_flagpoles(batch_data, start_idx - context_start)
                
                # 过滤掉不在当前批次范围内的旗杆（避免重复计算上下文部分）
                valid_flagpoles = []
                batch_start_time = full_data.iloc[start_idx]['timestamp']
                batch_end_time = full_data.iloc[end_idx-1]['timestamp']
                
                for flagpole in batch_flagpoles:
                    if batch_start_time <= flagpole.start_time <= batch_end_time:
                        valid_flagpoles.append(flagpole)
                
                all_flagpoles.extend(valid_flagpoles)
                logger.info(f"第{batch_count}批检测到{len(valid_flagpoles)}个旗杆")
                
            except Exception as e:
                logger.error(f"第{batch_count}批处理失败: {e}")
                continue
            
            # 进度报告
            if batch_count % 5 == 0:
                elapsed = time.time() - start_time
                estimated_total = elapsed * total_batches / batch_count
                remaining = estimated_total - elapsed
                logger.info(f"进度: {batch_count}/{total_batches} "
                           f"({batch_count/total_batches:.1%}), "
                           f"已用时{elapsed/60:.1f}分钟, "
                           f"预计还需{remaining/60:.1f}分钟")
        
        # 4. 去重和后处理
        logger.info("对检测结果进行去重和后处理...")
        final_flagpoles = self._remove_duplicate_flagpoles(all_flagpoles)
        
        total_time = time.time() - start_time
        logger.info(f"完整检测完成！")
        logger.info(f"  总用时: {total_time/60:.2f}分钟")
        logger.info(f"  检测到旗杆: {len(final_flagpoles)}个")
        
        # 5. 生成测试结果
        test_result = self._create_test_result(symbol, full_data, final_flagpoles, total_time)
        
        # 6. 保存结果
        if save_results:
            self._save_test_results(symbol, test_result, final_flagpoles)
        
        return test_result
    
    def _apply_relaxed_thresholds(self):
        """应用放宽的阈值"""
        original_method = self.flagpole_detector._get_fallback_thresholds
        
        def relaxed_thresholds():
            return {
                'slope_score_p90': 0.15,    # 从0.5降低到0.15
                'volume_burst_p85': 1.25,   # 从1.5降低到1.25
                'retrace_depth_p75': 0.45,  # 从0.3放宽到0.45
            }
        
        self.flagpole_detector._get_fallback_thresholds = relaxed_thresholds
        self._original_thresholds_method = original_method
    
    def _detect_batch_flagpoles(self, batch_data: pd.DataFrame, offset: int) -> list:
        """检测单个批次的旗杆"""
        try:
            flagpoles = self.flagpole_detector.detect_flagpoles(
                df=batch_data,
                current_regime=MarketRegime.UNKNOWN,
                timeframe="15m"
            )
            return flagpoles
        except Exception as e:
            logger.error(f"批次检测失败: {e}")
            return []
    
    def _remove_duplicate_flagpoles(self, flagpoles: list) -> list:
        """去除重复的旗杆"""
        if not flagpoles:
            return []
        
        # 按开始时间排序
        sorted_flagpoles = sorted(flagpoles, key=lambda x: x.start_time)
        
        # 去重逻辑：如果两个旗杆时间重叠，保留质量更高的
        final_flagpoles = []
        
        for candidate in sorted_flagpoles:
            is_duplicate = False
            
            for existing in final_flagpoles:
                # 检查时间重叠
                if not (candidate.end_time < existing.start_time or 
                       candidate.start_time > existing.end_time):
                    is_duplicate = True
                    
                    # 如果候选旗杆质量更高，替换现有的
                    if candidate.slope_score > existing.slope_score:
                        final_flagpoles.remove(existing)
                        final_flagpoles.append(candidate)
                    break
            
            if not is_duplicate:
                final_flagpoles.append(candidate)
        
        return sorted(final_flagpoles, key=lambda x: x.start_time)
    
    def _create_test_result(self, symbol: str, data: pd.DataFrame, 
                          flagpoles: list, total_time: float) -> dict:
        """创建测试结果"""
        result = {
            "symbol": symbol,
            "test_info": {
                "total_records": len(data),
                "start_date": str(data['timestamp'].min()),
                "end_date": str(data['timestamp'].max()),
                "timeframe": "15m",
                "processing_time_minutes": round(total_time / 60, 2)
            },
            "detection_stats": {
                "total_flagpoles": len(flagpoles),
                "up_flagpoles": len([f for f in flagpoles if f.direction == 'up']),
                "down_flagpoles": len([f for f in flagpoles if f.direction == 'down']),
                "detection_rate_per_1000": round(len(flagpoles) * 1000 / len(data), 2)
            },
            "flagpole_summary": {
                "avg_height_percent": np.mean([f.height_percent for f in flagpoles]) if flagpoles else 0,
                "avg_slope_score": np.mean([f.slope_score for f in flagpoles]) if flagpoles else 0,
                "avg_volume_burst": np.mean([f.volume_burst for f in flagpoles]) if flagpoles else 0,
                "height_range": {
                    "min": min([f.height_percent for f in flagpoles]) if flagpoles else 0,
                    "max": max([f.height_percent for f in flagpoles]) if flagpoles else 0
                }
            }
        }
        
        return result
    
    def _save_test_results(self, symbol: str, test_result: dict, flagpoles: list):
        """保存测试结果"""
        output_dir = "output/flagpole_tests/full_range"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存测试摘要
        summary_file = os.path.join(output_dir, f"{symbol}_full_range_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(test_result, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"测试摘要已保存到: {summary_file}")
        
        # 保存详细旗杆列表
        if flagpoles:
            detailed_flagpoles = []
            for i, flagpole in enumerate(flagpoles):
                detailed_flagpoles.append({
                    "id": i + 1,
                    "start_time": str(flagpole.start_time),
                    "end_time": str(flagpole.end_time),
                    "start_price": flagpole.start_price,
                    "end_price": flagpole.end_price,
                    "height": flagpole.height,
                    "height_percent": flagpole.height_percent,
                    "direction": flagpole.direction,
                    "bars_count": flagpole.bars_count,
                    "slope_score": flagpole.slope_score,
                    "volume_burst": flagpole.volume_burst,
                    "impulse_bar_ratio": flagpole.impulse_bar_ratio,
                    "retracement_ratio": flagpole.retracement_ratio,
                    "trend_strength": flagpole.trend_strength
                })
            
            detailed_file = os.path.join(output_dir, f"{symbol}_full_range_flagpoles.json")
            with open(detailed_file, 'w', encoding='utf-8') as f:
                json.dump(detailed_flagpoles, f, indent=2, ensure_ascii=False)
            
            # CSV格式
            csv_file = os.path.join(output_dir, f"{symbol}_full_range_flagpoles.csv")
            flagpoles_df = pd.DataFrame(detailed_flagpoles)
            flagpoles_df.to_csv(csv_file, index=False)
            
            logger.info(f"详细旗杆数据已保存到:")
            logger.info(f"  JSON: {detailed_file}")
            logger.info(f"  CSV: {csv_file}")
    
    def print_summary(self, test_result: dict):
        """打印测试摘要"""
        print("\\n" + "="*80)
        print("旗杆检测完整时间范围测试结果")
        print("="*80)
        
        info = test_result["test_info"]
        stats = test_result["detection_stats"]
        summary = test_result["flagpole_summary"]
        
        print(f"\\n数据信息:")
        print(f"  品种: {test_result['symbol']}")
        print(f"  总记录数: {info['total_records']:,}")
        print(f"  时间范围: {info['start_date']} 到 {info['end_date']}")
        print(f"  处理用时: {info['processing_time_minutes']}分钟")
        
        print(f"\\n检测统计:")
        print(f"  检测到旗杆总数: {stats['total_flagpoles']}")
        print(f"  上升旗杆: {stats['up_flagpoles']} ({stats['up_flagpoles']/max(stats['total_flagpoles'],1):.1%})")
        print(f"  下降旗杆: {stats['down_flagpoles']} ({stats['down_flagpoles']/max(stats['total_flagpoles'],1):.1%})")
        print(f"  检测率: {stats['detection_rate_per_1000']}个/千条记录")
        
        if stats['total_flagpoles'] > 0:
            print(f"\\n旗杆特征统计:")
            print(f"  平均高度: {summary['avg_height_percent']:.3%}")
            print(f"  高度范围: {summary['height_range']['min']:.3%} - {summary['height_range']['max']:.3%}")
            print(f"  平均斜率分: {summary['avg_slope_score']:.2f}")
            print(f"  平均量能爆发: {summary['avg_volume_burst']:.2f}x")


def main():
    """主函数"""
    # 配置日志
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add("logs/flagpole_full_range_test.log", level="DEBUG", rotation="10 MB")
    
    try:
        # 创建测试实例
        tester = FullRangeFlagpoleTest()
        
        # 运行完整测试
        result = tester.run_full_test(
            symbol="RBL8",
            use_relaxed_thresholds=True,  # 使用放宽阈值
            save_results=True
        )
        
        # 打印摘要
        tester.print_summary(result)
        
        print("\\n" + "="*80)
        print("完整时间范围旗杆检测测试完成！")
        print("结果文件保存在: output/flagpole_tests/full_range/")
        print("="*80)
        
    except Exception as e:
        logger.error(f"完整测试执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()