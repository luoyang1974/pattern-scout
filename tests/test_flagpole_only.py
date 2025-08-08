#!/usr/bin/env python3
"""
旗杆识别专项测试脚本
测试目标：
1. 只测试旗杆识别部分
2. 输出旗杆数据详细信息
3. 生成旗杆可视化K线图
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from typing import List, Dict, Any
import json
from loguru import logger

# 导入项目模块
from src.data.connectors.csv_connector import CSVDataConnector
from src.patterns.detectors.flagpole_detector import FlagpoleDetector
from src.patterns.base.market_regime_detector import BaselineManager
from src.data.models.base_models import Flagpole, MarketRegime


class FlagpoleTestSuite:
    """旗杆检测专项测试套件"""
    
    def __init__(self, data_path: str = "data/csv"):
        """
        初始化测试套件
        
        Args:
            data_path: CSV数据路径
        """
        self.data_path = data_path
        self.csv_connector = CSVDataConnector(data_path)
        
        # 连接到数据源
        if not self.csv_connector.connect():
            logger.error("Failed to connect to CSV data source")
        
        # 初始化基线管理器
        self.baseline_manager = BaselineManager()
        
        # 初始化旗杆检测器
        self.flagpole_detector = FlagpoleDetector(self.baseline_manager)
        
        # 测试结果存储
        self.test_results = {}
        
    def run_flagpole_detection_test(self, symbol: str, 
                                  start_date: str = None,
                                  end_date: str = None,
                                  save_data: bool = True,
                                  generate_charts: bool = True) -> Dict[str, Any]:
        """
        运行旗杆检测测试
        
        Args:
            symbol: 交易品种代码
            start_date: 开始日期
            end_date: 结束日期  
            save_data: 是否保存数据
            generate_charts: 是否生成图表
            
        Returns:
            测试结果字典
        """
        logger.info(f"开始旗杆检测测试: {symbol}")
        
        # 1. 加载数据
        logger.info("加载数据...")
        data = self.csv_connector.get_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        if data.empty:
            logger.error(f"无法加载数据: {symbol}")
            return {"error": f"无法加载数据: {symbol}"}
        
        logger.info(f"数据加载完成，共 {len(data)} 条记录")
        logger.info(f"数据时间范围: {data['timestamp'].min()} 到 {data['timestamp'].max()}")
        
        # 2. 市场状态检测
        logger.info("检测市场状态...")
        current_regime = self._detect_market_regime(data)
        logger.info(f"检测到市场状态: {current_regime.value}")
        
        # 3. 执行旗杆检测
        logger.info("执行旗杆检测...")
        flagpoles = self.flagpole_detector.detect_flagpoles(
            df=data,
            current_regime=current_regime,
            timeframe="15m"  # 假设为15分钟数据
        )
        
        logger.info(f"检测到 {len(flagpoles)} 个旗杆")
        
        # 4. 整理测试结果
        test_result = {
            "symbol": symbol,
            "data_info": {
                "total_records": len(data),
                "start_date": str(data['timestamp'].min()),
                "end_date": str(data['timestamp'].max()),
                "timeframe": "15m"
            },
            "market_regime": current_regime.value,
            "detection_stats": {
                "total_flagpoles": len(flagpoles),
                "up_flagpoles": len([f for f in flagpoles if f.direction == 'up']),
                "down_flagpoles": len([f for f in flagpoles if f.direction == 'down'])
            },
            "flagpoles": []
        }
        
        # 5. 处理每个检测到的旗杆
        for i, flagpole in enumerate(flagpoles):
            flagpole_info = {
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
            }
            test_result["flagpoles"].append(flagpole_info)
            
            # 打印旗杆详细信息
            logger.info(f"旗杆 #{i+1}: {flagpole.direction} "
                       f"高度={flagpole.height:.4f}({flagpole.height_percent:.2%}) "
                       f"斜率分={flagpole.slope_score:.2f} "
                       f"量能爆发={flagpole.volume_burst:.2f}")
        
        # 6. 保存数据
        if save_data:
            self._save_flagpole_data(symbol, test_result)
        
        # 7. 生成图表
        if generate_charts:
            self._generate_flagpole_charts(symbol, data, flagpoles)
        
        # 8. 获取检测器统计信息
        detection_stats = self.flagpole_detector.get_detection_stats()
        test_result["detector_stats"] = detection_stats
        
        # 保存到测试结果
        self.test_results[symbol] = test_result
        
        return test_result
    
    def _detect_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """检测市场状态"""
        # 简化的市场状态检测逻辑
        # 计算最近20期的ATR
        if len(data) < 20:
            return MarketRegime.UNKNOWN
        
        # 计算ATR
        high_low = data['high'] - data['low']
        high_close_prev = abs(data['high'] - data['close'].shift(1))
        low_close_prev = abs(data['low'] - data['close'].shift(1))
        
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        
        # 最近ATR与长期ATR的比较
        recent_atr = atr.tail(5).mean()
        long_term_atr = atr.tail(50).mean()
        
        if pd.isna(recent_atr) or pd.isna(long_term_atr):
            return MarketRegime.UNKNOWN
        
        volatility_ratio = recent_atr / long_term_atr
        
        if volatility_ratio > 1.5:
            return MarketRegime.HIGH_VOLATILITY
        elif volatility_ratio < 0.7:
            return MarketRegime.LOW_VOLATILITY
        else:
            return MarketRegime.UNKNOWN
    
    def _save_flagpole_data(self, symbol: str, test_result: Dict[str, Any]):
        """保存旗杆数据到文件"""
        output_dir = "output/flagpole_tests"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存JSON格式详细数据
        json_file = os.path.join(output_dir, f"{symbol}_flagpole_test_result.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(test_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"旗杆检测结果已保存到: {json_file}")
        
        # 保存CSV格式旗杆列表
        if test_result["flagpoles"]:
            csv_file = os.path.join(output_dir, f"{symbol}_flagpoles.csv")
            flagpoles_df = pd.DataFrame(test_result["flagpoles"])
            flagpoles_df.to_csv(csv_file, index=False)
            logger.info(f"旗杆列表已保存到: {csv_file}")
    
    def _generate_flagpole_charts(self, symbol: str, data: pd.DataFrame, flagpoles: List[Flagpole]):
        """生成旗杆可视化图表"""
        if not flagpoles:
            logger.warning("没有检测到旗杆，跳过图表生成")
            return
        
        output_dir = "output/flagpole_tests/charts"
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        # 1. K线图 + 旗杆标记
        self._plot_candlestick_with_flagpoles(ax1, data, flagpoles, symbol)
        
        # 2. 成交量图 + 旗杆成交量分析
        self._plot_volume_with_flagpoles(ax2, data, flagpoles, symbol)
        
        plt.tight_layout()
        
        # 保存图表
        chart_file = os.path.join(output_dir, f"{symbol}_flagpole_analysis.png")
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"旗杆分析图表已保存到: {chart_file}")
        
        # 生成单个旗杆详细图表
        self._generate_individual_flagpole_charts(symbol, data, flagpoles, output_dir)
    
    def _plot_candlestick_with_flagpoles(self, ax, data: pd.DataFrame, flagpoles: List[Flagpole], symbol: str):
        """绘制K线图和旗杆标记"""
        # 绘制K线图
        for i, (_, row) in enumerate(data.iterrows()):
            color = 'red' if row['close'] >= row['open'] else 'green'
            
            # 绘制影线
            ax.plot([i, i], [row['low'], row['high']], color='black', linewidth=0.5)
            
            # 绘制实体
            body_height = abs(row['close'] - row['open'])
            bottom = min(row['open'], row['close'])
            
            rect = plt.Rectangle((i-0.4, bottom), 0.8, body_height, 
                               facecolor=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
        
        # 标记旗杆
        data_indexed = data.reset_index(drop=True)
        timestamp_to_index = {ts: i for i, ts in enumerate(data['timestamp'])}
        
        colors = ['blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for i, flagpole in enumerate(flagpoles):
            color = colors[i % len(colors)]
            
            # 找到旗杆在数据中的索引范围
            start_idx = None
            end_idx = None
            
            for j, ts in enumerate(data['timestamp']):
                if ts == flagpole.start_time:
                    start_idx = j
                if ts == flagpole.end_time:
                    end_idx = j
                    break
            
            if start_idx is not None and end_idx is not None:
                # 绘制旗杆区域
                ax.axvspan(start_idx, end_idx, alpha=0.3, color=color, 
                          label=f'旗杆{i+1}-{flagpole.direction}')
                
                # 标记起始和结束点
                ax.plot(start_idx, flagpole.start_price, 'o', color=color, markersize=8, 
                       markeredgecolor='black', markeredgewidth=1)
                ax.plot(end_idx, flagpole.end_price, 's', color=color, markersize=8, 
                       markeredgecolor='black', markeredgewidth=1)
                
                # 添加文本标注
                mid_idx = (start_idx + end_idx) / 2
                mid_price = (flagpole.start_price + flagpole.end_price) / 2
                ax.annotate(f'旗杆{i+1}\\n{flagpole.height_percent:.1%}', 
                           xy=(mid_idx, mid_price), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
        
        ax.set_title(f'{symbol} - 旗杆检测结果 (共{len(flagpoles)}个旗杆)', fontsize=14, fontweight='bold')
        ax.set_ylabel('价格', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)
    
    def _plot_volume_with_flagpoles(self, ax, data: pd.DataFrame, flagpoles: List[Flagpole], symbol: str):
        """绘制成交量图和旗杆成交量分析"""
        if 'volume' not in data.columns:
            ax.text(0.5, 0.5, '无成交量数据', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            return
        
        # 绘制成交量柱状图
        ax.bar(range(len(data)), data['volume'], alpha=0.6, color='gray', width=0.8)
        
        # 计算成交量移动平均线
        volume_ma = data['volume'].rolling(window=20).mean()
        ax.plot(range(len(data)), volume_ma, color='red', linewidth=1, label='成交量MA20')
        
        # 标记旗杆期间的成交量
        colors = ['blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for i, flagpole in enumerate(flagpoles):
            color = colors[i % len(colors)]
            
            # 找到旗杆在数据中的索引范围
            start_idx = None
            end_idx = None
            
            for j, ts in enumerate(data['timestamp']):
                if ts == flagpole.start_time:
                    start_idx = j
                if ts == flagpole.end_time:
                    end_idx = j
                    break
            
            if start_idx is not None and end_idx is not None:
                # 突出显示旗杆期间的成交量
                ax.axvspan(start_idx, end_idx, alpha=0.2, color=color)
                
                # 标记成交量爆发情况
                flagpole_volume = data.iloc[start_idx:end_idx+1]['volume']
                max_volume = flagpole_volume.max()
                max_volume_idx = start_idx + flagpole_volume.argmax()
                
                ax.plot(max_volume_idx, max_volume, '^', color=color, markersize=10,
                       markeredgecolor='black', markeredgewidth=1)
                
                ax.annotate(f'量能爆发\\n{flagpole.volume_burst:.1f}x', 
                           xy=(max_volume_idx, max_volume), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
        
        ax.set_title('成交量分析 (旗杆期间量能爆发)', fontsize=12, fontweight='bold')
        ax.set_ylabel('成交量', fontsize=12)
        ax.set_xlabel('时间索引', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    def _generate_individual_flagpole_charts(self, symbol: str, data: pd.DataFrame, 
                                           flagpoles: List[Flagpole], output_dir: str):
        """生成单个旗杆的详细图表"""
        for i, flagpole in enumerate(flagpoles):
            # 找到旗杆在数据中的位置
            start_idx = None
            end_idx = None
            
            for j, ts in enumerate(data['timestamp']):
                if ts == flagpole.start_time:
                    start_idx = j
                if ts == flagpole.end_time:
                    end_idx = j
                    break
            
            if start_idx is None or end_idx is None:
                continue
            
            # 扩展显示范围（前后各20个K线）
            display_start = max(0, start_idx - 20)
            display_end = min(len(data) - 1, end_idx + 20)
            
            display_data = data.iloc[display_start:display_end + 1].copy()
            display_data = display_data.reset_index(drop=True)
            
            # 调整旗杆索引到新的显示范围
            flagpole_start_in_display = start_idx - display_start  
            flagpole_end_in_display = end_idx - display_start
            
            # 创建单个旗杆详细图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
            
            # K线图
            for k, (_, row) in enumerate(display_data.iterrows()):
                color = 'red' if row['close'] >= row['open'] else 'green'
                
                # 突出显示旗杆期间
                if flagpole_start_in_display <= k <= flagpole_end_in_display:
                    edge_color = 'blue'
                    edge_width = 1.5
                    alpha = 1.0
                else:
                    edge_color = 'black'
                    edge_width = 0.5
                    alpha = 0.7
                
                # 绘制影线
                ax1.plot([k, k], [row['low'], row['high']], color=edge_color, linewidth=edge_width)
                
                # 绘制实体
                body_height = abs(row['close'] - row['open'])
                bottom = min(row['open'], row['close'])
                
                rect = plt.Rectangle((k-0.4, bottom), 0.8, body_height, 
                                   facecolor=color, alpha=alpha, edgecolor=edge_color, 
                                   linewidth=edge_width)
                ax1.add_patch(rect)
            
            # 标记旗杆边界
            ax1.axvspan(flagpole_start_in_display, flagpole_end_in_display, 
                       alpha=0.2, color='blue', label='旗杆区间')
            
            # 标记起止点
            ax1.plot(flagpole_start_in_display, flagpole.start_price, 'go', 
                    markersize=10, label='起始点')
            ax1.plot(flagpole_end_in_display, flagpole.end_price, 'ro', 
                    markersize=10, label='结束点')
            
            ax1.set_title(f'{symbol} - 旗杆#{i+1} 详细分析 '
                         f'({flagpole.direction}, {flagpole.height_percent:.2%})', 
                         fontsize=14, fontweight='bold')
            ax1.set_ylabel('价格', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 成交量图
            if 'volume' in display_data.columns:
                bars = ax2.bar(range(len(display_data)), display_data['volume'], 
                              alpha=0.6, color='gray', width=0.8)
                
                # 突出显示旗杆期间成交量
                for k in range(flagpole_start_in_display, flagpole_end_in_display + 1):
                    if k < len(bars):
                        bars[k].set_color('orange')
                        bars[k].set_alpha(0.8)
                
                ax2.set_title('成交量 (旗杆期间高亮显示)', fontsize=12)
                ax2.set_ylabel('成交量', fontsize=12)
                ax2.grid(True, alpha=0.3)
            
            # 添加旗杆统计信息
            stats_text = (f'旗杆统计:\\n'
                         f'高度: {flagpole.height:.4f} ({flagpole.height_percent:.2%})\\n'
                         f'K线数: {flagpole.bars_count}\\n'
                         f'斜率分: {flagpole.slope_score:.2f}\\n'
                         f'量能爆发: {flagpole.volume_burst:.2f}x\\n'
                         f'动量占比: {flagpole.impulse_bar_ratio:.1%}\\n'
                         f'回撤比例: {flagpole.retracement_ratio:.1%}\\n'
                         f'趋势强度: {flagpole.trend_strength:.3f}')
            
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # 保存单个旗杆图表
            individual_chart_file = os.path.join(output_dir, 
                                               f"{symbol}_flagpole_{i+1}_{flagpole.direction}.png")
            plt.savefig(individual_chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"旗杆#{i+1}详细图表已保存到: {individual_chart_file}")
    
    def print_test_summary(self, symbol: str = None):
        """打印测试摘要"""
        if symbol and symbol in self.test_results:
            result = self.test_results[symbol]
            self._print_single_test_summary(symbol, result)
        else:
            # 打印所有测试摘要
            for sym, result in self.test_results.items():
                self._print_single_test_summary(sym, result)
                print("-" * 60)
    
    def _print_single_test_summary(self, symbol: str, result: Dict[str, Any]):
        """打印单个测试的摘要"""
        print(f"\\n{'='*60}")
        print(f"旗杆检测测试摘要 - {symbol}")
        print(f"{'='*60}")
        
        # 数据信息
        data_info = result["data_info"]
        print(f"数据信息:")
        print(f"  总记录数: {data_info['total_records']}")
        print(f"  时间范围: {data_info['start_date']} 到 {data_info['end_date']}")
        print(f"  时间周期: {data_info['timeframe']}")
        print(f"  市场状态: {result['market_regime']}")
        
        # 检测统计
        stats = result["detection_stats"]
        print(f"\\n检测统计:")
        print(f"  检测到旗杆总数: {stats['total_flagpoles']}")
        print(f"  上升旗杆: {stats['up_flagpoles']}")
        print(f"  下降旗杆: {stats['down_flagpoles']}")
        
        if result["flagpoles"]:
            print(f"\\n旗杆详细列表:")
            print(f"{'ID':<3} {'方向':<4} {'高度%':<8} {'K线数':<6} {'斜率分':<8} {'量能爆发':<8} {'动量占比':<8}")
            print("-" * 55)
            
            for flagpole in result["flagpoles"]:
                print(f"{flagpole['id']:<3} "
                     f"{flagpole['direction']:<4} "
                     f"{flagpole['height_percent']:<8.2%} "
                     f"{flagpole['bars_count']:<6} "
                     f"{flagpole['slope_score']:<8.2f} "
                     f"{flagpole['volume_burst']:<8.2f} "
                     f"{flagpole['impulse_bar_ratio']:<8.1%}")
        else:
            print("\\n未检测到旗杆")


def main():
    """主函数"""
    # 配置日志
    logger.add("logs/flagpole_test.log", rotation="10 MB", retention="7 days")
    
    # 创建测试套件
    test_suite = FlagpoleTestSuite(data_path="data/csv")
    
    # 运行测试 - 可以修改这里的参数
    symbol = "RBL8"  # 测试品种
    start_date = None  # 开始日期，None表示使用所有数据
    end_date = None    # 结束日期，None表示使用所有数据
    
    try:
        result = test_suite.run_flagpole_detection_test(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            save_data=True,
            generate_charts=True
        )
        
        # 打印摘要
        test_suite.print_test_summary(symbol)
        
        print(f"\\n测试完成！")
        print(f"结果文件保存在: output/flagpole_tests/")
        print(f"图表文件保存在: output/flagpole_tests/charts/")
        
    except Exception as e:
        logger.error(f"测试执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()