#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的旗杆识别和可视化脚本
使用系统现有检测器进行高效识别
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.connectors.csv_connector import CSVDataConnector
from src.patterns.detectors.flagpole_detector import FlagpoleDetector
from src.patterns.base.market_regime_detector import RegimeDetector
from src.patterns.base.robust_statistics import RobustStatistics

class OptimizedFlagpoleVisualizer:
    """优化的旗杆可视化器 - 使用系统现有检测器"""
    
    def __init__(self):
        # 数据连接器
        self.csv_connector = CSVDataConnector()
        
        # 输出目录
        self.output_dir = Path(project_root) / "output" / "flagpole_tests" / "optimized_flagpoles"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 系统检测器
        self.flagpole_detector = FlagpoleDetector()
        
        print("优化旗杆可视化器初始化完成")
        print(f"输出目录: {self.output_dir}")
    
    def get_complete_data(self, symbol: str) -> pd.DataFrame:
        """获取完整数据集"""
        try:
            # 连接数据源
            self.csv_connector.connect()
            
            # 获取完整数据
            df = self.csv_connector.get_data(symbol)
            if df.empty:
                print(f"未找到 {symbol} 的数据")
                return pd.DataFrame()
            
            print(f"完整数据: {len(df)} 条记录")
            print(f"数据时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            print(f"获取数据失败: {str(e)}")
            return pd.DataFrame()
    
    def detect_all_flagpoles_optimized(self, df: pd.DataFrame) -> List[Dict]:
        """使用系统检测器优化检测所有旗杆"""
        try:
            print("开始使用系统检测器检测旗杆...")
            
            # 检测时间周期
            timeframe = self._detect_timeframe(df)
            print(f"检测到时间周期: {timeframe}")
            
            # 使用系统旗杆检测器
            flagpoles = self.flagpole_detector.detect_flagpoles(df, timeframe)
            
            print(f"系统检测器找到 {len(flagpoles)} 个旗杆")
            
            # 转换为兼容格式
            converted_flagpoles = []
            for flagpole in flagpoles:
                try:
                    converted = {
                        'start_idx': flagpole.start_idx,
                        'end_idx': flagpole.end_idx,
                        'direction': flagpole.direction,
                        'height_pct': flagpole.height_percentage,
                        'slope_score': getattr(flagpole, 'slope_score', 0.5),
                        'volume_burst': getattr(flagpole, 'volume_burst_ratio', 1.5),
                        'volume_spike': getattr(flagpole, 'volume_spike_ratio', 2.0),
                        'retrace_pct': getattr(flagpole, 'max_retracement_percentage', 10.0),
                        'volatility': getattr(flagpole, 'volatility', 1.0),
                        'quality_score': flagpole.confidence,
                        'start_price': flagpole.start_price,
                        'end_price': flagpole.end_price,
                        'length': flagpole.length,
                        'detection_params': {
                            'min_height_pct': 0.5,
                            'min_slope_score': 0.1,
                            'min_volume_burst': 1.2,
                            'max_retrace_pct': 30,
                            'min_length': 3,
                            'max_length': 10
                        }
                    }
                    converted_flagpoles.append(converted)
                except Exception as e:
                    print(f"转换旗杆数据失败: {e}")
                    continue
            
            print(f"成功转换 {len(converted_flagpoles)} 个旗杆")
            return converted_flagpoles
            
        except Exception as e:
            print(f"旗杆检测失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _detect_timeframe(self, df: pd.DataFrame) -> str:
        """检测时间周期"""
        try:
            if len(df) < 2:
                return "15m"
            
            # 计算时间间隔
            time_diff = (df['timestamp'].iloc[1] - df['timestamp'].iloc[0]).total_seconds()
            
            # 时间周期映射
            timeframe_map = {
                60: "1m",
                300: "5m", 
                900: "15m",
                1800: "30m",
                3600: "1h",
                14400: "4h",
                86400: "1d"
            }
            
            # 找到最接近的时间周期
            closest_timeframe = "15m"
            min_diff = float('inf')
            for seconds, timeframe in timeframe_map.items():
                diff = abs(time_diff - seconds)
                if diff < min_diff:
                    min_diff = diff
                    closest_timeframe = timeframe
            
            return closest_timeframe
            
        except Exception:
            return "15m"
    
    def create_flagpole_chart(self, df: pd.DataFrame, flagpole: Dict, index: int) -> str:
        """创建单个旗杆图表"""
        try:
            # 获取旗杆信息
            start_idx = flagpole['start_idx']
            end_idx = flagpole['end_idx']
            direction = flagpole['direction']
            
            # 显示范围：旗杆前后各50个周期
            context = 50
            display_start = max(0, start_idx - context)
            display_end = min(len(df), end_idx + context)
            
            display_df = df.iloc[display_start:display_end].copy()
            adj_start = start_idx - display_start
            adj_end = end_idx - display_start
            
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                         gridspec_kw={'height_ratios': [3.5, 1]})
            
            # 调整子图布局
            plt.subplots_adjust(right=0.72)
            
            # === K线图 ===
            for i, (_, row) in enumerate(display_df.iterrows()):
                # K线颜色：上涨红色，下跌绿色
                color = '#FF4444' if row['close'] >= row['open'] else '#00AA00'
                
                # K线实体
                body_height = abs(row['close'] - row['open'])
                body_bottom = min(row['open'], row['close'])
                
                if body_height > 0:
                    ax1.add_patch(Rectangle((i-0.3, body_bottom), 0.6, body_height, 
                                          facecolor=color, alpha=0.7, edgecolor=color))
                else:
                    ax1.plot([i-0.3, i+0.3], [row['close'], row['close']], 
                            color=color, linewidth=2)
                
                # 影线
                ax1.plot([i, i], [row['low'], row['high']], color=color, linewidth=1)
            
            # 标记旗杆区域
            flagpole_color = '#FF6666' if direction == 'up' else '#66AA66'
            ax1.axvspan(adj_start, adj_end, alpha=0.25, color=flagpole_color, label='旗杆区域')
            
            # 标记起终点
            start_price = display_df.iloc[adj_start]['close']
            end_price = display_df.iloc[adj_end]['close']
            
            ax1.plot(adj_start, start_price, marker='^' if direction == 'up' else 'v', 
                    color='darkred' if direction == 'up' else 'darkgreen', 
                    markersize=12, label='起点', markeredgecolor='white', markeredgewidth=1)
            ax1.plot(adj_end, end_price, marker='s', 
                    color='darkred' if direction == 'up' else 'darkgreen', 
                    markersize=10, label='终点', markeredgecolor='white', markeredgewidth=1)
            
            # 趋势线
            ax1.plot([adj_start, adj_end], [start_price, end_price], 
                    color='black', linestyle='--', linewidth=2.5, alpha=0.8)
            
            # 标题
            start_time = display_df.iloc[adj_start]['timestamp']
            direction_cn = "上升" if direction == 'up' else "下降"
            
            title = (f'旗杆 #{index+1:03d} - {direction_cn}旗杆详细分析\\n'
                    f'时间: {start_time.strftime("%Y-%m-%d %H:%M")} | '
                    f'高度: {flagpole["height_pct"]:.2f}% | 长度: {flagpole["length"]} 周期')
            
            ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            # === 添加右侧统一信息标注 ===
            self._add_right_side_annotations(ax1, flagpole, direction_cn)
            
            ax1.set_ylabel('价格', fontsize=11)
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left', fontsize=10)
            
            # === 成交量图 ===
            colors = ['#FF4444' if display_df.iloc[i]['close'] >= display_df.iloc[i]['open'] 
                     else '#00AA00' for i in range(len(display_df))]
            
            ax2.bar(range(len(display_df)), display_df['volume'], 
                   color=colors, alpha=0.7)
            
            # 标记旗杆成交量区域
            ax2.axvspan(adj_start, adj_end, alpha=0.25, color=flagpole_color)
            
            # 标记量能爆发点
            if flagpole['volume_spike'] > 1.5:
                max_vol_idx = adj_start + np.argmax(
                    display_df.iloc[adj_start:adj_end+1]['volume'].values
                )
                max_vol = display_df.iloc[max_vol_idx]['volume']
                ax2.plot(max_vol_idx, max_vol, marker='*', color='orange', 
                        markersize=15, label=f'量能爆发 ({flagpole["volume_spike"]:.2f}x)')
                ax2.legend(loc='upper right', fontsize=9)
            
            ax2.set_title('成交量分析', fontsize=11)
            ax2.set_xlabel('时间位置', fontsize=11)
            ax2.set_ylabel('成交量', fontsize=11)
            ax2.grid(True, alpha=0.3)
            
            # 时间标签
            time_points = [0, len(display_df)//3, 2*len(display_df)//3, len(display_df)-1]
            time_labels = [display_df.iloc[i]['timestamp'].strftime('%m-%d %H:%M') 
                          for i in time_points if i < len(display_df)]
            
            for ax in [ax1, ax2]:
                ax.set_xticks(time_points[:len(time_labels)])
                ax.set_xticklabels(time_labels, rotation=45, fontsize=10)
            
            plt.tight_layout()
            
            # 保存
            filename = f"flagpole_{index+1:03d}_{direction}_{start_time.strftime('%Y%m%d_%H%M')}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=120, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            print(f"创建图表失败: {str(e)}")
            return ""
    
    def _add_right_side_annotations(self, ax, flagpole: Dict, direction_cn: str):
        """在K线图右侧添加统一的信息标注"""
        try:
            x_pos = 1.02
            annotations = []
            
            # 旗杆核心信息
            annotations.append({
                'text': f'【旗杆信息】',
                'style': {'fontsize': 12, 'fontweight': 'bold', 'color': 'black'}
            })
            annotations.append({
                'text': f'方向: {direction_cn}旗杆',
                'style': {'fontsize': 11, 'color': 'darkred' if direction_cn == '上升' else 'darkgreen'}
            })
            annotations.append({
                'text': f'高度: {flagpole["height_pct"]:.2f}%',
                'style': {'fontsize': 11, 'color': 'black'}
            })
            annotations.append({
                'text': f'长度: {flagpole["length"]} 周期',
                'style': {'fontsize': 11, 'color': 'black'}
            })
            annotations.append({
                'text': f'质量得分: {flagpole["quality_score"]:.2f}',
                'style': {'fontsize': 11, 'color': 'green' if flagpole["quality_score"] > 0.5 else 'orange'}
            })
            
            # 检测参数
            annotations.append({
                'text': f'\\n【系统检测器】',
                'style': {'fontsize': 12, 'fontweight': 'bold', 'color': 'black'}
            })
            annotations.append({
                'text': f'斜率分数: {flagpole["slope_score"]:.3f}',
                'style': {'fontsize': 10, 'color': 'gray'}
            })
            annotations.append({
                'text': f'回撤幅度: {flagpole["retrace_pct"]:.1f}%',
                'style': {'fontsize': 10, 'color': 'gray'}
            })
            annotations.append({
                'text': f'波动率: {flagpole["volatility"]:.2f}%',
                'style': {'fontsize': 10, 'color': 'gray'}
            })
            
            # 成交量信息
            annotations.append({
                'text': f'\\n【成交量分析】',
                'style': {'fontsize': 12, 'fontweight': 'bold', 'color': 'black'}
            })
            annotations.append({
                'text': f'量能爆发: {flagpole["volume_burst"]:.2f}x',
                'style': {'fontsize': 11, 'color': 'red' if flagpole["volume_burst"] > 2 else 'black'}
            })
            annotations.append({
                'text': f'最大量能: {flagpole["volume_spike"]:.2f}x',
                'style': {'fontsize': 11, 'color': 'red' if flagpole["volume_spike"] > 3 else 'black'}
            })
            
            # 添加标注
            start_y = 0.95
            line_height = 0.04
            
            for i, annotation in enumerate(annotations):
                y_pos = start_y - i * line_height
                ax.text(x_pos, y_pos, annotation['text'],
                       transform=ax.transAxes,
                       verticalalignment='top',
                       horizontalalignment='left',
                       **annotation['style'])
            
            # 添加边框
            from matplotlib.patches import Rectangle
            box_height = len(annotations) * line_height + 0.02
            box_rect = Rectangle((x_pos - 0.01, start_y - box_height), 
                               0.25, box_height,
                               transform=ax.transAxes,
                               facecolor='white',
                               edgecolor='lightgray',
                               alpha=0.9,
                               linewidth=1)
            ax.add_patch(box_rect)
            
        except Exception as e:
            print(f"添加右侧标注失败: {str(e)}")
    
    def run_optimized_detection_and_visualization(self, symbol: str = "RBL8") -> Dict[str, Any]:
        """运行优化的旗杆检测和可视化"""
        print("\\n开始优化旗杆检测和可视化...")
        
        # 获取完整数据
        df = self.get_complete_data(symbol)
        if df.empty:
            return {}
        
        # 使用系统检测器检测旗杆
        flagpoles = self.detect_all_flagpoles_optimized(df)
        
        if not flagpoles:
            print("未检测到旗杆")
            return {}
        
        results = {
            'total_flagpoles': len(flagpoles),
            'charts_created': 0,
            'successful_charts': [],
            'failed_charts': [],
            'output_directory': str(self.output_dir),
            'data_info': {
                'symbol': symbol,
                'total_records': len(df),
                'start_date': df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S'),
                'end_date': df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        # 统计分析
        up_flagpoles = [f for f in flagpoles if f['direction'] == 'up']
        down_flagpoles = [f for f in flagpoles if f['direction'] == 'down']
        
        print(f"\\n旗杆统计:")
        print(f"总数: {len(flagpoles)} 个")
        print(f"上升旗杆: {len(up_flagpoles)} 个 ({len(up_flagpoles)/len(flagpoles)*100:.1f}%)")
        print(f"下降旗杆: {len(down_flagpoles)} 个 ({len(down_flagpoles)/len(flagpoles)*100:.1f}%)")
        
        if flagpoles:
            print(f"平均高度: {np.mean([f['height_pct'] for f in flagpoles]):.2f}%")
            print(f"平均质量得分: {np.mean([f['quality_score'] for f in flagpoles]):.2f}")
        
        # 创建所有图表
        print(f"\\n开始创建 {len(flagpoles)} 个图表...")
        
        for i, flagpole in enumerate(flagpoles):
            direction_cn = "上升" if flagpole['direction'] == 'up' else "下降"
            
            # 显示进度
            if i % 10 == 0 or i == len(flagpoles) - 1:
                print(f"创建进度: {(i+1)/len(flagpoles)*100:.1f}% ({i+1}/{len(flagpoles)})")
            
            chart_path = self.create_flagpole_chart(df, flagpole, i)
            if chart_path:
                results['successful_charts'].append({
                    'index': i + 1,
                    'direction': flagpole['direction'],
                    'height_pct': flagpole['height_pct'],
                    'quality_score': flagpole['quality_score'],
                    'chart_path': chart_path,
                    'start_time': df.iloc[flagpole['start_idx']]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                })
                results['charts_created'] += 1
            else:
                results['failed_charts'].append({
                    'index': i + 1,
                    'direction': flagpole['direction'],
                    'error': '图表创建失败'
                })
        
        # 生成汇总报告
        self._generate_optimized_report(results, flagpoles)
        
        print(f"\\n优化检测和可视化完成!")
        print(f"成功创建 {results['charts_created']}/{len(flagpoles)} 个图表")
        print(f"输出目录: {results['output_directory']}")
        
        return results
    
    def _generate_optimized_report(self, results: Dict[str, Any], flagpoles: List[Dict]):
        """生成优化的汇总报告"""
        try:
            report_path = self.output_dir / f"optimized_flagpoles_report.md"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"# {results['data_info']['symbol']} - 优化旗杆检测和可视化报告\\n\\n")
                f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
                f.write(f"**检测方式**: 使用系统高效检测器\\n\\n")
                
                f.write("## 数据信息\\n\\n")
                f.write(f"- **品种代码**: {results['data_info']['symbol']}\\n")
                f.write(f"- **数据记录**: {results['data_info']['total_records']:,} 条\\n")
                f.write(f"- **时间范围**: {results['data_info']['start_date']} 至 {results['data_info']['end_date']}\\n\\n")
                
                f.write("## 检测结果统计\\n\\n")
                f.write(f"- **总旗杆数**: {results['total_flagpoles']} 个\\n")
                f.write(f"- **成功创建图表**: {results['charts_created']} 个\\n")
                f.write(f"- **失败**: {len(results['failed_charts'])} 个\\n")
                f.write(f"- **成功率**: {results['charts_created']/results['total_flagpoles']*100:.1f}%\\n\\n")
                
                # 详细统计
                if flagpoles:
                    up_count = len([f for f in flagpoles if f['direction'] == 'up'])
                    down_count = len([f for f in flagpoles if f['direction'] == 'down'])
                    avg_height = np.mean([f['height_pct'] for f in flagpoles])
                    avg_quality = np.mean([f['quality_score'] for f in flagpoles])
                    max_height = max([f['height_pct'] for f in flagpoles])
                    min_height = min([f['height_pct'] for f in flagpoles])
                    
                    f.write("## 质量分析\\n\\n")
                    f.write(f"- **上升旗杆**: {up_count} 个 ({up_count/len(flagpoles)*100:.1f}%)\\n")
                    f.write(f"- **下降旗杆**: {down_count} 个 ({down_count/len(flagpoles)*100:.1f}%)\\n")
                    f.write(f"- **平均高度**: {avg_height:.2f}%\\n")
                    f.write(f"- **高度范围**: {min_height:.2f}% ~ {max_height:.2f}%\\n")
                    f.write(f"- **平均质量得分**: {avg_quality:.2f}\\n\\n")
                
                f.write("## 系统检测器优势\\n\\n")
                f.write("- **高效处理**: 使用现有优化算法，处理速度快\\n")
                f.write("- **质量保证**: 基于动态基线系统，检测质量高\\n")
                f.write("- **智能过滤**: 自动去除低质量和重叠形态\\n")
                f.write("- **参数优化**: 根据时间周期自动调整检测参数\\n\\n")
                
                f.write("## 输出文件\\n\\n")
                f.write(f"所有图表文件已保存到: `{results['output_directory']}`\\n")
                
            print(f"优化汇总报告已生成: {report_path}")
            
        except Exception as e:
            print(f"生成汇总报告失败: {str(e)}")

def main():
    """主函数"""
    try:
        visualizer = OptimizedFlagpoleVisualizer()
        
        # 运行优化检测和可视化
        results = visualizer.run_optimized_detection_and_visualization("RBL8")
        
        if results:
            print(f"\\n=== 优化检测结果 ===")
            print(f"数据范围: {results['data_info']['start_date']} 至 {results['data_info']['end_date']}")
            print(f"总记录数: {results['data_info']['total_records']:,} 条")
            print(f"检测到旗杆: {results['total_flagpoles']} 个")
            print(f"成功创建图表: {results['charts_created']} 个")
            print(f"输出目录: {results['output_directory']}")
            
    except Exception as e:
        print(f"程序执行失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()