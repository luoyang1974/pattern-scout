#!/usr/bin/env python3
"""
单个旗杆独立可视化测试脚本（简化版）
使用经验证的检测逻辑为每个旗杆创建独立的K线图表
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

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.connectors.csv_connector import CSVDataConnector

class SimpleFlagpoleDetector:
    """简化的旗杆检测器"""
    
    def __init__(self):
        # 使用放宽的阈值（与完整测试保持一致）
        self.relaxed_thresholds = {
            'slope_score_p90': 0.1,   # 从0.5放宽到0.1
            'volume_burst_p85': 1.2,  # 从1.5放宽到1.2
            'retrace_depth_p75': 0.5, # 从0.3放宽到0.5
        }
        
        # ATR周期
        self.atr_period = 14
        
    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """计算ATR"""
        high_low = df['high'] - df['low']
        high_prev_close = np.abs(df['high'] - df['close'].shift(1))
        low_prev_close = np.abs(df['low'] - df['close'].shift(1))
        
        ranges = pd.concat([high_low, high_prev_close, low_prev_close], axis=1)
        true_range = ranges.max(axis=1)
        
        return true_range.rolling(window=self.atr_period).mean()
        
    def detect_flagpoles(self, df: pd.DataFrame) -> List[Dict]:
        """检测旗杆"""
        flagpoles = []
        
        if len(df) < 100:
            return flagpoles
        
        # 计算技术指标
        df = df.copy()
        df['atr'] = self.calculate_atr(df)
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        
        # 计算价格变化
        df['price_change'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
        df['abs_price_change'] = abs(df['price_change'])
        
        # 寻找可能的旗杆
        min_length = 3
        max_length = 20
        
        for i in range(50, len(df) - 50):
            
            # 尝试不同的旗杆长度
            for length in range(min_length, max_length + 1):
                if i + length >= len(df):
                    continue
                
                start_idx = i
                end_idx = i + length
                
                # 获取旗杆数据
                pole_data = df.iloc[start_idx:end_idx + 1]
                
                if len(pole_data) < min_length:
                    continue
                
                # 计算旗杆特征
                start_price = pole_data.iloc[0]['close']
                end_price = pole_data.iloc[-1]['close']
                height_pct = abs(end_price - start_price) / start_price * 100
                
                # 基本条件筛选
                if height_pct < 0.25:  # 最小高度0.25%
                    continue
                    
                # 判断方向
                if end_price > start_price:
                    direction = 'up'
                    # 检查是否为持续上升
                    low_point = pole_data['low'].min()
                    if low_point < start_price * (1 - self.relaxed_thresholds['retrace_depth_p75']):
                        continue
                else:
                    direction = 'down'
                    # 检查是否为持续下降
                    high_point = pole_data['high'].max()
                    if high_point > start_price * (1 + self.relaxed_thresholds['retrace_depth_p75']):
                        continue
                
                # 计算斜率分数
                x_values = np.arange(len(pole_data))
                y_values = pole_data['close'].values
                
                # 简单线性回归
                if len(x_values) > 1:
                    slope, intercept = np.polyfit(x_values, y_values, 1)
                    slope_score = abs(slope) / start_price * 100  # 转换为百分比
                else:
                    slope_score = 0
                
                # 斜率筛选
                if slope_score < self.relaxed_thresholds['slope_score_p90']:
                    continue
                
                # 计算成交量爆发
                avg_volume = df.iloc[start_idx - 20:start_idx]['volume'].mean() if start_idx >= 20 else df['volume'].mean()
                max_volume = pole_data['volume'].max()
                volume_burst = max_volume / avg_volume if avg_volume > 0 else 1.0
                
                # 成交量筛选
                if volume_burst < self.relaxed_thresholds['volume_burst_p85']:
                    continue
                
                # 创建旗杆记录
                flagpole = {
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'direction': direction,
                    'height_pct': height_pct,
                    'slope_score': slope_score,
                    'volume_burst': volume_burst,
                    'start_price': start_price,
                    'end_price': end_price,
                    'length': length
                }
                
                # 检查是否与已有旗杆重叠
                overlap = False
                for existing in flagpoles:
                    if (start_idx <= existing['end_idx'] and end_idx >= existing['start_idx']):
                        # 如果重叠，保留质量更高的
                        existing_score = existing['height_pct'] * existing['slope_score'] * existing['volume_burst']
                        current_score = height_pct * slope_score * volume_burst
                        if current_score > existing_score:
                            flagpoles.remove(existing)
                        else:
                            overlap = True
                        break
                
                if not overlap:
                    flagpoles.append(flagpole)
        
        # 按时间排序
        flagpoles.sort(key=lambda x: x['start_idx'])
        
        return flagpoles

class IndividualFlagpoleVisualizer:
    """单个旗杆可视化器（简化版）"""
    
    def __init__(self):
        # 数据连接器
        self.csv_connector = CSVDataConnector()
        
        # 输出目录
        self.output_dir = Path(project_root) / "output" / "flagpole_tests" / "individual_charts"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 旗杆检测器
        self.detector = SimpleFlagpoleDetector()
        
        print("初始化完成")
        print(f"输出目录: {self.output_dir}")
        
    def detect_flagpoles_with_context(self, symbol: str) -> Tuple[pd.DataFrame, List[Dict]]:
        """检测旗杆并获取上下文数据"""
        try:
            # 连接数据源
            self.csv_connector.connect()
            
            # 获取数据
            df = self.csv_connector.get_data(symbol)
            if df.empty:
                print(f"未找到 {symbol} 的数据")
                return pd.DataFrame(), []
            
            print(f"加载数据: {len(df)} 条记录")
            print(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
            
            # 检测旗杆
            flagpoles = self.detector.detect_flagpoles(df)
            
            print(f"检测到 {len(flagpoles)} 个旗杆")
            
            return df, flagpoles
            
        except Exception as e:
            print(f"检测失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame(), []
    
    def create_individual_chart(self, df: pd.DataFrame, flagpole: Dict, index: int, 
                              context_periods: int = 100) -> str:
        """为单个旗杆创建独立的K线图"""
        try:
            # 获取旗杆信息
            start_idx = flagpole['start_idx']
            end_idx = flagpole['end_idx']
            direction = flagpole['direction']
            height_pct = flagpole['height_pct']
            slope_score = flagpole.get('slope_score', 0)
            volume_burst = flagpole.get('volume_burst', 1)
            
            # 计算显示范围（前后各context_periods个数据点）
            display_start = max(0, start_idx - context_periods)
            display_end = min(len(df), end_idx + context_periods)
            
            # 提取显示数据
            display_df = df.iloc[display_start:display_end].copy()
            
            # 调整旗杆索引到显示范围内
            flagpole_start_in_display = start_idx - display_start
            flagpole_end_in_display = end_idx - display_start
            
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                         gridspec_kw={'height_ratios': [3, 1]})
            
            # === K线图 ===
            # 绘制K线
            for i, (_, row) in enumerate(display_df.iterrows()):
                color = 'red' if row['close'] >= row['open'] else 'blue'
                
                # K线实体
                body_height = abs(row['close'] - row['open'])
                body_bottom = min(row['open'], row['close'])
                
                ax1.add_patch(Rectangle((i-0.3, body_bottom), 0.6, body_height, 
                                      facecolor=color, alpha=0.7))
                
                # 上下影线
                ax1.plot([i, i], [row['low'], row['high']], color=color, linewidth=1)
            
            # 标记旗杆区域
            flagpole_color = 'red' if direction == 'up' else 'blue'
            ax1.axvspan(flagpole_start_in_display, flagpole_end_in_display, 
                       alpha=0.2, color=flagpole_color, label='旗杆区域')
            
            # 标记旗杆起始点和结束点
            start_price = display_df.iloc[flagpole_start_in_display]['close']
            end_price = display_df.iloc[flagpole_end_in_display]['close']
            
            marker_up = '^' if direction == 'up' else 'v'
            marker_down = 's'
            
            ax1.plot(flagpole_start_in_display, start_price, marker=marker_up, 
                    color=flagpole_color, markersize=12, label='旗杆起点')
            ax1.plot(flagpole_end_in_display, end_price, marker=marker_down, 
                    color=flagpole_color, markersize=10, label='旗杆终点')
            
            # 绘制旗杆趋势线
            ax1.plot([flagpole_start_in_display, flagpole_end_in_display], 
                    [start_price, end_price], 
                    color=flagpole_color, linestyle='--', linewidth=2, alpha=0.8)
            
            # 图表标题和标签
            start_time = display_df.iloc[flagpole_start_in_display]['timestamp']
            end_time = display_df.iloc[flagpole_end_in_display]['timestamp']
            
            direction_cn = "上升" if direction == 'up' else "下降"
            ax1.set_title(f'旗杆 #{index+1:03d} - {direction_cn}旗杆详细分析\\n'
                         f'时间: {start_time.strftime("%Y-%m-%d %H:%M")} - {end_time.strftime("%Y-%m-%d %H:%M")}\\n'
                         f'高度: {height_pct:.2f}% | 斜率分: {slope_score:.3f} | 量能爆发: {volume_burst:.2f}x', 
                         fontsize=12, fontweight='bold')
            
            ax1.set_xlabel('时间位置')
            ax1.set_ylabel('价格')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 设置x轴标签（显示关键时间点）
            time_indices = [0, len(display_df)//4, len(display_df)//2, 
                           3*len(display_df)//4, len(display_df)-1]
            time_labels = [display_df.iloc[i]['timestamp'].strftime('%m-%d %H:%M') 
                          for i in time_indices]
            ax1.set_xticks(time_indices)
            ax1.set_xticklabels(time_labels, rotation=45)
            
            # === 成交量图 ===
            volume_colors = ['red' if display_df.iloc[i]['close'] >= display_df.iloc[i]['open'] 
                           else 'blue' for i in range(len(display_df))]
            
            ax2.bar(range(len(display_df)), display_df['volume'], 
                   color=volume_colors, alpha=0.7)
            
            # 标记旗杆期间的成交量
            ax2.axvspan(flagpole_start_in_display, flagpole_end_in_display, 
                       alpha=0.2, color=flagpole_color)
            
            # 标记量能爆发点
            if volume_burst > 1.2:  # 如果有明显的量能爆发
                max_vol_idx = flagpole_start_in_display + np.argmax(
                    display_df.iloc[flagpole_start_in_display:flagpole_end_in_display+1]['volume'].values
                )
                max_vol = display_df.iloc[max_vol_idx]['volume']
                ax2.plot(max_vol_idx, max_vol, marker='*', color='orange', 
                        markersize=15, label=f'量能爆发 ({volume_burst:.2f}x)')
            
            ax2.set_title('成交量分析')
            ax2.set_xlabel('时间位置')
            ax2.set_ylabel('成交量')
            ax2.grid(True, alpha=0.3)
            if volume_burst > 1.2:
                ax2.legend()
            
            # 设置x轴标签
            ax2.set_xticks(time_indices)
            ax2.set_xticklabels(time_labels, rotation=45)
            
            plt.tight_layout()
            
            # 保存图表
            filename = f"flagpole_{index+1:03d}_{direction}_{start_time.strftime('%Y%m%d_%H%M')}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            print(f"创建图表失败 (旗杆 #{index+1}): {str(e)}")
            import traceback
            traceback.print_exc()
            return ""
    
    def visualize_all_flagpoles(self, symbol: str = "RBL8", 
                               max_charts: Optional[int] = None) -> Dict[str, Any]:
        """为所有旗杆创建独立的可视化图表"""
        print(f"\\n开始为 {symbol} 创建所有旗杆的独立可视化图表...")
        
        # 检测旗杆
        df, flagpoles = self.detect_flagpoles_with_context(symbol)
        
        if not flagpoles:
            print("未检测到旗杆")
            return {}
        
        total_flagpoles = len(flagpoles)
        charts_to_create = total_flagpoles if max_charts is None else min(max_charts, total_flagpoles)
        
        print(f"准备为 {charts_to_create}/{total_flagpoles} 个旗杆创建图表...")
        
        results = {
            'total_flagpoles': total_flagpoles,
            'charts_created': 0,
            'successful_charts': [],
            'failed_charts': [],
            'output_directory': str(self.output_dir)
        }
        
        # 为每个旗杆创建图表
        for i, flagpole in enumerate(flagpoles[:charts_to_create]):
            print(f"\\n处理旗杆 #{i+1}/{charts_to_create}...")
            print(f"   方向: {'上升' if flagpole['direction'] == 'up' else '下降'}")
            print(f"   高度: {flagpole['height_pct']:.2f}%")
            print(f"   时间: 索引 {flagpole['start_idx']}-{flagpole['end_idx']}")
            
            chart_path = self.create_individual_chart(df, flagpole, i)
            
            if chart_path:
                results['successful_charts'].append({
                    'index': i + 1,
                    'direction': flagpole['direction'],
                    'height_pct': flagpole['height_pct'],
                    'chart_path': chart_path
                })
                results['charts_created'] += 1
                print(f"   图表已创建: {Path(chart_path).name}")
            else:
                results['failed_charts'].append(i + 1)
                print(f"   图表创建失败")
        
        # 生成汇总报告
        self._generate_summary_report(results, symbol)
        
        print(f"\\n处理完成!")
        print(f"成功创建 {results['charts_created']}/{total_flagpoles} 个旗杆图表")
        print(f"输出目录: {self.output_dir}")
        
        return results
    
    def _generate_summary_report(self, results: Dict[str, Any], symbol: str):
        """生成汇总报告"""
        try:
            report_path = self.output_dir / f"{symbol}_individual_charts_summary.md"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"# {symbol} - 单个旗杆可视化汇总报告\\n\\n")
                f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
                
                f.write("## 处理统计\\n\\n")
                f.write(f"- **总旗杆数**: {results['total_flagpoles']} 个\\n")
                f.write(f"- **成功创建图表**: {results['charts_created']} 个\\n")
                f.write(f"- **失败数量**: {len(results['failed_charts'])} 个\\n")
                f.write(f"- **成功率**: {results['charts_created']/results['total_flagpoles']*100:.1f}%\\n\\n")
                
                f.write("## 输出文件\\n\\n")
                f.write(f"所有图表文件保存在: `{results['output_directory']}`\\n\\n")
                
                if results['successful_charts']:
                    f.write("### 成功创建的图表\\n\\n")
                    f.write("| 序号 | 方向 | 高度 | 文件名 |\\n")
                    f.write("|------|------|------|--------|\\n")
                    
                    for chart in results['successful_charts'][:50]:  # 只显示前50个
                        filename = Path(chart['chart_path']).name
                        direction_cn = "上升" if chart['direction'] == 'up' else "下降"
                        f.write(f"| {chart['index']} | {direction_cn} | {chart['height_pct']:.2f}% | `{filename}` |\\n")
                    
                    if len(results['successful_charts']) > 50:
                        f.write(f"| ... | ... | ... | ... |\\n")
                        f.write(f"| 共 {len(results['successful_charts'])} 个图表文件 | | | |\\n")
                
                if results['failed_charts']:
                    f.write("\\n### 失败的图表\\n\\n")
                    f.write(f"失败的旗杆序号: {', '.join(map(str, results['failed_charts']))}\\n")
                
                f.write("\\n## 使用说明\\n\\n")
                f.write("每个图表文件包含:\\n")
                f.write("- **上半部分**: K线图 + 旗杆区域标记\\n")
                f.write("- **下半部分**: 成交量 + 量能爆发标记\\n")
                f.write("- **文件命名**: `flagpole_序号_方向_时间.png`\\n")
                f.write("- **显示范围**: 旗杆前后各100个时间点的上下文\\n\\n")
            
            print(f"汇总报告已生成: {report_path}")
            
        except Exception as e:
            print(f"生成汇总报告失败: {str(e)}")

def main():
    """主函数"""
    try:
        visualizer = IndividualFlagpoleVisualizer()
        
        # 创建所有旗杆的独立可视化
        # 可以设置max_charts参数来限制创建的图表数量（测试用）
        results = visualizer.visualize_all_flagpoles(
            symbol="RBL8",
            max_charts=20  # 先创建前20个进行测试
        )
        
        print(f"\\n最终结果:")
        print(f"总旗杆数: {results.get('total_flagpoles', 0)}")
        print(f"成功创建: {results.get('charts_created', 0)} 个图表")
        print(f"输出目录: {results.get('output_directory', 'unknown')}")
        
    except Exception as e:
        print(f"程序执行失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()