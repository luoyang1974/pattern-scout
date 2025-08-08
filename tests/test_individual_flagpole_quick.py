#!/usr/bin/env python3
"""
快速单个旗杆可视化测试脚本
使用部分数据快速测试单个旗杆图表生成功能
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

class QuickFlagpoleDetector:
    """快速旗杆检测器（使用部分数据）"""
    
    def __init__(self):
        # 使用放宽的阈值
        self.relaxed_thresholds = {
            'min_height_pct': 0.25,
            'min_slope_score': 0.1,
            'min_volume_burst': 1.2,
            'max_retrace': 0.5,
        }
        
    def detect_flagpoles(self, df: pd.DataFrame) -> List[Dict]:
        """快速检测旗杆（简化算法）"""
        flagpoles = []
        
        if len(df) < 50:
            return flagpoles
        
        # 计算基础指标
        df = df.copy()
        df['volume_ma'] = df['volume'].rolling(window=10).mean()
        df['price_change_pct'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1) * 100
        
        # 寻找明显的价格跳跃
        for i in range(20, len(df) - 20):
            
            # 寻找3-10个周期的快速移动
            for length in range(3, 11):
                if i + length >= len(df):
                    continue
                
                start_idx = i
                end_idx = i + length
                
                start_price = df.iloc[start_idx]['close']
                end_price = df.iloc[end_idx]['close']
                
                # 计算高度
                height_pct = abs(end_price - start_price) / start_price * 100
                
                if height_pct < self.relaxed_thresholds['min_height_pct']:
                    continue
                
                # 判断方向
                direction = 'up' if end_price > start_price else 'down'
                
                # 检查成交量
                pole_volume = df.iloc[start_idx:end_idx + 1]['volume'].mean()
                base_volume = df.iloc[start_idx - 10:start_idx]['volume'].mean()
                volume_burst = pole_volume / base_volume if base_volume > 0 else 1.0
                
                if volume_burst < self.relaxed_thresholds['min_volume_burst']:
                    continue
                
                # 计算斜率分数
                x_values = np.arange(length + 1)
                y_values = df.iloc[start_idx:end_idx + 1]['close'].values
                slope = (y_values[-1] - y_values[0]) / len(y_values)
                slope_score = abs(slope) / start_price * 100
                
                if slope_score < self.relaxed_thresholds['min_slope_score']:
                    continue
                
                # 检查回撤
                if direction == 'up':
                    min_price = df.iloc[start_idx:end_idx + 1]['low'].min()
                    if min_price < start_price * (1 - self.relaxed_thresholds['max_retrace'] / 100):
                        continue
                else:
                    max_price = df.iloc[start_idx:end_idx + 1]['high'].max()
                    if max_price > start_price * (1 + self.relaxed_thresholds['max_retrace'] / 100):
                        continue
                
                # 检查重叠
                overlap = False
                for existing in flagpoles:
                    if not (end_idx < existing['start_idx'] or start_idx > existing['end_idx']):
                        overlap = True
                        break
                
                if not overlap:
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
                    flagpoles.append(flagpole)
        
        # 按时间排序并限制数量
        flagpoles.sort(key=lambda x: x['start_idx'])
        return flagpoles[:20]  # 只返回前20个

class QuickFlagpoleVisualizer:
    """快速单个旗杆可视化器"""
    
    def __init__(self):
        # 数据连接器
        self.csv_connector = CSVDataConnector()
        
        # 输出目录
        self.output_dir = Path(project_root) / "output" / "flagpole_tests" / "individual_charts_quick"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 旗杆检测器
        self.detector = QuickFlagpoleDetector()
        
        print("快速可视化器初始化完成")
        print(f"输出目录: {self.output_dir}")
        
    def get_sample_data(self, symbol: str, max_records: int = 2000) -> pd.DataFrame:
        """获取样本数据（限制数据量）"""
        try:
            # 连接数据源
            self.csv_connector.connect()
            
            # 获取完整数据
            df = self.csv_connector.get_data(symbol)
            if df.empty:
                print(f"未找到 {symbol} 的数据")
                return pd.DataFrame()
            
            print(f"原始数据: {len(df)} 条记录")
            
            # 取最近的数据进行测试
            if len(df) > max_records:
                df = df.tail(max_records).copy()
                print(f"取最近 {max_records} 条数据进行测试")
            
            print(f"测试数据范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            print(f"获取数据失败: {str(e)}")
            return pd.DataFrame()
    
    def create_individual_chart(self, df: pd.DataFrame, flagpole: Dict, index: int) -> str:
        """为单个旗杆创建K线图"""
        try:
            # 获取旗杆信息
            start_idx = flagpole['start_idx']
            end_idx = flagpole['end_idx']
            direction = flagpole['direction']
            height_pct = flagpole['height_pct']
            
            # 计算显示范围
            context = 50
            display_start = max(0, start_idx - context)
            display_end = min(len(df), end_idx + context)
            
            display_df = df.iloc[display_start:display_end].copy()
            adj_start = start_idx - display_start
            adj_end = end_idx - display_start
            
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                         gridspec_kw={'height_ratios': [3, 1]})
            
            # === K线图 ===
            for i, (_, row) in enumerate(display_df.iterrows()):
                color = '#FF4444' if row['close'] >= row['open'] else '#4444FF'
                
                # K线实体
                body_height = abs(row['close'] - row['open'])
                body_bottom = min(row['open'], row['close'])
                
                if body_height > 0:
                    ax1.add_patch(Rectangle((i-0.4, body_bottom), 0.8, body_height, 
                                          facecolor=color, alpha=0.7, edgecolor=color))
                else:
                    # 一字线
                    ax1.plot([i-0.4, i+0.4], [row['close'], row['close']], 
                            color=color, linewidth=2)
                
                # 影线
                ax1.plot([i, i], [row['low'], row['high']], color=color, linewidth=1)
            
            # 标记旗杆区域
            flagpole_color = '#FF6666' if direction == 'up' else '#6666FF'
            ax1.axvspan(adj_start, adj_end, alpha=0.25, color=flagpole_color, label='旗杆区域')
            
            # 标记起终点
            start_price = display_df.iloc[adj_start]['close']
            end_price = display_df.iloc[adj_end]['close']
            
            ax1.plot(adj_start, start_price, marker='^' if direction == 'up' else 'v', 
                    color='darkred' if direction == 'up' else 'darkblue', 
                    markersize=10, label='起点')
            ax1.plot(adj_end, end_price, marker='s', 
                    color='darkred' if direction == 'up' else 'darkblue', 
                    markersize=8, label='终点')
            
            # 趋势线
            ax1.plot([adj_start, adj_end], [start_price, end_price], 
                    color='black', linestyle='--', linewidth=2, alpha=0.7)
            
            # 标题
            start_time = display_df.iloc[adj_start]['timestamp']
            direction_cn = "上升" if direction == 'up' else "下降"
            
            ax1.set_title(f'旗杆 #{index+1:03d} - {direction_cn}旗杆 ({height_pct:.2f}%)\\n'
                         f'时间: {start_time.strftime("%Y-%m-%d %H:%M")} | '
                         f'长度: {end_idx - start_idx + 1} 周期', 
                         fontsize=11, fontweight='bold')
            
            ax1.set_ylabel('价格')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')
            
            # === 成交量图 ===
            colors = ['#FF4444' if display_df.iloc[i]['close'] >= display_df.iloc[i]['open'] 
                     else '#4444FF' for i in range(len(display_df))]
            
            ax2.bar(range(len(display_df)), display_df['volume'], 
                   color=colors, alpha=0.7)
            
            # 标记旗杆成交量区域
            ax2.axvspan(adj_start, adj_end, alpha=0.25, color=flagpole_color)
            
            ax2.set_title('成交量')
            ax2.set_xlabel('时间位置')
            ax2.set_ylabel('成交量')
            ax2.grid(True, alpha=0.3)
            
            # 时间标签
            time_points = [0, len(display_df)//3, 2*len(display_df)//3, len(display_df)-1]
            time_labels = [display_df.iloc[i]['timestamp'].strftime('%m-%d %H:%M') 
                          for i in time_points if i < len(display_df)]
            
            for ax in [ax1, ax2]:
                ax.set_xticks(time_points[:len(time_labels)])
                ax.set_xticklabels(time_labels, rotation=45)
            
            plt.tight_layout()
            
            # 保存
            filename = f"quick_flagpole_{index+1:03d}_{direction}_{start_time.strftime('%Y%m%d_%H%M')}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            print(f"创建图表失败: {str(e)}")
            return ""
    
    def run_quick_test(self, symbol: str = "RBL8") -> Dict[str, Any]:
        """运行快速测试"""
        print(f"\\n开始快速旗杆可视化测试...")
        
        # 获取样本数据
        df = self.get_sample_data(symbol, max_records=2000)
        if df.empty:
            return {}
        
        # 检测旗杆
        flagpoles = self.detector.detect_flagpoles(df)
        print(f"检测到 {len(flagpoles)} 个旗杆")
        
        if not flagpoles:
            print("未检测到旗杆")
            return {}
        
        results = {
            'total_flagpoles': len(flagpoles),
            'charts_created': 0,
            'successful_charts': [],
            'output_directory': str(self.output_dir)
        }
        
        # 创建图表
        for i, flagpole in enumerate(flagpoles):
            print(f"创建图表 #{i+1}: {flagpole['direction']}旗杆, 高度{flagpole['height_pct']:.2f}%")
            
            chart_path = self.create_individual_chart(df, flagpole, i)
            if chart_path:
                results['successful_charts'].append({
                    'index': i + 1,
                    'direction': flagpole['direction'],
                    'height_pct': flagpole['height_pct'],
                    'chart_path': chart_path
                })
                results['charts_created'] += 1
                print(f"  成功: {Path(chart_path).name}")
            else:
                print(f"  失败")
        
        print(f"\\n快速测试完成!")
        print(f"成功创建 {results['charts_created']}/{len(flagpoles)} 个图表")
        print(f"输出目录: {results['output_directory']}")
        
        return results

def main():
    """主函数"""
    try:
        visualizer = QuickFlagpoleVisualizer()
        
        # 运行快速测试
        results = visualizer.run_quick_test("RBL8")
        
        if results:
            print(f"\\n=== 测试结果 ===")
            print(f"总旗杆数: {results['total_flagpoles']}")
            print(f"成功创建: {results['charts_created']} 个图表")
            print(f"输出目录: {results['output_directory']}")
            
    except Exception as e:
        print(f"程序执行失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()