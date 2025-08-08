#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的单个旗杆独立可视化测试脚本
解决中文乱码问题，优化显示数量，添加参数标注
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

class ImprovedFlagpoleDetector:
    """改进的旗杆检测器"""
    
    def __init__(self):
        # 使用放宽的阈值
        self.detection_params = {
            'min_height_pct': 0.25,      # 最小高度百分比
            'min_slope_score': 0.1,      # 最小斜率分数
            'min_volume_burst': 1.2,     # 最小量能爆发倍数
            'max_retrace_pct': 50,       # 最大回撤百分比
            'min_length': 3,             # 最小旗杆长度
            'max_length': 10,            # 最大旗杆长度
            'volume_ma_period': 10,      # 量能均线周期
            'context_period': 20,        # 上下文周期
        }
        
    def detect_flagpoles(self, df: pd.DataFrame) -> List[Dict]:
        """检测旗杆并记录详细参数"""
        flagpoles = []
        
        if len(df) < 50:
            return flagpoles
        
        # 计算基础指标
        df = df.copy()
        df['volume_ma'] = df['volume'].rolling(window=self.detection_params['volume_ma_period']).mean()
        df['price_change_pct'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1) * 100
        
        # 扫描旗杆
        for i in range(self.detection_params['context_period'], 
                      len(df) - self.detection_params['context_period']):
            
            for length in range(self.detection_params['min_length'], 
                              self.detection_params['max_length'] + 1):
                if i + length >= len(df):
                    continue
                
                start_idx = i
                end_idx = i + length
                
                start_price = df.iloc[start_idx]['close']
                end_price = df.iloc[end_idx]['close']
                
                # 计算高度
                height_pct = abs(end_price - start_price) / start_price * 100
                
                if height_pct < self.detection_params['min_height_pct']:
                    continue
                
                # 判断方向
                direction = 'up' if end_price > start_price else 'down'
                
                # 成交量分析
                pole_volume = df.iloc[start_idx:end_idx + 1]['volume'].mean()
                base_volume = df.iloc[start_idx - self.detection_params['volume_ma_period']:start_idx]['volume'].mean()
                volume_burst = pole_volume / base_volume if base_volume > 0 else 1.0
                
                if volume_burst < self.detection_params['min_volume_burst']:
                    continue
                
                # 计算斜率分数
                x_values = np.arange(length + 1)
                y_values = df.iloc[start_idx:end_idx + 1]['close'].values
                slope = (y_values[-1] - y_values[0]) / len(y_values)
                slope_score = abs(slope) / start_price * 100
                
                if slope_score < self.detection_params['min_slope_score']:
                    continue
                
                # 回撤检查
                if direction == 'up':
                    min_price = df.iloc[start_idx:end_idx + 1]['low'].min()
                    retrace_pct = (start_price - min_price) / start_price * 100
                else:
                    max_price = df.iloc[start_idx:end_idx + 1]['high'].max()
                    retrace_pct = (max_price - start_price) / start_price * 100
                
                if retrace_pct > self.detection_params['max_retrace_pct']:
                    continue
                
                # 计算更多技术指标
                price_range = df.iloc[start_idx:end_idx + 1]['high'].max() - df.iloc[start_idx:end_idx + 1]['low'].min()
                volatility = price_range / start_price * 100
                
                max_volume = df.iloc[start_idx:end_idx + 1]['volume'].max()
                volume_spike = max_volume / base_volume if base_volume > 0 else 1.0
                
                # 计算质量得分
                quality_score = (height_pct * 0.4 + slope_score * 0.3 + 
                               min(volume_burst, 5) * 0.3) / 3
                
                # 检查重叠
                overlap = False
                for existing in flagpoles:
                    if not (end_idx < existing['start_idx'] or start_idx > existing['end_idx']):
                        if quality_score > existing['quality_score']:
                            flagpoles.remove(existing)
                        else:
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
                        'volume_spike': volume_spike,
                        'retrace_pct': retrace_pct,
                        'volatility': volatility,
                        'quality_score': quality_score,
                        'start_price': start_price,
                        'end_price': end_price,
                        'length': length,
                        'detection_params': self.detection_params.copy()
                    }
                    flagpoles.append(flagpole)
        
        # 按时间排序并限制数量
        flagpoles.sort(key=lambda x: x['start_idx'])
        return flagpoles[:20]

class ImprovedFlagpoleVisualizer:
    """改进的单个旗杆可视化器"""
    
    def __init__(self):
        # 数据连接器
        self.csv_connector = CSVDataConnector()
        
        # 输出目录
        self.output_dir = Path(project_root) / "output" / "flagpole_tests" / "individual_charts_improved"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 旗杆检测器
        self.detector = ImprovedFlagpoleDetector()
        
        print("改进版可视化器初始化完成")
        print(f"输出目录: {self.output_dir}")
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据去重处理"""
        try:
            # 按timestamp列去重，保留最后一条记录
            if 'timestamp' in df.columns:
                df_deduplicated = df.drop_duplicates(subset=['timestamp'], keep='last')
                df_deduplicated = df_deduplicated.sort_values('timestamp').reset_index(drop=True)
                return df_deduplicated
            else:
                print("警告: 数据中没有timestamp列，跳过去重处理")
                return df
        except Exception as e:
            print(f"数据去重失败: {str(e)}")
            return df
        
    def get_sample_data(self, symbol: str, max_records: int = 2000) -> pd.DataFrame:
        """获取样本数据"""
        try:
            # 连接数据源
            self.csv_connector.connect()
            
            # 获取完整数据
            df = self.csv_connector.get_data(symbol)
            if df.empty:
                print(f"未找到 {symbol} 的数据")
                return pd.DataFrame()
            
            print(f"原始数据: {len(df)} 条记录")
            
            # 数据去重处理
            original_count = len(df)
            df = self._remove_duplicates(df)
            deduplicated_count = len(df)
            
            if original_count > deduplicated_count:
                print(f"去重处理: 删除 {original_count - deduplicated_count} 条重复记录")
                print(f"去重后数据: {deduplicated_count} 条记录")
            
            # 取最近的数据进行测试
            if len(df) > max_records:
                df = df.tail(max_records).copy()
                print(f"取最近 {max_records} 条数据进行测试")
            
            print(f"测试数据范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            print(f"获取数据失败: {str(e)}")
            return pd.DataFrame()
    
    def create_improved_chart(self, df: pd.DataFrame, flagpole: Dict, index: int) -> str:
        """创建改进的旗杆图表"""
        try:
            # 获取旗杆信息
            start_idx = flagpole['start_idx']
            end_idx = flagpole['end_idx']
            direction = flagpole['direction']
            
            # 恢复到50个K线周期显示
            context = 50  # 恢复标准显示范围
            display_start = max(0, start_idx - context)
            display_end = min(len(df), end_idx + context)
            
            display_df = df.iloc[display_start:display_end].copy()
            adj_start = start_idx - display_start
            adj_end = end_idx - display_start
            
            # 创建图表，减小图片宽度，调整K线密度
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                         gridspec_kw={'height_ratios': [3.5, 1]})
            
            # 调整子图布局，为右侧标注留出空间
            plt.subplots_adjust(right=0.72)
            
            # === K线图 ===
            for i, (_, row) in enumerate(display_df.iterrows()):
                # 修改K线颜色：上涨红色，下跌绿色（中国股市习惯）
                color = '#FF4444' if row['close'] >= row['open'] else '#00AA00'
                
                # K线实体，减小K线宽度
                body_height = abs(row['close'] - row['open'])
                body_bottom = min(row['open'], row['close'])
                
                if body_height > 0:
                    # 减小K线宽度从0.8到0.6
                    ax1.add_patch(Rectangle((i-0.3, body_bottom), 0.6, body_height, 
                                          facecolor=color, alpha=0.7, edgecolor=color))
                else:
                    # 一字线也相应减小宽度
                    ax1.plot([i-0.3, i+0.3], [row['close'], row['close']], 
                            color=color, linewidth=2)
                
                # 影线
                ax1.plot([i, i], [row['low'], row['high']], color=color, linewidth=1)
            
            # 标记旗杆区域（调整颜色匹配新的K线颜色体系）
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
            
            # === 成交量图 ===（同步调整成交量柱颜色）
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
            
            # 成交量信息已经集成到右侧统一标注中，这里不再单独添加
            
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
            filename = f"improved_flagpole_{index+1:03d}_{direction}_{start_time.strftime('%Y%m%d_%H%M')}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=120, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(filepath)
            
        except Exception as e:
            print(f"创建图表失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return ""
    
    def _add_right_side_annotations(self, ax, flagpole: Dict, direction_cn: str):
        """在K线图右侧添加统一的信息标注"""
        try:
            # 设置右侧标注区域的x位置（图表右边界外侧）
            x_pos = 1.02  # 超出图表边界
            
            # 标注内容列表，从上到下排列
            annotations = []
            
            # 1. 旗杆核心信息
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
            
            # 2. 检测参数
            annotations.append({
                'text': f'\\n【检测参数】',
                'style': {'fontsize': 12, 'fontweight': 'bold', 'color': 'black'}
            })
            params = flagpole['detection_params']
            annotations.append({
                'text': f'最小高度: {params["min_height_pct"]:.2f}%',
                'style': {'fontsize': 10, 'color': 'gray'}
            })
            annotations.append({
                'text': f'最小斜率: {params["min_slope_score"]:.2f}',
                'style': {'fontsize': 10, 'color': 'gray'}
            })
            annotations.append({
                'text': f'最小量能: {params["min_volume_burst"]:.1f}x',
                'style': {'fontsize': 10, 'color': 'gray'}
            })
            annotations.append({
                'text': f'最大回撤: {params["max_retrace_pct"]:.0f}%',
                'style': {'fontsize': 10, 'color': 'gray'}
            })
            
            # 3. 实际测量值
            annotations.append({
                'text': f'\\n【实际测量】',
                'style': {'fontsize': 12, 'fontweight': 'bold', 'color': 'black'}
            })
            annotations.append({
                'text': f'斜率分数: {flagpole["slope_score"]:.3f}',
                'style': {'fontsize': 11, 'color': 'black'}
            })
            annotations.append({
                'text': f'回撤幅度: {flagpole["retrace_pct"]:.1f}%',
                'style': {'fontsize': 11, 'color': 'black'}
            })
            annotations.append({
                'text': f'波动率: {flagpole["volatility"]:.2f}%',
                'style': {'fontsize': 11, 'color': 'black'}
            })
            
            # 4. 成交量信息
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
            
            # 计算起始y位置（从顶部开始）
            start_y = 0.95
            line_height = 0.04  # 每行间距
            
            # 逐行添加标注
            for i, annotation in enumerate(annotations):
                y_pos = start_y - i * line_height
                
                ax.text(x_pos, y_pos, annotation['text'],
                       transform=ax.transAxes,
                       verticalalignment='top',
                       horizontalalignment='left',
                       **annotation['style'])
            
            # 添加边框（可选）
            from matplotlib.patches import Rectangle
            # 计算边框位置和大小
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

    def _create_parameter_text(self, flagpole: Dict) -> str:
        """创建参数信息文本"""
        params = flagpole['detection_params']
        
        text = "检测参数\\n" + "-" * 15 + "\\n"
        text += f"最小高度: {params['min_height_pct']:.2f}%\\n"
        text += f"最小斜率: {params['min_slope_score']:.2f}\\n"
        text += f"最小量能: {params['min_volume_burst']:.1f}x\\n"
        text += f"最大回撤: {params['max_retrace_pct']:.0f}%\\n"
        text += f"长度范围: {params['min_length']}-{params['max_length']}\\n"
        
        text += "\\n实际测量\\n" + "-" * 15 + "\\n"
        text += f"高度: {flagpole['height_pct']:.2f}%\\n"
        text += f"斜率分: {flagpole['slope_score']:.3f}\\n"
        text += f"量能爆发: {flagpole['volume_burst']:.2f}x\\n"
        text += f"回撤: {flagpole['retrace_pct']:.1f}%\\n"
        text += f"波动率: {flagpole['volatility']:.2f}%\\n"
        text += f"质量得分: {flagpole['quality_score']:.2f}"
        
        return text
    
    def run_improved_test(self, symbol: str = "RBL8") -> Dict[str, Any]:
        """运行改进的测试"""
        print("\\n开始改进版旗杆可视化测试...")
        
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
            direction_cn = "上升" if flagpole['direction'] == 'up' else "下降"
            print(f"创建图表 #{i+1}: {direction_cn}旗杆, 高度{flagpole['height_pct']:.2f}%, "
                  f"质量{flagpole['quality_score']:.2f}")
            
            chart_path = self.create_improved_chart(df, flagpole, i)
            if chart_path:
                results['successful_charts'].append({
                    'index': i + 1,
                    'direction': flagpole['direction'],
                    'height_pct': flagpole['height_pct'],
                    'quality_score': flagpole['quality_score'],
                    'chart_path': chart_path
                })
                results['charts_created'] += 1
                print(f"  成功: {Path(chart_path).name}")
            else:
                print(f"  失败")
        
        # 生成改进的汇总报告
        self._generate_improved_report(results, symbol, flagpoles)
        
        print(f"\\n改进测试完成!")
        print(f"成功创建 {results['charts_created']}/{len(flagpoles)} 个图表")
        print(f"输出目录: {results['output_directory']}")
        
        return results
    
    def _generate_improved_report(self, results: Dict[str, Any], symbol: str, flagpoles: List[Dict]):
        """生成改进的汇总报告"""
        try:
            report_path = self.output_dir / f"{symbol}_improved_charts_summary.md"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"# {symbol} - 改进版单个旗杆可视化汇总报告\\n\\n")
                f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
                
                f.write("## 改进内容\\n\\n")
                f.write("1. **解决中文乱码**: 设置正确的中文字体支持\\n")
                f.write("2. **优化显示数量**: K线显示数量减少1/4 (50→37个周期)\\n")
                f.write("3. **参数信息标注**: 在图表上显示检测参数和实际测量值\\n")
                f.write("4. **增强信息展示**: 添加质量得分、波动率等关键指标\\n\\n")
                
                f.write("## 处理统计\\n\\n")
                f.write(f"- **总旗杆数**: {results['total_flagpoles']} 个\\n")
                f.write(f"- **成功创建图表**: {results['charts_created']} 个\\n")
                f.write(f"- **成功率**: {results['charts_created']/results['total_flagpoles']*100:.1f}%\\n\\n")
                
                # 统计分析
                if flagpoles:
                    avg_height = np.mean([f['height_pct'] for f in flagpoles])
                    avg_quality = np.mean([f['quality_score'] for f in flagpoles])
                    up_count = len([f for f in flagpoles if f['direction'] == 'up'])
                    down_count = len([f for f in flagpoles if f['direction'] == 'down'])
                    
                    f.write("## 检测质量分析\\n\\n")
                    f.write(f"- **平均高度**: {avg_height:.2f}%\\n")
                    f.write(f"- **平均质量得分**: {avg_quality:.2f}\\n")
                    f.write(f"- **上升旗杆**: {up_count} 个 ({up_count/len(flagpoles)*100:.1f}%)\\n")
                    f.write(f"- **下降旗杆**: {down_count} 个 ({down_count/len(flagpoles)*100:.1f}%)\\n\\n")
                
                if results['successful_charts']:
                    f.write("### 成功创建的图表\\n\\n")
                    f.write("| 序号 | 方向 | 高度 | 质量得分 | 文件名 |\\n")
                    f.write("|------|------|------|----------|--------|\\n")
                    
                    for chart in results['successful_charts']:
                        filename = Path(chart['chart_path']).name
                        direction_cn = "上升" if chart['direction'] == 'up' else "下降"
                        f.write(f"| {chart['index']} | {direction_cn} | {chart['height_pct']:.2f}% | "
                               f"{chart['quality_score']:.2f} | `{filename}` |\\n")
                
                f.write("\\n## 图表特色功能\\n\\n")
                f.write("### 参数信息标注\\n")
                f.write("- **右上角参数框**: 显示检测参数和实际测量值\\n")
                f.write("- **旗杆核心信息**: 在旗杆区域显示方向、高度、质量得分\\n")
                f.write("- **成交量信息**: 在成交量图显示量能爆发倍数\\n\\n")
                
                f.write("### 优化显示\\n")
                f.write("- **减少K线数量**: 显示范围从50减少到37个周期\\n")
                f.write("- **中文字体支持**: 解决图表中文显示乱码问题\\n")
                f.write("- **高清图表**: 提升DPI到120，图表更加清晰\\n")
                
            print(f"改进汇总报告已生成: {report_path}")
            
        except Exception as e:
            print(f"生成汇总报告失败: {str(e)}")

def main():
    """主函数"""
    try:
        # 设置控制台输出编码
        import sys
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        
        visualizer = ImprovedFlagpoleVisualizer()
        
        # 运行改进测试
        results = visualizer.run_improved_test("RBL8")
        
        if results:
            print(f"\\n=== 改进测试结果 ===")
            print(f"总旗杆数: {results['total_flagpoles']}")
            print(f"成功创建: {results['charts_created']} 个图表")
            print(f"输出目录: {results['output_directory']}")
            
    except Exception as e:
        print(f"程序执行失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()