#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成示例形态K线图表
为高质量和部分中等质量形态生成K线图表作为示例
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path

# 设置控制台编码（Windows）
if sys.platform.startswith('win'):
    os.system('chcp 65001 > nul')
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# 添加src到Python路径
sys.path.append(str(__file__).replace('generate_sample_pattern_charts.py', ''))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class SamplePatternChartGenerator:
    """示例形态K线图表生成器"""
    
    def __init__(self, data_connector):
        self.data_connector = data_connector
        self.generated_charts = []
        
    def generate_sample_charts(self) -> Dict:
        """生成示例形态图表"""
        print("开始生成示例形态K线图表...")
        
        # 读取形态数据
        patterns_df = pd.read_csv('output/reports/pattern_detailed_list.csv')
        print(f"加载了 {len(patterns_df)} 个形态数据")
        
        # 获取价格数据
        from datetime import datetime
        start_date = datetime(2019, 1, 1)
        end_date = datetime.now()
        price_data = self.data_connector.get_data('RBL8', start_date, end_date)
        
        # 处理时间字段
        time_col = 'timestamp' if 'timestamp' in price_data.columns else 'Datetime'
        price_data[time_col] = pd.to_datetime(price_data[time_col])
        price_data = price_data.set_index(time_col).sort_index()
        
        print(f"加载了 {len(price_data)} 条价格数据")
        
        # 创建输出目录
        output_dir = Path('output/pattern_charts_sample')
        output_dir.mkdir(exist_ok=True)
        (output_dir / 'high_quality').mkdir(exist_ok=True)
        (output_dir / 'medium_quality').mkdir(exist_ok=True)
        
        total_generated = 0
        
        # 生成所有高质量形态图表 (9个)
        high_quality = patterns_df[patterns_df['pattern_quality'] == 'high']
        print(f"\\n生成所有 {len(high_quality)} 个高质量形态图表...")
        for idx, pattern in high_quality.iterrows():
            try:
                chart_info = self._generate_single_pattern_chart(
                    pattern, price_data, output_dir / 'high_quality', idx, 'high'
                )
                if chart_info:
                    self.generated_charts.append(chart_info)
                    total_generated += 1
                    print(f"  已生成高质量形态 {total_generated}/{len(high_quality)}: {chart_info['chart_filename']}")
            except Exception as e:
                print(f"生成高质量形态 {idx} 图表失败: {e}")
        
        # 生成部分中等质量形态图表 (前20个)
        medium_quality = patterns_df[patterns_df['pattern_quality'] == 'medium'].head(20)
        print(f"\\n生成前 {len(medium_quality)} 个中等质量形态图表...")
        for idx, pattern in medium_quality.iterrows():
            try:
                chart_info = self._generate_single_pattern_chart(
                    pattern, price_data, output_dir / 'medium_quality', idx, 'medium'
                )
                if chart_info:
                    self.generated_charts.append(chart_info)
                    total_generated += 1
                    print(f"  已生成中等质量形态 {total_generated - len(high_quality)}/{len(medium_quality)}: {chart_info['chart_filename']}")
            except Exception as e:
                print(f"生成中等质量形态 {idx} 图表失败: {e}")
        
        # 生成映射文件
        mapping_file = self._generate_mapping_file(output_dir)
        
        # 生成汇总文档
        summary_file = self._generate_summary_document(output_dir, patterns_df)
        
        result = {
            'total_patterns_available': len(patterns_df),
            'charts_generated': total_generated,
            'high_quality_charts': len([c for c in self.generated_charts if c['quality'] == 'high']),
            'medium_quality_charts': len([c for c in self.generated_charts if c['quality'] == 'medium']),
            'output_directory': str(output_dir),
            'mapping_file': mapping_file,
            'summary_file': summary_file
        }
        
        print(f"\\n✅ 示例图表生成完成!")
        print(f"   可用形态总数: {result['total_patterns_available']}")
        print(f"   生成示例图表: {result['charts_generated']} 个")
        print(f"   高质量: {result['high_quality_charts']} 个")
        print(f"   中等质量: {result['medium_quality_charts']} 个")
        
        return result
    
    def _generate_single_pattern_chart(self, pattern: pd.Series, price_data: pd.DataFrame,
                                     output_dir: Path, pattern_idx: int, quality: str) -> Optional[Dict]:
        """生成单个形态的K线图表"""
        
        # 获取形态时间信息
        flagpole_start = pd.to_datetime(pattern['flagpole_start_time'])
        flagpole_end = pd.to_datetime(pattern['flagpole_end_time'])
        pattern_end = flagpole_end + timedelta(minutes=15 * pattern['pattern_duration'])
        
        # 扩展显示窗口（前后各25个K线）
        display_start = flagpole_start - timedelta(minutes=15 * 25)
        display_end = pattern_end + timedelta(minutes=15 * 25)
        
        # 获取显示数据
        try:
            display_data = price_data[display_start:display_end]
            if len(display_data) < 15:
                return None
        except:
            return None
            
        # 获取形态数据
        flagpole_data = price_data[flagpole_start:flagpole_end]
        pattern_flag_data = price_data[flagpole_end:pattern_end]
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # 绘制K线图
        self._plot_candlestick_chart(ax1, display_data, flagpole_data, pattern_flag_data, pattern)
        
        # 绘制成交量图
        self._plot_volume_chart(ax2, display_data, flagpole_data, pattern_flag_data)
        
        # 设置图表信息
        pattern_type = pattern['pattern_type'].upper()
        direction = pattern['flagpole_direction'].upper() 
        confidence = pattern['confidence_score']
        height = pattern['flagpole_height_percent']
        
        # 生成文件名
        time_str = flagpole_start.strftime('%Y%m%d_%H%M')
        filename = f'{pattern_type}_{direction}_{time_str}_conf{confidence:.3f}.png'
        chart_path = output_dir / filename
        
        # 设置标题
        fig.suptitle(
            f'RBL8 {pattern_type} 形态分析 ({quality.upper()}) - {direction}方向\\n'
            f'置信度: {confidence:.3f} | 旗杆高度: {height:.2f}% | 持续时间: {pattern["pattern_duration"]}K线 | 时间: {flagpole_start.strftime("%Y-%m-%d %H:%M")}',
            fontsize=11, fontweight='bold'
        )
        
        plt.tight_layout()
        
        # 保存图表
        try:
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"保存图表失败: {e}")
            plt.close()
            return None
        
        # 返回图表信息
        chart_info = {
            'pattern_index': pattern_idx,
            'pattern_type': pattern_type,
            'direction': direction,
            'quality': quality,
            'confidence': confidence,
            'flagpole_height': height,
            'pattern_duration': pattern['pattern_duration'],
            'flagpole_start': flagpole_start.strftime('%Y-%m-%d %H:%M'),
            'flagpole_end': flagpole_end.strftime('%Y-%m-%d %H:%M'),
            'chart_filename': filename,
            'chart_path': str(chart_path),
            'file_size_kb': chart_path.stat().st_size // 1024
        }
        
        return chart_info
    
    def _plot_candlestick_chart(self, ax, display_data: pd.DataFrame, flagpole_data: pd.DataFrame,
                               pattern_data: pd.DataFrame, pattern: pd.Series):
        """绘制K线图"""
        
        # 绘制K线
        for timestamp, row in display_data.iterrows():
            open_price = row['open']
            high_price = row['high']
            low_price = row['low'] 
            close_price = row['close']
            
            # 确定颜色
            color = 'red' if close_price >= open_price else 'green'
            
            # 绘制影线
            ax.plot([timestamp, timestamp], [low_price, high_price], 
                   color=color, linewidth=0.8, alpha=0.8)
            
            # 绘制实体
            body_height = abs(close_price - open_price)
            if body_height > 0:
                body_bottom = min(open_price, close_price)
                ax.bar(timestamp, body_height, bottom=body_bottom,
                      width=timedelta(minutes=8), color=color, alpha=0.8)
        
        # 标记形态区域
        if len(flagpole_data) > 0:
            ax.axvspan(flagpole_data.index[0], flagpole_data.index[-1], 
                      alpha=0.2, color='red' if pattern['flagpole_direction'] == 'down' else 'green',
                      label=f'旗杆区域 ({pattern["flagpole_direction"].upper()})')
        
        if len(pattern_data) > 0:
            ax.axvspan(pattern_data.index[0], pattern_data.index[-1],
                      alpha=0.15, color='blue', label='旗面区域')
        
        # 绘制旗杆趋势线
        if len(flagpole_data) > 1:
            start_price = flagpole_data['close'].iloc[0]
            end_price = flagpole_data['close'].iloc[-1]
            ax.plot([flagpole_data.index[0], flagpole_data.index[-1]],
                   [start_price, end_price],
                   color='red' if pattern['flagpole_direction'] == 'down' else 'green',
                   linewidth=2.5, label='旗杆趋势线', alpha=0.9)
        
        # 绘制形态边界
        if pattern['pattern_type'] == 'pennant':
            self._draw_pennant_boundaries(ax, pattern_data)
        else:
            self._draw_flag_boundaries(ax, pattern_data)
        
        # 添加移动平均线
        if len(display_data) >= 20:
            ma20 = display_data['close'].rolling(window=20).mean()
            ax.plot(display_data.index, ma20, color='orange', alpha=0.7, 
                   linewidth=1.5, label='MA20')
        
        # 标记重要价位
        current_price = display_data['close'].iloc[-1]
        ax.axhline(y=current_price, color='gray', linestyle=':', alpha=0.5)
        
        # 设置坐标轴
        ax.set_ylabel('价格', fontsize=11)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 格式化时间轴
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\\n%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        
    def _draw_pennant_boundaries(self, ax, pattern_data: pd.DataFrame):
        """绘制三角旗边界"""
        if len(pattern_data) < 3:
            return
            
        times = pattern_data.index
        highs = pattern_data['high']
        lows = pattern_data['low']
        
        # 连接首尾高低点形成三角形
        if len(times) >= 2:
            ax.plot([times[0], times[-1]], [highs.iloc[0], highs.iloc[-1]], 
                   'b--', linewidth=2, alpha=0.8, label='三角旗上边界')
            ax.plot([times[0], times[-1]], [lows.iloc[0], lows.iloc[-1]], 
                   'b--', linewidth=2, alpha=0.8, label='三角旗下边界')
    
    def _draw_flag_boundaries(self, ax, pattern_data: pd.DataFrame):
        """绘制矩形旗边界"""
        if len(pattern_data) < 2:
            return
            
        avg_high = pattern_data['high'].mean()
        avg_low = pattern_data['low'].mean()
        
        times = pattern_data.index
        ax.plot([times[0], times[-1]], [avg_high, avg_high], 
               'b--', linewidth=2, alpha=0.8, label='矩形旗上边界')
        ax.plot([times[0], times[-1]], [avg_low, avg_low], 
               'b--', linewidth=2, alpha=0.8, label='矩形旗下边界')
    
    def _plot_volume_chart(self, ax, display_data: pd.DataFrame, 
                          flagpole_data: pd.DataFrame, pattern_data: pd.DataFrame):
        """绘制成交量图"""
        
        colors = ['red' if row['close'] >= row['open'] else 'green' 
                 for _, row in display_data.iterrows()]
        
        ax.bar(display_data.index, display_data['volume'], 
               color=colors, alpha=0.6, width=timedelta(minutes=8))
        
        # 标记区域
        if len(flagpole_data) > 0:
            ax.axvspan(flagpole_data.index[0], flagpole_data.index[-1], 
                      alpha=0.2, color='red')
        if len(pattern_data) > 0:
            ax.axvspan(pattern_data.index[0], pattern_data.index[-1], 
                      alpha=0.15, color='blue')
        
        # 成交量均线
        if len(display_data) >= 10:
            vol_ma = display_data['volume'].rolling(window=10).mean()
            ax.plot(display_data.index, vol_ma, color='purple', 
                   alpha=0.8, linewidth=1.5, label='成交量MA10')
        
        ax.set_ylabel('成交量', fontsize=11)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def _generate_mapping_file(self, output_dir: Path) -> str:
        """生成形态图表映射文件"""
        
        mapping_data = {
            'generated_at': datetime.now().isoformat(),
            'description': '形态K线图表示例生成结果',
            'total_charts': len(self.generated_charts),
            'charts_by_quality': {
                'high': len([c for c in self.generated_charts if c['quality'] == 'high']),
                'medium': len([c for c in self.generated_charts if c['quality'] == 'medium'])
            },
            'charts': self.generated_charts
        }
        
        # JSON映射文件
        mapping_file = output_dir / 'pattern_chart_mapping.json'
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, ensure_ascii=False, indent=2)
        
        # CSV映射文件
        csv_data = [{
            'pattern_index': c['pattern_index'],
            'pattern_type': c['pattern_type'],
            'direction': c['direction'], 
            'quality': c['quality'],
            'confidence': c['confidence'],
            'flagpole_height': c['flagpole_height'],
            'flagpole_start': c['flagpole_start'],
            'chart_filename': c['chart_filename']
        } for c in self.generated_charts]
        
        csv_file = output_dir / 'pattern_chart_mapping.csv'
        pd.DataFrame(csv_data).to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        return str(mapping_file)
    
    def _generate_summary_document(self, output_dir: Path, patterns_df: pd.DataFrame) -> str:
        """生成形态图表汇总文档"""
        
        summary = {
            'title': 'RBL8期货旗形形态K线图表生成报告',
            'generated_at': datetime.now().isoformat(),
            'overview': {
                'total_patterns_identified': len(patterns_df),
                'sample_charts_generated': len(self.generated_charts),
                'coverage': f"{len(self.generated_charts)}/{len(patterns_df)} ({len(self.generated_charts)/len(patterns_df)*100:.1f}%)"
            },
            'quality_distribution': {
                'available_patterns': patterns_df['pattern_quality'].value_counts().to_dict(),
                'generated_charts': {q: len([c for c in self.generated_charts if c['quality'] == q]) 
                                   for q in ['high', 'medium']}
            },
            'file_structure': {
                'high_quality/': '所有高质量形态图表 (9个)',
                'medium_quality/': '部分中等质量形态图表 (20个示例)',
                'pattern_chart_mapping.json': '完整的形态图表映射信息',
                'pattern_chart_mapping.csv': '简化的映射表格'
            },
            'naming_convention': {
                'format': '{PATTERN_TYPE}_{DIRECTION}_{TIMESTAMP}_conf{CONFIDENCE}.png',
                'examples': [c['chart_filename'] for c in self.generated_charts[:3]]
            },
            'technical_details': {
                'chart_resolution': '150 DPI',
                'chart_size': '14x9 inches',
                'time_window': '形态前后各25个K线',
                'indicators': ['MA20移动平均线', '成交量MA10', '形态边界线', '旗杆趋势线']
            }
        }
        
        summary_file = output_dir / 'generation_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        return str(summary_file)

def main():
    """主函数"""
    print("=== 示例形态K线图表生成系统 ===")
    
    # 导入数据连接器
    from src.data.connectors.csv_connector import CSVDataConnector
    
    # 初始化数据连接器
    data_connector = CSVDataConnector('data/csv')
    data_connector.connect()
    
    # 创建示例图表生成器
    generator = SamplePatternChartGenerator(data_connector)
    
    # 生成示例形态图表
    result = generator.generate_sample_charts()
    
    # 关闭连接
    data_connector.close()
    
    print("\\n" + "="*60)
    print("🎉 示例形态K线图表生成完成！")
    print(f"📁 输出目录: {result['output_directory']}")
    print(f"📊 生成图表: {result['charts_generated']} 个")
    print(f"🏆 高质量: {result['high_quality_charts']} 个")
    print(f"📈 中等质量: {result['medium_quality_charts']} 个") 
    print("="*60)

if __name__ == "__main__":
    main()