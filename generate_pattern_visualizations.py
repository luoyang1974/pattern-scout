#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
形态可视化与图表生成系统
为高质量旗形模式生成详细的可视化图表
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

# 设置控制台编码（Windows）
if sys.platform.startswith('win'):
    os.system('chcp 65001 > nul')
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# 添加src到Python路径
sys.path.append(str(__file__).replace('generate_pattern_visualizations.py', ''))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class PatternVisualizer:
    """形态可视化生成器"""
    
    def __init__(self, data_connector):
        self.data_connector = data_connector
        self.chart_count = 0
        
    def generate_top_patterns_charts(self, top_n: int = 10) -> List[str]:
        """为顶级旗形模式生成详细图表"""
        print(f"开始生成前 {top_n} 个高质量旗形的可视化图表...")
        
        # 读取高质量形态
        try:
            patterns_df = pd.read_csv('output/reports/high_quality_patterns.csv')
        except FileNotFoundError:
            print("未找到高质量形态文件")
            return []
            
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
        
        # 创建图表目录
        chart_dir = 'output/charts/patterns'
        os.makedirs(chart_dir, exist_ok=True)
        
        chart_paths = []
        patterns_to_visualize = patterns_df.head(top_n)
        
        for idx, pattern in patterns_to_visualize.iterrows():
            try:
                chart_path = self._generate_single_pattern_chart(pattern, price_data, chart_dir, idx)
                if chart_path:
                    chart_paths.append(chart_path)
                    self.chart_count += 1
                    print(f"已生成图表 {self.chart_count}/{top_n}: {chart_path}")
            except Exception as e:
                print(f"生成形态 {idx} 的图表时出错: {e}")
                continue
                
        print(f"✅ 完成 {len(chart_paths)} 个形态图表的生成")
        return chart_paths
    
    def _generate_single_pattern_chart(self, pattern: pd.Series, price_data: pd.DataFrame, 
                                     chart_dir: str, pattern_idx: int) -> Optional[str]:
        """生成单个形态的详细图表"""
        
        # 获取形态时间范围
        flagpole_start = pd.to_datetime(pattern['flagpole_start_time'])
        flagpole_end = pd.to_datetime(pattern['flagpole_end_time'])
        pattern_end = flagpole_end + timedelta(minutes=15 * pattern['pattern_duration'])
        
        # 扩展显示范围（前后各加20个K线）
        display_start = flagpole_start - timedelta(minutes=15 * 20)
        display_end = pattern_end + timedelta(minutes=15 * 20)
        
        # 获取显示数据
        display_data = price_data[display_start:display_end]
        if len(display_data) < 30:
            return None
            
        # 获取形态相关数据
        flagpole_data = price_data[flagpole_start:flagpole_end]
        pattern_data = price_data[flagpole_end:pattern_end]
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # 主价格图
        self._plot_price_chart(ax1, display_data, flagpole_data, pattern_data, pattern)
        
        # 成交量图
        self._plot_volume_chart(ax2, display_data, flagpole_data, pattern_data)
        
        # 设置标题和格式
        pattern_type = pattern['pattern_type'].upper()
        direction = pattern['flagpole_direction'].upper()
        confidence = pattern['confidence_score']
        height = pattern['flagpole_height_percent']
        
        fig.suptitle(
            f'RBL8 {pattern_type} 形态分析 - {direction}方向\n'
            f'置信度: {confidence:.3f} | 旗杆高度: {height:.2f}% | '
            f'时间: {flagpole_start.strftime("%Y-%m-%d %H:%M")}',
            fontsize=14, fontweight='bold'
        )
        
        plt.tight_layout()
        
        # 保存图表
        chart_filename = f'pattern_{pattern_idx+1:02d}_{pattern_type}_{direction}_{flagpole_start.strftime("%Y%m%d_%H%M")}.png'
        chart_path = os.path.join(chart_dir, chart_filename)
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    def _plot_price_chart(self, ax, display_data: pd.DataFrame, flagpole_data: pd.DataFrame, 
                         pattern_data: pd.DataFrame, pattern: pd.Series):
        """绘制价格图表"""
        
        # 基础K线图
        self._plot_candlesticks(ax, display_data)
        
        # 标记旗杆区域
        if len(flagpole_data) > 0:
            ax.axvspan(flagpole_data.index[0], flagpole_data.index[-1], 
                      alpha=0.2, color='red' if pattern['flagpole_direction'] == 'down' else 'green',
                      label='旗杆区域')
        
        # 标记旗面区域
        if len(pattern_data) > 0:
            ax.axvspan(pattern_data.index[0], pattern_data.index[-1], 
                      alpha=0.2, color='blue', label='旗面区域')
        
        # 绘制旗杆线条
        if len(flagpole_data) > 1:
            flagpole_start_price = flagpole_data['close'].iloc[0]
            flagpole_end_price = flagpole_data['close'].iloc[-1]
            ax.plot([flagpole_data.index[0], flagpole_data.index[-1]], 
                   [flagpole_start_price, flagpole_end_price],
                   color='red' if pattern['flagpole_direction'] == 'down' else 'green', 
                   linewidth=3, label='旗杆趋势', alpha=0.8)
        
        # 绘制形态边界线
        if pattern['pattern_type'] == 'pennant':
            self._draw_pennant_lines(ax, pattern_data)
        else:  # flag
            self._draw_flag_lines(ax, pattern_data)
            
        # 添加移动平均线
        if len(display_data) > 20:
            ma20 = display_data['close'].rolling(window=20).mean()
            ax.plot(display_data.index, ma20, color='orange', alpha=0.7, label='MA20')
        
        # 设置格式
        ax.set_ylabel('价格', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 格式化x轴
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_candlesticks(self, ax, data: pd.DataFrame):
        """绘制K线图"""
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            open_price = row['open']
            high_price = row['high'] 
            low_price = row['low']
            close_price = row['close']
            
            # 颜色选择
            color = 'red' if close_price >= open_price else 'green'
            
            # 绘制影线
            ax.plot([timestamp, timestamp], [low_price, high_price], 
                   color=color, linewidth=1, alpha=0.8)
            
            # 绘制实体
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            
            ax.bar(timestamp, body_height, bottom=body_bottom, 
                  width=timedelta(minutes=12), color=color, alpha=0.8)
    
    def _draw_pennant_lines(self, ax, pattern_data: pd.DataFrame):
        """绘制三角旗形的收敛线"""
        if len(pattern_data) < 4:
            return
            
        # 计算上下边界点
        highs = pattern_data['high']
        lows = pattern_data['low']
        times = pattern_data.index
        
        # 上边界线（连接高点）
        high_indices = [0, len(highs)//2, -1]
        upper_times = [times[i] for i in high_indices if i < len(times)]
        upper_prices = [highs.iloc[i] for i in high_indices if i < len(highs)]
        
        if len(upper_times) >= 2:
            ax.plot(upper_times, upper_prices, 'b--', linewidth=2, 
                   alpha=0.7, label='三角旗上边界')
        
        # 下边界线（连接低点）
        low_indices = [0, len(lows)//2, -1]
        lower_times = [times[i] for i in low_indices if i < len(times)]
        lower_prices = [lows.iloc[i] for i in low_indices if i < len(lows)]
        
        if len(lower_times) >= 2:
            ax.plot(lower_times, lower_prices, 'b--', linewidth=2, 
                   alpha=0.7, label='三角旗下边界')
    
    def _draw_flag_lines(self, ax, pattern_data: pd.DataFrame):
        """绘制矩形旗的平行线"""
        if len(pattern_data) < 4:
            return
            
        # 计算平行通道
        highs = pattern_data['high']
        lows = pattern_data['low']
        
        upper_level = highs.mean()
        lower_level = lows.mean()
        
        ax.axhline(y=upper_level, color='blue', linestyle='--', 
                  alpha=0.7, label='矩形旗上边界')
        ax.axhline(y=lower_level, color='blue', linestyle='--', 
                  alpha=0.7, label='矩形旗下边界')
    
    def _plot_volume_chart(self, ax, display_data: pd.DataFrame, flagpole_data: pd.DataFrame, 
                          pattern_data: pd.DataFrame):
        """绘制成交量图表"""
        
        # 基础成交量柱状图
        colors = ['red' if row['close'] >= row['open'] else 'green' 
                 for _, row in display_data.iterrows()]
        
        ax.bar(display_data.index, display_data['volume'], 
               color=colors, alpha=0.6, width=timedelta(minutes=12))
        
        # 标记旗杆和旗面区域
        if len(flagpole_data) > 0:
            ax.axvspan(flagpole_data.index[0], flagpole_data.index[-1], 
                      alpha=0.2, color='red')
        if len(pattern_data) > 0:
            ax.axvspan(pattern_data.index[0], pattern_data.index[-1], 
                      alpha=0.2, color='blue')
        
        # 成交量移动平均
        if len(display_data) > 20:
            vol_ma = display_data['volume'].rolling(window=20).mean()
            ax.plot(display_data.index, vol_ma, color='purple', 
                   alpha=0.7, label='成交量MA20')
        
        ax.set_ylabel('成交量', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 格式化x轴
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

def generate_pattern_summary_dashboard():
    """生成形态汇总仪表板"""
    print("生成形态汇总仪表板...")
    
    # 读取所有分析结果
    try:
        patterns_df = pd.read_csv('output/reports/pattern_detailed_list.csv')
        outcomes_df = pd.read_csv('output/reports/outcomes/pattern_outcome_analysis.csv')
    except FileNotFoundError as e:
        print(f"缺少必要文件: {e}")
        return None
        
    # 创建综合仪表板
    fig = plt.figure(figsize=(20, 16))
    
    # 创建网格布局
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # 1. 形态类型分布
    ax1 = fig.add_subplot(gs[0, 0])
    type_counts = patterns_df['pattern_type'].value_counts()
    ax1.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.set_title('形态类型分布', fontweight='bold')
    
    # 2. 质量等级分布
    ax2 = fig.add_subplot(gs[0, 1])
    quality_counts = patterns_df['pattern_quality'].value_counts()
    ax2.pie(quality_counts.values, labels=quality_counts.index, autopct='%1.1f%%', startangle=90)
    ax2.set_title('质量等级分布', fontweight='bold')
    
    # 3. 方向分布
    ax3 = fig.add_subplot(gs[0, 2])
    direction_counts = patterns_df['flagpole_direction'].value_counts()
    ax3.pie(direction_counts.values, labels=direction_counts.index, autopct='%1.1f%%', startangle=90)
    ax3.set_title('旗杆方向分布', fontweight='bold')
    
    # 4. 结局分析
    ax4 = fig.add_subplot(gs[0, 3])
    outcome_counts = outcomes_df['outcome_type'].value_counts()
    ax4.pie(outcome_counts.values, labels=outcome_counts.index, autopct='%1.1f%%', startangle=90)
    ax4.set_title('形态结局分布', fontweight='bold')
    
    # 5. 置信度分布
    ax5 = fig.add_subplot(gs[1, :2])
    ax5.hist(patterns_df['confidence_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax5.set_xlabel('置信度')
    ax5.set_ylabel('数量')
    ax5.set_title('置信度分布', fontweight='bold')
    ax5.axvline(patterns_df['confidence_score'].mean(), color='red', linestyle='--', 
                label=f'均值: {patterns_df["confidence_score"].mean():.3f}')
    ax5.legend()
    
    # 6. 旗杆高度分布
    ax6 = fig.add_subplot(gs[1, 2:])
    ax6.hist(patterns_df['flagpole_height_percent'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    ax6.set_xlabel('旗杆高度 (%)')
    ax6.set_ylabel('数量')
    ax6.set_title('旗杆高度分布', fontweight='bold')
    ax6.axvline(patterns_df['flagpole_height_percent'].mean(), color='red', linestyle='--',
                label=f'均值: {patterns_df["flagpole_height_percent"].mean():.2f}%')
    ax6.legend()
    
    # 7. 时间分布（月度）
    ax7 = fig.add_subplot(gs[2, :2])
    patterns_df['month'] = pd.to_datetime(patterns_df['flagpole_start_time']).dt.month
    monthly_counts = patterns_df['month'].value_counts().sort_index()
    ax7.bar(monthly_counts.index, monthly_counts.values, color='coral', alpha=0.7)
    ax7.set_xlabel('月份')
    ax7.set_ylabel('形态数量')
    ax7.set_title('形态月度分布', fontweight='bold')
    ax7.set_xticks(range(1, 13))
    
    # 8. 成功率统计
    ax8 = fig.add_subplot(gs[2, 2:])
    success_metrics = {
        '突破成功率': (outcomes_df['breakthrough_success'].sum() / len(outcomes_df) * 100),
        '成交量确认率': (outcomes_df['volume_confirm'].sum() / len(outcomes_df) * 100),
        '高质量占比': (len(patterns_df[patterns_df['pattern_quality'] == 'high']) / len(patterns_df) * 100)
    }
    
    bars = ax8.bar(success_metrics.keys(), success_metrics.values(), 
                   color=['green', 'blue', 'gold'], alpha=0.7)
    ax8.set_ylabel('百分比 (%)')
    ax8.set_title('关键成功率指标', fontweight='bold')
    
    # 在柱状图上添加数值
    for bar, value in zip(bars, success_metrics.values()):
        ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}%', ha='center', va='bottom')
    
    # 9. 统计摘要文本
    ax9 = fig.add_subplot(gs[3, :])
    ax9.axis('off')
    
    summary_text = f"""
    📊 RBL8 期货旗形形态识别分析报告 📊
    
    🔍 识别统计：
    • 总形态数量: {len(patterns_df)} 个
    • 高质量形态: {len(patterns_df[patterns_df['pattern_quality'] == 'high'])} 个 ({len(patterns_df[patterns_df['pattern_quality'] == 'high'])/len(patterns_df)*100:.1f}%)
    • 平均置信度: {patterns_df['confidence_score'].mean():.3f}
    • 平均旗杆高度: {patterns_df['flagpole_height_percent'].mean():.2f}%
    
    📈 结局分析：
    • 分析形态数量: {len(outcomes_df)} 个
    • 突破成功率: {outcomes_df['breakthrough_success'].sum()/len(outcomes_df)*100:.1f}%
    • 平均价格变动: {outcomes_df['price_move_percent'].mean():.2f}%
    
    🎯 形态分布：
    • 三角旗形占比: {type_counts.get('pennant', 0)/len(patterns_df)*100:.1f}%
    • 矩形旗形占比: {type_counts.get('flag', 0)/len(patterns_df)*100:.1f}%
    • 上涨方向: {direction_counts.get('up', 0)} 个，下跌方向: {direction_counts.get('down', 0)} 个
    """
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # 设置总标题
    fig.suptitle('PatternScout - RBL8旗形形态分析综合仪表板', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 保存仪表板
    dashboard_path = 'output/charts/pattern_analysis_dashboard.png'
    plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"综合仪表板已保存: {dashboard_path}")
    return dashboard_path

def main():
    """主函数"""
    print("=== 形态可视化与图表生成系统 ===")
    
    # 导入数据连接器
    from src.data.connectors.csv_connector import CSVDataConnector
    
    # 初始化数据连接器
    data_connector = CSVDataConnector('data/csv')
    data_connector.connect()
    
    # 创建可视化生成器
    visualizer = PatternVisualizer(data_connector)
    
    # 生成顶级形态图表
    top_charts = visualizer.generate_top_patterns_charts(top_n=10)
    
    # 生成综合仪表板
    dashboard_path = generate_pattern_summary_dashboard()
    
    print(f"\n✅ 形态可视化生成完成！")
    print(f"   • 生成个别形态图表: {len(top_charts)} 个")
    if dashboard_path:
        print(f"   • 综合仪表板: {dashboard_path}")
    
    data_connector.close()

if __name__ == "__main__":
    main()