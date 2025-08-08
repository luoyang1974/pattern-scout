#!/usr/bin/env python3
"""
完整时间范围旗杆可视化脚本
将所有488个旗杆绘制在K线图上
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from typing import List, Dict
import json
from loguru import logger

# 导入项目模块
from src.data.connectors.csv_connector import CSVDataConnector


class FullRangeVisualization:
    """完整时间范围旗杆可视化"""
    
    def __init__(self):
        self.csv_connector = CSVDataConnector("data/csv")
        if not self.csv_connector.connect():
            raise ConnectionError("无法连接到CSV数据源")
    
    def create_full_range_charts(self, symbol: str = "RBL8"):
        """创建完整时间范围的旗杆可视化图表"""
        logger.info("开始创建完整时间范围旗杆可视化图表")
        
        # 1. 加载完整数据
        logger.info("加载完整数据集...")
        full_data = self.csv_connector.get_data(symbol)
        
        if full_data.empty:
            raise ValueError(f"无法加载数据: {symbol}")
        
        logger.info(f"数据加载完成: {len(full_data)}条记录")
        
        # 2. 加载旗杆检测结果
        flagpoles_file = "output/flagpole_tests/full_range/RBL8_full_range_flagpoles.json"
        
        if not os.path.exists(flagpoles_file):
            raise FileNotFoundError("请先运行完整时间范围测试生成旗杆数据")
        
        with open(flagpoles_file, 'r', encoding='utf-8') as f:
            flagpoles_data = json.load(f)
        
        logger.info(f"加载旗杆数据: {len(flagpoles_data)}个旗杆")
        
        # 3. 创建输出目录
        output_dir = "output/flagpole_tests/full_range_charts"
        os.makedirs(output_dir, exist_ok=True)
        
        # 4. 生成多种可视化图表
        self._create_overview_chart(full_data, flagpoles_data, symbol, output_dir)
        self._create_yearly_charts(full_data, flagpoles_data, symbol, output_dir)
        self._create_density_heatmap(full_data, flagpoles_data, symbol, output_dir)
        self._create_statistics_chart(flagpoles_data, symbol, output_dir)
        
        logger.info("完整时间范围可视化图表创建完成")
        logger.info(f"图表保存在: {output_dir}")
    
    def _create_overview_chart(self, data: pd.DataFrame, flagpoles: List[Dict], 
                              symbol: str, output_dir: str):
        """创建完整时间范围总览图表"""
        logger.info("创建完整时间范围总览图表...")
        
        # 设置中文字体和图表样式
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 由于数据量太大，我们抽样显示（每50条取1条）
        sample_interval = 50
        sampled_data = data.iloc[::sample_interval].copy()
        sampled_data = sampled_data.reset_index(drop=True)
        
        # 创建图表
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 15))
        
        # 1. 价格走势图 + 旗杆标记
        self._plot_price_overview(ax1, sampled_data, flagpoles, sample_interval, 
                                 f'{symbol} - 完整时间范围价格走势与旗杆分布 (共{len(flagpoles)}个旗杆)')
        
        # 2. 成交量图 + 旗杆量能爆发
        self._plot_volume_overview(ax2, sampled_data, flagpoles, sample_interval, 
                                  '成交量与旗杆量能爆发分析')
        
        # 3. 旗杆分布密度图
        self._plot_flagpole_density(ax3, data, flagpoles, '旗杆时间分布密度')
        
        plt.tight_layout()
        
        # 保存图表
        overview_file = os.path.join(output_dir, f"{symbol}_complete_overview.png")
        plt.savefig(overview_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"完整总览图表已保存到: {overview_file}")
    
    def _plot_price_overview(self, ax, sampled_data: pd.DataFrame, flagpoles: List[Dict], 
                           sample_interval: int, title: str):
        """绘制价格走势总览"""
        # 绘制价格线
        ax.plot(range(len(sampled_data)), sampled_data['close'], 
                color='black', linewidth=0.8, alpha=0.8, label='收盘价')
        
        # 标记旗杆位置
        flagpole_colors = ['red' if f['direction'] == 'up' else 'blue' for f in flagpoles]
        flagpole_sizes = [max(20, min(100, f['height_percent'] * 5000)) for f in flagpoles]
        
        # 找到旗杆在抽样数据中的对应位置
        up_positions = []
        down_positions = []
        up_prices = []
        down_prices = []
        
        for flagpole in flagpoles:
            # 将旗杆时间转换为数据索引
            flagpole_time = pd.to_datetime(flagpole['start_time'])
            
            # 在抽样数据中找到最接近的时间点
            time_diffs = abs(sampled_data['timestamp'] - flagpole_time)
            closest_idx = time_diffs.idxmin()
            
            if flagpole['direction'] == 'up':
                up_positions.append(closest_idx)
                up_prices.append(flagpole['start_price'])
            else:
                down_positions.append(closest_idx)
                down_prices.append(flagpole['start_price'])
        
        # 绘制旗杆标记
        if up_positions:
            ax.scatter(up_positions, up_prices, c='red', marker='^', 
                      s=30, alpha=0.7, label=f'上升旗杆 ({len(up_positions)}个)')
        
        if down_positions:
            ax.scatter(down_positions, down_prices, c='blue', marker='v', 
                      s=30, alpha=0.7, label=f'下降旗杆 ({len(down_positions)}个)')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('价格', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 设置x轴时间标签
        if len(sampled_data) > 0:
            time_ticks = np.linspace(0, len(sampled_data)-1, 10, dtype=int)
            ax.set_xticks(time_ticks)
            ax.set_xticklabels([sampled_data.iloc[i]['timestamp'].strftime('%Y-%m') 
                               for i in time_ticks], rotation=45)
    
    def _plot_volume_overview(self, ax, sampled_data: pd.DataFrame, flagpoles: List[Dict], 
                            sample_interval: int, title: str):
        """绘制成交量总览"""
        if 'volume' not in sampled_data.columns:
            ax.text(0.5, 0.5, '无成交量数据', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=14)
            return
        
        # 绘制成交量柱状图
        ax.bar(range(len(sampled_data)), sampled_data['volume'], 
               alpha=0.6, color='gray', width=1.0, label='成交量')
        
        # 添加成交量移动平均线
        volume_ma = sampled_data['volume'].rolling(window=50).mean()
        ax.plot(range(len(volume_ma)), volume_ma, color='orange', 
                linewidth=2, label='成交量MA50', alpha=0.8)
        
        # 标记高量能爆发的旗杆
        high_volume_flagpoles = [f for f in flagpoles if f['volume_burst'] > 2.0]
        
        if high_volume_flagpoles:
            hv_positions = []
            hv_volumes = []
            
            for flagpole in high_volume_flagpoles:
                flagpole_time = pd.to_datetime(flagpole['start_time'])
                time_diffs = abs(sampled_data['timestamp'] - flagpole_time)
                closest_idx = time_diffs.idxmin()
                
                hv_positions.append(closest_idx)
                hv_volumes.append(sampled_data.iloc[closest_idx]['volume'])
            
            ax.scatter(hv_positions, hv_volumes, c='red', marker='*', 
                      s=100, alpha=0.8, label=f'高量能旗杆 (>{2.0}x, {len(high_volume_flagpoles)}个)')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('成交量', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 设置x轴时间标签
        time_ticks = np.linspace(0, len(sampled_data)-1, 10, dtype=int)
        ax.set_xticks(time_ticks)
        ax.set_xticklabels([sampled_data.iloc[i]['timestamp'].strftime('%Y-%m') 
                           for i in time_ticks], rotation=45)
    
    def _plot_flagpole_density(self, ax, full_data: pd.DataFrame, flagpoles: List[Dict], title: str):
        """绘制旗杆分布密度"""
        # 创建时间序列
        start_date = full_data['timestamp'].min()
        end_date = full_data['timestamp'].max()
        
        # 按月统计旗杆数量
        flagpole_times = [pd.to_datetime(f['start_time']) for f in flagpoles]
        flagpole_df = pd.DataFrame({'timestamp': flagpole_times, 'count': 1})
        
        # 按月汇总
        flagpole_df['year_month'] = flagpole_df['timestamp'].dt.to_period('M')
        monthly_counts = flagpole_df.groupby('year_month')['count'].sum()
        
        # 创建完整的月度时间序列
        full_months = pd.period_range(start_date, end_date, freq='M')
        monthly_data = pd.Series(0, index=full_months)
        
        # 填入旗杆数量
        for period, count in monthly_counts.items():
            if period in monthly_data.index:
                monthly_data[period] = count
        
        # 绘制密度图
        x_pos = range(len(monthly_data))
        colors = ['red' if count > monthly_data.median() else 'blue' for count in monthly_data]
        
        bars = ax.bar(x_pos, monthly_data.values, color=colors, alpha=0.7, width=0.8)
        
        # 添加平均线
        avg_count = monthly_data.mean()
        ax.axhline(y=avg_count, color='green', linestyle='--', linewidth=2, 
                   label=f'月均值: {avg_count:.1f}个')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('旗杆数量/月', fontsize=12)
        ax.set_xlabel('时间', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 设置x轴标签
        if len(monthly_data) > 0:
            step = max(1, len(monthly_data) // 12)  # 显示约12个标签
            tick_positions = range(0, len(monthly_data), step)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([str(monthly_data.index[i]) for i in tick_positions], 
                              rotation=45)
    
    def _create_yearly_charts(self, data: pd.DataFrame, flagpoles: List[Dict], 
                            symbol: str, output_dir: str):
        """创建年度分解图表"""
        logger.info("创建年度分解图表...")
        
        # 按年份分组旗杆
        yearly_flagpoles = {}
        for flagpole in flagpoles:
            year = pd.to_datetime(flagpole['start_time']).year
            if year not in yearly_flagpoles:
                yearly_flagpoles[year] = []
            yearly_flagpoles[year].append(flagpole)
        
        # 为每年创建图表
        for year, year_flagpoles in yearly_flagpoles.items():
            if len(year_flagpoles) < 5:  # 跳过旗杆数量太少的年份
                continue
                
            logger.info(f"创建{year}年图表 ({len(year_flagpoles)}个旗杆)")
            
            # 筛选该年的数据
            year_start = pd.Timestamp(f'{year}-01-01')
            year_end = pd.Timestamp(f'{year}-12-31 23:59:59')
            
            year_data = data[(data['timestamp'] >= year_start) & 
                           (data['timestamp'] <= year_end)].copy()
            
            if year_data.empty:
                continue
                
            year_data = year_data.reset_index(drop=True)
            
            # 创建年度图表
            self._create_single_year_chart(year_data, year_flagpoles, year, symbol, output_dir)
    
    def _create_single_year_chart(self, year_data: pd.DataFrame, year_flagpoles: List[Dict], 
                                year: int, symbol: str, output_dir: str):
        """创建单年图表"""
        # 抽样数据（每10条取1条）
        sample_interval = max(1, len(year_data) // 2000)  # 确保图表不超过2000个点
        sampled_data = year_data.iloc[::sample_interval].copy()
        sampled_data = sampled_data.reset_index(drop=True)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        # K线图
        self._plot_candlestick_yearly(ax1, sampled_data, year_flagpoles, sample_interval, 
                                    f'{symbol} - {year}年K线图与旗杆分布 ({len(year_flagpoles)}个旗杆)')
        
        # 成交量图
        self._plot_volume_yearly(ax2, sampled_data, year_flagpoles, sample_interval, 
                               f'{year}年成交量与旗杆量能分析')
        
        plt.tight_layout()
        
        # 保存图表
        yearly_file = os.path.join(output_dir, f"{symbol}_{year}_flagpoles.png")
        plt.savefig(yearly_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"{year}年图表已保存到: {yearly_file}")
    
    def _plot_candlestick_yearly(self, ax, data: pd.DataFrame, flagpoles: List[Dict], 
                               sample_interval: int, title: str):
        """绘制年度K线图"""
        # 绘制简化K线图
        for i, (_, row) in enumerate(data.iterrows()):
            color = 'red' if row['close'] >= row['open'] else 'green'
            
            # 绘制影线
            ax.plot([i, i], [row['low'], row['high']], color='black', linewidth=0.5)
            
            # 绘制实体
            body_height = abs(row['close'] - row['open'])
            bottom = min(row['open'], row['close'])
            
            if body_height > 0:
                rect = plt.Rectangle((i-0.4, bottom), 0.8, body_height, 
                                   facecolor=color, alpha=0.7, 
                                   edgecolor='black', linewidth=0.3)
                ax.add_patch(rect)
            else:
                # 十字星
                ax.plot([i-0.4, i+0.4], [row['close'], row['close']], 
                       color='black', linewidth=1)
        
        # 标记旗杆
        self._mark_flagpoles_yearly(ax, data, flagpoles)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('价格', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 设置时间轴
        if len(data) > 10:
            time_ticks = np.linspace(0, len(data)-1, 8, dtype=int)
            ax.set_xticks(time_ticks)
            ax.set_xticklabels([data.iloc[i]['timestamp'].strftime('%m-%d') 
                               for i in time_ticks], rotation=45)
    
    def _mark_flagpoles_yearly(self, ax, data: pd.DataFrame, flagpoles: List[Dict]):
        """在年度图表上标记旗杆"""
        colors = ['blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for i, flagpole in enumerate(flagpoles):
            color = colors[i % len(colors)]
            
            # 找到旗杆时间对应的索引
            start_time = pd.to_datetime(flagpole['start_time'])
            end_time = pd.to_datetime(flagpole['end_time'])
            
            start_idx = None
            end_idx = None
            
            for j, timestamp in enumerate(data['timestamp']):
                if abs((timestamp - start_time).total_seconds()) < 900:  # 15分钟内
                    start_idx = j
                if abs((timestamp - end_time).total_seconds()) < 900:
                    end_idx = j
            
            if start_idx is not None and end_idx is not None:
                # 标记旗杆区域
                ax.axvspan(start_idx - 0.5, end_idx + 0.5, alpha=0.2, color=color)
                
                # 标记起止点
                start_price = flagpole['start_price']
                end_price = flagpole['end_price']
                
                ax.plot(start_idx, start_price, 'o', color=color, markersize=6, 
                       markeredgecolor='black', markeredgewidth=1)
                ax.plot(end_idx, end_price, 's', color=color, markersize=6, 
                       markeredgecolor='black', markeredgewidth=1)
                
                # 添加标注（仅对前10个旗杆，避免过于拥挤）
                if i < 10:
                    mid_idx = (start_idx + end_idx) / 2
                    mid_price = (start_price + end_price) / 2
                    ax.annotate(f'{i+1}\\n{flagpole["height_percent"]:.1%}', 
                               xy=(mid_idx, mid_price), xytext=(3, 3), 
                               textcoords='offset points', fontsize=7,
                               bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.6))
    
    def _plot_volume_yearly(self, ax, data: pd.DataFrame, flagpoles: List[Dict], 
                          sample_interval: int, title: str):
        """绘制年度成交量图"""
        if 'volume' not in data.columns:
            ax.text(0.5, 0.5, '无成交量数据', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            return
        
        # 绘制成交量
        volumes = data['volume'].values
        ax.bar(range(len(volumes)), volumes, alpha=0.6, color='gray', width=0.8)
        
        # 标记旗杆的量能爆发
        for flagpole in flagpoles:
            start_time = pd.to_datetime(flagpole['start_time'])
            
            for j, timestamp in enumerate(data['timestamp']):
                if abs((timestamp - start_time).total_seconds()) < 900:
                    volume_burst = flagpole['volume_burst']
                    if volume_burst > 1.5:  # 只标记显著的量能爆发
                        color = 'red' if flagpole['direction'] == 'up' else 'blue'
                        ax.plot(j, volumes[j], '^', color=color, markersize=8,
                               markeredgecolor='black', markeredgewidth=1)
                    break
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('成交量', fontsize=12)
        ax.set_xlabel('时间', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 设置时间轴
        if len(data) > 10:
            time_ticks = np.linspace(0, len(data)-1, 8, dtype=int)
            ax.set_xticks(time_ticks)
            ax.set_xticklabels([data.iloc[i]['timestamp'].strftime('%m-%d') 
                               for i in time_ticks], rotation=45)
    
    def _create_density_heatmap(self, data: pd.DataFrame, flagpoles: List[Dict], 
                               symbol: str, output_dir: str):
        """创建旗杆分布热力图"""
        logger.info("创建旗杆分布热力图...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 月度热力图
        self._create_monthly_heatmap(ax1, flagpoles, '月度旗杆分布热力图')
        
        # 2. 周内分布
        self._create_weekly_distribution(ax2, flagpoles, '周内旗杆分布')
        
        # 3. 小时分布
        self._create_hourly_distribution(ax3, flagpoles, '小时旗杆分布')
        
        # 4. 高度分布
        self._create_height_distribution(ax4, flagpoles, '旗杆高度分布')
        
        plt.suptitle(f'{symbol} - 旗杆分布分析热力图', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        heatmap_file = os.path.join(output_dir, f"{symbol}_flagpole_heatmap.png")
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"热力图已保存到: {heatmap_file}")
    
    def _create_monthly_heatmap(self, ax, flagpoles: List[Dict], title: str):
        """创建月度热力图"""
        # 统计各月份旗杆数量
        monthly_counts = {}
        for flagpole in flagpoles:
            timestamp = pd.to_datetime(flagpole['start_time'])
            year_month = timestamp.strftime('%Y-%m')
            monthly_counts[year_month] = monthly_counts.get(year_month, 0) + 1
        
        if not monthly_counts:
            ax.text(0.5, 0.5, '无数据', transform=ax.transAxes, ha='center', va='center')
            return
        
        # 创建热力图数据
        months = sorted(monthly_counts.keys())
        counts = [monthly_counts[month] for month in months]
        
        # 简化显示（只显示部分月份标签）
        display_months = months[::max(1, len(months)//20)]  # 最多显示20个标签
        display_indices = [months.index(m) for m in display_months]
        
        bars = ax.bar(range(len(months)), counts, alpha=0.7)
        
        # 根据数量设置颜色
        max_count = max(counts)
        for bar, count in zip(bars, counts):
            intensity = count / max_count
            bar.set_color(plt.cm.Reds(0.3 + 0.7 * intensity))
        
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel('旗杆数量', fontsize=9)
        
        # 设置x轴标签
        ax.set_xticks(display_indices)
        ax.set_xticklabels([months[i] for i in display_indices], rotation=45, fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _create_weekly_distribution(self, ax, flagpoles: List[Dict], title: str):
        """创建周内分布图"""
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_counts = [0] * 7
        
        for flagpole in flagpoles:
            timestamp = pd.to_datetime(flagpole['start_time'])
            weekday_counts[timestamp.weekday()] += 1
        
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, 7))
        bars = ax.bar(weekdays, weekday_counts, color=colors, alpha=0.8)
        
        # 添加数值标签
        for bar, count in zip(bars, weekday_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   str(count), ha='center', va='bottom', fontsize=9)
        
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel('旗杆数量', fontsize=9)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    def _create_hourly_distribution(self, ax, flagpoles: List[Dict], title: str):
        """创建小时分布图"""
        hourly_counts = [0] * 24
        
        for flagpole in flagpoles:
            timestamp = pd.to_datetime(flagpole['start_time'])
            hourly_counts[timestamp.hour] += 1
        
        # 创建颜色映射
        max_count = max(hourly_counts) if hourly_counts else 1
        colors = [plt.cm.viridis(count / max_count) for count in hourly_counts]
        
        bars = ax.bar(range(24), hourly_counts, color=colors, alpha=0.8)
        
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel('旗杆数量', fontsize=9)
        ax.set_xlabel('小时', fontsize=9)
        ax.set_xticks(range(0, 24, 3))
        ax.grid(True, alpha=0.3)
    
    def _create_height_distribution(self, ax, flagpoles: List[Dict], title: str):
        """创建高度分布图"""
        heights = [f['height_percent'] for f in flagpoles]
        
        if not heights:
            ax.text(0.5, 0.5, '无数据', transform=ax.transAxes, ha='center', va='center')
            return
        
        # 创建直方图
        n_bins = min(20, len(set(heights)))  # 最多20个分组
        counts, bins, patches = ax.hist(heights, bins=n_bins, alpha=0.7, edgecolor='black')
        
        # 设置颜色渐变
        max_count = max(counts)
        for patch, count in zip(patches, counts):
            intensity = count / max_count
            patch.set_facecolor(plt.cm.Oranges(0.3 + 0.7 * intensity))
        
        # 添加统计信息
        mean_height = np.mean(heights)
        median_height = np.median(heights)
        
        ax.axvline(mean_height, color='red', linestyle='--', 
                   label=f'平均值: {mean_height:.2%}')
        ax.axvline(median_height, color='blue', linestyle='--', 
                   label=f'中位数: {median_height:.2%}')
        
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel('数量', fontsize=9)
        ax.set_xlabel('旗杆高度 (%)', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _create_statistics_chart(self, flagpoles: List[Dict], symbol: str, output_dir: str):
        """创建统计分析图表"""
        logger.info("创建统计分析图表...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 方向分布饼图
        directions = [f['direction'] for f in flagpoles]
        direction_counts = pd.Series(directions).value_counts()
        
        ax1.pie(direction_counts.values, labels=['上升旗杆', '下降旗杆'], 
                autopct='%1.1f%%', startangle=90,
                colors=['red', 'blue'], alpha=0.8)
        ax1.set_title('旗杆方向分布', fontsize=12, fontweight='bold')
        
        # 2. 斜率分分布
        slope_scores = [f['slope_score'] for f in flagpoles]
        ax2.hist(slope_scores, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(np.mean(slope_scores), color='red', linestyle='--', 
                   label=f'平均值: {np.mean(slope_scores):.2f}')
        ax2.set_title('斜率分分布', fontsize=12, fontweight='bold')
        ax2.set_xlabel('斜率分')
        ax2.set_ylabel('数量')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 量能爆发分布
        volume_bursts = [f['volume_burst'] for f in flagpoles]
        ax3.hist(volume_bursts, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax3.axvline(np.mean(volume_bursts), color='red', linestyle='--', 
                   label=f'平均值: {np.mean(volume_bursts):.2f}')
        ax3.set_title('量能爆发分布', fontsize=12, fontweight='bold')
        ax3.set_xlabel('量能爆发倍数')
        ax3.set_ylabel('数量')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. K线数分布
        bars_counts = [f['bars_count'] for f in flagpoles]
        bars_distribution = pd.Series(bars_counts).value_counts().sort_index()
        
        ax4.bar(bars_distribution.index, bars_distribution.values, 
                alpha=0.7, color='purple', edgecolor='black')
        ax4.set_title('旗杆K线数分布', fontsize=12, fontweight='bold')
        ax4.set_xlabel('K线数量')
        ax4.set_ylabel('旗杆数量')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'{symbol} - 旗杆统计分析 (总计{len(flagpoles)}个)', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        stats_file = os.path.join(output_dir, f"{symbol}_flagpole_statistics.png")
        plt.savefig(stats_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"统计图表已保存到: {stats_file}")


def main():
    """主函数"""
    # 配置日志
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add("logs/full_range_visualization.log", level="DEBUG")
    
    try:
        # 创建可视化实例
        visualizer = FullRangeVisualization()
        
        # 创建完整可视化图表
        visualizer.create_full_range_charts("RBL8")
        
        print("\\n" + "="*80)
        print("完整时间范围旗杆可视化完成！")
        print("="*80)
        print("生成的图表包括:")
        print("1. RBL8_complete_overview.png - 完整时间范围总览图")
        print("2. RBL8_[年份]_flagpoles.png - 各年度详细图表")
        print("3. RBL8_flagpole_heatmap.png - 分布热力图")
        print("4. RBL8_flagpole_statistics.png - 统计分析图")
        print("\\n图表保存在: output/flagpole_tests/full_range_charts/")
        print("="*80)
        
    except Exception as e:
        logger.error(f"可视化生成失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()