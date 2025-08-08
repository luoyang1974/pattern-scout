#!/usr/bin/env python3
"""
简化版完整时间范围旗杆可视化
基于完整测试的摘要信息创建可视化
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import json
from loguru import logger

# 导入项目模块
from src.data.connectors.csv_connector import CSVDataConnector
from src.patterns.detectors.flagpole_detector import FlagpoleDetector
from src.patterns.base.market_regime_detector import BaselineManager
from src.data.models.base_models import MarketRegime


class SimpleFullRangeVisualization:
    """简化版完整时间范围可视化"""
    
    def __init__(self):
        self.csv_connector = CSVDataConnector("data/csv")
        if not self.csv_connector.connect():
            raise ConnectionError("无法连接到CSV数据源")
            
        # 初始化检测器（用于重新生成部分数据）
        self.baseline_manager = BaselineManager()
        self.flagpole_detector = FlagpoleDetector(self.baseline_manager)
        self._apply_relaxed_thresholds()
    
    def _apply_relaxed_thresholds(self):
        """应用放宽的阈值"""
        def relaxed_thresholds():
            return {
                'slope_score_p90': 0.15,    
                'volume_burst_p85': 1.25,   
                'retrace_depth_p75': 0.45,  
            }
        
        self.flagpole_detector._get_fallback_thresholds = relaxed_thresholds
    
    def create_comprehensive_visualization(self, symbol: str = "RBL8"):
        """创建综合可视化图表"""
        logger.info("开始创建完整时间范围综合可视化")
        
        # 1. 加载完整数据
        logger.info("加载完整数据集...")
        full_data = self.csv_connector.get_data(symbol)
        
        if full_data.empty:
            raise ValueError(f"无法加载数据: {symbol}")
        
        logger.info(f"数据加载完成: {len(full_data)}条记录")
        
        # 2. 创建输出目录
        output_dir = "output/flagpole_tests/comprehensive_charts"
        os.makedirs(output_dir, exist_ok=True)
        
        # 3. 分时段检测和可视化
        self._create_period_based_visualization(full_data, symbol, output_dir)
        
        # 4. 创建整体统计图表
        self._create_overall_analysis(full_data, symbol, output_dir)
        
        logger.info("综合可视化图表创建完成")
        logger.info(f"图表保存在: {output_dir}")
    
    def _create_period_based_visualization(self, data: pd.DataFrame, symbol: str, output_dir: str):
        """创建分时段的可视化"""
        logger.info("创建分时段可视化图表...")
        
        # 按年度分割数据
        data['year'] = data['timestamp'].dt.year
        years = sorted(data['year'].unique())
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 为每个年度创建检测和可视化
        all_flagpoles = []
        
        for year in years:
            if year < 2019 or year > 2025:  # 跳过边界年份
                continue
                
            logger.info(f"处理{year}年数据...")
            
            # 提取年度数据
            year_data = data[data['year'] == year].copy()
            if len(year_data) < 100:  # 数据太少跳过
                continue
                
            year_data = year_data.reset_index(drop=True)
            
            # 检测该年的旗杆
            try:
                year_flagpoles = self.flagpole_detector.detect_flagpoles(
                    df=year_data,
                    current_regime=MarketRegime.UNKNOWN,
                    timeframe="15m"
                )
                
                all_flagpoles.extend(year_flagpoles)
                logger.info(f"{year}年检测到{len(year_flagpoles)}个旗杆")
                
                # 创建年度图表
                if len(year_flagpoles) > 0:
                    self._create_yearly_chart(year_data, year_flagpoles, year, symbol, output_dir)
                    
            except Exception as e:
                logger.error(f"{year}年检测失败: {e}")
                continue
        
        # 5. 创建总体概览图表
        logger.info(f"总共检测到{len(all_flagpoles)}个旗杆，创建总体概览...")
        self._create_master_overview(data, all_flagpoles, symbol, output_dir)
        
        return all_flagpoles
    
    def _create_yearly_chart(self, year_data: pd.DataFrame, flagpoles: list, 
                           year: int, symbol: str, output_dir: str):
        """创建年度图表"""
        # 数据抽样（避免图表过于密集）
        sample_ratio = max(1, len(year_data) // 2000)
        sampled_data = year_data.iloc[::sample_ratio].copy()
        sampled_data = sampled_data.reset_index(drop=True)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        # 绘制价格走势
        ax1.plot(range(len(sampled_data)), sampled_data['close'], 
                color='black', linewidth=0.8, alpha=0.8, label='收盘价')
        
        # 标记旗杆
        self._mark_flagpoles_on_chart(ax1, sampled_data, flagpoles, sample_ratio)
        
        ax1.set_title(f'{symbol} - {year}年价格走势与旗杆分布 (共{len(flagpoles)}个旗杆)', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('价格', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 设置时间轴
        if len(sampled_data) > 10:
            time_ticks = np.linspace(0, len(sampled_data)-1, 8, dtype=int)
            ax1.set_xticks(time_ticks)
            ax1.set_xticklabels([sampled_data.iloc[i]['timestamp'].strftime('%m-%d') 
                               for i in time_ticks], rotation=45)
        
        # 绘制成交量
        if 'volume' in sampled_data.columns:
            ax2.bar(range(len(sampled_data)), sampled_data['volume'], 
                   alpha=0.6, color='gray', width=0.8)
            
            # 标记旗杆的量能爆发
            self._mark_volume_bursts(ax2, sampled_data, flagpoles, sample_ratio)
            
            ax2.set_title(f'{year}年成交量与量能爆发', fontsize=12, fontweight='bold')
            ax2.set_ylabel('成交量', fontsize=12)
            ax2.set_xlabel('时间', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # 时间轴
            ax2.set_xticks(time_ticks)
            ax2.set_xticklabels([sampled_data.iloc[i]['timestamp'].strftime('%m-%d') 
                               for i in time_ticks], rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        yearly_file = os.path.join(output_dir, f"{symbol}_{year}_comprehensive.png")
        plt.savefig(yearly_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"{year}年图表已保存: {yearly_file}")
    
    def _mark_flagpoles_on_chart(self, ax, sampled_data: pd.DataFrame, flagpoles: list, sample_ratio: int):
        """在图表上标记旗杆"""
        up_positions = []
        down_positions = []
        up_prices = []
        down_prices = []
        
        for flagpole in flagpoles:
            # 找到最接近的时间点
            flagpole_time = flagpole.start_time
            time_diffs = abs(sampled_data['timestamp'] - flagpole_time)
            
            if len(time_diffs) > 0:
                closest_idx = time_diffs.idxmin()
                
                if flagpole.direction == 'up':
                    up_positions.append(closest_idx)
                    up_prices.append(flagpole.start_price)
                else:
                    down_positions.append(closest_idx)
                    down_prices.append(flagpole.start_price)
        
        # 绘制旗杆标记
        if up_positions:
            ax.scatter(up_positions, up_prices, c='red', marker='^', 
                      s=40, alpha=0.8, label=f'上升旗杆 ({len(up_positions)}个)')
        
        if down_positions:
            ax.scatter(down_positions, down_prices, c='blue', marker='v', 
                      s=40, alpha=0.8, label=f'下降旗杆 ({len(down_positions)}个)')
    
    def _mark_volume_bursts(self, ax, sampled_data: pd.DataFrame, flagpoles: list, sample_ratio: int):
        """标记量能爆发"""
        if 'volume' not in sampled_data.columns:
            return
            
        high_volume_positions = []
        high_volumes = []
        
        for flagpole in flagpoles:
            if flagpole.volume_burst > 1.5:  # 只标记显著量能爆发
                flagpole_time = flagpole.start_time
                time_diffs = abs(sampled_data['timestamp'] - flagpole_time)
                
                if len(time_diffs) > 0:
                    closest_idx = time_diffs.idxmin()
                    high_volume_positions.append(closest_idx)
                    high_volumes.append(sampled_data.iloc[closest_idx]['volume'])
        
        if high_volume_positions:
            ax.scatter(high_volume_positions, high_volumes, c='red', marker='*', 
                      s=60, alpha=0.8, label=f'量能爆发 ({len(high_volume_positions)}个)')
            ax.legend()
    
    def _create_master_overview(self, data: pd.DataFrame, all_flagpoles: list, 
                               symbol: str, output_dir: str):
        """创建主总览图表"""
        logger.info("创建主总览图表...")
        
        # 大幅抽样（每100条取1条）
        sample_ratio = 100
        sampled_data = data.iloc[::sample_ratio].copy()
        sampled_data = sampled_data.reset_index(drop=True)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 15))
        
        # 1. 完整价格走势
        ax1.plot(range(len(sampled_data)), sampled_data['close'], 
                color='black', linewidth=0.6, alpha=0.9, label='收盘价')
        
        # 标记所有旗杆
        self._mark_flagpoles_on_chart(ax1, sampled_data, all_flagpoles, sample_ratio)
        
        ax1.set_title(f'{symbol} - 完整时间范围价格走势 (2019-2025, 共{len(all_flagpoles)}个旗杆)', 
                     fontsize=16, fontweight='bold')
        ax1.set_ylabel('价格', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 时间轴
        time_ticks = np.linspace(0, len(sampled_data)-1, 10, dtype=int)
        ax1.set_xticks(time_ticks)
        ax1.set_xticklabels([sampled_data.iloc[i]['timestamp'].strftime('%Y-%m') 
                           for i in time_ticks], rotation=45)
        
        # 2. 旗杆分布密度
        self._plot_flagpole_density_timeline(ax2, all_flagpoles, 
                                           '旗杆时间分布密度 (月度统计)')
        
        # 3. 旗杆特征分析
        self._plot_flagpole_characteristics(ax3, all_flagpoles, 
                                          '旗杆高度分布与质量分析')
        
        plt.tight_layout()
        
        # 保存主览图
        master_file = os.path.join(output_dir, f"{symbol}_master_overview.png")
        plt.savefig(master_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"主总览图表已保存: {master_file}")
    
    def _plot_flagpole_density_timeline(self, ax, flagpoles: list, title: str):
        """绘制旗杆密度时间线"""
        if not flagpoles:
            ax.text(0.5, 0.5, '无旗杆数据', transform=ax.transAxes, ha='center', va='center')
            return
        
        # 按月统计
        flagpole_times = [f.start_time for f in flagpoles]
        flagpole_df = pd.DataFrame({'timestamp': flagpole_times})
        flagpole_df['year_month'] = flagpole_df['timestamp'].dt.to_period('M')
        
        monthly_counts = flagpole_df.groupby('year_month').size()
        
        # 创建连续的月度时间序列
        start_period = monthly_counts.index.min()
        end_period = monthly_counts.index.max()
        all_months = pd.period_range(start_period, end_period, freq='M')
        
        full_monthly = pd.Series(0, index=all_months)
        for period, count in monthly_counts.items():
            full_monthly[period] = count
        
        # 绘制
        x_pos = range(len(full_monthly))
        colors = ['red' if count > full_monthly.median() else 'blue' 
                 for count in full_monthly.values]
        
        bars = ax.bar(x_pos, full_monthly.values, color=colors, alpha=0.7)
        
        # 添加统计线
        avg_count = full_monthly.mean()
        ax.axhline(y=avg_count, color='green', linestyle='--', linewidth=2, 
                   label=f'月均值: {avg_count:.1f}个')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('旗杆数量/月', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 时间轴标签
        step = max(1, len(full_monthly) // 15)
        tick_positions = range(0, len(full_monthly), step)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(full_monthly.index[i]) for i in tick_positions], 
                          rotation=45)
    
    def _plot_flagpole_characteristics(self, ax, flagpoles: list, title: str):
        """绘制旗杆特征分析"""
        if not flagpoles:
            ax.text(0.5, 0.5, '无旗杆数据', transform=ax.transAxes, ha='center', va='center')
            return
        
        # 提取特征数据
        heights = [f.height_percent for f in flagpoles]
        slope_scores = [f.slope_score for f in flagpoles]
        
        # 创建散点图
        colors = ['red' if f.direction == 'up' else 'blue' for f in flagpoles]
        sizes = [max(10, min(100, h * 10000)) for h in heights]  # 根据高度调整点大小
        
        scatter = ax.scatter(slope_scores, heights, c=colors, s=sizes, alpha=0.6, 
                           edgecolors='black', linewidth=0.5)
        
        # 添加统计信息
        avg_height = np.mean(heights)
        avg_slope = np.mean(slope_scores)
        
        ax.axhline(y=avg_height, color='green', linestyle='--', alpha=0.7,
                   label=f'平均高度: {avg_height:.2%}')
        ax.axvline(x=avg_slope, color='orange', linestyle='--', alpha=0.7,
                   label=f'平均斜率分: {avg_slope:.2f}')
        
        # 添加象限标注
        ax.text(0.05, 0.95, f'高质量区间\\n(高度>{avg_height:.2%}, 斜率>{avg_slope:.2f})', 
                transform=ax.transAxes, fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('斜率分', fontsize=12)
        ax.set_ylabel('旗杆高度 (%)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_overall_analysis(self, data: pd.DataFrame, symbol: str, output_dir: str):
        """创建整体分析图表"""
        logger.info("创建整体分析图表...")
        
        # 创建基于摘要数据的分析图表
        try:
            summary_file = "output/flagpole_tests/full_range/RBL8_full_range_summary.json"
            if os.path.exists(summary_file):
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
                
                self._create_summary_charts(summary_data, symbol, output_dir)
        except Exception as e:
            logger.warning(f"无法创建基于摘要的图表: {e}")
        
        # 创建基于数据的统计图表
        self._create_data_statistics(data, symbol, output_dir)
    
    def _create_summary_charts(self, summary: dict, symbol: str, output_dir: str):
        """基于摘要数据创建图表"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 检测统计
        stats = summary["detection_stats"]
        categories = ['上升旗杆', '下降旗杆']
        values = [stats['up_flagpoles'], stats['down_flagpoles']]
        colors = ['red', 'blue']
        
        ax1.pie(values, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('旗杆方向分布', fontsize=12, fontweight='bold')
        
        # 2. 时间统计
        test_info = summary["test_info"]
        time_stats = [
            ['总记录数', test_info['total_records']],
            ['检测旗杆数', stats['total_flagpoles']],
            ['检测率(/千条)', stats['detection_rate_per_1000']],
            ['处理时间(分钟)', test_info['processing_time_minutes']]
        ]
        
        ax2.axis('tight')
        ax2.axis('off')
        table = ax2.table(cellText=time_stats, 
                         colLabels=['指标', '数值'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax2.set_title('测试统计摘要', fontsize=12, fontweight='bold')
        
        # 3. 质量分析
        flagpole_summary = summary["flagpole_summary"]
        quality_data = [
            flagpole_summary['avg_height_percent'],
            flagpole_summary['height_range']['min'],
            flagpole_summary['height_range']['max'],
            flagpole_summary['avg_slope_score']
        ]
        quality_labels = ['平均高度', '最小高度', '最大高度', '平均斜率分']
        
        bars = ax3.bar(quality_labels, quality_data, color=['green', 'blue', 'red', 'orange'], alpha=0.7)
        ax3.set_title('旗杆质量指标', fontsize=12, fontweight='bold')
        ax3.set_ylabel('数值')
        ax3.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars, quality_data):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 4. 检测效率
        efficiency_data = [
            test_info['total_records'],
            stats['total_flagpoles'],
            test_info['processing_time_minutes']
        ]
        efficiency_labels = ['数据量\\n(万条)', '检测数量\\n(个)', '处理时间\\n(分钟)']
        
        # 标准化数据用于比较
        normalized_data = [
            efficiency_data[0] / 10000,  # 转换为万条
            efficiency_data[1] * 10,     # 放大10倍便于显示
            efficiency_data[2] * 10      # 放大10倍便于显示
        ]
        
        bars = ax4.bar(efficiency_labels, normalized_data, 
                      color=['purple', 'green', 'orange'], alpha=0.7)
        ax4.set_title('处理效率分析', fontsize=12, fontweight='bold')
        ax4.set_ylabel('标准化数值')
        
        plt.suptitle(f'{symbol} - 完整时间范围检测分析总结', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        summary_chart_file = os.path.join(output_dir, f"{symbol}_analysis_summary.png")
        plt.savefig(summary_chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"分析摘要图表已保存: {summary_chart_file}")
    
    def _create_data_statistics(self, data: pd.DataFrame, symbol: str, output_dir: str):
        """创建数据统计图表"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 价格统计
        price_stats = data['close'].describe()
        ax1.boxplot([data['close']], labels=['收盘价'])
        ax1.set_title('价格分布箱线图', fontsize=12, fontweight='bold')
        ax1.set_ylabel('价格')
        ax1.grid(True, alpha=0.3)
        
        # 2. 成交量分布
        if 'volume' in data.columns:
            volume_data = data['volume'].dropna()
            ax2.hist(volume_data, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax2.axvline(volume_data.mean(), color='red', linestyle='--', 
                       label=f'平均值: {volume_data.mean():.0f}')
            ax2.set_title('成交量分布', fontsize=12, fontweight='bold')
            ax2.set_xlabel('成交量')
            ax2.set_ylabel('频次')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. 年度价格走势
        data['year'] = data['timestamp'].dt.year
        yearly_avg = data.groupby('year')['close'].mean()
        
        ax3.plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=2, markersize=6)
        ax3.set_title('年度平均价格走势', fontsize=12, fontweight='bold')
        ax3.set_xlabel('年份')
        ax3.set_ylabel('平均价格')
        ax3.grid(True, alpha=0.3)
        
        # 4. 市场活跃度
        data['month'] = data['timestamp'].dt.to_period('M')
        monthly_volatility = data.groupby('month')['close'].std()
        
        recent_volatility = monthly_volatility.tail(24)  # 最近24个月
        ax4.plot(range(len(recent_volatility)), recent_volatility.values, linewidth=1.5)
        ax4.set_title('近期市场波动率趋势', fontsize=12, fontweight='bold')
        ax4.set_xlabel('时间 (月)')
        ax4.set_ylabel('价格标准差')
        ax4.grid(True, alpha=0.3)
        
        # 设置x轴标签
        if len(recent_volatility) > 6:
            step = len(recent_volatility) // 6
            ticks = range(0, len(recent_volatility), step)
            ax4.set_xticks(ticks)
            ax4.set_xticklabels([str(recent_volatility.index[i]) for i in ticks], rotation=45)
        
        plt.suptitle(f'{symbol} - 基础数据统计分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        data_stats_file = os.path.join(output_dir, f"{symbol}_data_statistics.png")
        plt.savefig(data_stats_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"数据统计图表已保存: {data_stats_file}")


def main():
    """主函数"""
    # 配置日志
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add("logs/simple_full_range_viz.log", level="DEBUG")
    
    try:
        # 创建可视化实例
        visualizer = SimpleFullRangeVisualization()
        
        # 创建综合可视化图表
        visualizer.create_comprehensive_visualization("RBL8")
        
        print("\\n" + "="*80)
        print("完整时间范围旗杆可视化完成！")
        print("="*80)
        print("生成的图表包括:")
        print("1. RBL8_master_overview.png - 完整时间范围主总览图")
        print("2. RBL8_[年份]_comprehensive.png - 各年度综合分析图")
        print("3. RBL8_analysis_summary.png - 检测结果分析摘要")
        print("4. RBL8_data_statistics.png - 基础数据统计分析")
        print("\\n所有图表保存在: output/flagpole_tests/comprehensive_charts/")
        print("="*80)
        
    except Exception as e:
        logger.error(f"可视化生成失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()