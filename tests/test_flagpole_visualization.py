#!/usr/bin/env python3
"""
旗杆可视化测试脚本
生成找到的旗杆的K线图可视化
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List
from loguru import logger

# 导入项目模块
from src.data.connectors.csv_connector import CSVDataConnector
from src.patterns.detectors.flagpole_detector import FlagpoleDetector
from src.patterns.base.market_regime_detector import BaselineManager
from src.data.models.base_models import Flagpole, MarketRegime


def generate_flagpole_visualization():
    """生成旗杆可视化图表"""
    logger.info("开始生成旗杆可视化图表")
    
    # 1. 加载数据和检测旗杆（复用快速测试的逻辑）
    csv_connector = CSVDataConnector("data/csv")
    if not csv_connector.connect():
        logger.error("无法连接到CSV数据源")
        return
    
    symbol = "RBL8"
    all_data = csv_connector.get_data(symbol)
    
    if all_data.empty:
        logger.error(f"无法加载数据: {symbol}")
        return
    
    # 使用最后1000条数据
    data = all_data.tail(1000).reset_index(drop=True)
    logger.info(f"使用数据: {len(data)} 条记录")
    
    # 2. 检测旗杆
    baseline_manager = BaselineManager()
    flagpole_detector = FlagpoleDetector(baseline_manager)
    
    # 使用放宽的阈值
    original_method = flagpole_detector._get_fallback_thresholds
    def relaxed_thresholds():
        return {
            'slope_score_p90': 0.1,   
            'volume_burst_p85': 1.2,  
            'retrace_depth_p75': 0.5,  
        }
    flagpole_detector._get_fallback_thresholds = relaxed_thresholds
    
    flagpoles = flagpole_detector.detect_flagpoles(
        df=data,
        current_regime=MarketRegime.UNKNOWN,
        timeframe="15m"
    )
    
    logger.info(f"检测到 {len(flagpoles)} 个旗杆")
    
    if not flagpoles:
        logger.warning("未检测到旗杆，无法生成可视化图表")
        return
    
    # 3. 生成可视化图表
    output_dir = "output/flagpole_tests/charts"
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 生成总览图表
    create_overview_chart(data, flagpoles, symbol, output_dir)
    
    # 生成单个旗杆详细图表
    create_individual_charts(data, flagpoles, symbol, output_dir)
    
    logger.info("可视化图表生成完成")
    
    # 恢复原始方法
    flagpole_detector._get_fallback_thresholds = original_method


def create_overview_chart(data: pd.DataFrame, flagpoles: List[Flagpole], symbol: str, output_dir: str):
    """创建总览图表"""
    logger.info("创建总览图表")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # 绘制K线图
    plot_candlestick(ax1, data, f'{symbol} - 旗杆检测总览')
    
    # 标记旗杆
    mark_flagpoles_on_chart(ax1, data, flagpoles)
    
    # 绘制成交量
    plot_volume(ax2, data, '成交量')
    mark_flagpoles_volume(ax2, data, flagpoles)
    
    plt.tight_layout()
    
    # 保存图表
    chart_file = os.path.join(output_dir, f"{symbol}_flagpole_overview.png")
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"总览图表已保存到: {chart_file}")


def create_individual_charts(data: pd.DataFrame, flagpoles: List[Flagpole], symbol: str, output_dir: str):
    """创建单个旗杆详细图表"""
    logger.info("创建单个旗杆详细图表")
    
    for i, flagpole in enumerate(flagpoles):
        # 找到旗杆在数据中的位置
        start_idx = None
        end_idx = None
        
        for j, timestamp in enumerate(data['timestamp']):
            if timestamp == flagpole.start_time:
                start_idx = j
            if timestamp == flagpole.end_time:
                end_idx = j
                break
        
        if start_idx is None or end_idx is None:
            logger.warning(f"无法找到旗杆#{i+1}的时间索引")
            continue
        
        # 扩展显示范围（前后各30个K线）
        display_start = max(0, start_idx - 30)
        display_end = min(len(data) - 1, end_idx + 30)
        
        display_data = data.iloc[display_start:display_end + 1].copy()
        display_data = display_data.reset_index(drop=True)
        
        # 调整旗杆索引到新的显示范围
        flagpole_start_in_display = start_idx - display_start  
        flagpole_end_in_display = end_idx - display_start
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # K线图
        plot_candlestick(ax1, display_data, 
                        f'{symbol} - 旗杆#{i+1} ({flagpole.direction}, {flagpole.height_percent:.2%})')
        
        # 突出显示旗杆区间
        ax1.axvspan(flagpole_start_in_display - 0.5, flagpole_end_in_display + 0.5, 
                   alpha=0.2, color='blue', label='旗杆区间')
        
        # 标记起止点
        start_price = flagpole.start_price
        end_price = flagpole.end_price
        ax1.plot(flagpole_start_in_display, start_price, 'go', markersize=12, 
                label=f'起始点 ({start_price:.2f})', markeredgecolor='black', markeredgewidth=2)
        ax1.plot(flagpole_end_in_display, end_price, 'ro', markersize=12, 
                label=f'结束点 ({end_price:.2f})', markeredgecolor='black', markeredgewidth=2)
        
        # 绘制旗杆趋势线
        ax1.plot([flagpole_start_in_display, flagpole_end_in_display], 
                [start_price, end_price], 'b--', linewidth=2, alpha=0.8, label='旗杆趋势线')
        
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 成交量图
        plot_volume(ax2, display_data, '成交量分析')
        
        # 突出显示旗杆期间成交量
        if 'volume' in display_data.columns:
            volumes = display_data['volume'].values
            colors = ['orange' if flagpole_start_in_display <= j <= flagpole_end_in_display else 'gray' 
                     for j in range(len(volumes))]
            alphas = [0.8 if flagpole_start_in_display <= j <= flagpole_end_in_display else 0.5 
                     for j in range(len(volumes))]
            
            ax2.clear()
            bars = ax2.bar(range(len(volumes)), volumes, color=colors, alpha=0.6)
            
            for j, (bar, alpha) in enumerate(zip(bars, alphas)):
                bar.set_alpha(alpha)
            
            ax2.set_title('成交量分析 (旗杆期间高亮显示)')
            ax2.set_ylabel('成交量')
            ax2.grid(True, alpha=0.3)
        
        # 添加统计信息文本框
        stats_text = create_stats_text(flagpole)
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图表
        chart_file = os.path.join(output_dir, 
                                f"{symbol}_flagpole_{i+1}_{flagpole.direction}_{flagpole.height_percent:.1%}.png")
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"旗杆#{i+1}详细图表已保存到: {chart_file}")


def plot_candlestick(ax, data: pd.DataFrame, title: str):
    """绘制K线图"""
    for i, (_, row) in enumerate(data.iterrows()):
        # 确定颜色
        if row['close'] >= row['open']:
            color = 'red'  # 阳线
            edge_color = 'darkred'
        else:
            color = 'green'  # 阴线
            edge_color = 'darkgreen'
        
        # 绘制影线
        ax.plot([i, i], [row['low'], row['high']], color='black', linewidth=1)
        
        # 绘制实体
        body_height = abs(row['close'] - row['open'])
        bottom = min(row['open'], row['close'])
        
        if body_height > 0:
            rect = plt.Rectangle((i-0.4, bottom), 0.8, body_height, 
                               facecolor=color, alpha=0.8, 
                               edgecolor=edge_color, linewidth=0.5)
            ax.add_patch(rect)
        else:
            # 十字星
            ax.plot([i-0.4, i+0.4], [row['close'], row['close']], 
                   color=edge_color, linewidth=1)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('价格', fontsize=12)
    ax.grid(True, alpha=0.3)


def plot_volume(ax, data: pd.DataFrame, title: str):
    """绘制成交量图"""
    if 'volume' not in data.columns:
        ax.text(0.5, 0.5, '无成交量数据', transform=ax.transAxes, 
               ha='center', va='center', fontsize=12)
        return
    
    volumes = data['volume'].values
    ax.bar(range(len(volumes)), volumes, alpha=0.6, color='gray', width=0.8)
    
    # 添加成交量移动平均线
    if len(volumes) >= 20:
        volume_ma = pd.Series(volumes).rolling(window=20).mean()
        ax.plot(range(len(volume_ma)), volume_ma, color='red', linewidth=1, 
               label='成交量MA20', alpha=0.8)
        ax.legend()
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('成交量', fontsize=12)
    ax.grid(True, alpha=0.3)


def mark_flagpoles_on_chart(ax, data: pd.DataFrame, flagpoles: List[Flagpole]):
    """在K线图上标记旗杆"""
    colors = ['blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, flagpole in enumerate(flagpoles):
        color = colors[i % len(colors)]
        
        # 找到旗杆在数据中的索引
        start_idx = None
        end_idx = None
        
        for j, timestamp in enumerate(data['timestamp']):
            if timestamp == flagpole.start_time:
                start_idx = j
            if timestamp == flagpole.end_time:
                end_idx = j
                break
        
        if start_idx is not None and end_idx is not None:
            # 标记旗杆区域
            ax.axvspan(start_idx - 0.5, end_idx + 0.5, alpha=0.2, color=color, 
                      label=f'旗杆{i+1}-{flagpole.direction}')
            
            # 标记关键点
            ax.plot(start_idx, flagpole.start_price, 'o', color=color, 
                   markersize=8, markeredgecolor='black', markeredgewidth=1)
            ax.plot(end_idx, flagpole.end_price, 's', color=color, 
                   markersize=8, markeredgecolor='black', markeredgewidth=1)
            
            # 添加文本标注
            mid_idx = (start_idx + end_idx) / 2
            mid_price = (flagpole.start_price + flagpole.end_price) / 2
            ax.annotate(f'#{i+1}\\n{flagpole.height_percent:.1%}', 
                       xy=(mid_idx, mid_price), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
    
    if flagpoles:
        ax.legend(loc='upper left', fontsize=8)


def mark_flagpoles_volume(ax, data: pd.DataFrame, flagpoles: List[Flagpole]):
    """在成交量图上标记旗杆"""
    if 'volume' not in data.columns:
        return
    
    colors = ['blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, flagpole in enumerate(flagpoles):
        color = colors[i % len(colors)]
        
        # 找到旗杆在数据中的索引
        start_idx = None
        end_idx = None
        
        for j, timestamp in enumerate(data['timestamp']):
            if timestamp == flagpole.start_time:
                start_idx = j
            if timestamp == flagpole.end_time:
                end_idx = j
                break
        
        if start_idx is not None and end_idx is not None:
            # 标记旗杆期间成交量
            ax.axvspan(start_idx - 0.5, end_idx + 0.5, alpha=0.2, color=color)
            
            # 找到期间最大成交量点
            flagpole_volumes = data.iloc[start_idx:end_idx+1]['volume']
            max_volume_idx = start_idx + flagpole_volumes.argmax()
            max_volume = flagpole_volumes.max()
            
            ax.plot(max_volume_idx, max_volume, '^', color=color, markersize=10,
                   markeredgecolor='black', markeredgewidth=1)
            
            ax.annotate(f'#{i+1}\\n{flagpole.volume_burst:.1f}x', 
                       xy=(max_volume_idx, max_volume), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))


def create_stats_text(flagpole: Flagpole) -> str:
    """创建统计信息文本"""
    return (f'旗杆统计信息:\\n'
           f'方向: {flagpole.direction}\\n'
           f'高度: {flagpole.height:.4f} ({flagpole.height_percent:.2%})\\n'
           f'K线数: {flagpole.bars_count}\\n'
           f'斜率分: {flagpole.slope_score:.2f}\\n'
           f'量能爆发: {flagpole.volume_burst:.2f}x\\n'
           f'动量占比: {flagpole.impulse_bar_ratio:.1%}\\n'
           f'回撤比例: {flagpole.retracement_ratio:.1%}\\n'
           f'趋势强度: {flagpole.trend_strength:.3f}\\n'
           f'时间范围: {flagpole.start_time.strftime("%m-%d %H:%M")} 到 {flagpole.end_time.strftime("%m-%d %H:%M")}')


def main():
    """主函数"""
    # 配置日志
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add("logs/flagpole_visualization.log", level="DEBUG")
    
    try:
        generate_flagpole_visualization()
        print("\\n可视化图表生成完成！")
        print("文件保存在: output/flagpole_tests/charts/")
        print("- 总览图表: RBL8_flagpole_overview.png")
        print("- 单个旗杆详细图表: RBL8_flagpole_[编号]_[方向]_[高度].png")
    except Exception as e:
        logger.error(f"可视化生成失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()