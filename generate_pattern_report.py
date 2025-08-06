#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成旗形识别报告
"""
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def generate_pattern_report():
    """生成旗形识别报告"""
    print("正在生成旗形识别报告...")
    
    # 连接数据库
    conn = sqlite3.connect('output/data/patterns.db')
    
    # 读取所有旗形记录
    df = pd.read_sql_query("""
        SELECT * FROM patterns 
        ORDER BY flagpole_start_time
    """, conn)
    
    print(f"发现 {len(df)} 个旗形模式")
    
    # 统计信息
    print("\n=== 旗形识别结果汇总 ===")
    print(f"总计旗形数量: {len(df)}")
    print(f"品种: {df['symbol'].unique().tolist()}")
    
    # 按类型统计
    type_counts = df['pattern_type'].value_counts()
    print(f"\n按形态类型统计:")
    for pattern_type, count in type_counts.items():
        print(f"  {pattern_type}: {count} 个")
    
    # 按质量统计
    quality_counts = df['pattern_quality'].value_counts()
    print(f"\n按质量等级统计:")
    for quality, count in quality_counts.items():
        print(f"  {quality}: {count} 个")
    
    # 置信度统计
    print(f"\n置信度统计:")
    print(f"  平均置信度: {df['confidence_score'].mean():.3f}")
    print(f"  最高置信度: {df['confidence_score'].max():.3f}")
    print(f"  最低置信度: {df['confidence_score'].min():.3f}")
    
    # 旗杆方向统计
    direction_counts = df['flagpole_direction'].value_counts()
    print(f"\n旗杆方向统计:")
    for direction, count in direction_counts.items():
        print(f"  {direction}: {count} 个")
    
    # 旗杆高度统计
    print(f"\n旗杆高度统计 (%):")
    print(f"  平均高度: {df['flagpole_height_percent'].mean():.2f}%")
    print(f"  最大高度: {df['flagpole_height_percent'].max():.2f}%")
    print(f"  最小高度: {df['flagpole_height_percent'].min():.2f}%")
    
    # 形态持续时间统计
    print(f"\n形态持续时间统计 (K线数):")
    print(f"  平均持续时间: {df['pattern_duration'].mean():.1f} K线")
    print(f"  最长持续时间: {df['pattern_duration'].max()} K线")
    print(f"  最短持续时间: {df['pattern_duration'].min()} K线")
    
    # 生成图表
    generate_pattern_charts(df)
    
    # 生成详细列表
    generate_pattern_list(df)
    
    conn.close()
    print("\n报告生成完成!")

def generate_pattern_charts(df):
    """生成旗形统计图表"""
    print("\n正在生成统计图表...")
    
    # 创建图表目录
    os.makedirs('output/reports', exist_ok=True)
    
    # 设置图表样式
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('RBL8 旗形模式识别统计报告', fontsize=16, fontweight='bold')
    
    # 1. 形态类型分布
    ax1 = axes[0, 0]
    type_counts = df['pattern_type'].value_counts()
    colors = ['#2E8B57', '#FF6347']
    wedges, texts, autotexts = ax1.pie(type_counts.values, labels=type_counts.index, 
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('形态类型分布')
    
    # 2. 质量等级分布
    ax2 = axes[0, 1]
    quality_counts = df['pattern_quality'].value_counts()
    colors = ['#FFD700', '#87CEEB', '#DDA0DD']
    wedges, texts, autotexts = ax2.pie(quality_counts.values, labels=quality_counts.index, 
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('质量等级分布')
    
    # 3. 旗杆方向分布
    ax3 = axes[0, 2]
    direction_counts = df['flagpole_direction'].value_counts()
    colors = ['#FF4500', '#32CD32']
    wedges, texts, autotexts = ax3.pie(direction_counts.values, labels=direction_counts.index, 
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    ax3.set_title('旗杆方向分布')
    
    # 4. 置信度分布
    ax4 = axes[1, 0]
    ax4.hist(df['confidence_score'], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('置信度')
    ax4.set_ylabel('数量')
    ax4.set_title('置信度分布')
    ax4.axvline(df['confidence_score'].mean(), color='red', linestyle='--', label=f'平均值: {df["confidence_score"].mean():.3f}')
    ax4.legend()
    
    # 5. 旗杆高度分布
    ax5 = axes[1, 1]
    ax5.hist(df['flagpole_height_percent'], bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
    ax5.set_xlabel('旗杆高度 (%)')
    ax5.set_ylabel('数量')
    ax5.set_title('旗杆高度分布')
    ax5.axvline(df['flagpole_height_percent'].mean(), color='red', linestyle='--', 
                label=f'平均值: {df["flagpole_height_percent"].mean():.2f}%')
    ax5.legend()
    
    # 6. 形态持续时间分布
    ax6 = axes[1, 2]
    ax6.hist(df['pattern_duration'], bins=20, color='lightcoral', alpha=0.7, edgecolor='black')
    ax6.set_xlabel('持续时间 (K线数)')
    ax6.set_ylabel('数量')
    ax6.set_title('形态持续时间分布')
    ax6.axvline(df['pattern_duration'].mean(), color='red', linestyle='--', 
                label=f'平均值: {df["pattern_duration"].mean():.1f}')
    ax6.legend()
    
    plt.tight_layout()
    
    # 保存图表
    chart_path = 'output/reports/pattern_statistics.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"统计图表已保存: {chart_path}")

def generate_pattern_list(df):
    """生成详细的旗形列表"""
    print("\n正在生成详细旗形列表...")
    
    # 按置信度排序
    df_sorted = df.sort_values('confidence_score', ascending=False)
    
    # 生成CSV报告
    report_columns = [
        'symbol', 'pattern_type', 'flagpole_direction', 'flagpole_start_time', 
        'flagpole_end_time', 'flagpole_height_percent', 'pattern_duration',
        'confidence_score', 'pattern_quality', 'detection_date'
    ]
    
    report_path = 'output/reports/pattern_detailed_list.csv'
    df_sorted[report_columns].to_csv(report_path, index=False, encoding='utf-8-sig')
    print(f"详细旗形列表已保存: {report_path}")
    
    # 生成高质量旗形列表 (置信度 > 0.8)
    high_quality = df_sorted[df_sorted['confidence_score'] > 0.8]
    if len(high_quality) > 0:
        high_quality_path = 'output/reports/high_quality_patterns.csv'
        high_quality[report_columns].to_csv(high_quality_path, index=False, encoding='utf-8-sig')
        print(f"高质量旗形列表已保存: {high_quality_path} ({len(high_quality)}个模式)")
    
    # 打印前10个高置信度旗形
    print("\n=== 前10个高置信度旗形 ===")
    for i, row in df_sorted.head(10).iterrows():
        print(f"{row['pattern_type']:>7} | 置信度: {row['confidence_score']:.3f} | "
              f"方向: {row['flagpole_direction']:>4} | 高度: {row['flagpole_height_percent']:>6.2f}% | "
              f"时间: {row['flagpole_start_time']}")

if __name__ == "__main__":
    generate_pattern_report()