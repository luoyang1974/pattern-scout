#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
形态结局分析与追踪系统
执行六分类结局追踪：强势延续、标准延续、突破停滞、假突破反转、内部瘫解、反向运行
"""
import sys
import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

# 设置控制台编码（Windows）
if sys.platform.startswith('win'):
    os.system('chcp 65001 > nul')
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# 添加src到Python路径
sys.path.append(str(__file__).replace('analyze_pattern_outcomes.py', ''))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class PatternOutcome(Enum):
    """形态结局六分类"""
    STRONG_CONTINUATION = "强势延续"      # 突破后持续强势运行
    STANDARD_CONTINUATION = "标准延续"    # 突破后正常延续
    BREAKTHROUGH_STAGNATION = "突破停滞"  # 突破后缺乏后续动力
    FALSE_BREAKTHROUGH_REVERSAL = "假突破反转"  # 假突破后反向运行
    INTERNAL_BREAKDOWN = "内部瘫解"       # 形态内部失效
    REVERSE_MOVEMENT = "反向运行"         # 完全反向运行

@dataclass
class OutcomeAnalysis:
    """结局分析结果"""
    pattern_id: str
    outcome_type: PatternOutcome
    breakthrough_success: bool
    price_move_percent: float
    volume_confirm: bool
    time_to_outcome: int  # K线数
    confidence_score: float
    analysis_date: datetime

class PatternOutcomeAnalyzer:
    """形态结局分析器"""
    
    def __init__(self, data_connector):
        self.data_connector = data_connector
        self.outcome_results = []
        
    def analyze_all_patterns(self) -> List[OutcomeAnalysis]:
        """分析所有形态的结局"""
        print("开始执行形态结局分析...")
        
        # 从CSV文件读取形态数据
        try:
            patterns_df = pd.read_csv('output/reports/pattern_detailed_list.csv')
            # 过滤高质量和中等质量的形态
            patterns_df = patterns_df[patterns_df['pattern_quality'].isin(['high', 'medium'])]
            patterns_df = patterns_df.sort_values('flagpole_start_time')
        except FileNotFoundError:
            print("未找到形态详细列表文件，请先运行形态识别")
            return []
        
        print(f"找到 {len(patterns_df)} 个需要分析结局的形态")
        
        # 获取价格数据
        from datetime import datetime
        start_date = datetime(2019, 1, 1)
        end_date = datetime.now()
        price_data = self.data_connector.get_data('RBL8', start_date, end_date)
        # 检查时间字段名
        time_col = 'timestamp' if 'timestamp' in price_data.columns else 'Datetime'
        price_data[time_col] = pd.to_datetime(price_data[time_col])
        price_data = price_data.set_index(time_col).sort_index()
        
        # 分析每个形态的结局
        for idx, pattern in patterns_df.iterrows():
            try:
                outcome = self._analyze_single_pattern(pattern, price_data, idx)
                if outcome:
                    self.outcome_results.append(outcome)
                    if len(self.outcome_results) % 10 == 0:
                        print(f"已分析 {len(self.outcome_results)} 个形态结局...")
            except Exception as e:
                print(f"分析形态 {idx} 时出错: {e}")
                continue
                
        print(f"结局分析完成，共分析 {len(self.outcome_results)} 个形态")
        return self.outcome_results
    
    def _analyze_single_pattern(self, pattern: pd.Series, price_data: pd.DataFrame, pattern_idx: int) -> Optional[OutcomeAnalysis]:
        """分析单个形态的结局"""
        
        # 获取形态时间信息
        flagpole_start = pd.to_datetime(pattern['flagpole_start_time'])
        flagpole_end = pd.to_datetime(pattern['flagpole_end_time'])
        
        # 估算形态结束时间（旗杆结束后 + 形态持续时间）
        pattern_end = flagpole_end + timedelta(minutes=15 * pattern['pattern_duration'])
        
        # 获取分析窗口（形态结束后30个K线）
        analysis_end = pattern_end + timedelta(minutes=15 * 30)
        
        # 检查数据是否足够
        if analysis_end > price_data.index.max():
            return None
            
        # 获取相关价格数据
        pattern_data = price_data[flagpole_start:pattern_end]
        outcome_data = price_data[pattern_end:analysis_end]
        
        if len(pattern_data) < 5 or len(outcome_data) < 10:
            return None
            
        # 确定突破方向和关键价位
        direction = pattern['flagpole_direction']
        pattern_high = pattern_data['high'].max()
        pattern_low = pattern_data['low'].min()
        
        if direction == 'up':
            breakthrough_level = pattern_high
            expected_direction = 1
        else:
            breakthrough_level = pattern_low  
            expected_direction = -1
            
        # 分析结局
        outcome_type, breakthrough_success, price_move_percent, volume_confirm, time_to_outcome, confidence = \
            self._classify_outcome(pattern_data, outcome_data, direction, breakthrough_level, expected_direction)
            
        return OutcomeAnalysis(
            pattern_id=f"pattern_{pattern_idx}",
            outcome_type=outcome_type,
            breakthrough_success=breakthrough_success,
            price_move_percent=price_move_percent,
            volume_confirm=volume_confirm,
            time_to_outcome=time_to_outcome,
            confidence_score=confidence,
            analysis_date=datetime.now()
        )
    
    def _classify_outcome(self, pattern_data: pd.DataFrame, outcome_data: pd.DataFrame, 
                         direction: str, breakthrough_level: float, expected_direction: int) -> Tuple:
        """分类形态结局"""
        
        # 基础数据
        pattern_close = pattern_data['close'].iloc[-1]
        outcome_prices = outcome_data['close']
        outcome_volumes = outcome_data['volume']
        pattern_avg_volume = pattern_data['volume'].mean()
        
        # 检查是否突破关键价位
        if direction == 'up':
            breakthrough = outcome_data['high'].max() > breakthrough_level * 1.001  # 0.1%缓冲
            max_move = (outcome_data['high'].max() - pattern_close) / pattern_close * 100
        else:
            breakthrough = outcome_data['low'].min() < breakthrough_level * 0.999
            max_move = (pattern_close - outcome_data['low'].min()) / pattern_close * 100
            
        # 最终价格变动
        final_price = outcome_prices.iloc[-1]
        final_move = (final_price - pattern_close) / pattern_close * 100 * expected_direction
        
        # 成交量确认
        breakthrough_volume_confirm = outcome_volumes.iloc[:5].mean() > pattern_avg_volume * 1.2
        
        # 时间到结局
        time_to_outcome = len(outcome_data)
        
        # 分类逻辑
        if breakthrough and final_move > 2.0:
            if max_move > 5.0:
                outcome_type = PatternOutcome.STRONG_CONTINUATION
                confidence = 0.9
            else:
                outcome_type = PatternOutcome.STANDARD_CONTINUATION  
                confidence = 0.8
        elif breakthrough and final_move > 0.5:
            outcome_type = PatternOutcome.BREAKTHROUGH_STAGNATION
            confidence = 0.7
        elif breakthrough and final_move < -1.0:
            outcome_type = PatternOutcome.FALSE_BREAKTHROUGH_REVERSAL
            confidence = 0.8
        elif not breakthrough and final_move < -1.0:
            outcome_type = PatternOutcome.INTERNAL_BREAKDOWN
            confidence = 0.7
        else:
            outcome_type = PatternOutcome.REVERSE_MOVEMENT
            confidence = 0.6
            
        return (outcome_type, breakthrough, max_move, breakthrough_volume_confirm, 
                time_to_outcome, confidence)

def generate_outcome_analysis_report(outcomes: List[OutcomeAnalysis]):
    """生成结局分析报告"""
    print("\n生成结局分析报告...")
    
    # 创建报告目录
    os.makedirs('output/reports/outcomes', exist_ok=True)
    
    # 统计分析
    outcome_counts = {}
    total_patterns = len(outcomes)
    
    for outcome in outcomes:
        outcome_type = outcome.outcome_type.value
        outcome_counts[outcome_type] = outcome_counts.get(outcome_type, 0) + 1
    
    print(f"\n=== 形态结局分析汇总 ===")
    print(f"总分析形态数量: {total_patterns}")
    
    print("\n六分类结局统计:")
    for outcome_type, count in outcome_counts.items():
        percentage = count / total_patterns * 100
        print(f"  {outcome_type}: {count} 个 ({percentage:.1f}%)")
    
    # 成功率统计
    successful_outcomes = [o for o in outcomes if o.outcome_type in [
        PatternOutcome.STRONG_CONTINUATION, PatternOutcome.STANDARD_CONTINUATION
    ]]
    success_rate = len(successful_outcomes) / total_patterns * 100
    print(f"\n形态成功率: {success_rate:.1f}%")
    
    # 突破成功率
    breakthrough_success = [o for o in outcomes if o.breakthrough_success]
    breakthrough_rate = len(breakthrough_success) / total_patterns * 100
    print(f"突破成功率: {breakthrough_rate:.1f}%")
    
    # 平均价格变动
    avg_price_move = np.mean([o.price_move_percent for o in outcomes])
    print(f"平均价格变动: {avg_price_move:.2f}%")
    
    # 生成详细CSV报告
    outcome_data = []
    for outcome in outcomes:
        outcome_data.append({
            'pattern_id': outcome.pattern_id,
            'outcome_type': outcome.outcome_type.value,
            'breakthrough_success': outcome.breakthrough_success,
            'price_move_percent': outcome.price_move_percent,
            'volume_confirm': outcome.volume_confirm,
            'time_to_outcome': outcome.time_to_outcome,
            'confidence_score': outcome.confidence_score,
            'analysis_date': outcome.analysis_date.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    outcome_df = pd.DataFrame(outcome_data)
    report_path = 'output/reports/outcomes/pattern_outcome_analysis.csv'
    outcome_df.to_csv(report_path, index=False, encoding='utf-8-sig')
    print(f"\n详细结局分析报告已保存: {report_path}")
    
    # 生成可视化图表
    generate_outcome_charts(outcome_counts, outcomes)
    
    # 保存到数据库
    save_outcomes_to_database(outcomes)
    
def generate_outcome_charts(outcome_counts: Dict, outcomes: List[OutcomeAnalysis]):
    """生成结局分析可视化图表"""
    print("生成结局分析图表...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('形态结局分析报告', fontsize=16, fontweight='bold')
    
    # 1. 结局类型分布饼图
    ax1 = axes[0, 0]
    colors = ['#2E8B57', '#32CD32', '#FFD700', '#FF6347', '#FF4500', '#8B0000']
    labels = list(outcome_counts.keys())
    sizes = list(outcome_counts.values())
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                       colors=colors[:len(labels)], startangle=90)
    ax1.set_title('形态结局类型分布')
    
    # 2. 价格变动分布直方图
    ax2 = axes[0, 1]
    price_moves = [o.price_move_percent for o in outcomes]
    ax2.hist(price_moves, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('价格变动 (%)')
    ax2.set_ylabel('数量')
    ax2.set_title('价格变动分布')
    ax2.axvline(np.mean(price_moves), color='red', linestyle='--', 
                label=f'平均值: {np.mean(price_moves):.2f}%')
    ax2.legend()
    
    # 3. 成功率对比柱状图
    ax3 = axes[1, 0]
    success_categories = ['形态成功', '突破成功', '成交量确认']
    success_rates = [
        len([o for o in outcomes if o.outcome_type in [PatternOutcome.STRONG_CONTINUATION, PatternOutcome.STANDARD_CONTINUATION]]) / len(outcomes) * 100,
        len([o for o in outcomes if o.breakthrough_success]) / len(outcomes) * 100,
        len([o for o in outcomes if o.volume_confirm]) / len(outcomes) * 100
    ]
    
    bars = ax3.bar(success_categories, success_rates, color=['green', 'blue', 'orange'], alpha=0.7)
    ax3.set_ylabel('成功率 (%)')
    ax3.set_title('各类成功率统计')
    
    # 在柱状图上显示数值
    for bar, rate in zip(bars, success_rates):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{rate:.1f}%', ha='center', va='bottom')
    
    # 4. 时间到结局分布
    ax4 = axes[1, 1]
    time_to_outcomes = [o.time_to_outcome for o in outcomes]
    ax4.hist(time_to_outcomes, bins=15, color='lightcoral', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('时间到结局 (K线数)')
    ax4.set_ylabel('数量')
    ax4.set_title('时间到结局分布')
    ax4.axvline(np.mean(time_to_outcomes), color='red', linestyle='--',
                label=f'平均值: {np.mean(time_to_outcomes):.1f}')
    ax4.legend()
    
    plt.tight_layout()
    
    # 保存图表
    chart_path = 'output/reports/outcomes/outcome_analysis_charts.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"结局分析图表已保存: {chart_path}")

def save_outcomes_to_database(outcomes: List[OutcomeAnalysis]):
    """将结局分析结果保存到数据库"""
    print("保存结局分析结果到数据库...")
    
    conn = sqlite3.connect('output/data/patterns.db')
    cursor = conn.cursor()
    
    # 创建结局分析表（如果不存在）
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pattern_outcomes (
            id TEXT PRIMARY KEY,
            pattern_id TEXT,
            outcome_type TEXT,
            breakthrough_success BOOLEAN,
            price_move_percent REAL,
            volume_confirm BOOLEAN,
            time_to_outcome INTEGER,
            confidence_score REAL,
            analysis_date TEXT,
            FOREIGN KEY (pattern_id) REFERENCES patterns(id)
        )
    ''')
    
    # 清除旧的结局分析数据
    cursor.execute('DELETE FROM pattern_outcomes')
    
    # 插入新的结局分析数据
    for outcome in outcomes:
        outcome_id = f"outcome_{outcome.pattern_id}"
        cursor.execute('''
            INSERT OR REPLACE INTO pattern_outcomes 
            (id, pattern_id, outcome_type, breakthrough_success, price_move_percent,
             volume_confirm, time_to_outcome, confidence_score, analysis_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            outcome_id, outcome.pattern_id, outcome.outcome_type.value,
            outcome.breakthrough_success, outcome.price_move_percent,
            outcome.volume_confirm, outcome.time_to_outcome,
            outcome.confidence_score, outcome.analysis_date.strftime('%Y-%m-%d %H:%M:%S')
        ))
    
    conn.commit()
    conn.close()
    
    print(f"已保存 {len(outcomes)} 个结局分析结果到数据库")

def main():
    """主函数"""
    print("=== 形态结局分析与追踪系统 ===")
    
    # 导入数据连接器
    from src.data.connectors.csv_connector import CSVDataConnector
    
    # 初始化数据连接器
    data_connector = CSVDataConnector('data/csv')
    data_connector.connect()
    
    # 创建分析器并执行分析
    analyzer = PatternOutcomeAnalyzer(data_connector)
    outcomes = analyzer.analyze_all_patterns()
    
    if outcomes:
        # 生成分析报告
        generate_outcome_analysis_report(outcomes)
        print("\n✅ 形态结局分析完成！")
    else:
        print("⚠️ 没有找到可分析的形态数据")
    
    data_connector.close()

if __name__ == "__main__":
    main()