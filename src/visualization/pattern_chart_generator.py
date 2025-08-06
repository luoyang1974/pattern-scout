"""
阶段5：形态可视化及图表输出模块
专注于生成标准化的形态可视化图表
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import mplfinance as mpf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path
import platform
import os

from src.data.models.base_models import (
    PatternRecord, PatternOutcomeAnalysis, MarketSnapshot,
    InvalidationSignal, FlagSubType, PatternOutcome
)
from src.data.connectors.csv_connector import CSVDataConnector
from loguru import logger


class PatternChartGenerator:
    """
    形态图表生成器（阶段5）
    专注于可视化图表生成
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化图表生成器
        
        Args:
            config: 图表配置参数
        """
        self.config = config or {}
        self._setup_chinese_fonts()
        self._setup_default_styles()
        
        # 图表输出路径
        self.charts_base_path = self.config.get("charts_path", "output/charts")
        Path(self.charts_base_path).mkdir(parents=True, exist_ok=True)
        
        # 初始化数据连接器
        try:
            self.data_connector = CSVDataConnector("data/csv/")
            self.data_connector.connect()
        except Exception as e:
            logger.warning(f"数据连接器初始化失败: {e}")
            self.data_connector = None
    
    def _setup_chinese_fonts(self):
        """配置中文字体支持"""
        try:
            if platform.system() == 'Windows':
                # Windows系统字体
                font_dirs = [r'C:\Windows\Fonts', r'C:\Windows\System32\Fonts']
                chinese_font_files = ['msyh.ttc', 'simhei.ttf', 'simsun.ttc', 'kaiti.ttf']
                
                font_found = False
                for font_dir in font_dirs:
                    if not os.path.exists(font_dir):
                        continue
                    for font_file in chinese_font_files:
                        font_path = os.path.join(font_dir, font_file)
                        if os.path.exists(font_path):
                            try:
                                self.chinese_font_prop = font_manager.FontProperties(fname=font_path, size=10)
                                mpl.rcParams['font.sans-serif'] = [font_manager.FontProperties(fname=font_path).get_name()]
                                mpl.rcParams['axes.unicode_minus'] = False
                                font_found = True
                                logger.info(f"中文字体加载成功: {font_file}")
                                break
                            except Exception as e:
                                logger.warning(f"加载字体 {font_file} 失败: {e}")
                                continue
                    if font_found:
                        break
                
                if not font_found:
                    logger.warning("未找到中文字体，使用系统默认字体")
                    self.chinese_font_prop = font_manager.FontProperties(size=10)
            else:
                # 非Windows系统
                self.chinese_font_prop = font_manager.FontProperties(size=10)
                
        except Exception as e:
            logger.error(f"中文字体配置失败: {e}")
            self.chinese_font_prop = font_manager.FontProperties(size=10)
    
    def _setup_default_styles(self):
        """设置默认样式"""
        self.default_style = {
            'figure_size': (16, 12),
            'dpi': 100,
            'title_fontsize': 14,
            'label_fontsize': 12,
            'legend_fontsize': 10,
            'line_width': 2,
            'colors': {
                'flagpole': '#2E86C1',      # 蓝色
                'flag_boundary': '#E74C3C', # 红色
                'pennant_boundary': '#F39C12', # 橙色
                'breakout_level': '#27AE60', # 绿色
                'invalidation_level': '#8E44AD', # 紫色
                'target_projection': '#F1C40F', # 黄色
                'volume_high': '#3498DB',    # 浅蓝
                'volume_low': '#95A5A6'     # 灰色
            }
        }
    
    def generate_pattern_chart(self, pattern: PatternRecord,
                             df: pd.DataFrame,
                             outcome_analysis: Optional[PatternOutcomeAnalysis] = None,
                             market_snapshot: Optional[MarketSnapshot] = None,
                             invalidation_signals: Optional[List[InvalidationSignal]] = None,
                             save_chart: bool = True) -> str:
        """
        生成单个形态的标准化图表
        
        Args:
            pattern: 形态记录
            df: OHLCV数据
            outcome_analysis: 结局分析（可选）
            market_snapshot: 市场快照（可选）
            invalidation_signals: 失效信号列表（可选）
            save_chart: 是否保存图表
            
        Returns:
            图表文件路径
        """
        try:
            # 准备数据
            chart_data = self._prepare_chart_data(pattern, df)
            
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.default_style['figure_size'], 
                                          dpi=self.default_style['dpi'],
                                          gridspec_kw={'height_ratios': [3, 1]})
            
            # 绘制K线图和形态
            self._plot_candlesticks(ax1, chart_data)
            self._plot_pattern_elements(ax1, pattern, chart_data)
            
            # 绘制关键水平线（如果有结局分析）
            if outcome_analysis:
                self._plot_outcome_levels(ax1, outcome_analysis, chart_data)
            
            # 绘制失效信号
            if invalidation_signals:
                self._plot_invalidation_signals(ax1, invalidation_signals, chart_data)
            
            # 绘制成交量
            self._plot_volume(ax2, chart_data, pattern)
            
            # 设置标题和标签
            title = self._create_chart_title(pattern, outcome_analysis)
            ax1.set_title(title, fontproperties=self.chinese_font_prop, 
                         fontsize=self.default_style['title_fontsize'], pad=20)
            
            # 添加信息框
            self._add_info_box(ax1, pattern, outcome_analysis, market_snapshot)
            
            # 格式化图表
            self._format_chart(ax1, ax2, chart_data)
            
            # 保存图表
            chart_path = ""\n            if save_chart:
                chart_path = self._save_chart(fig, pattern, outcome_analysis)
            
            plt.close(fig)
            
            return chart_path
            
        except Exception as e:
            logger.error(f"生成形态图表失败 {pattern.id}: {e}")
            return ""
    
    def _prepare_chart_data(self, pattern: PatternRecord, df: pd.DataFrame) -> pd.DataFrame:
        """
        准备图表数据
        包含形态前后的上下文窗口
        """
        # 确定时间窗口
        flagpole_start = pattern.flagpole.start_time
        pattern_end = pattern.pattern_boundaries[-1].end_time if pattern.pattern_boundaries else pattern.flagpole.end_time
        
        # 上下文窗口：形态发生前的 pole_bars_count * 2 根K线，形态后的适当数据
        pole_duration = (pattern.flagpole.end_time - flagpole_start).total_seconds() / 60  # 分钟
        context_before = flagpole_start - pd.Timedelta(minutes=pole_duration * 2)
        context_after = pattern_end + pd.Timedelta(minutes=pole_duration * 4)
        
        # 过滤数据
        chart_data = df[(df['timestamp'] >= context_before) & (df['timestamp'] <= context_after)].copy()
        
        if chart_data.empty:
            logger.warning(f"图表数据为空，使用全部数据 {pattern.id}")
            chart_data = df.copy()
        
        # 确保时间列为datetime类型
        chart_data['timestamp'] = pd.to_datetime(chart_data['timestamp'])
        chart_data = chart_data.sort_values('timestamp')
        
        return chart_data
    
    def _plot_candlesticks(self, ax, chart_data: pd.DataFrame):
        """绘制K线图"""
        # 使用matplotlib绘制K线
        for idx, row in chart_data.iterrows():
            timestamp = row['timestamp']
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']
            
            # K线颜色
            color = '#E74C3C' if close_price < open_price else '#27AE60'  # 红跌绿涨
            
            # 绘制上下影线
            ax.plot([timestamp, timestamp], [low_price, high_price], color='black', linewidth=1)
            
            # 绘制实体
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            rect = plt.Rectangle((timestamp, body_bottom), pd.Timedelta(minutes=10), body_height,
                               facecolor=color, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
    
    def _plot_pattern_elements(self, ax, pattern: PatternRecord, chart_data: pd.DataFrame):
        """绘制形态元素"""
        colors = self.default_style['colors']
        
        # 绘制旗杆
        flagpole_start = pattern.flagpole.start_time
        flagpole_end = pattern.flagpole.end_time
        flagpole_start_price = pattern.flagpole.start_price
        flagpole_end_price = pattern.flagpole.end_price
        
        ax.plot([flagpole_start, flagpole_end], [flagpole_start_price, flagpole_end_price],
               color=colors['flagpole'], linewidth=self.default_style['line_width'] + 1,
               label='旗杆', marker='o', markersize=6)
        
        # 绘制形态边界
        if pattern.pattern_boundaries:
            for i, boundary in enumerate(pattern.pattern_boundaries):
                color_key = 'flag_boundary' if pattern.sub_type == FlagSubType.FLAG else 'pennant_boundary'
                line_style = '-' if i == 0 else '--'  # 上边界实线，下边界虚线
                
                ax.plot([boundary.start_time, boundary.end_time], 
                       [boundary.start_price, boundary.end_price],
                       color=colors[color_key], linestyle=line_style,
                       linewidth=self.default_style['line_width'],
                       label=f'{"上边界" if i == 0 else "下边界"}')\n    \n    def _plot_outcome_levels(self, ax, outcome_analysis: PatternOutcomeAnalysis, chart_data: pd.DataFrame):
        """绘制结局分析相关的水平线"""
        colors = self.default_style['colors']
        
        # 时间范围
        x_min = chart_data['timestamp'].min()
        x_max = chart_data['timestamp'].max()
        
        # 突破监测位
        ax.axhline(y=outcome_analysis.breakout_level, color=colors['breakout_level'], 
                  linestyle='--', linewidth=1.5, alpha=0.8, label='突破监测位')
        
        # 形态失效位
        ax.axhline(y=outcome_analysis.invalidation_level, color=colors['invalidation_level'], 
                  linestyle='--', linewidth=1.5, alpha=0.8, label='形态失效位')
        
        # 目标映射一
        ax.axhline(y=outcome_analysis.target_projection_1, color=colors['target_projection'], 
                  linestyle=':', linewidth=1.5, alpha=0.8, label='目标映射一')
        
        # 实际高低点标记
        if not np.isnan(outcome_analysis.actual_high):
            ax.scatter([x_max], [outcome_analysis.actual_high], color='red', s=50, 
                      marker='^', label=f'实际高点: {outcome_analysis.actual_high:.2f}')
        
        if not np.isnan(outcome_analysis.actual_low):
            ax.scatter([x_max], [outcome_analysis.actual_low], color='blue', s=50, 
                      marker='v', label=f'实际低点: {outcome_analysis.actual_low:.2f}')
    
    def _plot_invalidation_signals(self, ax, signals: List[InvalidationSignal], chart_data: pd.DataFrame):
        """绘制失效信号"""
        for signal in signals:
            # 在信号时间点添加标记
            signal_time = signal.detection_time
            # 找到对应的价格数据
            signal_data = chart_data[chart_data['timestamp'] <= signal_time]
            if not signal_data.empty:
                signal_price = signal_data.iloc[-1]['close']
                ax.scatter([signal_time], [signal_price], color='red', s=80, 
                          marker='x', linewidth=3, label=f'失效信号: {signal.signal_type.value}')
    
    def _plot_volume(self, ax, chart_data: pd.DataFrame, pattern: PatternRecord):
        """绘制成交量"""
        colors = self.default_style['colors']
        
        # 成交量柱状图
        for idx, row in chart_data.iterrows():
            timestamp = row['timestamp']
            volume = row['volume']
            
            # 判断是否在旗杆或旗面区间内
            in_flagpole = (pattern.flagpole.start_time <= timestamp <= pattern.flagpole.end_time)
            in_flag = False
            if pattern.pattern_boundaries:
                pattern_start = pattern.flagpole.end_time
                pattern_end = pattern.pattern_boundaries[-1].end_time
                in_flag = (pattern_start <= timestamp <= pattern_end)
            
            # 选择颜色
            if in_flagpole:
                color = colors['volume_high']  # 旗杆区间高亮
            elif in_flag:
                color = colors['volume_low']   # 旗面区间低亮
            else:
                color = '#BDC3C7'  # 其他区间灰色
            
            ax.bar(timestamp, volume, color=color, alpha=0.7, width=pd.Timedelta(minutes=10))
        
        ax.set_ylabel('成交量', fontproperties=self.chinese_font_prop, 
                     fontsize=self.default_style['label_fontsize'])
    
    def _create_chart_title(self, pattern: PatternRecord, outcome_analysis: Optional[PatternOutcomeAnalysis] = None) -> str:
        """创建图表标题"""
        # 基础标题
        sub_type = pattern.sub_type.value if pattern.sub_type else pattern.pattern_type
        type_names = {
            'flag': '矩形旗形',
            'pennant': '三角旗形'
        }
        pattern_name = type_names.get(sub_type, sub_type)
        direction = '上升' if pattern.flagpole.direction == 'up' else '下降'
        
        title = f"ID-{pattern.id}: {pattern.symbol} - {direction}{pattern_name}"
        
        # 添加结局信息
        if outcome_analysis:
            outcome_names = {
                PatternOutcome.STRONG_CONTINUATION.value: '强势延续',
                PatternOutcome.STANDARD_CONTINUATION.value: '标准延续',
                PatternOutcome.BREAKOUT_STAGNATION.value: '突破停滞',
                PatternOutcome.FAILED_BREAKOUT.value: '假突破反转',
                PatternOutcome.INTERNAL_COLLAPSE.value: '内部瓦解',
                PatternOutcome.OPPOSITE_RUN.value: '反向运行'
            }
            outcome_name = outcome_names.get(outcome_analysis.outcome.value, outcome_analysis.outcome.value)
            title += f" - 结局: {outcome_name}"
        
        return title
    
    def _add_info_box(self, ax, pattern: PatternRecord, 
                     outcome_analysis: Optional[PatternOutcomeAnalysis] = None,
                     market_snapshot: Optional[MarketSnapshot] = None):
        """添加信息框显示关键指标"""
        info_lines = []
        
        # 形态DNA信息
        info_lines.append(f"置信度: {pattern.confidence_score:.2f}")
        info_lines.append(f"质量等级: {pattern.pattern_quality}")
        info_lines.append(f"旗杆高度: {pattern.flagpole.height_percent:.1f}%")
        info_lines.append(f"成交量比率: {pattern.flagpole.volume_ratio:.1f}")
        
        # 结局分析信息
        if outcome_analysis:
            info_lines.append("") # 空行分隔
            info_lines.append(f"监控期: {outcome_analysis.monitoring_duration}根K线")
            if outcome_analysis.success_ratio:
                info_lines.append(f"成功比率: {outcome_analysis.success_ratio:.2f}")
            if outcome_analysis.final_return:
                info_lines.append(f"最终回报: {outcome_analysis.final_return:.2f}")
        
        # 市场环境信息
        if market_snapshot:
            info_lines.append("") # 空行分隔
            info_lines.append(f"市场状态: {market_snapshot.regime.value}")
            info_lines.append(f"波动率: {market_snapshot.volatility_percentile:.0f}分位")\n        
        # 创建信息框
        info_text = "\\n".join(info_lines)
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontproperties=self.chinese_font_prop)
    
    def _format_chart(self, ax1, ax2, chart_data: pd.DataFrame):
        """格式化图表显示"""
        # 设置x轴时间格式
        import matplotlib.dates as mdates
        
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        # 设置y轴标签
        ax1.set_ylabel('价格', fontproperties=self.chinese_font_prop, 
                      fontsize=self.default_style['label_fontsize'])
        
        # 图例
        ax1.legend(prop=self.chinese_font_prop, fontsize=self.default_style['legend_fontsize'],
                  loc='upper left')
        
        # 紧凑布局
        plt.tight_layout()
    
    def _save_chart(self, fig, pattern: PatternRecord, 
                   outcome_analysis: Optional[PatternOutcomeAnalysis] = None) -> str:
        """保存图表文件"""
        try:
            # 创建文件名
            filename = self._create_chart_filename(pattern, outcome_analysis)
            
            # 确定输出路径
            sub_type = pattern.sub_type.value if pattern.sub_type else pattern.pattern_type
            chart_dir = Path(self.charts_base_path) / sub_type
            chart_dir.mkdir(parents=True, exist_ok=True)
            
            chart_path = chart_dir / f"{filename}.png"
            
            # 保存图表
            fig.savefig(chart_path, dpi=self.default_style['dpi'], 
                       bbox_inches='tight', facecolor='white')
            
            logger.info(f"图表保存成功: {chart_path}")
            return str(chart_path)
            
        except Exception as e:
            logger.error(f"保存图表失败: {e}")
            return ""
    
    def _create_chart_filename(self, pattern: PatternRecord, 
                              outcome_analysis: Optional[PatternOutcomeAnalysis] = None) -> str:
        """创建图表文件名"""
        # 基础文件名：形态ID_结局分类_确认时间戳
        base_name = pattern.id
        
        if outcome_analysis:
            outcome_name = outcome_analysis.outcome.value
            timestamp = outcome_analysis.analysis_date.strftime('%Y%m%d_%H%M')
            filename = f"{base_name}_{outcome_name}_{timestamp}"
        else:
            timestamp = pattern.detection_date.strftime('%Y%m%d_%H%M')
            filename = f"{base_name}_detected_{timestamp}"
        
        # 清理文件名
        illegal_chars = '<>:"/\\|?*'
        for char in illegal_chars:
            filename = filename.replace(char, '_')
        
        return filename
    
    def generate_summary_chart(self, patterns: List[PatternRecord],
                              outcomes: List[PatternOutcomeAnalysis] = None,
                              save_chart: bool = True) -> str:
        """
        生成汇总图表
        
        Args:
            patterns: 形态记录列表
            outcomes: 结局分析列表
            save_chart: 是否保存图表
            
        Returns:
            图表文件路径
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 形态类型分布
            self._plot_pattern_type_distribution(ax1, patterns)
            
            # 置信度分布
            self._plot_confidence_distribution(ax2, patterns)
            
            # 结局分布（如果有）
            if outcomes:
                self._plot_outcome_distribution(ax3, outcomes)
                self._plot_success_metrics(ax4, outcomes)
            else:
                ax3.text(0.5, 0.5, '暂无结局数据', ha='center', va='center',
                        fontproperties=self.chinese_font_prop, fontsize=14)
                ax4.text(0.5, 0.5, '暂无结局数据', ha='center', va='center',
                        fontproperties=self.chinese_font_prop, fontsize=14)
            
            # 设置总标题
            fig.suptitle(f'形态识别汇总报告 - 共{len(patterns)}个形态', 
                        fontproperties=self.chinese_font_prop, fontsize=16)
            
            plt.tight_layout()
            
            # 保存图表
            chart_path = ""
            if save_chart:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                summary_dir = Path(self.charts_base_path) / "summary"
                summary_dir.mkdir(parents=True, exist_ok=True)
                chart_path = summary_dir / f"summary_{timestamp}.png"
                
                fig.savefig(chart_path, dpi=self.default_style['dpi'], 
                           bbox_inches='tight', facecolor='white')
                logger.info(f"汇总图表保存成功: {chart_path}")
            
            plt.close(fig)
            return str(chart_path)
            
        except Exception as e:
            logger.error(f"生成汇总图表失败: {e}")
            return ""
    
    def _plot_pattern_type_distribution(self, ax, patterns: List[PatternRecord]):
        """绘制形态类型分布"""
        type_counts = {}
        for pattern in patterns:
            sub_type = pattern.sub_type.value if pattern.sub_type else pattern.pattern_type
            type_counts[sub_type] = type_counts.get(sub_type, 0) + 1
        
        types = list(type_counts.keys())
        counts = list(type_counts.values())
        
        ax.pie(counts, labels=types, autopct='%1.1f%%', startangle=90)
        ax.set_title('形态类型分布', fontproperties=self.chinese_font_prop)
    
    def _plot_confidence_distribution(self, ax, patterns: List[PatternRecord]):
        """绘制置信度分布"""
        confidences = [p.confidence_score for p in patterns]
        
        ax.hist(confidences, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('置信度', fontproperties=self.chinese_font_prop)
        ax.set_ylabel('数量', fontproperties=self.chinese_font_prop)
        ax.set_title('置信度分布', fontproperties=self.chinese_font_prop)
        ax.grid(True, alpha=0.3)
    
    def _plot_outcome_distribution(self, ax, outcomes: List[PatternOutcomeAnalysis]):
        """绘制结局分布"""
        outcome_counts = {}
        outcome_names = {
            PatternOutcome.STRONG_CONTINUATION.value: '强势延续',
            PatternOutcome.STANDARD_CONTINUATION.value: '标准延续',
            PatternOutcome.BREAKOUT_STAGNATION.value: '突破停滞',
            PatternOutcome.FAILED_BREAKOUT.value: '假突破反转',
            PatternOutcome.INTERNAL_COLLAPSE.value: '内部瓦解',
            PatternOutcome.OPPOSITE_RUN.value: '反向运行'
        }
        
        for outcome in outcomes:
            outcome_key = outcome.outcome.value
            outcome_name = outcome_names.get(outcome_key, outcome_key)
            outcome_counts[outcome_name] = outcome_counts.get(outcome_name, 0) + 1
        
        names = list(outcome_counts.keys())
        counts = list(outcome_counts.values())
        
        colors = ['#2ECC71', '#3498DB', '#F39C12', '#E74C3C', '#9B59B6', '#1ABC9C']
        ax.bar(names, counts, color=colors[:len(names)])
        ax.set_xlabel('结局类型', fontproperties=self.chinese_font_prop)
        ax.set_ylabel('数量', fontproperties=self.chinese_font_prop)
        ax.set_title('结局分布', fontproperties=self.chinese_font_prop)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    def _plot_success_metrics(self, ax, outcomes: List[PatternOutcomeAnalysis]):
        """绘制成功指标"""
        success_outcomes = [
            PatternOutcome.STRONG_CONTINUATION.value,
            PatternOutcome.STANDARD_CONTINUATION.value
        ]
        
        total = len(outcomes)
        successful = len([o for o in outcomes if o.outcome.value in success_outcomes])
        success_rate = successful / total if total > 0 else 0
        
        # 成功率饼图
        labels = ['成功', '失败']
        sizes = [successful, total - successful]
        colors = ['#2ECC71', '#E74C3C']
        
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title(f'成功率: {success_rate:.1%}', fontproperties=self.chinese_font_prop)
    
    def batch_generate_charts(self, patterns: List[PatternRecord],
                             df_dict: Dict[str, pd.DataFrame],
                             outcomes: Optional[List[PatternOutcomeAnalysis]] = None,
                             snapshots: Optional[List[MarketSnapshot]] = None,
                             signals_list: Optional[List[List[InvalidationSignal]]] = None) -> Dict[str, str]:
        """
        批量生成形态图表
        
        Args:
            patterns: 形态记录列表
            df_dict: symbol到数据的映射
            outcomes: 结局分析列表
            snapshots: 市场快照列表
            signals_list: 失效信号列表的列表
            
        Returns:
            形态ID到图表路径的映射
        """
        chart_paths = {}
        
        logger.info(f"开始批量生成 {len(patterns)} 个形态图表")
        
        for i, pattern in enumerate(patterns):
            try:
                # 获取对应数据
                symbol = pattern.symbol
                df = df_dict.get(symbol)
                if df is None:
                    logger.warning(f"未找到 {symbol} 的数据，跳过图表生成")
                    continue
                
                outcome = outcomes[i] if outcomes and i < len(outcomes) else None
                snapshot = snapshots[i] if snapshots and i < len(snapshots) else None
                signals = signals_list[i] if signals_list and i < len(signals_list) else None
                
                # 生成图表
                chart_path = self.generate_pattern_chart(
                    pattern, df, outcome, snapshot, signals
                )
                
                if chart_path:
                    chart_paths[pattern.id] = chart_path
                
            except Exception as e:
                logger.error(f"生成形态 {pattern.id} 图表失败: {e}")
                continue
        
        # 生成汇总图表
        summary_path = self.generate_summary_chart(patterns, outcomes)
        if summary_path:
            chart_paths['summary'] = summary_path
        
        logger.info(f"批量图表生成完成: {len(chart_paths)} 个图表")
        return chart_paths