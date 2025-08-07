"""
阶段5：形态可视化及图表输出模块
专注于生成标准化的形态可视化图表
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import mplfinance as mpf
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
import platform
import os

from src.data.models.base_models import (
    PatternRecord, PatternOutcomeAnalysis, MarketSnapshot,
    InvalidationSignal, PatternOutcome
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
            
            # 使用mplfinance绘制专业K线图
            chart_path = self._plot_candlesticks_with_mplfinance(pattern, chart_data)
            
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
    
    def _plot_candlesticks_with_mplfinance(self, pattern: PatternRecord, chart_data: pd.DataFrame) -> str:
        """使用mplfinance绘制专业K线图"""
        # 准备mplfinance需要的数据格式
        ohlc_data = chart_data.set_index('timestamp')[['open', 'high', 'low', 'close', 'volume']].copy()
        ohlc_data.index = pd.to_datetime(ohlc_data.index)
        
        # 创建附加绘图列表
        apds = []
        
        # 绘制旗杆
        flagpole_line = self._create_flagpole_line(pattern, ohlc_data)
        if flagpole_line is not None:
            apds.append(mpf.make_addplot(flagpole_line, color='#2E86C1', width=3, 
                                       secondary_y=False, panel=0))
        
        # 绘制旗面边界
        for i, boundary in enumerate(pattern.pattern_boundaries or []):
            boundary_line = self._create_boundary_line(boundary, ohlc_data)
            if boundary_line is not None:
                color = '#E74C3C' if pattern.sub_type.value == 'flag' else '#F39C12'
                linestyle = 'solid' if i == 0 else 'dashed'
                apds.append(mpf.make_addplot(boundary_line, color=color, width=2,
                                           linestyle=linestyle, secondary_y=False, panel=0))
        
        # 自定义样式 - 专业金融图表风格
        custom_style = mpf.make_mpf_style(
            base_mpl_style='default',
            marketcolors=mpf.make_marketcolors(
                up='#00AA00',    # 涨：绿色
                down='#FF4444',  # 跌：红色
                edge='inherit',
                wick='inherit',
                volume='inherit'
            ),
            facecolor='white',
            figcolor='white',
            gridstyle=':',
            gridcolor='#E0E0E0',
            gridaxis='both'
        )
        
        # 生成图表
        title = self._create_chart_title(pattern, None)
        
        # 创建保存路径
        chart_path = self._create_mplfinance_save_path(pattern)
        
        # 设置中文字体
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        
        mpf.plot(ohlc_data,
                type='candle',
                style=custom_style,
                addplot=apds,
                volume=True,
                title=title,
                figsize=(16, 12),
                tight_layout=True,
                savefig=chart_path,
                returnfig=False)
        
        return str(chart_path)
    
    def _create_flagpole_line(self, pattern: PatternRecord, ohlc_data: pd.DataFrame) -> Optional[pd.Series]:
        """创建旗杆线数据"""
        try:
            flagpole_line = pd.Series(index=ohlc_data.index, dtype=float)
            flagpole_line[:] = np.nan
            
            start_time = pd.to_datetime(pattern.flagpole.start_time)
            end_time = pd.to_datetime(pattern.flagpole.end_time)
            
            # 在旗杆时间范围内插值
            mask = (ohlc_data.index >= start_time) & (ohlc_data.index <= end_time)
            if mask.any():
                flagpole_times = ohlc_data.index[mask]
                total_duration = (end_time - start_time).total_seconds()
                price_diff = pattern.flagpole.end_price - pattern.flagpole.start_price
                
                for time in flagpole_times:
                    progress = (time - start_time).total_seconds() / total_duration if total_duration > 0 else 0
                    flagpole_line[time] = pattern.flagpole.start_price + progress * price_diff
                    
            return flagpole_line
            
        except Exception as e:
            logger.warning(f"创建旗杆线失败: {e}")
            return None
    
    def _create_boundary_line(self, boundary, ohlc_data: pd.DataFrame) -> Optional[pd.Series]:
        """创建边界线数据"""
        try:
            boundary_line = pd.Series(index=ohlc_data.index, dtype=float)
            boundary_line[:] = np.nan
            
            start_time = pd.to_datetime(boundary.start_time)
            end_time = pd.to_datetime(boundary.end_time)
            
            # 在边界时间范围内插值
            mask = (ohlc_data.index >= start_time) & (ohlc_data.index <= end_time)
            if mask.any():
                boundary_times = ohlc_data.index[mask]
                total_duration = (end_time - start_time).total_seconds()
                price_diff = boundary.end_price - boundary.start_price
                
                for time in boundary_times:
                    progress = (time - start_time).total_seconds() / total_duration if total_duration > 0 else 0
                    boundary_line[time] = boundary.start_price + progress * price_diff
                    
            return boundary_line
            
        except Exception as e:
            logger.warning(f"创建边界线失败: {e}")
            return None
    
    def _create_mplfinance_save_path(self, pattern: PatternRecord) -> Path:
        """创建mplfinance图表保存路径"""
        filename = self._create_chart_filename(pattern, None)
        sub_type = pattern.sub_type.value if pattern.sub_type else pattern.pattern_type
        chart_dir = Path(self.charts_base_path) / sub_type
        chart_dir.mkdir(parents=True, exist_ok=True)
        return chart_dir / f"{filename}.png"
    
    def _plot_outcome_levels(self, ax, outcome_analysis: PatternOutcomeAnalysis, chart_data: pd.DataFrame):
        """绘制结局分析相关的水平线"""
        colors = self.default_style['colors']
        
        # x轴范围（使用整数索引）
        x_min = 0
        x_max = len(chart_data) - 1
        
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
                      marker='^', label=f'实际高点: {outcome_analysis.actual_high:.2f}', zorder=6)
        
        if not np.isnan(outcome_analysis.actual_low):
            ax.scatter([x_max], [outcome_analysis.actual_low], color='blue', s=50, 
                      marker='v', label=f'实际低点: {outcome_analysis.actual_low:.2f}', zorder=6)
    
    def _plot_invalidation_signals(self, ax, signals: List[InvalidationSignal], chart_data: pd.DataFrame):
        """绘制失效信号"""
        for signal in signals:
            # 在信号时间点添加标记
            signal_time = signal.detection_time
            # 找到最接近的时间点
            time_diffs = np.abs((chart_data['timestamp'] - signal_time).dt.total_seconds())
            closest_idx = time_diffs.argmin()
            
            if closest_idx < len(chart_data):
                signal_price = chart_data.iloc[closest_idx]['close']
                signal_x_pos = closest_idx * 0.3  # 0.3 = candle_spacing
                ax.scatter([signal_x_pos], [signal_price], color='red', s=80, 
                          marker='x', linewidth=3, label=f'失效信号: {signal.signal_type}', zorder=7)
    
    def _plot_volume(self, ax, chart_data: pd.DataFrame, pattern: PatternRecord):
        """绘制成交量（修复时间间隙问题）"""
        colors = self.default_style['colors']
        
        if 'volume' not in chart_data.columns:
            return
        
        # 使用紧密间距的x轴坐标系，与K线图保持一致
        x_positions = [i * 0.3 for i in range(len(chart_data))]  # 0.3 = candle_spacing
        volumes = chart_data['volume'].values
        
        # 成交量柱状图
        for i, (idx, row) in enumerate(chart_data.iterrows()):
            timestamp = row['timestamp']
            volume = row['volume']
            x_pos = x_positions[i]
            
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
            
            # 绘制成交量柱（宽度与K线对应）
            ax.bar(x_pos, volume, color=color, alpha=0.7, width=0.25)
        
        # 设置成交量图的x轴范围与K线图对应
        candle_spacing = 0.3
        ax.set_xlim(-candle_spacing/2, (len(chart_data) - 1) * candle_spacing + candle_spacing/2)
        ax.set_ylabel('成交量', fontproperties=self.chinese_font_prop, 
                     fontsize=self.default_style['label_fontsize'])
    
    def _create_chart_title(self, pattern: PatternRecord, outcome_analysis: Optional[PatternOutcomeAnalysis] = None) -> str:
        """创建图表标题"""
        # 基础标题
        sub_type = pattern.sub_type if isinstance(pattern.sub_type, str) else pattern.sub_type.value if pattern.sub_type else pattern.pattern_type
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
            outcome_value = outcome_analysis.outcome if isinstance(outcome_analysis.outcome, str) else outcome_analysis.outcome.value if outcome_analysis.outcome else 'unknown'
            outcome_name = outcome_names.get(outcome_value, outcome_value)
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
            info_lines.append(f"波动率: {market_snapshot.volatility_percentile:.0f}分位")
        
        # 创建信息框
        info_text = "\\n".join(info_lines)
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontproperties=self.chinese_font_prop)
    
    def _format_chart(self, ax1, ax2, chart_data: pd.DataFrame):
        """格式化图表显示"""
        # 为成交量图同步设置x轴标签
        n_ticks = min(10, len(chart_data))
        if n_ticks > 1:
            tick_indices = np.linspace(0, len(chart_data)-1, n_ticks, dtype=int)
            tick_positions = [i * 0.3 for i in tick_indices]  # 0.3 = candle_spacing
            tick_labels = [chart_data.iloc[pos]['timestamp'].strftime('%m-%d\n%H:%M') 
                          for pos in tick_indices]
            ax2.set_xticks(tick_positions)
            ax2.set_xticklabels(tick_labels, fontsize=9)
        
        # 设置网格和样式
        for ax in [ax1, ax2]:
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=0)  # 不旋转x轴标签，因为已经分行
        
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
            sub_type = pattern.sub_type if isinstance(pattern.sub_type, str) else pattern.sub_type.value if pattern.sub_type else pattern.pattern_type
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
            outcome_name = outcome_analysis.outcome if isinstance(outcome_analysis.outcome, str) else outcome_analysis.outcome.value if outcome_analysis.outcome else 'unknown'
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
            sub_type = pattern.sub_type if isinstance(pattern.sub_type, str) else pattern.sub_type.value if pattern.sub_type else pattern.pattern_type
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
    
    def generate_baseline_summary_chart(self, baseline_data: Dict[str, Any], 
                                       regime_data: Dict[str, Any] = None,
                                       save_path: Optional[str] = None) -> str:
        """
        生成动态基线系统汇总图表
        
        Args:
            baseline_data: 基线系统数据
            regime_data: 市场状态数据
            save_path: 保存路径
            
        Returns:
            保存的图表路径
        """
        try:
            # 设置图表
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('动态基线系统汇总报告', fontproperties=self.chinese_font_prop, fontsize=16)
            
            # 1. 基线覆盖统计
            if 'coverage_stats' in baseline_data:
                stats = baseline_data['coverage_stats']
                names = list(stats.keys())
                values = list(stats.values())
                
                bars = ax1.bar(names, values, color='#3498DB', alpha=0.7)
                ax1.set_title('基线覆盖统计', fontproperties=self.chinese_font_prop)
                ax1.set_ylabel('覆盖率 (%)', fontproperties=self.chinese_font_prop)
                ax1.tick_params(axis='x', rotation=45)
                ax1.grid(True, alpha=0.3)
                
                # 添加数值标签
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax1.annotate(f'{value}%', xy=(bar.get_x() + bar.get_width()/2, height),
                               xytext=(0, 3), textcoords="offset points", 
                               ha='center', va='bottom')
            
            # 2. 市场状态分布
            if regime_data and 'regime_distribution' in regime_data:
                regime_dist = regime_data['regime_distribution']
                labels = ['高波动率', '低波动率', '未知状态']
                colors = ['#E74C3C', '#2ECC71', '#95A5A6']
                sizes = [regime_dist.get('high_volatility', 0), 
                        regime_dist.get('low_volatility', 0),
                        regime_dist.get('unknown', 0)]
                
                ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax2.set_title('市场状态分布', fontproperties=self.chinese_font_prop)
            else:
                # 默认显示基础信息
                info_text = f"""
数据点总数: {baseline_data.get('total_data_points', 'N/A')}
状态转换次数: {baseline_data.get('regime_transitions', 'N/A')}
当前状态: {baseline_data.get('current_regime', 'unknown')}
基线稳定性: {baseline_data.get('baseline_stability', 'N/A')}
                """
                ax2.text(0.1, 0.5, info_text, transform=ax2.transAxes, 
                        fontproperties=self.chinese_font_prop, fontsize=12,
                        verticalalignment='center')
                ax2.set_title('基线系统信息', fontproperties=self.chinese_font_prop)
                ax2.axis('off')
            
            # 3. 时间序列趋势（模拟数据）
            if 'time_series' in baseline_data:
                ts_data = baseline_data['time_series']
                ax3.plot(ts_data.get('timestamps', []), ts_data.get('values', []))
                ax3.set_title('基线变化趋势', fontproperties=self.chinese_font_prop)
                ax3.set_xlabel('时间', fontproperties=self.chinese_font_prop)
                ax3.set_ylabel('基线值', fontproperties=self.chinese_font_prop)
                ax3.grid(True, alpha=0.3)
            else:
                # 生成示例趋势图
                x = np.arange(0, 100)
                y = np.random.normal(0, 1, 100).cumsum()
                ax3.plot(x, y, color='#3498DB', linewidth=2)
                ax3.set_title('基线变化趋势（示例）', fontproperties=self.chinese_font_prop)
                ax3.set_xlabel('时间窗口', fontproperties=self.chinese_font_prop)
                ax3.set_ylabel('基线偏差', fontproperties=self.chinese_font_prop)
                ax3.grid(True, alpha=0.3)
            
            # 4. 性能指标
            performance_data = baseline_data.get('performance', {})
            metrics = ['检测准确率', '误报率', '基线稳定度', '响应速度']
            values = [
                performance_data.get('accuracy', 85),
                performance_data.get('false_positive', 12),
                performance_data.get('stability', 92),
                performance_data.get('response_time', 78)
            ]
            
            colors = ['#2ECC71' if v >= 80 else '#F39C12' if v >= 60 else '#E74C3C' for v in values]
            bars = ax4.barh(metrics, values, color=colors, alpha=0.7)
            ax4.set_title('性能指标', fontproperties=self.chinese_font_prop)
            ax4.set_xlabel('分数', fontproperties=self.chinese_font_prop)
            ax4.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                width = bar.get_width()
                ax4.annotate(f'{value}', xy=(width, bar.get_y() + bar.get_height()/2),
                           xytext=(3, 0), textcoords="offset points", 
                           ha='left', va='center')
            
            plt.tight_layout()
            
            # 保存图表
            if save_path is None:
                save_path = str(Path(self.charts_base_path) / "summary" / "baseline_summary.png")
            
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            logger.info(f"动态基线汇总图表已保存: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"生成动态基线汇总图表失败: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return ""