import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import mplfinance as mpf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

from src.data.models.base_models import (
    PatternRecord, TrendLine, PatternOutcomeAnalysis, 
    MarketSnapshot, InvalidationSignal, FlagSubType
)
from src.data.connectors.csv_connector import CSVDataConnector
from src.storage.dataset_manager import DatasetManager
from src.visualization.dynamic_chart_methods import DynamicChartMethods
from loguru import logger
import json


class DynamicChartGenerator(DynamicChartMethods):
    """动态基线系统图表生成器"""

    def __init__(self, config: Dict[str, Any] = None):
        # 配置中文字体支持
        self._setup_chinese_fonts()
        """
        初始化图表生成器

        Args:
            config: 图表配置参数
        """
        # 如果传入的config有'output'键，则提取output配置
        if config and "output" in config:
            self.config = config
        else:
            # 否则直接使用传入的配置作为output配置
            self.config = {"output": config or {}}

        # 合并默认配置
        default_config = self._get_default_config()
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value
            elif isinstance(value, dict) and isinstance(self.config[key], dict):
                # 深度合并字典
                for sub_key, sub_value in value.items():
                    if sub_key not in self.config[key]:
                        self.config[key][sub_key] = sub_value
        
        # 初始化可视化层级
        self.visualization_layers = {
            'baseline_indicators': True,
            'market_regime': True,
            'invalidation_signals': True,
            'outcome_tracking': True,
            'dynamic_thresholds': True
        }

        # 初始化数据连接器
        self.data_connector = CSVDataConnector("data/csv/")
        
        # 初始化中文字体属性
        self.chinese_font_prop = getattr(self, 'chinese_font_prop', None)
        try:
            self.data_connector.connect()
        except Exception as e:
            logger.warning(f"Failed to connect data connector: {e}")

        # 设置图表输出路径
        self.charts_base_path = self.config.get("charts_path", "output/charts")
    
    def _setup_chinese_fonts(self):
        """配置中文字体支持"""
        try:
            # Windows系统的字体路径
            import platform
            import os
            
            if platform.system() == 'Windows':
                # Windows系统字体目录
                font_dirs = [
                    r'C:\Windows\Fonts',
                    r'C:\Windows\System32\Fonts'
                ]
                
                # 尝试直接加载中文字体文件
                chinese_font_files = [
                    'msyh.ttc',      # 微软雅黑
                    'simhei.ttf',    # 黑体
                    'simsun.ttc',    # 宋体
                    'kaiti.ttf',     # 楷体
                ]
                
                font_found = False
                for font_dir in font_dirs:
                    if not os.path.exists(font_dir):
                        continue
                    for font_file in chinese_font_files:
                        font_path = os.path.join(font_dir, font_file)
                        if os.path.exists(font_path):
                            # 直接使用字体文件路径
                            from matplotlib.font_manager import FontProperties
                            self.chinese_font_prop = FontProperties(fname=font_path)
                            
                            # 设置全局字体
                            plt.rcParams['font.family'] = ['sans-serif']
                            plt.rcParams['font.sans-serif'] = [self.chinese_font_prop.get_name(), 'SimHei', 'Microsoft YaHei']
                            plt.rcParams['axes.unicode_minus'] = False
                            mpl.rcParams['font.family'] = ['sans-serif']
                            mpl.rcParams['font.sans-serif'] = [self.chinese_font_prop.get_name(), 'SimHei', 'Microsoft YaHei']
                            mpl.rcParams['axes.unicode_minus'] = False
                            
                            logger.info(f"Chinese font configured with file: {font_path}")
                            font_found = True
                            break
                    if font_found:
                        break
                
                if not font_found:
                    # 如果没有找到字体文件，尝试使用系统字体名
                    self._setup_system_fonts()
            else:
                # 非Windows系统
                self._setup_system_fonts()
                
        except Exception as e:
            logger.error(f"Failed to setup Chinese fonts: {e}")
            # 使用默认字体
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']
            self.chinese_font_prop = None
    
    def _setup_system_fonts(self):
        """设置系统字体"""
        # 设置matplotlib全局中文字体
        # 尝试常见的中文字体
        chinese_fonts = [
            'SimHei',  # 黑体
            'Microsoft YaHei',  # 微软雅黑
            'KaiTi',  # 楷体
            'FangSong',  # 仿宋
            'SimSun',  # 宋体
            'Arial Unicode MS',  # macOS
            'PingFang SC',  # macOS
            'Noto Sans CJK SC',  # Linux
            'Source Han Sans SC'  # Linux
        ]
        
        # 查找可用的中文字体
        available_fonts = [f.name for f in font_manager.fontManager.ttflist]
        chinese_font = None
        
        for font in chinese_fonts:
            if font in available_fonts:
                chinese_font = font
                break
        
        if chinese_font:
            # 设置全局中文字体
            plt.rcParams['font.sans-serif'] = [chinese_font]
            plt.rcParams['axes.unicode_minus'] = False
            mpl.rcParams['font.sans-serif'] = [chinese_font]
            mpl.rcParams['axes.unicode_minus'] = False
            
            from matplotlib.font_manager import FontProperties
            self.chinese_font_prop = FontProperties(family=chinese_font)
            logger.info(f"Chinese font configured: {chinese_font}")
        else:
            # 如果没有找到中文字体，使用DejaVu Sans作为备选
            logger.warning("No Chinese font found, using DejaVu Sans")
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']
            self.chinese_font_prop = None

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "output": {"width": 1600, "height": 1000, "dpi": 300, "format": "png"},
            "style": {
                "background_color": "white",
                "grid_color": "#e0e0e0",
                "text_color": "#333333",
                "up_color": "#26a69a",
                "down_color": "#ef5350",
                "volume_color": "#42a5f5",
                "flagpole_color": "#ff6b35",
                "flag_color": "#2196f3",
                "boundary_color": "#ff9800",
                "breakthrough_color": "#F39C12",
            },
            "pattern_types": {
                "flag_pattern": {
                    "name_cn": "旗形形态",
                    "color_flagpole": "#ff6b35",
                    "color_upper": "#2196f3",
                    "color_lower": "#ff9800",
                    "subtypes": {
                        "flag": {
                            "name_cn": "矩形旗",
                            "color_boundary": "#2196f3"
                        },
                        "pennant": {
                            "name_cn": "三角旗", 
                            "color_boundary": "#4caf50"
                        }
                    }
                }
            },
            "dynamic_baseline": {
                "regime_colors": {
                    "high_volatility": "#ff5722",
                    "low_volatility": "#4caf50",
                    "unknown": "#9e9e9e"
                },
                "threshold_colors": {
                    "slope_score": "#ff9800",
                    "volume_burst": "#e91e63",
                    "retrace_depth": "#9c27b0"
                },
                "invalidation_colors": {
                    "fake_breakout": "#f44336",
                    "volatility_decay": "#ff5722",
                    "volume_divergence": "#795548"
                }
            },
            "outcome_tracking": {
                "outcome_colors": {
                    "strong_continuation": "#4caf50",
                    "standard_continuation": "#8bc34a",
                    "breakout_stagnation": "#ff9800",
                    "failed_breakout": "#f44336",
                    "internal_collapse": "#9c27b0",
                    "opposite_run": "#e91e63",
                    "monitoring": "#607d8b"
                }
            },
        }

    def generate_pattern_chart(
        self,
        df: pd.DataFrame,
        pattern: PatternRecord,
        output_path: Optional[str] = None,
    ) -> str:
        """
        生成形态图表

        Args:
            df: 价格数据
            pattern: 形态记录
            output_path: 输出路径

        Returns:
            生成的图表文件路径
        """
        try:
            if self.config["output"]["format"].lower() == "html":
                return self._generate_interactive_chart(df, pattern, output_path)
            else:
                return self._generate_static_chart(df, pattern, output_path)
        except Exception as e:
            logger.error(f"Failed to generate chart: {e}")
            raise

    def _generate_static_chart(
        self,
        df: pd.DataFrame,
        pattern: PatternRecord,
        output_path: Optional[str] = None,
    ) -> str:
        """生成静态图表（使用matplotlib和mplfinance）"""

        # 准备数据
        chart_df = self._prepare_chart_data(df, pattern)

        # 设置TradingView风格
        style = mpf.make_mpf_style(
            base_mpf_style="charles",
            marketcolors=mpf.make_marketcolors(
                up=self.config["style"]["up_color"],
                down=self.config["style"]["down_color"],
                edge={
                    "up": self.config["style"]["up_color"],
                    "down": self.config["style"]["down_color"],
                },
                wick={
                    "up": self.config["style"]["up_color"],
                    "down": self.config["style"]["down_color"],
                },
                volume=self.config["style"]["volume_color"],
            ),
            gridstyle="--",
            gridcolor=self.config["style"]["grid_color"],
            facecolor=self.config["style"]["background_color"],
            figcolor=self.config["style"]["background_color"],
        )

        # 创建附加图线
        apds = self._create_additional_plots(chart_df, pattern)

        # 确保输出路径正确
        if not output_path:
            Path(self.charts_base_path).mkdir(parents=True, exist_ok=True)
            output_path = f"{self.charts_base_path}/{pattern.id}.{self.config['output']['format']}"
        else:
            # 确保输出目录存在
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # 生成图表
        fig, axes = mpf.plot(
            chart_df,
            type="candle",
            style=style,
            title=f"{pattern.symbol} - {pattern.pattern_type.title()} Pattern (ID: {pattern.id[:8]})",
            volume=True,
            figsize=(16, 10),
            addplot=apds,
            returnfig=True,
        )

        # 添加自定义标注
        self._add_pattern_annotations(axes, chart_df, pattern)

        # 保存图表
        fig.savefig(output_path, dpi=self.config["output"]["dpi"], bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Static chart saved: {output_path}")
        return output_path
    
    def _generate_dynamic_static_chart(
        self,
        df: pd.DataFrame,
        pattern: PatternRecord,
        market_snapshot: Optional[MarketSnapshot] = None,
        outcome_analysis: Optional[PatternOutcomeAnalysis] = None,
        baseline_data: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """生成动态基线系统静态图表"""
        
        # 准备数据
        chart_df = self._prepare_chart_data(df, pattern)
        
        # 设置动态基线风格
        style = self._create_dynamic_baseline_style()
        
        # 创建附加图线
        apds = self._create_dynamic_additional_plots(
            chart_df, pattern, market_snapshot, baseline_data
        )
        
        # 确保输出路径正确
        if not output_path:
            Path(self.charts_base_path).mkdir(parents=True, exist_ok=True)
            sub_type_name = pattern.sub_type if hasattr(pattern, 'sub_type') else 'flag'
            output_path = f"{self.charts_base_path}/{sub_type_name}/{pattern.id}.{self.config['output']['format']}"
        else:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 创建标题
        title = self._create_dynamic_chart_title(pattern, market_snapshot)
        
        # 生成图表
        fig, axes = mpf.plot(
            chart_df,
            type="candle",
            style=style,
            title=title,
            volume=True,
            figsize=(18, 12),
            addplot=apds,
            returnfig=True,
        )
        
        # 添加动态基线标注
        self._add_dynamic_pattern_annotations(
            axes, chart_df, pattern, market_snapshot, outcome_analysis, baseline_data
        )
        
        # 保存图表
        fig.savefig(output_path, dpi=self.config["output"]["dpi"], bbox_inches="tight")
        plt.close(fig)
        
        logger.info(f"Dynamic static chart saved: {output_path}")
        return output_path
    
    def _generate_dynamic_interactive_chart(
        self,
        df: pd.DataFrame,
        pattern: PatternRecord,
        market_snapshot: Optional[MarketSnapshot] = None,
        outcome_analysis: Optional[PatternOutcomeAnalysis] = None,
        baseline_data: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """生成动态基线系统交互式图表"""
        
        # 准备数据
        chart_df = self._prepare_chart_data(df, pattern)
        
        # 创建子图
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=["Price & Pattern", "Volume", "Market Regime"],
            row_heights=[0.6, 0.2, 0.2],
        )
        
        # 添加K线图
        fig.add_trace(
            go.Candlestick(
                x=chart_df.index,
                open=chart_df["Open"],
                high=chart_df["High"],
                low=chart_df["Low"],
                close=chart_df["Close"],
                name="Price",
                increasing_line_color="green",
                decreasing_line_color="red",
            ),
            row=1, col=1,
        )
        
        # 添加成交量
        colors = [
            "green" if c >= o else "red"
            for c, o in zip(chart_df["Close"], chart_df["Open"])
        ]
        fig.add_trace(
            go.Bar(
                x=chart_df.index,
                y=chart_df["Volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.7,
            ),
            row=2, col=1,
        )
        
        # 添加形态标注
        self._add_dynamic_plotly_annotations(
            fig, chart_df, pattern, market_snapshot, outcome_analysis, baseline_data
        )
        
        # 设置布局
        title = self._create_dynamic_chart_title(pattern, market_snapshot)
        fig.update_layout(
            title=title,
            xaxis_rangeslider_visible=False,
            width=self.config["output"]["width"],
            height=self.config["output"]["height"] + 200,  # 增加高度以适应第三个子图
            template="plotly_white",
        )
        
        # 保存图表
        if not output_path:
            Path(self.charts_base_path).mkdir(parents=True, exist_ok=True)
            sub_type_name = getattr(pattern, 'sub_type', 'flag')
            output_path = f"{self.charts_base_path}/{sub_type_name}/{pattern.id}.html"
        
        fig.write_html(output_path)
        logger.info(f"Dynamic interactive chart saved: {output_path}")
        return output_path
    
    def _add_dynamic_plotly_annotations(
        self, 
        fig, 
        chart_df: pd.DataFrame, 
        pattern: PatternRecord,
        market_snapshot: Optional[MarketSnapshot] = None,
        outcome_analysis: Optional[PatternOutcomeAnalysis] = None,
        baseline_data: Optional[Dict[str, Any]] = None
    ):
        """添加动态Plotly标注"""
        try:
            # 1. 旗杆区域高亮
            fig.add_vrect(
                x0=pattern.flagpole.start_time,
                x1=pattern.flagpole.end_time,
                fillcolor=self.config["style"]["flagpole_color"],
                opacity=0.2,
                layer="below",
                line_width=0,
                row=1, col=1,
            )
            
            # 2. 添加边界线
            if pattern.pattern_boundaries:
                boundary_colors = self._get_boundary_colors(pattern)
                for i, boundary in enumerate(pattern.pattern_boundaries):
                    color = boundary_colors[i] if i < len(boundary_colors) else "#ff9800"
                    fig.add_trace(
                        go.Scatter(
                            x=[boundary.start_time, boundary.end_time],
                            y=[boundary.start_price, boundary.end_price],
                            mode="lines",
                            line=dict(color=color, width=2, dash="dash"),
                            name=f"Boundary {i + 1}",
                            showlegend=i == 0,
                        ),
                        row=1, col=1,
                    )
            
            # 3. 添加旗杆标注
            fig.add_annotation(
                x=pattern.flagpole.start_time,
                y=pattern.flagpole.start_price,
                text=f"旗杆<br>{pattern.flagpole.height_percent:.1f}%",
                showarrow=True,
                arrowhead=2,
                arrowcolor=self.config["style"]["flagpole_color"],
                bgcolor="white",
                bordercolor=self.config["style"]["flagpole_color"],
                row=1, col=1,
            )
            
            # 4. 添加失效信号标记
            if hasattr(pattern, 'invalidation_signals') and self.visualization_layers['invalidation_signals']:
                for signal in pattern.invalidation_signals:
                    if signal.trigger_time in chart_df.index:
                        color = self.config["dynamic_baseline"]["invalidation_colors"].get(
                            signal.signal_type, "#f44336"
                        )
                        size = 15 if signal.is_critical else 10
                        
                        fig.add_trace(
                            go.Scatter(
                                x=[signal.trigger_time],
                                y=[signal.trigger_price],
                                mode="markers",
                                marker=dict(
                                    size=size,
                                    color=color,
                                    symbol="x" if signal.is_critical else "circle"
                                ),
                                name=f"{signal.signal_type}",
                                showlegend=True,
                            ),
                            row=1, col=1,
                        )
            
            # 5. 市场状态显示（第三个子图）
            if market_snapshot and self.visualization_layers['market_regime']:
                self._add_market_regime_subplot(fig, chart_df, market_snapshot)
            
        except Exception as e:
            logger.warning(f"Failed to add dynamic Plotly annotations: {e}")
    
    def _add_market_regime_subplot(self, fig, chart_df: pd.DataFrame, 
                                  market_snapshot: MarketSnapshot):
        """添加市场状态子图"""
        try:
            # 创建虚拟的状态时间序列（简化实现）
            regime_values = [1 if market_snapshot.current_regime == "high_volatility" else 0] * len(chart_df)
            regime_color = self.config["dynamic_baseline"]["regime_colors"].get(
                market_snapshot.current_regime, "#9e9e9e"
            )
            
            fig.add_trace(
                go.Scatter(
                    x=chart_df.index,
                    y=regime_values,
                    mode="lines",
                    fill="tozeroy",
                    line=dict(color=regime_color),
                    name="Market Regime",
                ),
                row=3, col=1,
            )
            
        except Exception as e:
            logger.warning(f"Failed to add market regime subplot: {e}")

    def _generate_interactive_chart(
        self,
        df: pd.DataFrame,
        pattern: PatternRecord,
        output_path: Optional[str] = None,
    ) -> str:
        """生成交互式图表（使用Plotly）"""

        # 准备数据
        chart_df = self._prepare_chart_data(df, pattern)

        # 创建子图
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=["Price", "Volume"],
            row_width=[0.7, 0.3],
        )

        # 添加K线图
        fig.add_trace(
            go.Candlestick(
                x=chart_df.index,
                open=chart_df["Open"],
                high=chart_df["High"],
                low=chart_df["Low"],
                close=chart_df["Close"],
                name="Price",
                increasing_line_color="green",
                decreasing_line_color="red",
            ),
            row=1,
            col=1,
        )

        # 添加成交量
        colors = [
            "green" if c >= o else "red"
            for c, o in zip(chart_df["Close"], chart_df["Open"])
        ]

        fig.add_trace(
            go.Bar(
                x=chart_df.index,
                y=chart_df["Volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.7,
            ),
            row=2,
            col=1,
        )

        # 添加形态标注
        self._add_plotly_pattern_annotations(fig, chart_df, pattern)

        # 设置布局
        fig.update_layout(
            title=f"{pattern.symbol} - {pattern.pattern_type.title()} Pattern (Confidence: {pattern.confidence_score:.2%})",
            xaxis_rangeslider_visible=False,
            width=self.config["output"]["width"],
            height=self.config["output"]["height"],
            template="plotly_white",
        )

        # 保存图表
        if not output_path:
            Path(self.charts_base_path).mkdir(parents=True, exist_ok=True)
            output_path = f"{self.charts_base_path}/{pattern.id}.html"

        fig.write_html(output_path)

        logger.info(f"Interactive chart saved: {output_path}")
        return output_path

    def _prepare_chart_data(
        self, df: pd.DataFrame, pattern: PatternRecord
    ) -> pd.DataFrame:
        """准备图表数据"""
        # 确定图表时间范围（形态前后各加少量数据点用于上下文）
        flagpole_start = pattern.flagpole.start_time
        pattern_end = pattern.flagpole.end_time + pd.Timedelta(
            days=pattern.pattern_duration
        )

        # 动态计算扩展时间范围，基于实际形态时间
        flagpole_duration = pattern.flagpole.end_time - pattern.flagpole.start_time

        # 前后缓冲时间：旗杆时间的50%，但不超过2天，不少于6小时
        buffer_duration = max(
            pd.Timedelta(hours=6), min(flagpole_duration * 0.5, pd.Timedelta(days=2))
        )

        extended_start = flagpole_start - buffer_duration
        extended_end = pattern_end + buffer_duration

        # 过滤数据
        mask = (df["timestamp"] >= extended_start) & (df["timestamp"] <= extended_end)
        chart_df = df.loc[mask].copy()

        # 重新索引并重命名列以适应mplfinance
        chart_df = chart_df.set_index("timestamp")
        chart_df = chart_df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )

        return chart_df

    def _create_additional_plots(
        self, chart_df: pd.DataFrame, pattern: PatternRecord
    ) -> List:
        """创建附加图线"""
        apds = []

        # 添加旗杆标记 - 修复为直线绘制
        flagpole_mask = (chart_df.index >= pattern.flagpole.start_time) & (
            chart_df.index <= pattern.flagpole.end_time
        )

        if flagpole_mask.any():
            flagpole_times = chart_df.index[flagpole_mask]
            total_duration = (
                pattern.flagpole.end_time - pattern.flagpole.start_time
            ).total_seconds()

            flagpole_line = pd.Series(index=chart_df.index, dtype=float)
            # 计算直线上每个点的价格
            for time_point in flagpole_times:
                progress = (
                    (time_point - pattern.flagpole.start_time).total_seconds()
                    / total_duration
                    if total_duration > 0
                    else 0
                )
                price = (
                    pattern.flagpole.start_price
                    + (pattern.flagpole.end_price - pattern.flagpole.start_price)
                    * progress
                )
                flagpole_line.loc[time_point] = price

            apds.append(
                mpf.make_addplot(
                    flagpole_line,
                    type="line",
                    color=self.config["style"]["flagpole_color"],
                    width=1.5,
                    alpha=0.9,
                )
            )

        # 添加边界线
        if pattern.pattern_boundaries:
            for boundary in pattern.pattern_boundaries:
                boundary_line = self._create_trend_line_series(chart_df, boundary)
                if boundary_line is not None:
                    apds.append(
                        mpf.make_addplot(
                            boundary_line,
                            type="line",
                            color=self.config["style"]["boundary_color"],
                            width=1.5,
                            linestyle="--",
                            alpha=0.85,
                        )
                    )

        return apds

    def _create_trend_line_series(
        self, chart_df: pd.DataFrame, trend_line: TrendLine
    ) -> Optional[pd.Series]:
        """创建趋势线数据系列"""
        try:
            # 找到趋势线时间范围内的数据点
            mask = (chart_df.index >= trend_line.start_time) & (
                chart_df.index <= trend_line.end_time
            )

            if not mask.any():
                return None

            # 计算趋势线上每个点的价格
            time_points = chart_df.index[mask]
            total_duration = (
                trend_line.end_time - trend_line.start_time
            ).total_seconds()

            trend_prices = []
            for time_point in time_points:
                progress = (
                    time_point - trend_line.start_time
                ).total_seconds() / total_duration
                price = (
                    trend_line.start_price
                    + (trend_line.end_price - trend_line.start_price) * progress
                )
                trend_prices.append(price)

            # 创建完整的series，只在趋势线时间范围内有值
            trend_series = pd.Series(index=chart_df.index, dtype=float)
            trend_series.loc[time_points] = trend_prices

            return trend_series

        except Exception as e:
            logger.warning(f"Failed to create trend line series: {e}")
            return None

    def _add_pattern_annotations(
        self, axes, chart_df: pd.DataFrame, pattern: PatternRecord
    ):
        """添加形态标注（matplotlib版本）"""
        try:
            ax_main = axes[0]  # 主价格图

            # 标记旗杆起始点
            flagpole_start_idx = chart_df.index.get_loc(
                pattern.flagpole.start_time, method="nearest"
            )

            # 添加旗杆箭头
            ax_main.annotate(
                f"Flagpole Start\n{pattern.flagpole.height_percent:.1f}%",
                xy=(flagpole_start_idx, chart_df.iloc[flagpole_start_idx]["Close"]),
                xytext=(
                    flagpole_start_idx - 5,
                    chart_df.iloc[flagpole_start_idx]["Close"],
                ),
                arrowprops=dict(
                    arrowstyle="->", color=self.config["style"]["flagpole_color"], lw=2
                ),
                fontsize=10,
                ha="center",
            )

            # 添加置信度信息
            ax_main.text(
                0.02,
                0.98,
                f"Pattern ID: {pattern.id[:8]}\n"
                f"Confidence: {pattern.confidence_score:.2%}\n"
                f"Quality: {pattern.pattern_quality}",
                transform=ax_main.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                fontsize=10,
            )

        except Exception as e:
            logger.warning(f"Failed to add pattern annotations: {e}")

    def _add_plotly_pattern_annotations(
        self, fig, chart_df: pd.DataFrame, pattern: PatternRecord
    ):
        """添加形态标注（Plotly版本）"""
        try:
            # 添加旗杆区域高亮
            fig.add_vrect(
                x0=pattern.flagpole.start_time,
                x1=pattern.flagpole.end_time,
                fillcolor=self.config["style"]["flagpole_color"],
                opacity=0.2,
                layer="below",
                line_width=0,
                row=1,
                col=1,
            )

            # 添加边界线
            if pattern.pattern_boundaries:
                for i, boundary in enumerate(pattern.pattern_boundaries):
                    # 创建边界线数据点
                    x_vals = [boundary.start_time, boundary.end_time]
                    y_vals = [boundary.start_price, boundary.end_price]

                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=y_vals,
                            mode="lines",
                            line=dict(
                                color=self.config["style"]["boundary_color"],
                                width=2,
                                dash="dash",
                            ),
                            name=f"Boundary {i + 1}",
                            showlegend=i == 0,
                        ),
                        row=1,
                        col=1,
                    )

            # 添加文本标注
            fig.add_annotation(
                x=pattern.flagpole.start_time,
                y=pattern.flagpole.start_price,
                text=f"Flagpole<br>{pattern.flagpole.height_percent:.1f}%",
                showarrow=True,
                arrowhead=2,
                arrowcolor=self.config["style"]["flagpole_color"],
                bgcolor="white",
                bordercolor=self.config["style"]["flagpole_color"],
                row=1,
                col=1,
            )

        except Exception as e:
            logger.warning(f"Failed to add Plotly pattern annotations: {e}")

    def generate_summary_chart(
        self, patterns: List[PatternRecord], output_path: Optional[str] = None
    ) -> str:
        """
        生成形态汇总图表

        Args:
            patterns: 形态列表
            output_path: 输出路径

        Returns:
            生成的图表文件路径
        """
        try:
            # 创建汇总统计（用于统计图表）
            _ = self._calculate_pattern_statistics(patterns)

            # 创建子图
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(
                "Pattern Scout - Analysis Summary", fontsize=16, fontweight="bold"
            )

            # 1. 形态质量分布
            qualities = [p.pattern_quality for p in patterns]
            quality_counts = pd.Series(qualities).value_counts()
            axes[0, 0].pie(
                quality_counts.values, labels=quality_counts.index, autopct="%1.1f%%"
            )
            axes[0, 0].set_title("Pattern Quality Distribution")

            # 2. 置信度分布
            confidences = [p.confidence_score for p in patterns]
            axes[0, 1].hist(confidences, bins=20, alpha=0.7, color="skyblue")
            axes[0, 1].set_title("Confidence Score Distribution")
            axes[0, 1].set_xlabel("Confidence Score")
            axes[0, 1].set_ylabel("Count")

            # 3. 旗杆高度分布
            flagpole_heights = [p.flagpole.height_percent for p in patterns]
            axes[1, 0].hist(flagpole_heights, bins=15, alpha=0.7, color="lightgreen")
            axes[1, 0].set_title("Flagpole Height Distribution")
            axes[1, 0].set_xlabel("Height (%)")
            axes[1, 0].set_ylabel("Count")

            # 4. 统计信息文本
            axes[1, 1].axis("off")
            stats_text = f"""
            Pattern Statistics:
            
            Total Patterns: {len(patterns)}
            
            Quality Breakdown:
            High: {quality_counts.get("high", 0)}
            Medium: {quality_counts.get("medium", 0)}
            Low: {quality_counts.get("low", 0)}
            
            Average Confidence: {np.mean(confidences):.2%}
            Average Flagpole Height: {np.mean(flagpole_heights):.1f}%
            
            Time Range:
            From: {min(p.flagpole.start_time for p in patterns).strftime("%Y-%m-%d")}
            To: {max(p.flagpole.start_time for p in patterns).strftime("%Y-%m-%d")}
            """

            axes[1, 1].text(
                0.1,
                0.9,
                stats_text,
                transform=axes[1, 1].transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
            )

            plt.tight_layout()

            # 保存图表
            if not output_path:
                Path(self.charts_base_path).mkdir(parents=True, exist_ok=True)
                output_path = f"{self.charts_base_path}/pattern_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

            plt.savefig(
                output_path, dpi=self.config["output"]["dpi"], bbox_inches="tight"
            )
            plt.close()

            logger.info(f"Summary chart saved: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to generate summary chart: {e}")
            raise

    def _calculate_pattern_statistics(
        self, patterns: List[PatternRecord]
    ) -> Dict[str, Any]:
        """计算形态统计信息"""
        if not patterns:
            return {}

        return {
            "total_count": len(patterns),
            "quality_distribution": pd.Series(
                [p.pattern_quality for p in patterns]
            ).value_counts(),
            "avg_confidence": np.mean([p.confidence_score for p in patterns]),
            "avg_flagpole_height": np.mean(
                [p.flagpole.height_percent for p in patterns]
            ),
            "time_range": {
                "start": min(p.flagpole.start_time for p in patterns),
                "end": max(p.flagpole.start_time for p in patterns),
            },
        }

    def load_pattern_data(self, pattern_file: str) -> dict:
        """加载形态数据"""
        try:
            with open(pattern_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load pattern data from {pattern_file}: {e}")
            return {}

    def create_meaningful_filename(self, pattern_data: dict) -> str:
        """创建有意义的文件名：资产名称_起始时间_形态名称"""
        try:
            # 提取资产名称（去掉-15min后缀）
            symbol = pattern_data.get("symbol", "").replace("-15min", "").strip()
            if not symbol:
                symbol = "unknown_symbol"

            # 提取旗杆起始时间并格式化
            start_time_str = pattern_data.get("flagpole", {}).get("start_time", "")
            if start_time_str:
                start_time = datetime.fromisoformat(start_time_str)
                time_str = start_time.strftime("%Y%m%d_%H%M")
            else:
                time_str = datetime.now().strftime("%Y%m%d_%H%M")

            # 形态名称映射
            pattern_type = pattern_data.get("pattern_type", "flag")
            pattern_config = self.config["pattern_types"].get(pattern_type, {})
            pattern_name = pattern_config.get("name_cn", pattern_type)
            if not pattern_name or not pattern_name.strip():
                pattern_name = "unknown_pattern"

            # 方向信息
            direction = pattern_data.get("flagpole", {}).get("direction", "up")
            direction_name = "上升" if direction == "up" else "下降"

            # 组合文件名并验证
            filename = f"{symbol}_{time_str}_{direction_name}{pattern_name}"

            # 最终验证：确保文件名不为空且不包含非法字符
            if filename and filename.strip():
                # 清理文件名中的非法字符
                filename = filename.strip()
                illegal_chars = '<>:"/\\|?*'
                for char in illegal_chars:
                    filename = filename.replace(char, "_")
                return filename
            else:
                raise ValueError("Generated filename is empty")

        except Exception as e:
            logger.warning(f"Failed to create meaningful filename: {e}")
            # 回退到使用ID，并确保不返回空字符串
            fallback_id = pattern_data.get("id", "unknown_pattern")
            if fallback_id and fallback_id.strip():  # 确保非空且非纯空白
                return fallback_id.strip()
            else:
                return "unknown_pattern"

    def generate_pattern_chart_from_data(
        self, pattern_data: dict, save_path: Optional[str] = None
    ) -> Optional[str]:
        """从形态数据生成TradingView风格图表"""
        try:
            # 确保保存路径正确
            if not save_path:
                # 如果没有指定保存路径，使用默认路径
                filename = self.create_meaningful_filename(pattern_data)
                pattern_type = pattern_data.get("pattern_type", "flag")
                type_dir = Path(f"{self.charts_base_path}/{pattern_type}")
                type_dir.mkdir(parents=True, exist_ok=True)
                save_path = str(type_dir / f"{filename}.png")
            else:
                # 确保保存目录存在
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            pattern_type = pattern_data.get("pattern_type", "flag")
            pattern_config = self.config["pattern_types"].get(
                pattern_type, self.config["pattern_types"]["flag"]
            )

            # 解析时间 - 兼容不同数据结构
            if "flagpole" in pattern_data and isinstance(
                pattern_data["flagpole"], dict
            ):
                flagpole_start = datetime.fromisoformat(
                    pattern_data["flagpole"]["start_time"]
                )
                flagpole_end = datetime.fromisoformat(
                    pattern_data["flagpole"]["end_time"]
                )
            else:
                # 使用扁平化结构
                flagpole_start = datetime.fromisoformat(
                    pattern_data["flagpole_start_time"]
                )
                flagpole_end = datetime.fromisoformat(pattern_data["flagpole_end_time"])

            # 动态计算时间范围，基于实际形态时间
            flagpole_duration = flagpole_end - flagpole_start
            pattern_duration_td = pd.Timedelta(
                days=pattern_data.get("pattern_duration", 3)
            )

            # 前后缓冲时间：旗杆时间的50%，但不超过2天，不少于6小时
            buffer_duration = max(
                pd.Timedelta(hours=6),
                min(flagpole_duration * 0.5, pd.Timedelta(days=2)),
            )

            extended_start = flagpole_start - buffer_duration
            extended_end = flagpole_end + pattern_duration_td + buffer_duration

            # 获取价格数据
            df = self.data_connector.get_data(
                pattern_data["symbol"], extended_start, extended_end
            )

            if df.empty:
                logger.warning(
                    f"No data found for pattern {pattern_data.get('id', 'unknown')}"
                )
                return None

            # 准备mplfinance数据格式
            df_plot = df.set_index("timestamp")
            df_plot = df_plot.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                }
            )

            # 创建附加图线
            addplots = []

            # 旗杆标记 - 修复为直线绘制
            # 兼容不同的数据结构格式
            if "flagpole" in pattern_data and isinstance(
                pattern_data["flagpole"], dict
            ):
                flagpole_start_price = pattern_data["flagpole"]["start_price"]
                flagpole_end_price = pattern_data["flagpole"]["end_price"]
            else:
                # 使用扁平化结构（从数据库查询获得的格式）
                # 需要从价格数据中计算起止价格
                flagpole_start_price = None
                flagpole_end_price = None

            # 创建旗杆直线数据
            flagpole_mask = (df_plot.index >= flagpole_start) & (
                df_plot.index <= flagpole_end
            )
            if flagpole_mask.any():
                flagpole_times = df_plot.index[flagpole_mask]
                total_duration = (flagpole_end - flagpole_start).total_seconds()

                # 如果没有明确的起止价格，从数据中获取
                if flagpole_start_price is None or flagpole_end_price is None:
                    flagpole_data = df_plot.loc[flagpole_mask]
                    if not flagpole_data.empty:
                        flagpole_start_price = flagpole_data.iloc[0]["Close"]
                        flagpole_end_price = flagpole_data.iloc[-1]["Close"]
                    else:
                        # 无法获取价格数据，跳过旗杆绘制
                        flagpole_start_price = flagpole_end_price = 0

                flagpole_line = pd.Series(index=df_plot.index, dtype=float)
                # 计算直线上每个点的价格
                for time_point in flagpole_times:
                    progress = (
                        (time_point - flagpole_start).total_seconds() / total_duration
                        if total_duration > 0
                        else 0
                    )
                    price = (
                        flagpole_start_price
                        + (flagpole_end_price - flagpole_start_price) * progress
                    )
                    flagpole_line.loc[time_point] = price

                addplots.append(
                    mpf.make_addplot(
                        flagpole_line,
                        type="line",
                        color=pattern_config["color_flagpole"],
                        width=1.5,
                        alpha=0.9,
                    )
                )

            # 边界线
            if (
                "pattern_boundaries" in pattern_data
                and pattern_data["pattern_boundaries"]
            ):
                for boundary in pattern_data["pattern_boundaries"]:
                    start_time = datetime.fromisoformat(boundary["start_time"])
                    end_time = datetime.fromisoformat(boundary["end_time"])

                    # 创建边界线数据
                    boundary_mask = (df_plot.index >= start_time) & (
                        df_plot.index <= end_time
                    )
                    if boundary_mask.any():
                        time_points = df_plot.index[boundary_mask]
                        total_duration = (end_time - start_time).total_seconds()

                        boundary_prices = []
                        for time_point in time_points:
                            progress = (
                                time_point - start_time
                            ).total_seconds() / total_duration
                            price = (
                                boundary["start_price"]
                                + (boundary["end_price"] - boundary["start_price"])
                                * progress
                            )
                            boundary_prices.append(price)

                        boundary_line = pd.Series(index=df_plot.index, dtype=float)
                        boundary_line.loc[time_points] = boundary_prices

                        # 根据边界线的位置选择颜色
                        is_upper = boundary.get("type", "upper") == "upper"
                        boundary_color = pattern_config.get(
                            "color_upper" if is_upper else "color_lower", "#2196f3"
                        )

                        addplots.append(
                            mpf.make_addplot(
                                boundary_line,
                                type="line",
                                color=boundary_color,
                                width=1.5,
                                linestyle="--",
                                alpha=0.85,
                            )
                        )

            # 设置TradingView风格
            style = mpf.make_mpf_style(
                base_mpf_style="charles",
                marketcolors=mpf.make_marketcolors(
                    up=self.config["style"]["up_color"],
                    down=self.config["style"]["down_color"],
                    edge={
                        "up": self.config["style"]["up_color"],
                        "down": self.config["style"]["down_color"],
                    },
                    wick={
                        "up": self.config["style"]["up_color"],
                        "down": self.config["style"]["down_color"],
                    },
                    volume=self.config["style"]["volume_color"],
                ),
                gridstyle="--",
                gridcolor=self.config["style"]["grid_color"],
                facecolor=self.config["style"]["background_color"],
                figcolor=self.config["style"]["background_color"],
            )

            # 创建标题 - 兼容不同数据结构
            if "flagpole" in pattern_data and isinstance(
                pattern_data["flagpole"], dict
            ):
                direction = pattern_data["flagpole"]["direction"]
                height_percent = pattern_data["flagpole"]["height_percent"]
            else:
                direction = pattern_data.get("flagpole_direction", "up")
                height_percent = pattern_data.get("flagpole_height_percent", 0)

            direction_text = "上升" if direction == "up" else "下降"
            pattern_name = pattern_config["name_cn"]
            title = (
                f"{pattern_data['symbol']} - {direction_text}{pattern_name}\\n"
                f"置信度: {pattern_data['confidence_score']:.3f} | "
                f"质量: {pattern_data['pattern_quality']} | "
                f"旗杆高度: {height_percent:.2f}%"
            )

            # 绘制图表
            fig, axes = mpf.plot(
                df_plot,
                type="candle",
                style=style,
                addplot=addplots,
                volume=True,
                title=title,
                figsize=(16, 10),
                tight_layout=True,
                returnfig=True,
                warn_too_much_data=1000,
            )

            # 手动保存图表
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"TradingView style chart saved: {save_path}")
                plt.close(fig)
            else:
                plt.show()

            return save_path

        except Exception as e:
            logger.error(f"Failed to generate chart from pattern data: {e}")
            return None

    def generate_classified_charts(
        self, patterns_dir: str = "output/data/patterns"
    ) -> Dict[str, Any]:
        """按形态类型分类生成图表"""
        try:
            patterns_path = Path(patterns_dir)
            pattern_files = list(patterns_path.glob("*.json"))

            if not pattern_files:
                logger.warning("No pattern files found!")
                return {}

            # 按类型分类形态
            classified_patterns = {"flag": [], "pennant": []}

            for pattern_file in pattern_files:
                pattern_data = self.load_pattern_data(pattern_file)
                if pattern_data:
                    pattern_type = pattern_data.get("pattern_type", "flag")
                    if pattern_type in classified_patterns:
                        classified_patterns[pattern_type].append(pattern_data)

            results = {}
            total_charts = 0

            for pattern_type, patterns in classified_patterns.items():
                if not patterns:
                    continue

                logger.info(f"Generating {len(patterns)} {pattern_type} charts...")

                # 创建类型目录
                type_dir = Path(f"{self.charts_base_path}/{pattern_type}")
                type_dir.mkdir(parents=True, exist_ok=True)

                charts_generated = 0
                chart_paths = []

                for i, pattern_data in enumerate(patterns, 1):
                    filename = self.create_meaningful_filename(pattern_data)
                    save_path = type_dir / f"{filename}.png"

                    logger.info(
                        f"Generating {i}/{len(patterns)} {pattern_type} chart: {filename}"
                    )

                    result = self.generate_pattern_chart_from_data(
                        pattern_data, str(save_path)
                    )
                    if result:
                        charts_generated += 1
                        chart_paths.append(str(save_path))
                        logger.info(f"TradingView style chart saved: {save_path}")

                results[pattern_type] = {
                    "charts_generated": charts_generated,
                    "total_patterns": len(patterns),
                    "charts": chart_paths,
                }
                total_charts += charts_generated

            logger.info(
                f"Classification chart generation completed. Generated {total_charts} charts"
            )
            return results

        except Exception as e:
            logger.error(f"Failed to generate classified charts: {e}")
            return {}

    def create_summary_report(self):
        """创建汇总报告"""
        try:
            dataset_manager = DatasetManager("output/data")
            pattern_dicts = dataset_manager.query_patterns()

            if not pattern_dicts:
                logger.warning("No patterns found for summary report")
                return

            # 统计信息
            pattern_stats = {
                "total": len(pattern_dicts),
                "by_type": {},
                "by_quality": {},
                "by_direction": {},
            }

            for pattern_dict in pattern_dicts:
                # 按类型统计
                pattern_type = pattern_dict.get("pattern_type", "unknown")
                pattern_stats["by_type"][pattern_type] = (
                    pattern_stats["by_type"].get(pattern_type, 0) + 1
                )

                # 按质量统计
                quality = pattern_dict.get("pattern_quality", "unknown")
                pattern_stats["by_quality"][quality] = (
                    pattern_stats["by_quality"].get(quality, 0) + 1
                )

                # 按方向统计
                flagpole_data = pattern_dict.get("flagpole", {})
                direction = flagpole_data.get("direction", "unknown")
                pattern_stats["by_direction"][direction] = (
                    pattern_stats["by_direction"].get(direction, 0) + 1
                )

            # 生成文本报告
            report_lines = [
                "\\n=== PatternScout TradingView风格形态分析报告 ===",
                f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"总计检测形态数量: {pattern_stats['total']}",
                "\\n=== 形态类型分布 ===",
            ]

            for pattern_type, count in pattern_stats["by_type"].items():
                percentage = (count / pattern_stats["total"]) * 100
                type_name = (
                    self.config["pattern_types"]
                    .get(pattern_type, {})
                    .get("name_cn", pattern_type)
                )
                report_lines.append(f"{type_name}: {count} ({percentage:.1f}%)")

            report_lines.extend(
                [
                    "\\n=== 质量分布 ===",
                ]
            )

            for quality, count in pattern_stats["by_quality"].items():
                percentage = (count / pattern_stats["total"]) * 100
                report_lines.append(f"{quality.upper()}: {count} ({percentage:.1f}%)")

            report_lines.extend(
                [
                    "\\n=== 方向分布 ===",
                ]
            )

            for direction, count in pattern_stats["by_direction"].items():
                percentage = (count / pattern_stats["total"]) * 100
                direction_name = (
                    "上升"
                    if direction == "up"
                    else "下降"
                    if direction == "down"
                    else direction
                )
                report_lines.append(f"{direction_name}: {count} ({percentage:.1f}%)")

            # 计算平均置信度
            confidence_scores = [
                p.get("confidence_score", 0)
                for p in pattern_dicts
                if p.get("confidence_score") is not None
            ]
            if confidence_scores:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                max_confidence = max(confidence_scores)
            else:
                avg_confidence = 0
                max_confidence = 0

            report_lines.extend(
                [
                    "\\n=== 统计指标 ===",
                    f"平均置信度: {avg_confidence:.3f}",
                    f"最高置信度: {max_confidence:.3f}",
                ]
            )

            report_text = "\\n".join(report_lines)
            print(report_text)

            logger.info("Summary report generated successfully")

        except Exception as e:
            logger.error(f"Failed to create summary report: {e}")
    
    def generate_outcome_analysis_chart(
        self,
        df: pd.DataFrame,
        pattern: PatternRecord,
        outcome_analysis: PatternOutcomeAnalysis,
        monitoring_data: pd.DataFrame,
        output_path: Optional[str] = None,
    ) -> str:
        """
        生成结局分析图表
        
        Args:
            df: 原始价格数据
            pattern: 形态记录
            outcome_analysis: 结局分析
            monitoring_data: 监控期间数据
            output_path: 输出路径
            
        Returns:
            生成的图表文件路径
        """
        try:
            # 合并数据：形态期 + 监控期
            extended_df = pd.concat([df, monitoring_data]).drop_duplicates('timestamp').sort_values('timestamp')
            
            # 生成动态图表
            return self.generate_dynamic_pattern_chart(
                extended_df, pattern, None, outcome_analysis, None, output_path
            )
            
        except Exception as e:
            logger.error(f"Failed to generate outcome analysis chart: {e}")
            raise
    
    def generate_baseline_summary_chart(
        self,
        baseline_data: Dict[str, Any],
        regime_history: List[Dict[str, Any]],
        output_path: Optional[str] = None,
    ) -> str:
        """
        生成基线数据汇总图表
        
        Args:
            baseline_data: 基线数据
            regime_history: 市场状态历史
            output_path: 输出路径
            
        Returns:
            生成的图表文件路径
        """
        try:
            # 创建子图
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(
                "Dynamic Baseline System - Summary Report", fontsize=16, fontweight="bold"
            )
            
            # 1. 市场状态分布
            if regime_history:
                regime_counts = {}
                for regime_record in regime_history:
                    regime = regime_record.get('regime', 'unknown')
                    regime_counts[regime] = regime_counts.get(regime, 0) + 1
                
                if regime_counts:
                    axes[0, 0].pie(
                        regime_counts.values(), 
                        labels=[
                            {
                                'high_volatility': '高波动',
                                'low_volatility': '低波动', 
                                'unknown': '未知'
                            }.get(k, k) for k in regime_counts.keys()
                        ],
                        autopct='%1.1f%%',
                        colors=[
                            self.config['dynamic_baseline']['regime_colors'].get(k, '#9e9e9e')
                            for k in regime_counts.keys()
                        ]
                    )
                # 设置中文标题
                title_text = '市场状态分布'
                if self.chinese_font_prop:
                    axes[0, 0].set_title(title_text, fontproperties=self.chinese_font_prop)
                else:
                    axes[0, 0].set_title(title_text)
            
            # 2. 动态阈值趋势（简化实现）
            trend_text = '动态阈值趋势\n（需要实际数据支持）'
            title2_text = '动态阈值趋势'
            if self.chinese_font_prop:
                axes[0, 1].text(
                    0.5, 0.5, 
                    trend_text,
                    transform=axes[0, 1].transAxes,
                    ha='center', va='center',
                    fontsize=12,
                    fontproperties=self.chinese_font_prop
                )
                axes[0, 1].set_title(title2_text, fontproperties=self.chinese_font_prop)
            else:
                axes[0, 1].text(
                    0.5, 0.5, 
                    trend_text,
                    transform=axes[0, 1].transAxes,
                    ha='center', va='center',
                    fontsize=12
                )
                axes[0, 1].set_title(title2_text)
            
            # 3. 基线覆盖统计
            if baseline_data:
                coverage_stats = baseline_data.get('coverage_stats', {})
                if coverage_stats:
                    metrics = list(coverage_stats.keys())
                    values = list(coverage_stats.values())
                    axes[1, 0].bar(metrics, values, alpha=0.7)
                    title3_text = '基线覆盖统计'
                    if self.chinese_font_prop:
                        axes[1, 0].set_title(title3_text, fontproperties=self.chinese_font_prop)
                    else:
                        axes[1, 0].set_title(title3_text)
                    axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 4. 系统状态信息
            status_text = f"""
            系统状态报告:
            
            基线数据点数: {baseline_data.get('total_data_points', 'N/A')}
            有效状态记录: {len(regime_history)}
            生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            状态转换次数: {baseline_data.get('regime_transitions', 'N/A')}
            当前活跃状态: {baseline_data.get('current_regime', 'unknown')}
            """
            
            if self.chinese_font_prop:
                axes[1, 1].text(
                    0.1, 0.9,
                    status_text,
                    transform=axes[1, 1].transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
                    fontproperties=self.chinese_font_prop
                )
            else:
                axes[1, 1].text(
                    0.1, 0.9,
                    status_text,
                    transform=axes[1, 1].transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
                )
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            # 保存图表
            if not output_path:
                Path(self.charts_base_path).mkdir(parents=True, exist_ok=True)
                output_path = f"{self.charts_base_path}/baseline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            plt.savefig(
                output_path, dpi=self.config["output"]["dpi"], bbox_inches="tight"
            )
            plt.close()
            
            logger.info(f"Baseline summary chart saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to generate baseline summary chart: {e}")
            raise
    
    def generate_dynamic_chart_from_scan_result(
        self,
        scan_result: Dict[str, Any],
        df: pd.DataFrame,
        output_dir: Optional[str] = None
    ) -> List[str]:
        """
        从扫描结果生成动态图表
        
        Args:
            scan_result: 动态扫描结果
            df: 价格数据
            output_dir: 输出目录
            
        Returns:
            生成的图表文件路径列表
        """
        try:
            if not scan_result.get('success', False):
                logger.warning("Scan result indicates failure, skipping chart generation")
                return []
            
            patterns = scan_result.get('patterns', [])
            if not patterns:
                logger.info("No patterns found in scan result")
                return []
            
            chart_paths = []
            market_snapshot_dict = scan_result.get('market_snapshot', {})
            
            for pattern_dict in patterns:
                try:
                    # 转换为 PatternRecord 对象（简化实现）
                    # 在实际实现中需要更完整的转换逻辑
                    pattern = self._dict_to_pattern_record(pattern_dict)
                    
                    # 创建输出路径
                    if output_dir:
                        Path(output_dir).mkdir(parents=True, exist_ok=True)
                        sub_type_dir = Path(output_dir) / getattr(pattern, 'sub_type', 'flag')
                        sub_type_dir.mkdir(parents=True, exist_ok=True)
                        
                        filename = self.create_dynamic_meaningful_filename(pattern)
                        chart_path = str(sub_type_dir / f"{filename}.{self.config['output']['format']}")
                    else:
                        chart_path = None
                    
                    # 生成图表
                    result_path = self.generate_dynamic_pattern_chart(
                        df, pattern, 
                        market_snapshot=self._dict_to_market_snapshot(market_snapshot_dict),
                        output_path=chart_path
                    )
                    
                    if result_path:
                        chart_paths.append(result_path)
                        logger.info(f"Generated dynamic chart: {result_path}")
                    
                except Exception as pattern_error:
                    logger.error(f"Failed to generate chart for pattern {pattern_dict.get('id', 'unknown')}: {pattern_error}")
                    continue
            
            logger.info(f"Generated {len(chart_paths)} dynamic charts from scan result")
            return chart_paths
            
        except Exception as e:
            logger.error(f"Failed to generate charts from scan result: {e}")
            return []
    
    def _dict_to_pattern_record(self, pattern_dict: Dict[str, Any]):
        """将字典转换为 PatternRecord 对象（简化实现）"""
        # 这里需要根据实际的 PatternRecord 类结构进行实现
        # 简化示例：
        return pattern_dict  # 这里需要实际的转换逻辑
    
    def _dict_to_market_snapshot(self, snapshot_dict: Dict[str, Any]) -> Optional[MarketSnapshot]:
        """将字典转换为 MarketSnapshot 对象（简化实现）"""
        if not snapshot_dict:
            return None
        # 这里需要根据实际的 MarketSnapshot 类结构进行实现
        return None  # 简化实现
    
    def create_dynamic_meaningful_filename(self, pattern) -> str:
        """为动态基线系统创建有意义的文件名"""
        try:
            # 获取基本信息
            symbol = pattern.symbol.replace("-15min", "").strip()
            if not symbol:
                symbol = "unknown_symbol"
            
            # 格式化时间
            time_str = pattern.flagpole.start_time.strftime("%Y%m%d_%H%M")
            
            # 获取形态类型名称
            sub_type = getattr(pattern, 'sub_type', 'flag')
            sub_type_config = self.config["pattern_types"]["flag_pattern"]["subtypes"].get(sub_type, {})
            pattern_name = sub_type_config.get("name_cn", sub_type)
            
            # 方向信息
            direction_name = "上升" if pattern.flagpole.direction == "up" else "下降"
            
            # 组合文件名
            filename = f"{symbol}_{time_str}_{direction_name}{pattern_name}"
            
            # 清理非法字符
            illegal_chars = '<>:"/\\|?*'
            for char in illegal_chars:
                filename = filename.replace(char, "_")
            
            return filename.strip() or pattern.id[:8]
            
        except Exception as e:
            logger.warning(f"Failed to create dynamic meaningful filename: {e}")
            return getattr(pattern, 'id', 'unknown_pattern')[:8]
    
    # 保留原有的方法以保持向后兼容
    def generate_pattern_chart(self, df: pd.DataFrame, pattern: PatternRecord, output_path: Optional[str] = None) -> str:
        """向后兼容的方法"""
        return self.generate_legacy_pattern_chart(df, pattern, output_path)
    
    def generate_legacy_pattern_chart(
        self,
        df: pd.DataFrame,
        pattern: PatternRecord,
        output_path: Optional[str] = None,
    ) -> str:
        """
        生成传统形态图表（向后兼容）
        """
        try:
            if self.config["output"]["format"].lower() == "html":
                return self._generate_interactive_chart(df, pattern, output_path)
            else:
                return self._generate_static_chart(df, pattern, output_path)
        except Exception as e:
            logger.error(f"Failed to generate legacy chart: {e}")
            raise


class ChartGenerator(DynamicChartGenerator):
    """向后兼容的图表生成器别名"""
    pass
