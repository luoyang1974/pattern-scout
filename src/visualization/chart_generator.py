import matplotlib.pyplot as plt
import mplfinance as mpf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

from src.data.models.base_models import PatternRecord, TrendLine
from src.data.connectors.csv_connector import CSVDataConnector
from src.storage.dataset_manager import DatasetManager
from loguru import logger
import json


class ChartGenerator:
    """图表生成器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化图表生成器
        
        Args:
            config: 图表配置参数
        """
        # 如果传入的config有'output'键，则提取output配置
        if config and 'output' in config:
            self.config = config
        else:
            # 否则直接使用传入的配置作为output配置
            self.config = {'output': config or {}}
        
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
        
        # 初始化数据连接器
        self.data_connector = CSVDataConnector("data/csv/")
        try:
            self.data_connector.connect()
        except Exception as e:
            logger.warning(f"Failed to connect data connector: {e}")
        
        # 设置图表输出路径
        self.charts_base_path = self.config.get('charts_path', 'output/charts')
        
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'output': {
                'width': 1600,
                'height': 1000,
                'dpi': 300,
                'format': 'png'
            },
            'style': {
                'background_color': 'white',
                'grid_color': '#e0e0e0',
                'text_color': '#333333',
                'up_color': '#26a69a',
                'down_color': '#ef5350',
                'volume_color': '#42a5f5',
                'flagpole_color': '#ff6b35',
                'flag_color': '#2196f3',
                'boundary_color': '#ff9800',
                'breakthrough_color': '#F39C12'
            },
            'pattern_types': {
                'flag': {
                    'name_cn': '旗形',
                    'color_flagpole': '#ff6b35',
                    'color_upper': '#2196f3',
                    'color_lower': '#ff9800'
                },
                'pennant': {
                    'name_cn': '三角旗形',
                    'color_flagpole': '#9c27b0',
                    'color_upper': '#4caf50',
                    'color_lower': '#f44336'
                }
            }
        }
    
    def generate_pattern_chart(self, df: pd.DataFrame, pattern: PatternRecord, 
                             output_path: Optional[str] = None) -> str:
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
            if self.config['output']['format'].lower() == 'html':
                return self._generate_interactive_chart(df, pattern, output_path)
            else:
                return self._generate_static_chart(df, pattern, output_path)
        except Exception as e:
            logger.error(f"Failed to generate chart: {e}")
            raise
    
    def _generate_static_chart(self, df: pd.DataFrame, pattern: PatternRecord,
                             output_path: Optional[str] = None) -> str:
        """生成静态图表（使用matplotlib和mplfinance）"""
        
        # 准备数据
        chart_df = self._prepare_chart_data(df, pattern)
        
        # 设置TradingView风格
        style = mpf.make_mpf_style(
            base_mpf_style='charles',
            marketcolors=mpf.make_marketcolors(
                up=self.config['style']['up_color'],
                down=self.config['style']['down_color'],
                edge={'up': self.config['style']['up_color'], 'down': self.config['style']['down_color']},
                wick={'up': self.config['style']['up_color'], 'down': self.config['style']['down_color']},
                volume=self.config['style']['volume_color']
            ),
            gridstyle='--',
            gridcolor=self.config['style']['grid_color'],
            facecolor=self.config['style']['background_color'],
            figcolor=self.config['style']['background_color']
        )
        
        # 创建附加图线
        apds = self._create_additional_plots(chart_df, pattern)
        
        # 生成图表
        fig, axes = mpf.plot(
            chart_df,
            type='candle',
            style=style,
            title=f'{pattern.symbol} - {pattern.pattern_type.title()} Pattern (ID: {pattern.id[:8]})',
            volume=True,
            figsize=(16, 10),
            addplot=apds,
            returnfig=True,
            savefig=dict(
                fname=output_path if output_path else f"charts/{pattern.id}.{self.config['output']['format']}",
                dpi=self.config['output']['dpi'],
                bbox_inches='tight'
            )
        )
        
        # 添加自定义标注
        self._add_pattern_annotations(axes, chart_df, pattern)
        
        # 保存图表
        if not output_path:
            Path(self.charts_base_path).mkdir(parents=True, exist_ok=True)
            output_path = f"{self.charts_base_path}/{pattern.id}.{self.config['output']['format']}"
        
        fig.savefig(output_path, dpi=self.config['output']['dpi'], bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Static chart saved: {output_path}")
        return output_path
    
    def _generate_interactive_chart(self, df: pd.DataFrame, pattern: PatternRecord,
                                  output_path: Optional[str] = None) -> str:
        """生成交互式图表（使用Plotly）"""
        
        # 准备数据
        chart_df = self._prepare_chart_data(df, pattern)
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=['Price', 'Volume'],
            row_width=[0.7, 0.3]
        )
        
        # 添加K线图
        fig.add_trace(
            go.Candlestick(
                x=chart_df.index,
                open=chart_df['Open'],
                high=chart_df['High'],
                low=chart_df['Low'],
                close=chart_df['Close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )
        
        # 添加成交量
        colors = ['green' if c >= o else 'red' 
                 for c, o in zip(chart_df['Close'], chart_df['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=chart_df.index,
                y=chart_df['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # 添加形态标注
        self._add_plotly_pattern_annotations(fig, chart_df, pattern)
        
        # 设置布局
        fig.update_layout(
            title=f'{pattern.symbol} - {pattern.pattern_type.title()} Pattern (Confidence: {pattern.confidence_score:.2%})',
            xaxis_rangeslider_visible=False,
            width=self.config['output']['width'],
            height=self.config['output']['height'],
            template='plotly_white'
        )
        
        # 保存图表
        if not output_path:
            Path(self.charts_base_path).mkdir(parents=True, exist_ok=True)
            output_path = f"{self.charts_base_path}/{pattern.id}.html"
        
        fig.write_html(output_path)
        
        logger.info(f"Interactive chart saved: {output_path}")
        return output_path
    
    def _prepare_chart_data(self, df: pd.DataFrame, pattern: PatternRecord) -> pd.DataFrame:
        """准备图表数据"""
        # 确定图表时间范围（形态前后各加少量数据点用于上下文）
        flagpole_start = pattern.flagpole.start_time
        pattern_end = pattern.flagpole.end_time + pd.Timedelta(days=pattern.pattern_duration)
        
        # 缩减扩展时间范围，只保留必要的上下文
        extended_start = flagpole_start - pd.Timedelta(days=3)
        extended_end = pattern_end + pd.Timedelta(days=5)
        
        # 过滤数据
        mask = (df['timestamp'] >= extended_start) & (df['timestamp'] <= extended_end)
        chart_df = df.loc[mask].copy()
        
        # 重新索引并重命名列以适应mplfinance
        chart_df = chart_df.set_index('timestamp')
        chart_df = chart_df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        return chart_df
    
    def _create_additional_plots(self, chart_df: pd.DataFrame, pattern: PatternRecord) -> List:
        """创建附加图线"""
        apds = []
        
        # 添加旗杆标记
        flagpole_mask = (
            (chart_df.index >= pattern.flagpole.start_time) &
            (chart_df.index <= pattern.flagpole.end_time)
        )
        
        if flagpole_mask.any():
            flagpole_line = pd.Series(
                data=chart_df.loc[flagpole_mask, 'Close'],
                index=chart_df.index
            )
            flagpole_line = flagpole_line.reindex(chart_df.index)
            
            apds.append(
                mpf.make_addplot(
                    flagpole_line,
                    type='line',
                    color=self.config['style']['flagpole_color'],
                    width=3,
                    alpha=0.7
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
                            type='line',
                            color=self.config['style']['boundary_color'],
                            width=2,
                            linestyle='--',
                            alpha=0.8
                        )
                    )
        
        return apds
    
    def _create_trend_line_series(self, chart_df: pd.DataFrame, trend_line: TrendLine) -> Optional[pd.Series]:
        """创建趋势线数据系列"""
        try:
            # 找到趋势线时间范围内的数据点
            mask = (
                (chart_df.index >= trend_line.start_time) &
                (chart_df.index <= trend_line.end_time)
            )
            
            if not mask.any():
                return None
            
            # 计算趋势线上每个点的价格
            time_points = chart_df.index[mask]
            total_duration = (trend_line.end_time - trend_line.start_time).total_seconds()
            
            trend_prices = []
            for time_point in time_points:
                progress = (time_point - trend_line.start_time).total_seconds() / total_duration
                price = trend_line.start_price + (trend_line.end_price - trend_line.start_price) * progress
                trend_prices.append(price)
            
            # 创建完整的series，只在趋势线时间范围内有值
            trend_series = pd.Series(index=chart_df.index, dtype=float)
            trend_series.loc[time_points] = trend_prices
            
            return trend_series
            
        except Exception as e:
            logger.warning(f"Failed to create trend line series: {e}")
            return None
    
    def _add_pattern_annotations(self, axes, chart_df: pd.DataFrame, pattern: PatternRecord):
        """添加形态标注（matplotlib版本）"""
        try:
            ax_main = axes[0]  # 主价格图
            
            # 标记旗杆起始点
            flagpole_start_idx = chart_df.index.get_loc(pattern.flagpole.start_time, method='nearest')
            flagpole_end_idx = chart_df.index.get_loc(pattern.flagpole.end_time, method='nearest')
            
            # 添加旗杆箭头
            ax_main.annotate(
                f'Flagpole Start\n{pattern.flagpole.height_percent:.1f}%',
                xy=(flagpole_start_idx, chart_df.iloc[flagpole_start_idx]['Close']),
                xytext=(flagpole_start_idx - 5, chart_df.iloc[flagpole_start_idx]['Close']),
                arrowprops=dict(
                    arrowstyle='->',
                    color=self.config['style']['flagpole_color'],
                    lw=2
                ),
                fontsize=10,
                ha='center'
            )
            
            # 添加置信度信息
            ax_main.text(
                0.02, 0.98,
                f'Pattern ID: {pattern.id[:8]}\n'
                f'Confidence: {pattern.confidence_score:.2%}\n'
                f'Quality: {pattern.pattern_quality}',
                transform=ax_main.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10
            )
            
        except Exception as e:
            logger.warning(f"Failed to add pattern annotations: {e}")
    
    def _add_plotly_pattern_annotations(self, fig, chart_df: pd.DataFrame, pattern: PatternRecord):
        """添加形态标注（Plotly版本）"""
        try:
            # 添加旗杆区域高亮
            fig.add_vrect(
                x0=pattern.flagpole.start_time,
                x1=pattern.flagpole.end_time,
                fillcolor=self.config['style']['flagpole_color'],
                opacity=0.2,
                layer="below",
                line_width=0,
                row=1, col=1
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
                            mode='lines',
                            line=dict(
                                color=self.config['style']['boundary_color'],
                                width=2,
                                dash='dash'
                            ),
                            name=f'Boundary {i+1}',
                            showlegend=i == 0
                        ),
                        row=1, col=1
                    )
            
            # 添加文本标注
            fig.add_annotation(
                x=pattern.flagpole.start_time,
                y=pattern.flagpole.start_price,
                text=f"Flagpole<br>{pattern.flagpole.height_percent:.1f}%",
                showarrow=True,
                arrowhead=2,
                arrowcolor=self.config['style']['flagpole_color'],
                bgcolor="white",
                bordercolor=self.config['style']['flagpole_color'],
                row=1, col=1
            )
            
        except Exception as e:
            logger.warning(f"Failed to add Plotly pattern annotations: {e}")
    
    def generate_summary_chart(self, patterns: List[PatternRecord], 
                             output_path: Optional[str] = None) -> str:
        """
        生成形态汇总图表
        
        Args:
            patterns: 形态列表
            output_path: 输出路径
            
        Returns:
            生成的图表文件路径
        """
        try:
            # 创建汇总统计
            pattern_stats = self._calculate_pattern_statistics(patterns)
            
            # 创建子图
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Pattern Scout - Analysis Summary', fontsize=16, fontweight='bold')
            
            # 1. 形态质量分布
            qualities = [p.pattern_quality for p in patterns]
            quality_counts = pd.Series(qualities).value_counts()
            axes[0, 0].pie(quality_counts.values, labels=quality_counts.index, autopct='%1.1f%%')
            axes[0, 0].set_title('Pattern Quality Distribution')
            
            # 2. 置信度分布
            confidences = [p.confidence_score for p in patterns]
            axes[0, 1].hist(confidences, bins=20, alpha=0.7, color='skyblue')
            axes[0, 1].set_title('Confidence Score Distribution')
            axes[0, 1].set_xlabel('Confidence Score')
            axes[0, 1].set_ylabel('Count')
            
            # 3. 旗杆高度分布
            flagpole_heights = [p.flagpole.height_percent for p in patterns]
            axes[1, 0].hist(flagpole_heights, bins=15, alpha=0.7, color='lightgreen')
            axes[1, 0].set_title('Flagpole Height Distribution')
            axes[1, 0].set_xlabel('Height (%)')
            axes[1, 0].set_ylabel('Count')
            
            # 4. 统计信息文本
            axes[1, 1].axis('off')
            stats_text = f"""
            Pattern Statistics:
            
            Total Patterns: {len(patterns)}
            
            Quality Breakdown:
            High: {quality_counts.get('high', 0)}
            Medium: {quality_counts.get('medium', 0)}
            Low: {quality_counts.get('low', 0)}
            
            Average Confidence: {np.mean(confidences):.2%}
            Average Flagpole Height: {np.mean(flagpole_heights):.1f}%
            
            Time Range:
            From: {min(p.flagpole.start_time for p in patterns).strftime('%Y-%m-%d')}
            To: {max(p.flagpole.start_time for p in patterns).strftime('%Y-%m-%d')}
            """
            
            axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            
            # 保存图表
            if not output_path:
                Path(self.charts_base_path).mkdir(parents=True, exist_ok=True)
                output_path = f"{self.charts_base_path}/pattern_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            plt.savefig(output_path, dpi=self.config['output']['dpi'], bbox_inches='tight')
            plt.close()
            
            logger.info(f"Summary chart saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to generate summary chart: {e}")
            raise
    
    def _calculate_pattern_statistics(self, patterns: List[PatternRecord]) -> Dict[str, Any]:
        """计算形态统计信息"""
        if not patterns:
            return {}
        
        return {
            'total_count': len(patterns),
            'quality_distribution': pd.Series([p.pattern_quality for p in patterns]).value_counts(),
            'avg_confidence': np.mean([p.confidence_score for p in patterns]),
            'avg_flagpole_height': np.mean([p.flagpole.height_percent for p in patterns]),
            'time_range': {
                'start': min(p.flagpole.start_time for p in patterns),
                'end': max(p.flagpole.start_time for p in patterns)
            }
        }
    
    def load_pattern_data(self, pattern_file: str) -> dict:
        """加载形态数据"""
        try:
            with open(pattern_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load pattern data from {pattern_file}: {e}")
            return {}
    
    def create_meaningful_filename(self, pattern_data: dict) -> str:
        """创建有意义的文件名：资产名称_起始时间_形态名称"""
        try:
            # 提取资产名称（去掉-15min后缀）
            symbol = pattern_data['symbol'].replace('-15min', '')
            
            # 提取旗杆起始时间并格式化
            start_time = datetime.fromisoformat(pattern_data['flagpole']['start_time'])
            time_str = start_time.strftime('%Y%m%d_%H%M')
            
            # 形态名称映射
            pattern_type = pattern_data['pattern_type']
            pattern_config = self.config['pattern_types'].get(pattern_type, {})
            pattern_name = pattern_config.get('name_cn', pattern_type)
            
            # 方向信息
            direction = pattern_data['flagpole']['direction']
            direction_name = '上升' if direction == 'up' else '下降'
            
            # 组合文件名
            filename = f"{symbol}_{time_str}_{direction_name}{pattern_name}"
            return filename
            
        except Exception as e:
            logger.warning(f"Failed to create meaningful filename: {e}")
            # 回退到使用ID
            return pattern_data.get('id', 'unknown')
    
    def generate_pattern_chart_from_data(self, pattern_data: dict, save_path: str = None) -> str:
        """从形态数据生成TradingView风格图表"""
        try:
            pattern_type = pattern_data.get('pattern_type', 'flag')
            pattern_config = self.config['pattern_types'].get(pattern_type, self.config['pattern_types']['flag'])
            
            # 解析时间
            flagpole_start = datetime.fromisoformat(pattern_data['flagpole']['start_time'])
            flagpole_end = datetime.fromisoformat(pattern_data['flagpole']['end_time'])
            
            # 缩减时间范围，只保留必要的时间段
            time_before = pd.Timedelta(days=3)  # 形态前3天
            time_after = pd.Timedelta(days=5)   # 形态后5天
            
            extended_start = flagpole_start - time_before
            extended_end = flagpole_end + time_after
            
            # 获取价格数据
            df = self.data_connector.get_data(pattern_data['symbol'], extended_start, extended_end)
            
            if df.empty:
                logger.warning(f"No data found for pattern {pattern_data.get('id', 'unknown')}")
                return None
            
            # 准备mplfinance数据格式
            df_plot = df.set_index('timestamp')
            df_plot = df_plot.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
            })
            
            # 创建附加图线
            addplots = []
            
            # 旗杆标记
            flagpole_mask = (df_plot.index >= flagpole_start) & (df_plot.index <= flagpole_end)
            if flagpole_mask.any():
                flagpole_line = pd.Series(index=df_plot.index, dtype=float)
                flagpole_line.loc[flagpole_mask] = df_plot.loc[flagpole_mask, 'Close']
                
                addplots.append(
                    mpf.make_addplot(flagpole_line, type='line', color=pattern_config['color_flagpole'], 
                                   width=4, alpha=0.8)
                )
            
            # 边界线
            if 'pattern_boundaries' in pattern_data and pattern_data['pattern_boundaries']:
                for boundary in pattern_data['pattern_boundaries']:
                    start_time = datetime.fromisoformat(boundary['start_time'])
                    end_time = datetime.fromisoformat(boundary['end_time'])
                    
                    # 创建边界线数据
                    boundary_mask = (df_plot.index >= start_time) & (df_plot.index <= end_time)
                    if boundary_mask.any():
                        time_points = df_plot.index[boundary_mask]
                        total_duration = (end_time - start_time).total_seconds()
                        
                        boundary_prices = []
                        for time_point in time_points:
                            progress = (time_point - start_time).total_seconds() / total_duration
                            price = boundary['start_price'] + (boundary['end_price'] - boundary['start_price']) * progress
                            boundary_prices.append(price)
                        
                        boundary_line = pd.Series(index=df_plot.index, dtype=float)
                        boundary_line.loc[time_points] = boundary_prices
                        
                        addplots.append(
                            mpf.make_addplot(boundary_line, type='line', 
                                           color=pattern_config.get('color_upper', '#2196f3'),
                                           width=2, linestyle='--', alpha=0.8)
                        )
            
            # 设置TradingView风格
            style = mpf.make_mpf_style(
                base_mpf_style='charles',
                marketcolors=mpf.make_marketcolors(
                    up=self.config['style']['up_color'],
                    down=self.config['style']['down_color'],
                    edge={'up': self.config['style']['up_color'], 'down': self.config['style']['down_color']},
                    wick={'up': self.config['style']['up_color'], 'down': self.config['style']['down_color']},
                    volume=self.config['style']['volume_color']
                ),
                gridstyle='--',
                gridcolor=self.config['style']['grid_color'],
                facecolor=self.config['style']['background_color'],
                figcolor=self.config['style']['background_color']
            )
            
            # 创建标题
            direction_text = "上升" if pattern_data['flagpole']['direction'] == 'up' else "下降"
            pattern_name = pattern_config['name_cn']
            title = (f"{pattern_data['symbol']} - {direction_text}{pattern_name}\\n"
                    f"置信度: {pattern_data['confidence_score']:.3f} | "
                    f"质量: {pattern_data['pattern_quality']} | "
                    f"旗杆高度: {pattern_data['flagpole']['height_percent']:.2f}%")
            
            # 绘制图表
            fig, axes = mpf.plot(
                df_plot,
                type='candle',
                style=style,
                addplot=addplots,
                volume=True,
                title=title,
                figsize=(16, 10),
                tight_layout=True,
                returnfig=True,
                warn_too_much_data=1000,
                savefig=dict(fname=save_path, dpi=300, bbox_inches='tight') if save_path else None
            )
            
            if save_path:
                logger.info(f"TradingView style chart saved: {save_path}")
                plt.close(fig)
            else:
                plt.show()
            
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to generate chart from pattern data: {e}")
            return None
    
    def generate_classified_charts(self, patterns_dir: str = "output/data/patterns") -> Dict[str, Any]:
        """按形态类型分类生成图表"""
        try:
            patterns_path = Path(patterns_dir)
            pattern_files = list(patterns_path.glob("*.json"))
            
            if not pattern_files:
                logger.warning("No pattern files found!")
                return {}
            
            # 按类型分类形态
            classified_patterns = {'flag': [], 'pennant': []}
            
            for pattern_file in pattern_files:
                pattern_data = self.load_pattern_data(pattern_file)
                if pattern_data:
                    pattern_type = pattern_data.get('pattern_type', 'flag')
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
                    
                    logger.info(f"Generating {i}/{len(patterns)} {pattern_type} chart: {filename}")
                    
                    result = self.generate_pattern_chart_from_data(pattern_data, str(save_path))
                    if result:
                        charts_generated += 1
                        chart_paths.append(str(save_path))
                        logger.info(f"TradingView style chart saved: {save_path}")
                
                results[pattern_type] = {
                    'charts_generated': charts_generated,
                    'total_patterns': len(patterns),
                    'charts': chart_paths
                }
                total_charts += charts_generated
            
            logger.info(f"Classification chart generation completed. Generated {total_charts} charts")
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
                'total': len(pattern_dicts),
                'by_type': {},
                'by_quality': {},
                'by_direction': {}
            }
            
            for pattern_dict in pattern_dicts:
                # 按类型统计
                pattern_type = pattern_dict.get('pattern_type', 'unknown')
                pattern_stats['by_type'][pattern_type] = pattern_stats['by_type'].get(pattern_type, 0) + 1
                
                # 按质量统计
                quality = pattern_dict.get('pattern_quality', 'unknown')
                pattern_stats['by_quality'][quality] = pattern_stats['by_quality'].get(quality, 0) + 1
                
                # 按方向统计
                flagpole_data = pattern_dict.get('flagpole', {})
                direction = flagpole_data.get('direction', 'unknown')
                pattern_stats['by_direction'][direction] = pattern_stats['by_direction'].get(direction, 0) + 1
            
            # 生成文本报告
            report_lines = [
                "\\n=== PatternScout TradingView风格形态分析报告 ===",
                f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"总计检测形态数量: {pattern_stats['total']}",
                "\\n=== 形态类型分布 ===",
            ]
            
            for pattern_type, count in pattern_stats['by_type'].items():
                percentage = (count / pattern_stats['total']) * 100
                type_name = self.config['pattern_types'].get(pattern_type, {}).get('name_cn', pattern_type)
                report_lines.append(f"{type_name}: {count} ({percentage:.1f}%)")
            
            report_lines.extend([
                "\\n=== 质量分布 ===",
            ])
            
            for quality, count in pattern_stats['by_quality'].items():
                percentage = (count / pattern_stats['total']) * 100
                report_lines.append(f"{quality.upper()}: {count} ({percentage:.1f}%)")
            
            report_lines.extend([
                "\\n=== 方向分布 ===",
            ])
            
            for direction, count in pattern_stats['by_direction'].items():
                percentage = (count / pattern_stats['total']) * 100
                direction_name = '上升' if direction == 'up' else '下降' if direction == 'down' else direction
                report_lines.append(f"{direction_name}: {count} ({percentage:.1f}%)")
            
            # 计算平均置信度
            confidence_scores = [p.get('confidence_score', 0) for p in pattern_dicts if p.get('confidence_score') is not None]
            if confidence_scores:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                max_confidence = max(confidence_scores)
            else:
                avg_confidence = 0
                max_confidence = 0
            
            report_lines.extend([
                "\\n=== 统计指标 ===",
                f"平均置信度: {avg_confidence:.3f}",
                f"最高置信度: {max_confidence:.3f}"
            ])
            
            report_text = "\\n".join(report_lines)
            print(report_text)
            
            logger.info("Summary report generated successfully")
            
        except Exception as e:
            logger.error(f"Failed to create summary report: {e}")