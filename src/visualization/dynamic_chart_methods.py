"""
动态基线系统图表生成方法扩展
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from loguru import logger
import mplfinance as mpf
from pathlib import Path


class DynamicChartMethods:
    """动态图表生成方法混入类"""
    
    def _create_dynamic_baseline_style(self):
        """创建动态基线风格"""
        return mpf.make_mpf_style(
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
            gridstyle="-.",
            gridcolor=self.config["style"]["grid_color"],
            facecolor=self.config["style"]["background_color"],
            figcolor=self.config["style"]["background_color"],
        )
    
    def _create_dynamic_chart_title(self, pattern, market_snapshot=None) -> str:
        """创建动态图表标题"""
        # 获取形态子类型名称
        sub_type = getattr(pattern, 'sub_type', 'flag')
        sub_type_config = self.config["pattern_types"]["flag_pattern"]["subtypes"].get(sub_type, {})
        sub_type_name = sub_type_config.get("name_cn", sub_type)
        
        # 基本信息
        direction_text = "上升" if pattern.flagpole.direction == "up" else "下降"
        
        title_parts = [
            f"{pattern.symbol} - {direction_text}{sub_type_name}",
            f"置信度: {pattern.confidence_score:.3f}",
            f"质量: {pattern.pattern_quality}",
            f"旗杆高度: {pattern.flagpole.height_percent:.2f}%"
        ]
        
        # 添加市场状态信息
        if market_snapshot:
            regime_text = "高波动" if market_snapshot.current_regime == "high_volatility" else "低波动"
            title_parts.append(f"市场状态: {regime_text}")
        
        return " | ".join(title_parts)
    
    def _create_dynamic_additional_plots(
        self, 
        chart_df: pd.DataFrame, 
        pattern,
        market_snapshot=None,
        baseline_data=None
    ) -> List:
        """创建动态附加图线"""
        apds = []
        
        # 1. 旗杆标记
        flagpole_line = self._create_flagpole_line(chart_df, pattern)
        if flagpole_line is not None:
            apds.append(
                mpf.make_addplot(
                    flagpole_line,
                    type="line",
                    color=self.config["style"]["flagpole_color"],
                    width=2.5,
                    alpha=0.9,
                )
            )
        
        # 2. 形态边界线
        if pattern.pattern_boundaries:
            boundary_colors = self._get_boundary_colors(pattern)
            for i, boundary in enumerate(pattern.pattern_boundaries):
                boundary_line = self._create_trend_line_series(chart_df, boundary)
                if boundary_line is not None:
                    color = boundary_colors[i] if i < len(boundary_colors) else "#ff9800"
                    apds.append(
                        mpf.make_addplot(
                            boundary_line,
                            type="line",
                            color=color,
                            width=2.0,
                            linestyle="--",
                            alpha=0.8,
                        )
                    )
        
        # 3. 动态阈值线（如果有基线数据）
        if baseline_data and self.visualization_layers['dynamic_thresholds']:
            threshold_lines = self._create_dynamic_threshold_lines(chart_df, baseline_data)
            apds.extend(threshold_lines)
        
        # 4. 失效信号标记
        if hasattr(pattern, 'invalidation_signals') and self.visualization_layers['invalidation_signals']:
            invalidation_markers = self._create_invalidation_markers(chart_df, pattern.invalidation_signals)
            apds.extend(invalidation_markers)
        
        return apds
    
    def _get_boundary_colors(self, pattern) -> List[str]:
        """获取边界线颜色"""
        sub_type = getattr(pattern, 'sub_type', 'flag')
        sub_type_config = self.config["pattern_types"]["flag_pattern"]["subtypes"].get(sub_type, {})
        base_color = sub_type_config.get("color_boundary", "#2196f3")
        
        # 为上下边界线生成不同色调
        colors = [base_color, self._adjust_color_brightness(base_color, 0.7)]
        return colors
    
    def _adjust_color_brightness(self, hex_color: str, factor: float) -> str:
        """调整颜色亮度"""
        try:
            # 简单的颜色调整，返回更深或更浅的颜色
            if factor < 1.0:
                return hex_color.replace('#', '#4d')  # 加深
            else:
                return hex_color.replace('#', '#b3')  # 加浅
        except:
            return hex_color
    
    def _create_flagpole_line(self, chart_df: pd.DataFrame, pattern) -> Optional[pd.Series]:
        """创建旗杆线"""
        flagpole_mask = (
            (chart_df.index >= pattern.flagpole.start_time) & 
            (chart_df.index <= pattern.flagpole.end_time)
        )
        
        if not flagpole_mask.any():
            return None
        
        flagpole_times = chart_df.index[flagpole_mask]
        total_duration = (pattern.flagpole.end_time - pattern.flagpole.start_time).total_seconds()
        
        flagpole_line = pd.Series(index=chart_df.index, dtype=float)
        
        for time_point in flagpole_times:
            progress = (
                (time_point - pattern.flagpole.start_time).total_seconds() / total_duration
                if total_duration > 0 else 0
            )
            price = (
                pattern.flagpole.start_price + 
                (pattern.flagpole.end_price - pattern.flagpole.start_price) * progress
            )
            flagpole_line.loc[time_point] = price
        
        return flagpole_line
    
    def _create_dynamic_threshold_lines(self, chart_df: pd.DataFrame, 
                                       baseline_data: Dict[str, Any]) -> List:
        """创建动态阈值线"""
        threshold_plots = []
        
        # 这里可以根据基线数据创建阈值线
        # 简化实现：返回空列表
        return threshold_plots
    
    def _create_invalidation_markers(self, chart_df: pd.DataFrame, 
                                    invalidation_signals) -> List:
        """创建失效信号标记"""
        markers = []
        
        for signal in invalidation_signals:
            # 查找信号时间点
            try:
                signal_time = signal.trigger_time
                if signal_time in chart_df.index:
                    signal_price = signal.trigger_price
                    
                    # 创建标记点序列
                    marker_series = pd.Series(index=chart_df.index, dtype=float)
                    marker_series.loc[signal_time] = signal_price
                    
                    # 根据信号类型选择颜色
                    color = self.config["dynamic_baseline"]["invalidation_colors"].get(
                        signal.signal_type, "#f44336"
                    )
                    
                    markers.append(
                        mpf.make_addplot(
                            marker_series,
                            type="scatter",
                            markersize=100 if signal.is_critical else 60,
                            color=color,
                            alpha=0.8,
                        )
                    )
            except Exception as e:
                logger.warning(f"Failed to create invalidation marker: {e}")
        
        return markers
    
    def _add_dynamic_pattern_annotations(
        self, 
        axes, 
        chart_df: pd.DataFrame, 
        pattern,
        market_snapshot=None,
        outcome_analysis=None,
        baseline_data=None
    ):
        """添加动态基线模式标注"""
        try:
            ax_main = axes[0]  # 主价格图
            
            # 1. 基本形态信息
            info_text = self._create_pattern_info_text(
                pattern, market_snapshot, outcome_analysis
            )
            
            ax_main.text(
                0.02, 0.98,
                info_text,
                transform=ax_main.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
                fontsize=10,
            )
            
            # 2. 动态阈值信息（如果有）
            if baseline_data and self.visualization_layers['baseline_indicators']:
                threshold_text = self._create_threshold_info_text(baseline_data)
                if threshold_text:
                    ax_main.text(
                        0.02, 0.78,
                        threshold_text,
                        transform=ax_main.transAxes,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
                        fontsize=9,
                    )
            
            # 3. 旗杆起始点标注
            self._add_flagpole_annotation(ax_main, chart_df, pattern)
            
            # 4. 结局分析标注（如果有）
            if outcome_analysis and self.visualization_layers['outcome_tracking']:
                self._add_outcome_annotation(ax_main, chart_df, outcome_analysis)
            
        except Exception as e:
            logger.warning(f"Failed to add dynamic pattern annotations: {e}")
    
    def _create_pattern_info_text(
        self, 
        pattern,
        market_snapshot=None,
        outcome_analysis=None
    ) -> str:
        """创建形态信息文本"""
        lines = [
            f"ID: {pattern.id[:8]}",
            f"置信度: {pattern.confidence_score:.3f}",
            f"质量: {pattern.pattern_quality}",
            f"旗杆高度: {pattern.flagpole.height_percent:.2f}%"
        ]
        
        # 添加市场状态
        if market_snapshot:
            regime_text = {
                "high_volatility": "高波动",
                "low_volatility": "低波动",
                "unknown": "未知"
            }.get(market_snapshot.current_regime, market_snapshot.current_regime)
            lines.append(f"市场状态: {regime_text}")
        
        # 添加结局信息
        if outcome_analysis:
            outcome_text = {
                "strong_continuation": "强势延续",
                "standard_continuation": "标准延续", 
                "breakout_stagnation": "突破停滞",
                "failed_breakout": "假突破反转",
                "internal_collapse": "内部瘫解",
                "opposite_run": "反向运行",
                "monitoring": "监控中"
            }.get(outcome_analysis.outcome.value, outcome_analysis.outcome.value)
            lines.append(f"结局: {outcome_text}")
        
        return "\n".join(lines)
    
    def _create_threshold_info_text(self, baseline_data: Dict[str, Any]) -> str:
        """创建阈值信息文本"""
        lines = ["动态阈值:"]
        
        # 简化实现：显示部分基线数据
        if 'regime_thresholds' in baseline_data:
            thresholds = baseline_data['regime_thresholds']
            for key, value in thresholds.items():
                if isinstance(value, (int, float)):
                    lines.append(f"{key}: {value:.3f}")
        
        return "\n".join(lines) if len(lines) > 1 else ""
    
    def _add_flagpole_annotation(self, ax_main, chart_df: pd.DataFrame, pattern):
        """添加旗杆标注"""
        try:
            flagpole_start_idx = chart_df.index.get_loc(
                pattern.flagpole.start_time, method="nearest"
            )
            
            ax_main.annotate(
                f"旗杆起点\n{pattern.flagpole.height_percent:.1f}%",
                xy=(flagpole_start_idx, chart_df.iloc[flagpole_start_idx]["Close"]),
                xytext=(flagpole_start_idx - 8, chart_df.iloc[flagpole_start_idx]["Close"]),
                arrowprops=dict(
                    arrowstyle="->", 
                    color=self.config["style"]["flagpole_color"], 
                    lw=2
                ),
                fontsize=10,
                ha="center",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
            )
        except Exception as e:
            logger.warning(f"Failed to add flagpole annotation: {e}")
    
    def _add_outcome_annotation(self, ax_main, chart_df: pd.DataFrame, 
                              outcome_analysis):
        """添加结局标注"""
        try:
            # 在图表右上角显示结局信息
            outcome_color = self.config["outcome_tracking"]["outcome_colors"].get(
                outcome_analysis.outcome.value, "#607d8b"
            )
            
            outcome_text = {
                "strong_continuation": "强势延续",
                "standard_continuation": "标准延续",
                "breakout_stagnation": "突破停滞",
                "failed_breakout": "假突破",
                "internal_collapse": "内部瘫解",
                "opposite_run": "反向运行"
            }.get(outcome_analysis.outcome.value, outcome_analysis.outcome.value)
            
            ax_main.text(
                0.98, 0.98,
                f"结局: {outcome_text}\n监控周期: {outcome_analysis.monitoring_duration}根K线",
                transform=ax_main.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor=outcome_color, alpha=0.8),
                fontsize=10,
                color="white"
            )
        except Exception as e:
            logger.warning(f"Failed to add outcome annotation: {e}")