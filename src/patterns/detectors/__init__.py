"""
形态检测器包
提供各种技术形态的检测器实现

重构说明：
- FlagDetector现在是统一的旗形检测器，支持矩形旗和三角旗检测
- PennantDetector已被移除，其功能集成到FlagDetector中
- PatternScanner已更新以使用新的统一架构
"""
from .flag_detector import FlagDetector
from .pattern_scanner import PatternScanner

__all__ = [
    'FlagDetector',
    'PatternScanner'
]