from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field


@dataclass
class PriceData:
    """OHLCV价格数据模型"""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def __post_init__(self):
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError("Invalid OHLC data: high/low values inconsistent")


@dataclass
class TrendLine:
    """趋势线模型"""
    start_time: datetime
    end_time: datetime
    start_price: float
    end_price: float
    slope: float
    r_squared: float  # 拟合度


@dataclass
class Flagpole:
    """旗杆模型"""
    start_time: datetime
    end_time: datetime
    start_price: float
    end_price: float
    height_percent: float
    direction: str  # 'up' or 'down'
    volume_ratio: float  # 相对平均成交量的比例


class PatternType:
    """形态类型枚举"""
    FLAG = "flag"
    PENNANT = "pennant"


@dataclass
class PatternRecord:
    """形态记录模型"""
    id: str
    symbol: str
    pattern_type: str
    detection_date: datetime
    
    # 形态特征
    flagpole: Flagpole
    pattern_boundaries: List[TrendLine]
    pattern_duration: int  # 天数
    
    # 突破分析
    breakthrough_date: Optional[datetime] = None
    breakthrough_price: Optional[float] = None
    result_type: Optional[str] = None  # 'continuation', 'reversal', 'sideways'
    
    # 评分
    confidence_score: float = 0.0
    pattern_quality: str = "unknown"  # 'high', 'medium', 'low'
    
    # 文件路径
    chart_path: Optional[str] = None
    data_path: Optional[str] = None


class BreakthroughResult(BaseModel):
    """突破结果模型"""
    pattern_id: str
    breakthrough_date: datetime
    breakthrough_price: float
    result_type: str = Field(..., pattern="^(continuation|reversal|sideways)$")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    analysis_details: dict = Field(default_factory=dict)


class AnalysisResult(BaseModel):
    """分析结果模型"""
    pattern: PatternRecord
    breakthrough: Optional[BreakthroughResult] = None
    scores: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)