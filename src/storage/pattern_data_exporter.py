"""
阶段4：形态数据输出模块
专注于结构化数据记录和导出功能
"""
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import uuid
from dataclasses import asdict

from src.data.models.base_models import (
    PatternRecord, PatternOutcomeAnalysis, MarketSnapshot,
    InvalidationSignal, PatternOutcome, MarketRegime
)
from loguru import logger


def datetime_serializer(obj):
    """自定义datetime序列化器"""
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif isinstance(obj, np.datetime64):
        return pd.Timestamp(obj).isoformat()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class PatternDataExporter:
    """
    形态数据输出器（阶段4）
    专注于数据结构化存储和导出
    """
    
    def __init__(self, output_root: str = "output/data"):
        """
        初始化数据输出器
        
        Args:
            output_root: 数据输出根目录
        """
        self.output_root = Path(output_root)
        self.db_path = self.output_root / "patterns.db"
        self._ensure_directories()
        self._init_database()
        
        # 数据缓存
        self.pattern_cache: Dict[str, PatternRecord] = {}
        self.outcome_cache: Dict[str, PatternOutcomeAnalysis] = {}
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        directories = [
            self.output_root,
            self.output_root / "patterns",
            self.output_root / "outcomes", 
            self.output_root / "snapshots",
            self.output_root / "exports",
            self.output_root / "backups"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"数据输出目录已初始化: {self.output_root}")
    
    def _get_db_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL") 
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=memory")
        return conn
    
    def _init_database(self):
        """初始化SQLite数据库"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # 创建形态记录表（增强版）
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS patterns (
                        id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        pattern_type TEXT NOT NULL,
                        sub_type TEXT,
                        detection_date TEXT NOT NULL,
                        flagpole_start_time TEXT NOT NULL,
                        flagpole_end_time TEXT NOT NULL,
                        flagpole_height_percent REAL NOT NULL,
                        flagpole_direction TEXT NOT NULL,
                        pattern_duration INTEGER NOT NULL,
                        confidence_score REAL NOT NULL,
                        pattern_quality TEXT NOT NULL,
                        market_regime TEXT,
                        invalidation_signals_count INTEGER DEFAULT 0,
                        chart_file_path TEXT,
                        data_file_path TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 创建形态DNA特征表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS pattern_dna (
                        pattern_id TEXT PRIMARY KEY,
                        slope_score REAL,
                        volume_burst REAL,
                        impulse_bars_percent REAL,
                        retrace_ratio REAL,
                        pole_bars_count INTEGER,
                        retrace_depth REAL,
                        volume_contraction REAL,
                        volatility_drop REAL,
                        channel_width REAL,
                        parallelism REAL,
                        convergence REAL,
                        flag_bars_count INTEGER,
                        FOREIGN KEY (pattern_id) REFERENCES patterns (id)
                    )
                """)
                
                # 创建结局分析表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS pattern_outcomes (
                        id TEXT PRIMARY KEY,
                        pattern_id TEXT NOT NULL,
                        outcome_classification TEXT NOT NULL,
                        analysis_date TEXT NOT NULL,
                        monitoring_duration INTEGER NOT NULL,
                        breakout_level REAL,
                        invalidation_level REAL,
                        target_projection_1 REAL,
                        risk_distance REAL,
                        actual_high REAL,
                        actual_low REAL,
                        breakthrough_occurred BOOLEAN,
                        breakthrough_direction TEXT,
                        success_ratio REAL,
                        holding_period INTEGER,
                        final_return REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (pattern_id) REFERENCES patterns (id)
                    )
                """)
                
                # 创建环境快照表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS market_snapshots (
                        id TEXT PRIMARY KEY,
                        pattern_id TEXT NOT NULL,
                        market_regime TEXT NOT NULL,
                        volatility_level REAL,
                        trend_strength REAL,
                        volume_profile TEXT,
                        snapshot_date TEXT NOT NULL,
                        additional_context TEXT,
                        FOREIGN KEY (pattern_id) REFERENCES patterns (id)
                    )
                """)
                
                # 创建失效信号表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS invalidation_signals (
                        id TEXT PRIMARY KEY,
                        pattern_id TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        signal_strength REAL,
                        detection_time TEXT NOT NULL,
                        description TEXT,
                        FOREIGN KEY (pattern_id) REFERENCES patterns (id)
                    )
                """)
                
                # 创建索引
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_symbol ON patterns (symbol)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns (pattern_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_sub_type ON patterns (sub_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_quality ON patterns (pattern_quality)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_regime ON patterns (market_regime)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_pattern_id ON pattern_outcomes (pattern_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_classification ON pattern_outcomes (outcome_classification)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_pattern_id ON market_snapshots (pattern_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_invalidation_pattern_id ON invalidation_signals (pattern_id)")
                
                conn.commit()
                
            logger.info("数据库初始化成功")
            
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise
    
    def save_pattern_record(self, pattern: PatternRecord, 
                           market_snapshot: Optional[MarketSnapshot] = None,
                           invalidation_signals: Optional[List[InvalidationSignal]] = None) -> bool:
        """
        保存形态记录及相关数据
        
        Args:
            pattern: 形态记录
            market_snapshot: 市场快照
            invalidation_signals: 失效信号列表
            
        Returns:
            保存是否成功
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # 保存形态主记录
                cursor.execute("""
                    INSERT OR REPLACE INTO patterns 
                    (id, symbol, pattern_type, sub_type, detection_date, 
                     flagpole_start_time, flagpole_end_time, flagpole_height_percent, 
                     flagpole_direction, pattern_duration, confidence_score, 
                     pattern_quality, market_regime, invalidation_signals_count,
                     chart_file_path, data_file_path, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern.id,
                    pattern.symbol,
                    pattern.pattern_type,
                    pattern.sub_type.value if pattern.sub_type else None,
                    pattern.detection_date.isoformat(),
                    pattern.flagpole.start_time.isoformat(),
                    pattern.flagpole.end_time.isoformat(),
                    pattern.flagpole.height_percent,
                    pattern.flagpole.direction,
                    pattern.pattern_duration,
                    pattern.confidence_score,
                    pattern.pattern_quality,
                    market_snapshot.regime.value if market_snapshot else None,
                    len(invalidation_signals) if invalidation_signals else 0,
                    pattern.chart_path,
                    pattern.data_path,
                    datetime.now().isoformat()
                ))
                
                # 保存形态DNA特征
                self._save_pattern_dna(cursor, pattern)
                
                # 保存市场快照
                if market_snapshot:
                    self._save_market_snapshot(cursor, pattern.id, market_snapshot)
                
                # 保存失效信号
                if invalidation_signals:
                    self._save_invalidation_signals(cursor, pattern.id, invalidation_signals)
                
                conn.commit()
            
            # 保存详细JSON文件
            self._save_pattern_json(pattern, market_snapshot, invalidation_signals)
            
            # 缓存形态记录
            self.pattern_cache[pattern.id] = pattern
            
            logger.info(f"形态记录 {pattern.id} 保存成功")
            return True
            
        except Exception as e:
            logger.error(f"保存形态记录 {pattern.id} 失败: {e}")
            return False
    
    def _save_pattern_dna(self, cursor, pattern: PatternRecord):
        """保存形态DNA特征"""
        # 从形态记录中提取DNA特征
        # 这里需要根据实际的PatternRecord结构调整
        flagpole = pattern.flagpole
        boundaries = pattern.pattern_boundaries or []
        
        # 计算DNA特征值（示例）
        dna_features = {
            'slope_score': getattr(flagpole, 'slope_score', None),
            'volume_burst': getattr(flagpole, 'volume_ratio', None),
            'impulse_bars_percent': None,  # 需要从详细数据计算
            'retrace_ratio': None,
            'pole_bars_count': getattr(flagpole, 'bars_count', None),
            'retrace_depth': None,
            'volume_contraction': None,
            'volatility_drop': None,
            'channel_width': None,
            'parallelism': boundaries[0].r_squared if boundaries else None,
            'convergence': None,
            'flag_bars_count': pattern.pattern_duration
        }
        
        cursor.execute("""
            INSERT OR REPLACE INTO pattern_dna 
            (pattern_id, slope_score, volume_burst, impulse_bars_percent, 
             retrace_ratio, pole_bars_count, retrace_depth, volume_contraction,
             volatility_drop, channel_width, parallelism, convergence, flag_bars_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pattern.id,
            dna_features['slope_score'],
            dna_features['volume_burst'],
            dna_features['impulse_bars_percent'],
            dna_features['retrace_ratio'],
            dna_features['pole_bars_count'],
            dna_features['retrace_depth'],
            dna_features['volume_contraction'],
            dna_features['volatility_drop'],
            dna_features['channel_width'],
            dna_features['parallelism'],
            dna_features['convergence'],
            dna_features['flag_bars_count']
        ))
    
    def _save_market_snapshot(self, cursor, pattern_id: str, snapshot: MarketSnapshot):
        """保存市场快照"""
        cursor.execute("""
            INSERT OR REPLACE INTO market_snapshots 
            (id, pattern_id, market_regime, volatility_level, trend_strength,
             volume_profile, snapshot_date, additional_context)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()),
            pattern_id,
            snapshot.regime.value,
            snapshot.volatility_percentile,
            snapshot.trend_strength,
            json.dumps(snapshot.volume_profile) if hasattr(snapshot, 'volume_profile') else None,
            snapshot.timestamp.isoformat(),
            json.dumps(snapshot.to_dict(), default=datetime_serializer)
        ))
    
    def _save_invalidation_signals(self, cursor, pattern_id: str, signals: List[InvalidationSignal]):
        """保存失效信号"""
        for signal in signals:
            cursor.execute("""
                INSERT INTO invalidation_signals 
                (id, pattern_id, signal_type, signal_strength, detection_time, description)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()),
                pattern_id,
                signal.signal_type.value,
                signal.strength,
                signal.detection_time.isoformat(),
                signal.description
            ))
    
    def save_outcome_analysis(self, outcome_analysis: PatternOutcomeAnalysis) -> bool:
        """
        保存结局分析
        
        Args:
            outcome_analysis: 结局分析记录
            
        Returns:
            保存是否成功
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO pattern_outcomes 
                    (id, pattern_id, outcome_classification, analysis_date, 
                     monitoring_duration, breakout_level, invalidation_level, 
                     target_projection_1, risk_distance, actual_high, actual_low,
                     breakthrough_occurred, breakthrough_direction, success_ratio,
                     holding_period, final_return)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()),
                    outcome_analysis.pattern_id,
                    outcome_analysis.outcome.value,
                    outcome_analysis.analysis_date.isoformat(),
                    outcome_analysis.monitoring_duration,
                    outcome_analysis.breakout_level,
                    outcome_analysis.invalidation_level,
                    outcome_analysis.target_projection_1,
                    outcome_analysis.risk_distance,
                    outcome_analysis.actual_high,
                    outcome_analysis.actual_low,
                    outcome_analysis.breakthrough_occurred,
                    outcome_analysis.breakthrough_direction,
                    outcome_analysis.success_ratio,
                    outcome_analysis.holding_period,
                    outcome_analysis.final_return
                ))
                
                conn.commit()
            
            # 保存详细JSON文件
            self._save_outcome_json(outcome_analysis)
            
            # 缓存结局分析
            self.outcome_cache[outcome_analysis.pattern_id] = outcome_analysis
            
            logger.info(f"结局分析 {outcome_analysis.pattern_id} 保存成功")
            return True
            
        except Exception as e:
            logger.error(f"保存结局分析 {outcome_analysis.pattern_id} 失败: {e}")
            return False
    
    def _save_pattern_json(self, pattern: PatternRecord, 
                          market_snapshot: Optional[MarketSnapshot],
                          invalidation_signals: Optional[List[InvalidationSignal]]):
        """保存形态JSON文件"""
        filename = self._create_meaningful_filename(pattern)
        pattern_file = self.output_root / "patterns" / f"{filename}.json"
        
        pattern_data = {
            'pattern': self._pattern_to_dict(pattern),
            'market_snapshot': market_snapshot.to_dict() if market_snapshot else None,
            'invalidation_signals': [signal.to_dict() for signal in invalidation_signals] if invalidation_signals else [],
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(pattern_file, 'w', encoding='utf-8') as f:
            json.dump(pattern_data, f, ensure_ascii=False, indent=2, default=datetime_serializer)
    
    def _save_outcome_json(self, outcome_analysis: PatternOutcomeAnalysis):
        """保存结局分析JSON文件"""
        filename = f"{outcome_analysis.pattern_id}_{outcome_analysis.outcome.value}_{outcome_analysis.analysis_date.strftime('%Y%m%d_%H%M')}"
        outcome_file = self.output_root / "outcomes" / f"{filename}.json"
        
        with open(outcome_file, 'w', encoding='utf-8') as f:
            json.dump(outcome_analysis.to_dict(), f, ensure_ascii=False, indent=2, default=datetime_serializer)
    
    def _create_meaningful_filename(self, pattern: PatternRecord) -> str:
        """创建有意义的文件名"""
        try:
            # 提取资产名称
            symbol = pattern.symbol.replace('-15min', '').replace('-5min', '').replace('-1h', '').strip()
            if not symbol:
                symbol = 'unknown_symbol'
            
            # 格式化时间
            start_time = pattern.flagpole.start_time
            time_str = start_time.strftime('%Y%m%d_%H%M')
            
            # 形态名称
            pattern_names = {
                'flag': '旗形',
                'pennant': '三角旗形'
            }
            sub_type = pattern.sub_type.value if pattern.sub_type else pattern.pattern_type
            pattern_name = pattern_names.get(sub_type, sub_type)
            
            # 方向
            direction = pattern.flagpole.direction
            direction_name = '上升' if direction == 'up' else '下降'
            
            # 组合文件名
            filename = f"{symbol}_{time_str}_{direction_name}{pattern_name}_{pattern.confidence_score:.2f}"
            
            # 清理非法字符
            illegal_chars = '<>:"/\\|?*'
            for char in illegal_chars:
                filename = filename.replace(char, '_')
                
            return filename
            
        except Exception as e:
            logger.error(f"创建文件名失败: {e}")
            return f"pattern_{pattern.id}"
    
    def _pattern_to_dict(self, pattern: PatternRecord) -> Dict[str, Any]:
        """将PatternRecord转换为字典"""
        def safe_isoformat(dt):
            if dt is None:
                return None
            if isinstance(dt, (datetime, pd.Timestamp)):
                return dt.isoformat()
            elif isinstance(dt, np.datetime64):
                return pd.Timestamp(dt).isoformat()
            else:
                return str(dt)
        
        return {
            'id': pattern.id,
            'symbol': pattern.symbol,
            'pattern_type': pattern.pattern_type,
            'sub_type': pattern.sub_type.value if pattern.sub_type else None,
            'detection_date': safe_isoformat(pattern.detection_date),
            'flagpole': {
                'start_time': safe_isoformat(pattern.flagpole.start_time),
                'end_time': safe_isoformat(pattern.flagpole.end_time),
                'start_price': pattern.flagpole.start_price,
                'end_price': pattern.flagpole.end_price,
                'height_percent': pattern.flagpole.height_percent,
                'height': pattern.flagpole.height,
                'direction': pattern.flagpole.direction,
                'volume_ratio': pattern.flagpole.volume_ratio
            },
            'pattern_boundaries': [
                {
                    'start_time': safe_isoformat(boundary.start_time),
                    'end_time': safe_isoformat(boundary.end_time),
                    'start_price': boundary.start_price,
                    'end_price': boundary.end_price,
                    'slope': boundary.slope,
                    'r_squared': boundary.r_squared
                } for boundary in (pattern.pattern_boundaries or [])
            ],
            'pattern_duration': pattern.pattern_duration,
            'confidence_score': pattern.confidence_score,
            'pattern_quality': pattern.pattern_quality,
            'chart_path': pattern.chart_path,
            'data_path': pattern.data_path
        }
    
    def batch_save_patterns(self, patterns: List[PatternRecord],
                           snapshots: Optional[List[MarketSnapshot]] = None,
                           signals_list: Optional[List[List[InvalidationSignal]]] = None) -> Dict[str, int]:
        """
        批量保存形态记录
        
        Args:
            patterns: 形态记录列表
            snapshots: 市场快照列表
            signals_list: 失效信号列表的列表
            
        Returns:
            保存统计信息
        """
        success_count = 0
        error_count = 0
        
        logger.info(f"开始批量保存 {len(patterns)} 个形态记录")
        
        for i, pattern in enumerate(patterns):
            snapshot = snapshots[i] if snapshots and i < len(snapshots) else None
            signals = signals_list[i] if signals_list and i < len(signals_list) else None
            
            if self.save_pattern_record(pattern, snapshot, signals):
                success_count += 1
            else:
                error_count += 1
        
        result = {
            'total': len(patterns),
            'success': success_count,
            'errors': error_count
        }
        
        logger.info(f"批量保存完成: {result}")
        return result
    
    def batch_save_outcomes(self, outcome_analyses: List[PatternOutcomeAnalysis]) -> Dict[str, int]:
        """
        批量保存结局分析
        
        Args:
            outcome_analyses: 结局分析列表
            
        Returns:
            保存统计信息
        """
        success_count = 0
        error_count = 0
        
        logger.info(f"开始批量保存 {len(outcome_analyses)} 个结局分析")
        
        for outcome in outcome_analyses:
            if self.save_outcome_analysis(outcome):
                success_count += 1
            else:
                error_count += 1
        
        result = {
            'total': len(outcome_analyses),
            'success': success_count,
            'errors': error_count
        }
        
        logger.info(f"批量保存完成: {result}")
        return result
    
    def export_comprehensive_dataset(self, output_path: Optional[str] = None) -> str:
        """
        导出综合数据集
        
        Args:
            output_path: 输出路径
            
        Returns:
            导出文件路径
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if not output_path:
                output_path = self.output_root / "exports" / f"comprehensive_dataset_{timestamp}.json"
            else:
                output_path = Path(output_path)
            
            # 从数据库查询完整数据
            with self._get_db_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # 查询所有数据表
                dataset = {
                    'export_info': {
                        'timestamp': timestamp,
                        'version': '3.0',
                        'description': '动态基线系统综合数据集'
                    },
                    'patterns': [],
                    'pattern_dna': [],
                    'outcomes': [],
                    'market_snapshots': [],
                    'invalidation_signals': []
                }
                
                # 导出形态记录
                cursor.execute("SELECT * FROM patterns ORDER BY detection_date DESC")
                dataset['patterns'] = [dict(row) for row in cursor.fetchall()]
                
                # 导出形态DNA
                cursor.execute("SELECT * FROM pattern_dna")
                dataset['pattern_dna'] = [dict(row) for row in cursor.fetchall()]
                
                # 导出结局分析
                cursor.execute("SELECT * FROM pattern_outcomes ORDER BY analysis_date DESC")
                dataset['outcomes'] = [dict(row) for row in cursor.fetchall()]
                
                # 导出市场快照
                cursor.execute("SELECT * FROM market_snapshots ORDER BY snapshot_date DESC")
                dataset['market_snapshots'] = [dict(row) for row in cursor.fetchall()]
                
                # 导出失效信号
                cursor.execute("SELECT * FROM invalidation_signals ORDER BY detection_time DESC")
                dataset['invalidation_signals'] = [dict(row) for row in cursor.fetchall()]
            
            # 写入文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"综合数据集导出成功: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"导出综合数据集失败: {e}")
            return ""
    
    def get_export_statistics(self) -> Dict[str, Any]:
        """获取导出统计信息"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # 形态统计
                cursor.execute("SELECT COUNT(*) FROM patterns")
                stats['total_patterns'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT sub_type, COUNT(*) FROM patterns GROUP BY sub_type")
                stats['pattern_types'] = dict(cursor.fetchall())
                
                # 结局统计
                cursor.execute("SELECT COUNT(*) FROM pattern_outcomes")
                stats['total_outcomes'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT outcome_classification, COUNT(*) FROM pattern_outcomes GROUP BY outcome_classification")
                stats['outcome_distribution'] = dict(cursor.fetchall())
                
                # 市场状态统计
                cursor.execute("SELECT market_regime, COUNT(*) FROM patterns WHERE market_regime IS NOT NULL GROUP BY market_regime")
                stats['regime_distribution'] = dict(cursor.fetchall())
                
                # 质量统计
                cursor.execute("SELECT pattern_quality, COUNT(*) FROM patterns GROUP BY pattern_quality")
                stats['quality_distribution'] = dict(cursor.fetchall())
                
                return stats
                
        except Exception as e:
            logger.error(f"获取导出统计失败: {e}")
            return {}
    
    def clear_cache(self):
        """清空缓存"""
        self.pattern_cache.clear()
        self.outcome_cache.clear()
        logger.info("数据缓存已清空")