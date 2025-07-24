import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import uuid
import shutil

from src.data.models.base_models import PatternRecord, AnalysisResult
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


class DatasetManager:
    """数据集管理系统"""
    
    def __init__(self, dataset_root: str = "output/data"):
        """
        初始化数据集管理器
        
        Args:
            dataset_root: 数据集根目录
        """
        self.dataset_root = Path(dataset_root)
        self.db_path = self.dataset_root / "patterns.db"
        self._ensure_directories()
        self._init_database()
        
    def _ensure_directories(self):
        """确保必要的目录存在"""
        directories = [
            self.dataset_root,
            self.dataset_root / "patterns",
            self.dataset_root / "analysis",
            self.dataset_root / "exports",
            self.dataset_root / "backups"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Dataset directories initialized at {self.dataset_root}")
    
    def _init_database(self):
        """初始化SQLite数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 创建形态记录表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS patterns (
                        id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        pattern_type TEXT NOT NULL,
                        detection_date TEXT NOT NULL,
                        flagpole_start_time TEXT NOT NULL,
                        flagpole_end_time TEXT NOT NULL,
                        flagpole_height_percent REAL NOT NULL,
                        flagpole_direction TEXT NOT NULL,
                        pattern_duration INTEGER NOT NULL,
                        confidence_score REAL NOT NULL,
                        pattern_quality TEXT NOT NULL,
                        chart_path TEXT,
                        data_path TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 创建突破分析表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS breakthrough_analysis (
                        id TEXT PRIMARY KEY,
                        pattern_id TEXT NOT NULL,
                        breakthrough_date TEXT,
                        breakthrough_price REAL,
                        result_type TEXT,
                        confidence_score REAL,
                        analysis_details TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (pattern_id) REFERENCES patterns (id)
                    )
                """)
                
                # 创建数据集版本表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS dataset_versions (
                        version TEXT PRIMARY KEY,
                        description TEXT,
                        pattern_count INTEGER,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT
                    )
                """)
                
                # 创建索引
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_symbol ON patterns (symbol)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns (pattern_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_quality ON patterns (pattern_quality)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_breakthrough_pattern_id ON breakthrough_analysis (pattern_id)")
                
                conn.commit()
                
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def create_meaningful_filename(self, pattern: PatternRecord) -> str:
        """创建有意义的文件名：资产名称_起始时间_形态名称"""
        try:
            # 提取资产名称（去掉-15min后缀）
            symbol = pattern.symbol.replace('-15min', '')
            
            # 提取旗杆起始时间并格式化
            start_time = pattern.flagpole.start_time
            time_str = start_time.strftime('%Y%m%d_%H%M')
            
            # 形态名称映射
            pattern_names = {
                'flag': '旗形',
                'pennant': '三角旗形'
            }
            pattern_name = pattern_names.get(pattern.pattern_type, pattern.pattern_type)
            
            # 方向信息
            direction_name = '上升' if pattern.flagpole.direction == 'up' else '下降'
            
            # 组合文件名
            filename = f"{symbol}_{time_str}_{direction_name}{pattern_name}"
            return filename
            
        except Exception as e:
            logger.error(f"创建文件名失败: {e}")
            # 回退到使用ID
            return pattern.id
    
    def save_pattern(self, pattern: PatternRecord) -> bool:
        """
        保存形态记录
        
        Args:
            pattern: 形态记录
            
        Returns:
            保存是否成功
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO patterns 
                    (id, symbol, pattern_type, detection_date, flagpole_start_time, 
                     flagpole_end_time, flagpole_height_percent, flagpole_direction, 
                     pattern_duration, confidence_score, pattern_quality, 
                     chart_path, data_path, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern.id,
                    pattern.symbol,
                    pattern.pattern_type,
                    pattern.detection_date.isoformat(),
                    pattern.flagpole.start_time.isoformat(),
                    pattern.flagpole.end_time.isoformat(),
                    pattern.flagpole.height_percent,
                    pattern.flagpole.direction,
                    pattern.pattern_duration,
                    pattern.confidence_score,
                    pattern.pattern_quality,
                    pattern.chart_path,
                    pattern.data_path,
                    datetime.now().isoformat()
                ))
                
                conn.commit()
            
            # 保存详细的JSON文件
            meaningful_filename = self.create_meaningful_filename(pattern)
            pattern_file = self.dataset_root / "patterns" / f"{meaningful_filename}.json"
            with open(pattern_file, 'w', encoding='utf-8') as f:
                json.dump(self._pattern_to_dict(pattern), f, ensure_ascii=False, indent=2, default=datetime_serializer)
            
            logger.info(f"Pattern {pattern.id} saved as {meaningful_filename}.json")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save pattern {pattern.id}: {e}")
            return False
    
    def save_analysis_result(self, analysis: AnalysisResult) -> bool:
        """
        保存分析结果
        
        Args:
            analysis: 分析结果
            
        Returns:
            保存是否成功
        """
        try:
            # 先保存形态记录
            self.save_pattern(analysis.pattern)
            
            # 保存突破分析
            if analysis.breakthrough:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO breakthrough_analysis 
                        (id, pattern_id, breakthrough_date, breakthrough_price, 
                         result_type, confidence_score, analysis_details)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        str(uuid.uuid4()),
                        analysis.breakthrough.pattern_id,
                        analysis.breakthrough.breakthrough_date.isoformat(),
                        analysis.breakthrough.breakthrough_price,
                        analysis.breakthrough.result_type,
                        analysis.breakthrough.confidence_score,
                        json.dumps(analysis.breakthrough.analysis_details, default=str)
                    ))
                    
                    conn.commit()
            
            # 保存完整的分析结果JSON
            analysis_file = self.dataset_root / "analysis" / f"{analysis.pattern.id}_analysis.json"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(self._analysis_to_dict(analysis), f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"Analysis result for pattern {analysis.pattern.id} saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save analysis result for pattern {analysis.pattern.id}: {e}")
            return False
    
    def batch_save_patterns(self, patterns: List[PatternRecord]) -> Dict[str, int]:
        """
        批量保存形态记录
        
        Args:
            patterns: 形态记录列表
            
        Returns:
            保存统计信息
        """
        success_count = 0
        error_count = 0
        
        logger.info(f"Starting batch save of {len(patterns)} patterns")
        
        for pattern in patterns:
            if self.save_pattern(pattern):
                success_count += 1
            else:
                error_count += 1
        
        result = {
            'total': len(patterns),
            'success': success_count,
            'errors': error_count
        }
        
        logger.info(f"Batch save completed: {result}")
        return result
    
    def batch_save_analysis_results(self, analysis_results: List[AnalysisResult]) -> Dict[str, int]:
        """
        批量保存分析结果
        
        Args:
            analysis_results: 分析结果列表
            
        Returns:
            保存统计信息
        """
        success_count = 0
        error_count = 0
        
        logger.info(f"Starting batch save of {len(analysis_results)} analysis results")
        
        for analysis in analysis_results:
            if self.save_analysis_result(analysis):
                success_count += 1
            else:
                error_count += 1
        
        result = {
            'total': len(analysis_results),
            'success': success_count,
            'errors': error_count
        }
        
        logger.info(f"Batch save completed: {result}")
        return result
    
    def query_patterns(self, 
                      symbol: Optional[str] = None,
                      pattern_type: Optional[str] = None,
                      pattern_quality: Optional[str] = None,
                      min_confidence: Optional[float] = None,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None,
                      limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        查询形态记录
        
        Args:
            symbol: 品种代码
            pattern_type: 形态类型
            pattern_quality: 形态质量
            min_confidence: 最小置信度
            start_date: 开始日期
            end_date: 结束日期
            limit: 结果数量限制
            
        Returns:
            查询结果列表
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row  # 使结果可以按列名访问
                cursor = conn.cursor()
                
                # 构建查询条件
                conditions = []
                params = []
                
                if symbol:
                    conditions.append("symbol = ?")
                    params.append(symbol)
                
                if pattern_type:
                    conditions.append("pattern_type = ?")
                    params.append(pattern_type)
                
                if pattern_quality:
                    conditions.append("pattern_quality = ?")
                    params.append(pattern_quality)
                
                if min_confidence is not None:
                    conditions.append("confidence_score >= ?")
                    params.append(min_confidence)
                
                if start_date:
                    conditions.append("detection_date >= ?")
                    params.append(start_date.isoformat())
                
                if end_date:
                    conditions.append("detection_date <= ?")
                    params.append(end_date.isoformat())
                
                # 构建完整查询
                query = "SELECT * FROM patterns"
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                query += " ORDER BY detection_date DESC"
                
                if limit:
                    query += f" LIMIT {limit}"
                
                cursor.execute(query, params)
                results = cursor.fetchall()
                
                # 转换为字典列表
                return [dict(row) for row in results]
                
        except Exception as e:
            logger.error(f"Failed to query patterns: {e}")
            return []
    
    def get_pattern_by_id(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取形态记录"""
        results = self.query_patterns()  # 查询所有，然后筛选
        for pattern in results:
            if pattern['id'] == pattern_id:
                return pattern
        return None
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 基本统计
                cursor.execute("SELECT COUNT(*) as total FROM patterns")
                total_patterns = cursor.fetchone()[0]
                
                # 按类型统计
                cursor.execute("""
                    SELECT pattern_type, COUNT(*) as count 
                    FROM patterns 
                    GROUP BY pattern_type
                """)
                type_distribution = dict(cursor.fetchall())
                
                # 按质量统计
                cursor.execute("""
                    SELECT pattern_quality, COUNT(*) as count 
                    FROM patterns 
                    GROUP BY pattern_quality
                """)
                quality_distribution = dict(cursor.fetchall())
                
                # 按品种统计
                cursor.execute("""
                    SELECT symbol, COUNT(*) as count 
                    FROM patterns 
                    GROUP BY symbol 
                    ORDER BY count DESC 
                    LIMIT 10
                """)
                top_symbols = dict(cursor.fetchall())
                
                # 置信度统计
                cursor.execute("""
                    SELECT AVG(confidence_score) as avg_confidence, 
                           MIN(confidence_score) as min_confidence,
                           MAX(confidence_score) as max_confidence
                    FROM patterns
                """)
                confidence_stats = cursor.fetchone()
                
                # 突破分析统计
                cursor.execute("SELECT COUNT(*) as total FROM breakthrough_analysis")
                total_analysis = cursor.fetchone()[0]
                
                cursor.execute("""
                    SELECT result_type, COUNT(*) as count 
                    FROM breakthrough_analysis 
                    WHERE result_type IS NOT NULL
                    GROUP BY result_type
                """)
                breakthrough_distribution = dict(cursor.fetchall())
                
                return {
                    'total_patterns': total_patterns,
                    'total_analysis': total_analysis,
                    'type_distribution': type_distribution,
                    'quality_distribution': quality_distribution,
                    'top_symbols': top_symbols,
                    'confidence_statistics': {
                        'average': confidence_stats[0] if confidence_stats[0] else 0,
                        'minimum': confidence_stats[1] if confidence_stats[1] else 0,
                        'maximum': confidence_stats[2] if confidence_stats[2] else 0
                    },
                    'breakthrough_distribution': breakthrough_distribution,
                    'last_updated': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get dataset statistics: {e}")
            return {}
    
    def export_dataset(self, export_format: str = "json", 
                      output_path: Optional[str] = None,
                      filters: Optional[Dict[str, Any]] = None) -> str:
        """
        导出数据集
        
        Args:
            export_format: 导出格式 ('json', 'csv', 'excel')
            output_path: 输出路径
            filters: 过滤条件
            
        Returns:
            导出文件路径
        """
        try:
            # 查询数据
            query_params = filters or {}
            patterns = self.query_patterns(**query_params)
            
            if not patterns:
                logger.warning("No patterns found for export")
                return ""
            
            # 生成输出文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if not output_path:
                output_path = self.dataset_root / "exports" / f"patterns_{timestamp}.{export_format}"
            else:
                output_path = Path(output_path)
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 根据格式导出
            if export_format.lower() == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(patterns, f, ensure_ascii=False, indent=2, default=str)
            
            elif export_format.lower() == 'csv':
                df = pd.DataFrame(patterns)
                df.to_csv(output_path, index=False, encoding='utf-8')
            
            elif export_format.lower() in ['excel', 'xlsx']:
                df = pd.DataFrame(patterns)
                df.to_excel(output_path, index=False)
            
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
            
            logger.info(f"Dataset exported to {output_path} ({len(patterns)} patterns)")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to export dataset: {e}")
            return ""
    
    def create_backup(self, version_name: Optional[str] = None) -> str:
        """
        创建数据集备份
        
        Args:
            version_name: 版本名称
            
        Returns:
            备份路径
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version_name = version_name or f"backup_{timestamp}"
            
            backup_dir = self.dataset_root / "backups" / version_name
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # 备份数据库
            shutil.copy2(self.db_path, backup_dir / "patterns.db")
            
            # 备份JSON文件
            patterns_dir = self.dataset_root / "patterns"
            if patterns_dir.exists():
                shutil.copytree(patterns_dir, backup_dir / "patterns", dirs_exist_ok=True)
            
            analysis_dir = self.dataset_root / "analysis"
            if analysis_dir.exists():
                shutil.copytree(analysis_dir, backup_dir / "analysis", dirs_exist_ok=True)
            
            # 记录版本信息
            stats = self.get_dataset_statistics()
            version_info = {
                'version': version_name,
                'created_at': timestamp,
                'pattern_count': stats.get('total_patterns', 0),
                'analysis_count': stats.get('total_analysis', 0),
                'backup_path': str(backup_dir)
            }
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO dataset_versions 
                    (version, description, pattern_count, metadata)
                    VALUES (?, ?, ?, ?)
                """, (
                    version_name,
                    f"Automatic backup created at {timestamp}",
                    stats.get('total_patterns', 0),
                    json.dumps(version_info, default=str)
                ))
                conn.commit()
            
            logger.info(f"Backup created: {backup_dir}")
            return str(backup_dir)
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return ""
    
    def _pattern_to_dict(self, pattern: PatternRecord) -> Dict[str, Any]:
        """将PatternRecord转换为字典"""
        def safe_isoformat(dt):
            """安全地转换时间为ISO格式"""
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
            'detection_date': safe_isoformat(pattern.detection_date),
            'flagpole': {
                'start_time': safe_isoformat(pattern.flagpole.start_time),
                'end_time': safe_isoformat(pattern.flagpole.end_time),
                'start_price': pattern.flagpole.start_price,
                'end_price': pattern.flagpole.end_price,
                'height_percent': pattern.flagpole.height_percent,
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
            'breakthrough_date': safe_isoformat(pattern.breakthrough_date),
            'breakthrough_price': pattern.breakthrough_price,
            'result_type': pattern.result_type,
            'confidence_score': pattern.confidence_score,
            'pattern_quality': pattern.pattern_quality,
            'chart_path': pattern.chart_path,
            'data_path': pattern.data_path
        }
    
    def _analysis_to_dict(self, analysis: AnalysisResult) -> Dict[str, Any]:
        """将AnalysisResult转换为字典"""
        return {
            'pattern': self._pattern_to_dict(analysis.pattern),
            'breakthrough': {
                'pattern_id': analysis.breakthrough.pattern_id,
                'breakthrough_date': analysis.breakthrough.breakthrough_date.isoformat(),
                'breakthrough_price': analysis.breakthrough.breakthrough_price,
                'result_type': analysis.breakthrough.result_type,
                'confidence_score': analysis.breakthrough.confidence_score,
                'analysis_details': analysis.breakthrough.analysis_details
            } if analysis.breakthrough else None,
            'scores': analysis.scores,
            'metadata': analysis.metadata
        }