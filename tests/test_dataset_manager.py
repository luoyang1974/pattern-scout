import unittest
import tempfile
import shutil
import os
from pathlib import Path
from datetime import datetime

from src.storage.dataset_manager import DatasetManager
from src.data.models.base_models import PatternRecord, Flagpole, PatternType, FlagSubType, AnalysisResult, BreakthroughResult


class TestDatasetManager(unittest.TestCase):
    """测试数据集管理器"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.manager = DatasetManager(dataset_root=self.temp_dir)
        
        # 创建测试形态记录
        self.test_flagpole = Flagpole(
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 5),
            start_price=100.0,
            end_price=110.0,
            height_percent=10.0,
            direction='up',
            volume_ratio=1.5
        )
        
        self.test_pattern = PatternRecord(
            id='test-pattern-001',
            symbol='TEST',
            pattern_type=PatternType.FLAG_PATTERN,
            sub_type=FlagSubType.FLAG,
            detection_date=datetime(2024, 1, 10),
            flagpole=self.test_flagpole,
            pattern_boundaries=[],
            pattern_duration=10,
            confidence_score=0.75,
            pattern_quality='high'
        )
        
        # 创建测试突破结果
        self.test_breakthrough = BreakthroughResult(
            pattern_id='test-pattern-001',
            breakthrough_date=datetime(2024, 1, 15),
            breakthrough_price=112.0,
            result_type='continuation',
            confidence_score=0.8,
            analysis_details={'volume_confirmation': 1.3}
        )
        
        self.test_analysis = AnalysisResult(
            pattern=self.test_pattern,
            breakthrough=self.test_breakthrough,
            scores={'breakthrough_strength': 0.8},
            metadata={'analysis_date': datetime.now()}
        )
    
    def tearDown(self):
        """清理测试环境"""
        # 确保数据库连接完全关闭
        if hasattr(self, 'manager'):
            # 强制关闭可能的数据幓连接
            try:
                import gc
                gc.collect()  # 强制垃圾回收
            except:
                pass
        
        # 使用更安全的目录删除方法
        try:
            if os.path.exists(self.temp_dir):
                # Windows特殊处理
                import time
                time.sleep(0.1)  # 短暂等待文件句柄释放
                
                # 尝试删除，如果失败则重试
                for attempt in range(3):
                    try:
                        shutil.rmtree(self.temp_dir)
                        break
                    except (PermissionError, OSError):
                        if attempt < 2:
                            time.sleep(0.2 * (attempt + 1))
                            continue
                        else:
                            # 最终失败时，至少清理数据幓文件
                            try:
                                db_path = Path(self.temp_dir) / "patterns.db"
                                if db_path.exists():
                                    db_path.unlink()
                            except:
                                pass
        except Exception as e:
            # 测试清理失败不应该影响测试结果
            import warnings
            warnings.warn(f"Failed to cleanup test directory: {e}")
    
    def test_init_database(self):
        """测试数据库初始化"""
        db_path = Path(self.temp_dir) / "patterns.db"
        self.assertTrue(db_path.exists())
    
    def test_save_pattern(self):
        """测试保存形态记录"""
        result = self.manager.save_pattern(self.test_pattern)
        self.assertTrue(result)
        
        # 验证JSON文件是否创建（使用实际的文件名生成逻辑）
        meaningful_filename = self.manager.create_meaningful_filename(self.test_pattern)
        json_path = Path(self.temp_dir) / "patterns" / f"{meaningful_filename}.json"
        self.assertTrue(json_path.exists(), f"Expected file {json_path} does not exist")
    
    def test_save_analysis_result(self):
        """测试保存分析结果"""
        result = self.manager.save_analysis_result(self.test_analysis)
        self.assertTrue(result)
        
        # 验证分析文件是否创建
        analysis_path = Path(self.temp_dir) / "analysis" / f"{self.test_pattern.id}_analysis.json"
        self.assertTrue(analysis_path.exists())
    
    def test_query_patterns(self):
        """测试查询形态记录"""
        # 先保存一个形态
        self.manager.save_pattern(self.test_pattern)
        
        # 查询所有形态
        all_patterns = self.manager.query_patterns()
        self.assertEqual(len(all_patterns), 1)
        self.assertEqual(all_patterns[0]['id'], self.test_pattern.id)
        
        # 按符号查询
        symbol_patterns = self.manager.query_patterns(symbol='TEST')
        self.assertEqual(len(symbol_patterns), 1)
        
        # 按符号查询（不存在的符号）
        no_patterns = self.manager.query_patterns(symbol='NONEXISTENT')
        self.assertEqual(len(no_patterns), 0)
        
        # 按置信度查询
        high_conf_patterns = self.manager.query_patterns(min_confidence=0.8)
        self.assertEqual(len(high_conf_patterns), 0)  # 我们的测试形态置信度是0.75
        
        low_conf_patterns = self.manager.query_patterns(min_confidence=0.7)
        self.assertEqual(len(low_conf_patterns), 1)
    
    def test_batch_save_patterns(self):
        """测试批量保存形态"""
        # 创建多个测试形态
        patterns = []
        for i in range(5):
            flagpole = Flagpole(
                start_time=datetime(2024, 1, i+1),
                end_time=datetime(2024, 1, i+5),
                start_price=100.0 + i,
                end_price=110.0 + i,
                height_percent=10.0,
                direction='up',
                volume_ratio=1.5
            )
            
            pattern = PatternRecord(
                id=f'test-pattern-{i:03d}',
                symbol=f'TEST{i}',
                pattern_type=PatternType.FLAG_PATTERN,
                sub_type=FlagSubType.PENNANT,
                detection_date=datetime(2024, 1, 10+i),
                flagpole=flagpole,
                pattern_boundaries=[],
                pattern_duration=10,
                confidence_score=0.7 + i*0.05,
                pattern_quality='medium'
            )
            patterns.append(pattern)
        
        # 批量保存
        result = self.manager.batch_save_patterns(patterns)
        
        self.assertEqual(result['total'], 5)
        self.assertEqual(result['success'], 5)
        self.assertEqual(result['errors'], 0)
        
        # 验证所有形态都被保存
        all_patterns = self.manager.query_patterns()
        self.assertEqual(len(all_patterns), 5)
    
    def test_get_dataset_statistics(self):
        """测试获取数据集统计"""
        # 保存一些测试数据
        self.manager.save_pattern(self.test_pattern)
        self.manager.save_analysis_result(self.test_analysis)
        
        stats = self.manager.get_dataset_statistics()
        
        self.assertIn('total_patterns', stats)
        self.assertIn('total_analysis', stats)
        self.assertEqual(stats['total_patterns'], 1)
        self.assertEqual(stats['total_analysis'], 1)
        
        self.assertIn('type_distribution', stats)
        self.assertIn('quality_distribution', stats)
        self.assertIn('confidence_statistics', stats)
    
    def test_export_dataset(self):
        """测试数据集导出"""
        # 保存测试数据
        self.manager.save_pattern(self.test_pattern)
        
        # 导出为JSON
        json_export_path = self.manager.export_dataset('json')
        self.assertTrue(Path(json_export_path).exists())
        
        # 导出为CSV
        csv_export_path = self.manager.export_dataset('csv')
        self.assertTrue(Path(csv_export_path).exists())
    
    def test_create_backup(self):
        """测试创建备份"""
        # 保存一些数据
        self.manager.save_pattern(self.test_pattern)
        
        # 创建备份
        backup_path = self.manager.create_backup('test_backup')
        
        self.assertTrue(Path(backup_path).exists())
        self.assertTrue((Path(backup_path) / 'patterns.db').exists())


class TestDatasetManagerEdgeCases(unittest.TestCase):
    """测试数据集管理器边界情况"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = DatasetManager(dataset_root=self.temp_dir)
    
    def tearDown(self):
        """清理测试环境"""
        # 确保数据幓连接完全关闭
        if hasattr(self, 'manager'):
            try:
                import gc
                gc.collect()  # 强制垃圾回收
            except:
                pass
        
        # 使用更安全的目录删除方法
        try:
            if os.path.exists(self.temp_dir):
                import time
                time.sleep(0.1)
                
                for attempt in range(3):
                    try:
                        shutil.rmtree(self.temp_dir)
                        break
                    except (PermissionError, OSError):
                        if attempt < 2:
                            time.sleep(0.2 * (attempt + 1))
                            continue
                        else:
                            try:
                                db_path = Path(self.temp_dir) / "patterns.db"
                                if db_path.exists():
                                    db_path.unlink()
                            except:
                                pass
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to cleanup test directory: {e}")
    
    def test_query_empty_database(self):
        """测试查询空数据库"""
        patterns = self.manager.query_patterns()
        self.assertEqual(len(patterns), 0)
    
    def test_get_statistics_empty_database(self):
        """测试获取空数据库统计"""
        stats = self.manager.get_dataset_statistics()
        self.assertEqual(stats['total_patterns'], 0)
        self.assertEqual(stats['total_analysis'], 0)
    
    def test_export_empty_dataset(self):
        """测试导出空数据集"""
        export_path = self.manager.export_dataset('json')
        self.assertEqual(export_path, "")  # 空数据集应该返回空字符串
    
    def test_invalid_export_format(self):
        """测试无效的导出格式"""
        # 先添加一些数据
        flagpole = Flagpole(
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 5),
            start_price=100.0,
            end_price=110.0,
            height_percent=10.0,
            direction='up',
            volume_ratio=1.5
        )
        
        pattern = PatternRecord(
            id='test-pattern-001',
            symbol='TEST',
            pattern_type=PatternType.FLAG,
            detection_date=datetime(2024, 1, 10),
            flagpole=flagpole,
            pattern_boundaries=[],
            pattern_duration=10,
            confidence_score=0.75,
            pattern_quality='high'
        )
        
        self.manager.save_pattern(pattern)
        
        # 尝试无效格式导出
        export_path = self.manager.export_dataset('invalid_format')
        self.assertEqual(export_path, "")


if __name__ == '__main__':
    unittest.main()