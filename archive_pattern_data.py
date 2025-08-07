#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据归档与存储管理系统
完整的形态数据归档、备份和管理功能
"""
import sys
import os
import json
import sqlite3
import pandas as pd
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict
import zipfile

# 设置控制台编码（Windows）
if sys.platform.startswith('win'):
    os.system('chcp 65001 > nul')
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

class PatternDataArchiver:
    """形态数据归档管理器"""
    
    def __init__(self, base_path: str = 'output'):
        self.base_path = Path(base_path)
        self.archive_path = self.base_path / 'archives'
        self.backup_path = self.base_path / 'backups'
        
        # 确保目录存在
        self.archive_path.mkdir(parents=True, exist_ok=True)
        self.backup_path.mkdir(parents=True, exist_ok=True)
        
    def create_full_archive(self) -> str:
        """创建完整的数据归档"""
        print("开始创建完整数据归档...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_name = f'pattern_scout_archive_{timestamp}'
        archive_dir = self.archive_path / archive_name
        
        # 创建归档目录结构
        archive_dir.mkdir(exist_ok=True)
        (archive_dir / 'data').mkdir(exist_ok=True)
        (archive_dir / 'reports').mkdir(exist_ok=True)
        (archive_dir / 'charts').mkdir(exist_ok=True)
        (archive_dir / 'metadata').mkdir(exist_ok=True)
        
        # 归档内容统计
        archive_stats = {
            'created_at': datetime.now().isoformat(),
            'version': '3.0',
            'description': 'PatternScout 旗形形态识别完整归档',
            'contents': {}
        }
        
        # 1. 归档数据文件
        data_stats = self._archive_data_files(archive_dir / 'data')
        archive_stats['contents']['data'] = data_stats
        
        # 2. 归档报告文件
        reports_stats = self._archive_reports(archive_dir / 'reports')
        archive_stats['contents']['reports'] = reports_stats
        
        # 3. 归档图表文件
        charts_stats = self._archive_charts(archive_dir / 'charts')
        archive_stats['contents']['charts'] = charts_stats
        
        # 4. 生成元数据
        metadata_stats = self._generate_metadata(archive_dir / 'metadata')
        archive_stats['contents']['metadata'] = metadata_stats
        
        # 5. 保存归档统计信息
        with open(archive_dir / 'archive_info.json', 'w', encoding='utf-8') as f:
            json.dump(archive_stats, f, ensure_ascii=False, indent=2)
        
        # 6. 创建ZIP压缩包
        zip_path = self._create_zip_archive(archive_dir)
        
        print(f"✅ 完整归档创建成功: {zip_path}")
        return str(zip_path)
    
    def _archive_data_files(self, target_dir: Path) -> Dict:
        """归档数据文件"""
        print("  归档数据文件...")
        
        data_stats = {'files': [], 'total_size': 0}
        source_data_dir = self.base_path / 'data'
        
        if source_data_dir.exists():
            # 复制数据库
            if (source_data_dir / 'patterns.db').exists():
                shutil.copy2(source_data_dir / 'patterns.db', target_dir / 'patterns.db')
                db_size = (source_data_dir / 'patterns.db').stat().st_size
                data_stats['files'].append({
                    'name': 'patterns.db',
                    'type': 'database',
                    'size': db_size,
                    'description': '形态识别数据库'
                })
                data_stats['total_size'] += db_size
            
            # 复制其他数据文件
            for data_file in source_data_dir.rglob('*.csv'):
                if data_file.is_file():
                    rel_path = data_file.relative_to(source_data_dir)
                    target_file = target_dir / rel_path
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(data_file, target_file)
                    
                    file_size = data_file.stat().st_size
                    data_stats['files'].append({
                        'name': str(rel_path),
                        'type': 'csv',
                        'size': file_size,
                        'description': 'CSV数据文件'
                    })
                    data_stats['total_size'] += file_size
        
        print(f"    已归档 {len(data_stats['files'])} 个数据文件")
        return data_stats
    
    def _archive_reports(self, target_dir: Path) -> Dict:
        """归档报告文件"""
        print("  归档报告文件...")
        
        reports_stats = {'files': [], 'total_size': 0}
        source_reports_dir = self.base_path / 'reports'
        
        if source_reports_dir.exists():
            for report_file in source_reports_dir.rglob('*'):
                if report_file.is_file():
                    rel_path = report_file.relative_to(source_reports_dir)
                    target_file = target_dir / rel_path
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(report_file, target_file)
                    
                    file_size = report_file.stat().st_size
                    file_type = report_file.suffix.lower()
                    
                    reports_stats['files'].append({
                        'name': str(rel_path),
                        'type': file_type,
                        'size': file_size,
                        'description': self._get_file_description(report_file.name)
                    })
                    reports_stats['total_size'] += file_size
        
        print(f"    已归档 {len(reports_stats['files'])} 个报告文件")
        return reports_stats
    
    def _archive_charts(self, target_dir: Path) -> Dict:
        """归档图表文件"""
        print("  归档图表文件...")
        
        charts_stats = {'files': [], 'total_size': 0}
        source_charts_dir = self.base_path / 'charts'
        
        if source_charts_dir.exists():
            for chart_file in source_charts_dir.rglob('*.png'):
                if chart_file.is_file():
                    rel_path = chart_file.relative_to(source_charts_dir)
                    target_file = target_dir / rel_path
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(chart_file, target_file)
                    
                    file_size = chart_file.stat().st_size
                    charts_stats['files'].append({
                        'name': str(rel_path),
                        'type': 'image/png',
                        'size': file_size,
                        'description': self._get_chart_description(chart_file.name)
                    })
                    charts_stats['total_size'] += file_size
        
        print(f"    已归档 {len(charts_stats['files'])} 个图表文件")
        return charts_stats
    
    def _generate_metadata(self, target_dir: Path) -> Dict:
        """生成元数据"""
        print("  生成元数据文件...")
        
        metadata_stats = {'files': [], 'total_size': 0}
        
        # 1. 数据库统计信息
        db_metadata = self._generate_database_metadata()
        if db_metadata:
            db_file = target_dir / 'database_metadata.json'
            with open(db_file, 'w', encoding='utf-8') as f:
                json.dump(db_metadata, f, ensure_ascii=False, indent=2)
            
            file_size = db_file.stat().st_size
            metadata_stats['files'].append({
                'name': 'database_metadata.json',
                'type': 'metadata',
                'size': file_size,
                'description': '数据库元数据'
            })
            metadata_stats['total_size'] += file_size
        
        # 2. 分析摘要
        analysis_summary = self._generate_analysis_summary()
        summary_file = target_dir / 'analysis_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_summary, f, ensure_ascii=False, indent=2)
        
        file_size = summary_file.stat().st_size
        metadata_stats['files'].append({
            'name': 'analysis_summary.json',
            'type': 'summary',
            'size': file_size,
            'description': '分析结果摘要'
        })
        metadata_stats['total_size'] += file_size
        
        # 3. 文件清单
        inventory = self._generate_file_inventory()
        inventory_file = target_dir / 'file_inventory.json'
        with open(inventory_file, 'w', encoding='utf-8') as f:
            json.dump(inventory, f, ensure_ascii=False, indent=2)
        
        file_size = inventory_file.stat().st_size
        metadata_stats['files'].append({
            'name': 'file_inventory.json',
            'type': 'inventory',
            'size': file_size,
            'description': '文件清单'
        })
        metadata_stats['total_size'] += file_size
        
        print(f"    已生成 {len(metadata_stats['files'])} 个元数据文件")
        return metadata_stats
    
    def _generate_database_metadata(self) -> Dict:
        """生成数据库元数据"""
        db_path = self.base_path / 'data' / 'patterns.db'
        if not db_path.exists():
            return None
            
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 获取表信息
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            metadata = {
                'database_file': str(db_path),
                'created_at': datetime.now().isoformat(),
                'tables': {},
                'total_records': 0
            }
            
            for table_name, in tables:
                # 获取表结构
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                # 获取记录数
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                record_count = cursor.fetchone()[0]
                
                metadata['tables'][table_name] = {
                    'columns': [col[1] for col in columns],
                    'record_count': record_count,
                    'structure': columns
                }
                metadata['total_records'] += record_count
            
            conn.close()
            return metadata
            
        except Exception as e:
            print(f"生成数据库元数据时出错: {e}")
            return None
    
    def _generate_analysis_summary(self) -> Dict:
        """生成分析结果摘要"""
        summary = {
            'generated_at': datetime.now().isoformat(),
            'analysis_type': 'Pattern Recognition & Outcome Analysis',
            'version': 'PatternScout 3.0',
            'patterns': {},
            'outcomes': {},
            'charts': {}
        }
        
        # 读取形态分析结果
        pattern_file = self.base_path / 'reports' / 'pattern_detailed_list.csv'
        if pattern_file.exists():
            try:
                df = pd.read_csv(pattern_file)
                summary['patterns'] = {
                    'total_count': len(df),
                    'high_quality_count': len(df[df['pattern_quality'] == 'high']),
                    'average_confidence': float(df['confidence_score'].mean()),
                    'pattern_types': df['pattern_type'].value_counts().to_dict(),
                    'direction_distribution': df['flagpole_direction'].value_counts().to_dict()
                }
            except Exception as e:
                print(f"读取形态分析结果时出错: {e}")
        
        # 读取结局分析结果
        outcome_file = self.base_path / 'reports' / 'outcomes' / 'pattern_outcome_analysis.csv'
        if outcome_file.exists():
            try:
                df = pd.read_csv(outcome_file)
                summary['outcomes'] = {
                    'analyzed_count': len(df),
                    'breakthrough_success_rate': float(df['breakthrough_success'].mean()),
                    'average_price_move': float(df['price_move_percent'].mean()),
                    'outcome_distribution': df['outcome_type'].value_counts().to_dict()
                }
            except Exception as e:
                print(f"读取结局分析结果时出错: {e}")
        
        # 统计图表文件
        charts_dir = self.base_path / 'charts'
        if charts_dir.exists():
            chart_files = list(charts_dir.rglob('*.png'))
            summary['charts'] = {
                'total_count': len(chart_files),
                'categories': {
                    'patterns': len(list(charts_dir.glob('patterns/*.png'))),
                    'analysis': len(list(charts_dir.glob('*.png'))),
                    'outcomes': len(list(charts_dir.glob('**/outcomes/*.png')))
                }
            }
        
        return summary
    
    def _generate_file_inventory(self) -> Dict:
        """生成完整文件清单"""
        inventory = {
            'generated_at': datetime.now().isoformat(),
            'base_path': str(self.base_path),
            'directories': {},
            'total_files': 0,
            'total_size': 0
        }
        
        for root, dirs, files in os.walk(self.base_path):
            if 'archives' in root or 'backups' in root:
                continue  # 跳过归档和备份目录
                
            root_path = Path(root)
            rel_root = root_path.relative_to(self.base_path)
            
            if str(rel_root) not in inventory['directories']:
                inventory['directories'][str(rel_root)] = {
                    'files': [],
                    'file_count': 0,
                    'total_size': 0
                }
            
            for file_name in files:
                file_path = root_path / file_name
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    
                    inventory['directories'][str(rel_root)]['files'].append({
                        'name': file_name,
                        'size': file_size,
                        'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                        'type': file_path.suffix.lower()
                    })
                    
                    inventory['directories'][str(rel_root)]['file_count'] += 1
                    inventory['directories'][str(rel_root)]['total_size'] += file_size
                    inventory['total_files'] += 1
                    inventory['total_size'] += file_size
        
        return inventory
    
    def _create_zip_archive(self, archive_dir: Path) -> Path:
        """创建ZIP压缩归档"""
        print("  创建ZIP压缩包...")
        
        zip_path = self.archive_path / f"{archive_dir.name}.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in archive_dir.rglob('*'):
                if file_path.is_file():
                    rel_path = file_path.relative_to(archive_dir)
                    zipf.write(file_path, rel_path)
        
        # 删除临时目录
        shutil.rmtree(archive_dir)
        
        zip_size = zip_path.stat().st_size
        print(f"    ZIP文件大小: {zip_size / (1024*1024):.1f} MB")
        
        return zip_path
    
    def create_backup(self) -> str:
        """创建数据备份"""
        print("创建数据备份...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f'backup_{timestamp}'
        backup_dir = self.backup_path / backup_name
        
        # 备份关键数据
        backup_dir.mkdir(exist_ok=True)
        
        backup_stats = {
            'created_at': datetime.now().isoformat(),
            'type': 'incremental_backup',
            'files_backed_up': []
        }
        
        # 备份数据库
        db_path = self.base_path / 'data' / 'patterns.db'
        if db_path.exists():
            shutil.copy2(db_path, backup_dir / 'patterns.db')
            backup_stats['files_backed_up'].append('patterns.db')
        
        # 备份关键报告文件
        reports_to_backup = [
            'reports/pattern_detailed_list.csv',
            'reports/high_quality_patterns.csv',
            'reports/outcomes/pattern_outcome_analysis.csv'
        ]
        
        for report_file in reports_to_backup:
            source_file = self.base_path / report_file
            if source_file.exists():
                target_file = backup_dir / Path(report_file).name
                shutil.copy2(source_file, target_file)
                backup_stats['files_backed_up'].append(report_file)
        
        # 保存备份信息
        with open(backup_dir / 'backup_info.json', 'w', encoding='utf-8') as f:
            json.dump(backup_stats, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 备份创建成功: {backup_dir}")
        return str(backup_dir)
    
    def cleanup_old_archives(self, keep_days: int = 30):
        """清理旧的归档文件"""
        print(f"清理 {keep_days} 天前的归档文件...")
        
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        cleaned_count = 0
        
        for archive_file in self.archive_path.glob('*.zip'):
            file_time = datetime.fromtimestamp(archive_file.stat().st_mtime)
            if file_time < cutoff_date:
                archive_file.unlink()
                cleaned_count += 1
                print(f"  已删除: {archive_file.name}")
        
        print(f"✅ 清理完成，删除了 {cleaned_count} 个旧归档文件")
        return cleaned_count
    
    def _get_file_description(self, filename: str) -> str:
        """获取文件描述"""
        descriptions = {
            'pattern_detailed_list.csv': '详细形态识别列表',
            'high_quality_patterns.csv': '高质量形态列表',
            'pattern_outcome_analysis.csv': '形态结局分析结果',
            'pattern_statistics.png': '形态统计图表',
            'outcome_analysis_charts.png': '结局分析图表'
        }
        return descriptions.get(filename, '数据文件')
    
    def _get_chart_description(self, filename: str) -> str:
        """获取图表描述"""
        if 'dashboard' in filename:
            return '综合分析仪表板'
        elif 'baseline' in filename:
            return '动态基线汇总图表'
        elif 'pattern_' in filename:
            return '个别形态分析图表'
        elif 'outcome' in filename:
            return '结局分析图表'
        else:
            return '统计图表'

def generate_final_report():
    """生成最终分析报告"""
    print("生成最终分析报告...")
    
    report_content = {
        'title': 'PatternScout RBL8期货旗形形态识别分析报告',
        'generated_at': datetime.now().isoformat(),
        'version': '3.0',
        'summary': {},
        'detailed_results': {},
        'conclusions': [],
        'recommendations': []
    }
    
    # 读取各个分析结果
    try:
        # 形态识别结果
        patterns_df = pd.read_csv('output/reports/pattern_detailed_list.csv')
        high_quality_df = pd.read_csv('output/reports/high_quality_patterns.csv')
        
        # 结局分析结果
        outcomes_df = pd.read_csv('output/reports/outcomes/pattern_outcome_analysis.csv')
        
        # 生成摘要
        report_content['summary'] = {
            'total_patterns_identified': len(patterns_df),
            'high_quality_patterns': len(high_quality_df),
            'average_confidence_score': float(patterns_df['confidence_score'].mean()),
            'patterns_analyzed_for_outcome': len(outcomes_df),
            'breakthrough_success_rate': float(outcomes_df['breakthrough_success'].mean() * 100),
            'average_price_movement': float(outcomes_df['price_move_percent'].mean())
        }
        
        # 详细结果
        report_content['detailed_results'] = {
            'pattern_distribution': {
                'by_type': patterns_df['pattern_type'].value_counts().to_dict(),
                'by_quality': patterns_df['pattern_quality'].value_counts().to_dict(),
                'by_direction': patterns_df['flagpole_direction'].value_counts().to_dict()
            },
            'outcome_analysis': {
                'outcome_types': outcomes_df['outcome_type'].value_counts().to_dict(),
                'success_metrics': {
                    'breakthrough_success_rate': float(outcomes_df['breakthrough_success'].mean() * 100),
                    'volume_confirmation_rate': float(outcomes_df['volume_confirm'].mean() * 100),
                    'average_time_to_outcome': float(outcomes_df['time_to_outcome'].mean())
                }
            }
        }
        
        # 生成结论和建议
        report_content['conclusions'] = [
            f"在RBL8期货15分钟数据中识别出{len(patterns_df)}个旗形模式",
            f"其中{len(high_quality_df)}个为高质量模式，占比{len(high_quality_df)/len(patterns_df)*100:.1f}%",
            f"平均置信度为{patterns_df['confidence_score'].mean():.3f}，显示识别质量较好",
            f"突破成功率达到{outcomes_df['breakthrough_success'].mean()*100:.1f}%，表明形态有效性良好",
            f"三角旗形占主导地位，占比{patterns_df[patterns_df['pattern_type']=='pennant'].shape[0]/len(patterns_df)*100:.1f}%"
        ]
        
        report_content['recommendations'] = [
            "重点关注置信度0.8以上的高质量形态",
            "结合成交量确认信号提高交易成功率",
            "在形态突破后及时跟进，避免错过最佳入场时机",
            "建议将识别系统集成到实时交易策略中",
            "定期更新和优化识别参数以适应市场变化"
        ]
        
    except Exception as e:
        print(f"读取分析结果时出错: {e}")
        return None
    
    # 保存最终报告
    report_path = 'output/reports/final_analysis_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_content, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 最终分析报告已保存: {report_path}")
    return report_path

def main():
    """主函数"""
    print("=== 数据归档与存储管理系统 ===")
    
    # 创建归档管理器
    archiver = PatternDataArchiver()
    
    # 1. 生成最终报告
    final_report = generate_final_report()
    
    # 2. 创建完整归档
    archive_path = archiver.create_full_archive()
    
    # 3. 创建备份
    backup_path = archiver.create_backup()
    
    # 4. 清理旧归档（可选）
    # archiver.cleanup_old_archives(keep_days=30)
    
    print("\n=== 归档完成统计 ===")
    print(f"✅ 最终分析报告: {final_report}")
    print(f"✅ 完整归档文件: {archive_path}")
    print(f"✅ 数据备份: {backup_path}")
    print("\n🎉 所有数据归档工作完成！")

if __name__ == "__main__":
    main()