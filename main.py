import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List

# 添加src到Python路径
sys.path.append(str(Path(__file__).parent))

from src.utils.config_manager import ConfigManager
from src.data.connectors.base_connector import DataConnectorFactory
from src.patterns.detectors.pattern_scanner import PatternScanner
from src.analysis.breakthrough_analyzer import BreakthroughAnalyzer
from src.visualization.chart_generator import ChartGenerator
from src.storage.dataset_manager import DatasetManager
from src.data.models.base_models import PatternRecord, PatternType, FlagSubType

from loguru import logger


class PatternScout:
    """PatternScout主程序类"""
    
    def __init__(self, config_file: str = "config.yaml"):
        """
        初始化PatternScout
        
        Args:
            config_file: 配置文件路径
        """
        self.config_manager = ConfigManager(config_file)
        
        # 设置日志
        self._setup_logging()
        
        # 验证配置
        if not self.config_manager.validate_config():
            logger.error("Configuration validation failed")
            sys.exit(1)
        
        # 初始化组件
        self.data_connector = None
        self.pattern_scanner = PatternScanner(self.config_manager.config)
        self.breakthrough_analyzer = BreakthroughAnalyzer(self.config_manager.get('breakthrough', {}))
        self.chart_generator = ChartGenerator(self.config_manager.get('output', {}))
        self.dataset_manager = DatasetManager(self.config_manager.get('output.data_path', 'output/data'))
        
        logger.info("PatternScout initialized successfully")
    
    def _setup_logging(self):
        """设置日志配置"""
        log_config = self.config_manager.get('logging', {})
        log_level = log_config.get('level', 'INFO')
        log_file = log_config.get('file', 'logs/pattern_scout.log')
        
        # 创建日志目录
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # 配置loguru
        logger.remove()  # 移除默认处理器
        
        # 添加控制台输出
        logger.add(
            sys.stdout,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        
        # 添加文件输出
        logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation=log_config.get('max_size', '10MB'),
            retention=log_config.get('backup_count', 5)
        )
    
    def _setup_data_connector(self, connector_type: str = None) -> bool:
        """
        设置数据连接器
        
        Args:
            connector_type: 连接器类型 ('csv' or 'mongodb')
            
        Returns:
            设置是否成功
        """
        try:
            # 如果没有指定类型，选择第一个启用的数据源
            if not connector_type:
                data_sources = self.config_manager.get('data_sources', {})
                for source_type, config in data_sources.items():
                    if config.get('enabled', False):
                        connector_type = source_type
                        break
            
            if not connector_type:
                logger.error("No enabled data source found")
                return False
            
            # 获取数据源配置
            source_config = self.config_manager.get(f'data_sources.{connector_type}', {})
            
            # 移除不需要的配置项并映射参数名
            connector_config = {}
            for k, v in source_config.items():
                if k == 'enabled':
                    continue
                elif k == 'directory' and connector_type == 'csv':
                    connector_config['data_directory'] = v
                else:
                    connector_config[k] = v
            
            # 创建数据连接器
            self.data_connector = DataConnectorFactory.create_connector(
                connector_type, **connector_config
            )
            
            # 建立连接
            if self.data_connector.connect():
                logger.info(f"Connected to {connector_type} data source")
                return True
            else:
                logger.error(f"Failed to connect to {connector_type} data source")
                return False
                
        except Exception as e:
            logger.error(f"Failed to setup data connector: {e}")
            return False
    
    def scan_patterns(self, 
                     symbols: List[str] = None,
                     start_date: datetime = None,
                     end_date: datetime = None,
                     pattern_types: List[str] = None) -> List[PatternRecord]:
        """
        扫描形态
        
        Args:
            symbols: 品种列表，None表示扫描所有
            start_date: 开始日期
            end_date: 结束日期
            pattern_types: 形态类型列表
            
        Returns:
            检测到的形态列表
        """
        logger.info("Starting pattern scanning...")
        
        if not self.data_connector:
            logger.error("Data connector not initialized")
            return []
        
        # 设置默认参数
        if not symbols:
            symbols = self.data_connector.get_symbols()
            logger.info(f"Scanning all available symbols: {len(symbols)} symbols")
        
        if not start_date:
            start_date = datetime.now() - timedelta(days=365)  # 默认扫描一年
        
        if not end_date:
            end_date = datetime.now()
        
        if not pattern_types:
            pattern_types = [PatternType.FLAG_PATTERN]  # 默认扫描旗形形态（包含矩形旗和三角旗）
        
        all_patterns = []
        
        for symbol in symbols:
            try:
                logger.info(f"Scanning patterns for {symbol}...")
                
                # 获取数据
                df = self.data_connector.get_data(symbol, start_date, end_date)
                
                if df.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue
                
                # 使用统一的形态扫描器检测形态
                patterns_result = self.pattern_scanner.scan(df, pattern_types)
                
                # 汇总所有检测到的形态
                for pattern_type, patterns in patterns_result.items():
                    all_patterns.extend(patterns)
                    
                    # 按子类型统计结果
                    flag_count = sum(1 for p in patterns if p.sub_type == FlagSubType.FLAG)
                    pennant_count = sum(1 for p in patterns if p.sub_type == FlagSubType.PENNANT)
                    
                    logger.info(f"Found {len(patterns)} {pattern_type} patterns for {symbol} "
                               f"(Flag: {flag_count}, Pennant: {pennant_count})")
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                continue
        
        logger.info(f"Pattern scanning completed. Total patterns found: {len(all_patterns)}")
        return all_patterns
    
    def generate_reports(self, patterns: List[PatternRecord], 
                        generate_charts: bool = True,
                        generate_summary: bool = True) -> dict:
        """
        生成报告
        
        Args:
            patterns: 形态列表
            generate_charts: 是否生成个别图表
            generate_summary: 是否生成汇总图表
            
        Returns:
            生成的文件路径字典
        """
        logger.info("Generating reports...")
        
        results = {
            'individual_charts': [],
            'summary_chart': None,
            'data_file': None
        }
        
        if not patterns:
            logger.warning("No patterns to generate reports for")
            return results
        
        try:
            # 使用新的TradingView风格图表生成器
            if generate_charts:
                logger.info("Generating TradingView style classified charts...")
                chart_results = self.chart_generator.generate_classified_charts()
                
                # 统计生成的图表
                total_charts = 0
                for pattern_type, result in chart_results.items():
                    total_charts += result['charts_generated']
                    results['individual_charts'].extend(result['charts'])
                
                logger.info(f"Generated {total_charts} TradingView style charts")
            
            # 生成汇总报告
            if generate_summary:
                logger.info("Generating pattern classification summary...")
                try:
                    self.chart_generator.create_summary_report()
                    results['summary_chart'] = "Pattern classification summary generated"
                except Exception as e:
                    logger.error(f"Failed to generate summary report: {e}")
            
            # 保存数据文件
            if patterns:
                logger.info("Saving patterns to dataset...")
                save_result = self.dataset_manager.batch_save_patterns(patterns)
                logger.info(f"Saved {save_result['success']}/{save_result['total']} patterns to dataset")
                results['data_file'] = f"Dataset: {save_result['success']} patterns saved"
            
            logger.info(f"Report generation completed. Generated {len(results['individual_charts'])} charts")
            
        except Exception as e:
            logger.error(f"Error generating reports: {e}")
        
        return results
    
    def run(self, 
            symbols: List[str] = None,
            start_date: datetime = None,
            end_date: datetime = None,
            data_source: str = None,
            pattern_types: List[str] = None,
            generate_charts: bool = True,
            generate_summary: bool = True) -> dict:
        """
        运行PatternScout
        
        Args:
            symbols: 品种列表
            start_date: 开始日期
            end_date: 结束日期
            data_source: 数据源类型
            pattern_types: 形态类型列表
            generate_charts: 是否生成图表
            generate_summary: 是否生成汇总
            
        Returns:
            运行结果
        """
        logger.info("Starting PatternScout execution...")
        
        try:
            # 设置数据连接器
            if not self._setup_data_connector(data_source):
                return {'error': 'Failed to setup data connector'}
            
            # 扫描形态
            patterns = self.scan_patterns(symbols, start_date, end_date, pattern_types)
            
            if not patterns:
                logger.warning("No patterns detected")
                return {'patterns': [], 'reports': {}}
            
            # 生成报告
            reports = self.generate_reports(patterns, generate_charts, generate_summary)
            
            # 清理资源
            if self.data_connector:
                self.data_connector.close()
            
            result = {
                'patterns': patterns,
                'reports': reports,
                'summary': {
                    'total_patterns': len(patterns),
                    'quality_distribution': {
                        quality: sum(1 for p in patterns if p.pattern_quality == quality)
                        for quality in ['high', 'medium', 'low']
                    }
                }
            }
            
            logger.info("PatternScout execution completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"PatternScout execution failed: {e}")
            return {'error': str(e)}


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PatternScout - Flag and Triangle Flag Pattern Recognition')
    
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--symbols', nargs='+', help='Symbols to scan (default: all available)')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--data-source', choices=['csv', 'mongodb'], help='Data source type')
    parser.add_argument('--pattern-types', nargs='+', choices=['flag_pattern', 'flag', 'pennant'], 
                        help='Pattern types to detect: flag_pattern (unified), flag (rectangles only), pennant (triangles only). Default: flag_pattern', default=None)
    parser.add_argument('--min-confidence', type=float, help='Minimum confidence score')
    parser.add_argument('--export-dataset', choices=['json', 'csv', 'excel'], 
                        help='Export dataset in specified format')
    parser.add_argument('--analyze-breakthrough', action='store_true', 
                        help='Perform breakthrough analysis on detected patterns')
    parser.add_argument('--no-charts', action='store_true', help='Skip individual chart generation')
    parser.add_argument('--no-summary', action='store_true', help='Skip summary generation')
    
    args = parser.parse_args()
    
    # 解析日期参数
    start_date = None
    end_date = None
    
    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        except ValueError:
            print(f"Invalid start date format: {args.start_date}")
            sys.exit(1)
    
    if args.end_date:
        try:
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        except ValueError:
            print(f"Invalid end date format: {args.end_date}")
            sys.exit(1)
    
    # 处理形态类型参数（向后兼容）
    pattern_types = args.pattern_types
    if pattern_types:
        # 转换旧的API到新的统一API
        if 'flag' in pattern_types or 'pennant' in pattern_types:
            logger.warning("Using legacy pattern type names. Consider using 'flag_pattern' for unified detection.")
            # 如果用户指定了flag或pennant，转换为统一的flag_pattern
            pattern_types = [PatternType.FLAG_PATTERN]
        elif 'flag_pattern' not in pattern_types:
            # 如果用户没有指定flag_pattern，添加它
            pattern_types.append(PatternType.FLAG_PATTERN)
    
    # 创建PatternScout实例
    scout = PatternScout(args.config)
    
    # 运行扫描
    result = scout.run(
        symbols=args.symbols,
        start_date=start_date,
        end_date=end_date,
        data_source=args.data_source,
        pattern_types=args.pattern_types,
        generate_charts=not args.no_charts,
        generate_summary=not args.no_summary
    )
    
    # 输出结果
    if 'error' in result:
        print(f"Error: {result['error']}")
        sys.exit(1)
    else:
        print("PatternScout completed successfully!")
        
        if 'summary' in result:
            print(f"Total patterns detected: {result['summary']['total_patterns']}")
            print(f"Quality distribution: {result['summary']['quality_distribution']}")
        else:
            print("No patterns detected")
        
        if result.get('reports', {}).get('summary_chart'):
            print(f"Summary chart: {result['reports']['summary_chart']}")
        
        if result.get('reports', {}).get('individual_charts'):
            print(f"Generated {len(result['reports']['individual_charts'])} individual charts")


if __name__ == "__main__":
    main()
