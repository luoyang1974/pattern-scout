import sys
import os
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List

# 设置控制台编码（Windows）
if sys.platform.startswith('win'):
    os.system('chcp 65001 > nul')
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import locale
        try:
            locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
        except locale.Error:
            pass  # 忽略locale错误

# 添加src到Python路径
sys.path.append(str(Path(__file__).parent))

from src.utils.config_manager import ConfigManager
from src.data.connectors.base_connector import DataConnectorFactory
from src.patterns.detectors.pattern_scanner import PatternScanner
from src.analysis.breakthrough_analyzer import BreakthroughAnalyzer
from src.visualization.pattern_chart_generator import PatternChartGenerator
from src.storage.dataset_manager import DatasetManager
from src.data.models.base_models import PatternType
from src.patterns.base.timeframe_manager import TimeframeManager

from loguru import logger

# 配置loguru日志格式
logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
)


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
        self.scanner = PatternScanner(self.config_manager.config)
        self.breakthrough_analyzer = BreakthroughAnalyzer(self.config_manager.get('breakthrough', {}))
        self.chart_generator = PatternChartGenerator(self.config_manager.get('output', {}))
        self.dataset_manager = DatasetManager(self.config_manager.get('output.data_path', 'output/data'))
        self.timeframe_manager = TimeframeManager()
        
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
                     enable_outcome_tracking: bool = True) -> dict:
        """
        扫描形态
        
        Args:
            symbols: 品种列表，None表示扫描所有
            start_date: 开始日期
            end_date: 结束日期
            enable_outcome_tracking: 是否启用结局追踪
            
        Returns:
            扫描结果字典
        """
        logger.info("Starting pattern scanning...")
        
        if not self.data_connector:
            logger.error("Data connector not initialized")
            return {'success': False, 'error': 'Data connector not initialized'}
        
        # 设置默认参数
        if not symbols:
            symbols = self.data_connector.get_symbols()
            logger.info(f"Scanning all available symbols: {len(symbols)} symbols")
        
        # 智能设置默认时间范围
        if not start_date or not end_date:
            if hasattr(self.data_connector, 'get_data_range') and symbols:
                # 使用第一个品种的数据范围作为参考
                first_symbol = symbols[0]
                try:
                    data_start, data_end = self.data_connector.get_data_range(first_symbol)
                    if data_start and data_end:
                        if not start_date:
                            start_date = data_start
                            logger.info(f"Using actual data start date: {start_date}")
                        if not end_date:
                            end_date = data_end
                            logger.info(f"Using actual data end date: {end_date}")
                    else:
                        raise ValueError("Unable to detect data range")
                except Exception as e:
                    logger.warning(f"Failed to get data range for {first_symbol}: {e}")
                    # 使用传统的默认值
                    if not start_date:
                        start_date = datetime.now() - timedelta(days=365)
                    if not end_date:
                        end_date = datetime.now()
            else:
                # 使用传统的默认值
                if not start_date:
                    start_date = datetime.now() - timedelta(days=365)
                if not end_date:
                    end_date = datetime.now()
        
        all_scan_results = []
        total_patterns = 0
        
        for symbol in symbols:
            try:
                logger.info(f"Scanning patterns for {symbol}...")
                
                # 获取数据
                df = self.data_connector.get_data(symbol, start_date, end_date)
                
                if df.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue
                
                # 添加品种信息到数据框
                df['symbol'] = symbol
                
                # 使用扫描器
                scan_result = self.scanner.scan(
                    df, enable_outcome_analysis=enable_outcome_tracking
                )
                
                if scan_result['success']:
                    all_scan_results.append(scan_result)
                    patterns_count = scan_result['patterns_detected']
                    total_patterns += patterns_count
                    
                    logger.info(
                        f"Scan completed for {symbol}: "
                        f"{patterns_count} patterns detected, "
                        f"Market regime: {scan_result['market_regime']}, "
                        f"Scan time: {scan_result['scan_time']:.2f}s"
                    )
                else:
                    logger.warning(f"Scan failed for {symbol}")
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                continue
        
        # 汇总结果
        summary_result = {
            'success': True,
            'total_symbols_scanned': len(symbols),
            'successful_scans': len(all_scan_results),
            'total_patterns_detected': total_patterns,
            'scan_results': all_scan_results,
            'system_status': self.scanner.get_system_status(),
            'performance_metrics': self.scanner.get_performance_metrics()
        }
        
        logger.info(
            f"Pattern scanning completed. "
            f"{total_patterns} total patterns detected across {len(all_scan_results)} successful scans"
        )
        
        return summary_result
    
    def generate_reports(self, scan_results: dict, 
                        generate_charts: bool = True,
                        generate_baseline_summary: bool = True,
                        generate_outcome_charts: bool = True) -> dict:
        """
        生成系统报告
        
        Args:
            scan_results: 扫描结果
            generate_charts: 是否生成形态图表
            generate_baseline_summary: 是否生成基线汇总
            generate_outcome_charts: 是否生成结局分析图表
            
        Returns:
            生成的文件路径字典
        """
        logger.info("Generating reports...")
        
        results = {
            'pattern_charts': [],
            'baseline_summary_chart': None,
            'outcome_charts': [],
            'data_exports': [],
            'system_metrics': {}
        }
        
        if not scan_results.get('success', False):
            logger.warning("No successful scan results to generate reports for")
            return results
        
        try:
            # 1. 生成形态图表
            if generate_charts:
                logger.info("Generating pattern charts...")
                
                for scan_result in scan_results.get('scan_results', []):
                    # 为每个扫描结果生成图表
                    # 这里需要原始数据，简化实现中假设可以重新获取
                    if scan_result.get('patterns'):
                        logger.info(f"Generating charts for scan with {len(scan_result['patterns'])} patterns")
                        # 这里可以调用图表生成方法
                        # chart_paths = self.chart_generator.generate_dynamic_chart_from_scan_result(...)
                        # results['pattern_charts'].extend(chart_paths)
            
            # 2. 生成基线系统汇总
            if generate_baseline_summary:
                logger.info("Generating baseline system summary chart...")
                try:
                    # 汇总基线数据
                    baseline_data = self._aggregate_baseline_data(scan_results)
                    regime_history = self._extract_regime_history(scan_results)
                    
                    summary_chart_path = self.chart_generator.generate_baseline_summary_chart(
                        baseline_data, regime_history
                    )
                    results['baseline_summary_chart'] = summary_chart_path
                    
                except Exception as e:
                    logger.error(f"Failed to generate baseline summary: {e}")
            
            # 3. 生成结局分析图表（如果有结局数据）
            if generate_outcome_charts:
                logger.info("Generating outcome analysis charts...")
                outcome_data = self.scanner.export_outcome_data()
                if outcome_data:
                    results['outcome_charts'] = f"Generated outcome analysis for {len(outcome_data)} patterns"
            
            # 4. 导出系统数据
            logger.info("Exporting system data...")
            
            # 导出基线数据
            baseline_export = self.scanner.export_baseline_data()
            results['data_exports'].append(f"Baseline data exported: {len(baseline_export.get('baselines', {}))} entries")
            
            # 导出结局数据
            outcome_export = self.scanner.export_outcome_data()
            results['data_exports'].append(f"Outcome data exported: {len(outcome_export)} entries")
            
            # 5. 系统性能指标
            results['system_metrics'] = scan_results.get('performance_metrics', {})
            
            logger.info("Report generation completed")
            
        except Exception as e:
            logger.error(f"Error generating reports: {e}")
        
        return results
    
    def _aggregate_baseline_data(self, scan_results: dict) -> dict:
        """汇总基线数据"""
        # 简化实现：从扫描结果中提取基线相关信息
        baseline_data = {
            'total_data_points': 0,
            'regime_transitions': 0,
            'current_regime': 'unknown',
            'coverage_stats': {}
        }
        
        for scan_result in scan_results.get('scan_results', []):
            # 汇总各项指标
            if 'baseline_summary' in scan_result:
                baseline_summary = scan_result['baseline_summary']
                # 这里可以根据实际的基线数据结构进行汇总
        
        return baseline_data
    
    def _extract_regime_history(self, scan_results: dict) -> List[dict]:
        """提取市场状态历史"""
        regime_history = []
        
        for scan_result in scan_results.get('scan_results', []):
            regime_history.append({
                'regime': scan_result.get('market_regime', 'unknown'),
                'timestamp': scan_result.get('scan_time', 0)
            })
        
        return regime_history
    
    def run(self, 
           symbols: List[str] = None,
           start_date: datetime = None,
           end_date: datetime = None,
           data_source: str = None,
           enable_outcome_tracking: bool = True,
           generate_charts: bool = True,
           generate_baseline_summary: bool = True,
           generate_outcome_charts: bool = True) -> dict:
        """
        运行PatternScout
        
        Args:
            symbols: 品种列表
            start_date: 开始日期
            end_date: 结束日期
            data_source: 数据源类型
            enable_outcome_tracking: 是否启用结局追踪
            generate_charts: 是否生成形态图表
            generate_baseline_summary: 是否生成基线汇总
            generate_outcome_charts: 是否生成结局图表
            
        Returns:
            运行结果
        """
        logger.info("Starting PatternScout execution...")
        
        try:
            # 设置数据连接器
            if not self._setup_data_connector(data_source):
                return {'error': 'Failed to setup data connector'}
            
            # 扫描形态
            scan_results = self.scan_patterns(
                symbols, start_date, end_date, enable_outcome_tracking
            )
            
            if not scan_results.get('success', False):
                logger.warning("Pattern scanning failed")
                return {'error': 'Pattern scanning failed', 'scan_results': scan_results}
            
            # 生成报告
            reports = self.generate_reports(
                scan_results, generate_charts, generate_baseline_summary, generate_outcome_charts
            )
            
            # 清理资源
            if self.data_connector:
                self.data_connector.close()
            
            result = {
                'success': True,
                'scan_results': scan_results,
                'reports': reports,
                'summary': {
                    'total_patterns': scan_results.get('total_patterns_detected', 0),
                    'successful_scans': scan_results.get('successful_scans', 0),
                    'total_symbols_scanned': scan_results.get('total_symbols_scanned', 0),
                    'system_status': scan_results.get('system_status', {}),
                    'performance_metrics': scan_results.get('performance_metrics', {})
                }
            }
            
            logger.info("PatternScout execution completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"PatternScout execution failed: {e}")
            return {'error': str(e)}
    


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PatternScout - Enhanced Flag Pattern Recognition with Dynamic Baseline System')
    
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--symbols', nargs='+', help='Symbols to scan (default: all available)')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--data-source', choices=['csv', 'mongodb'], help='Data source type')
    parser.add_argument('--pattern-types', nargs='+', choices=['flag_pattern'], 
                        help='Pattern types to detect (default: flag_pattern)', default=None)
    parser.add_argument('--min-confidence', type=float, help='Minimum confidence score')
    parser.add_argument('--export-dataset', choices=['json', 'csv', 'excel'], 
                        help='Export dataset in specified format')
    parser.add_argument('--no-charts', action='store_true', help='Skip chart generation')
    parser.add_argument('--no-summary', action='store_true', help='Skip summary generation')
    parser.add_argument('--disable-outcome-tracking', action='store_true', help='Disable outcome tracking')
    
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
    
    
    # 创建PatternScout实例
    scout = PatternScout(args.config)
    
    # 运行PatternScout
    result = scout.run(
        symbols=args.symbols,
        start_date=start_date,
        end_date=end_date,
        data_source=args.data_source,
        enable_outcome_tracking=not args.disable_outcome_tracking,
        generate_charts=not args.no_charts,
        generate_baseline_summary=not args.no_summary,
        generate_outcome_charts=not args.no_summary
    )
    
    # 输出结果
    if 'error' in result:
        print(f"Error: {result['error']}")
        sys.exit(1)
    else:
        print("PatternScout completed successfully!")
        
        if 'summary' in result:
            summary = result['summary']
            print(f"Total patterns detected: {summary.get('total_patterns', 0)}")
            print(f"Successful scans: {summary.get('successful_scans', 0)}/{summary.get('total_symbols_scanned', 0)}")
            print(f"System performance: {summary.get('performance_metrics', {})}")
            
            # 显示系统状态
            system_status = summary.get('system_status', {})
            if system_status:
                print(f"Market regime detector: {system_status.get('regime_detector', {})}")
                print(f"Baseline manager: {system_status.get('baseline_manager', {})}")
        else:
            print("No patterns detected")
        
        if result.get('reports'):
            reports = result['reports']
            if reports.get('baseline_summary_chart'):
                print(f"Baseline summary chart: {reports['baseline_summary_chart']}")
            
            if reports.get('pattern_charts'):
                print(f"Generated {len(reports['pattern_charts'])} pattern charts")
            
            if reports.get('outcome_charts'):
                print(f"Outcome analysis: {reports['outcome_charts']}")
            
            # 显示数据导出信息
            data_exports = reports.get('data_exports', [])
            for export_info in data_exports:
                print(f"Data export: {export_info}")


if __name__ == "__main__":
    main()