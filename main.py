"""
PatternScout主程序入口
支持传统模式和动态基线模式
"""
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# 设置控制台编码（Windows）
if sys.platform.startswith('win'):
    os.system('chcp 65001 > nul')
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7 的兼容性
        import locale
        locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')

# 添加src到Python路径
sys.path.append(str(Path(__file__).parent))

from main_dynamic import DynamicPatternScout
from loguru import logger

# 配置loguru日志格式
logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
)


class PatternScout(DynamicPatternScout):
    """
    PatternScout主程序类（向后兼容别名）
    
    该类继承自DynamicPatternScout，提供向后兼容的API
    建议使用DynamicPatternScout获得完整的动态基线功能
    """
    
    def __init__(self, config_file: str = "config.yaml"):
        super().__init__(config_file)
        logger.info("PatternScout initialized with dynamic baseline system")
    
    def scan_patterns(self, symbols=None, start_date=None, end_date=None, pattern_types=None):
        """向后兼容的扫描方法 - 自动使用动态基线系统"""
        logger.info("Using dynamic baseline scanning for enhanced pattern detection")
        result = self.scan_patterns_dynamic(symbols, start_date, end_date)
        
        # 提取patterns列表以保持向后兼容
        all_patterns = []
        for scan_result in result.get('scan_results', []):
            all_patterns.extend(scan_result.get('patterns', []))
        
        return all_patterns
    
    def generate_reports(self, patterns=None, generate_charts=True, generate_summary=True):
        """向后兼容的报告生成方法"""
        if patterns is None:
            # 如果没有传入patterns，尝试生成动态报告
            return {'message': 'Use generate_dynamic_reports for enhanced functionality'}
        
        return self._generate_legacy_reports(patterns, generate_charts, generate_summary)
    
    def run(self, symbols=None, start_date=None, end_date=None, data_source=None, 
            pattern_types=None, generate_charts=True, generate_summary=True):
        """向后兼容的运行方法 - 默认使用动态基线系统"""
        logger.info("Running PatternScout with dynamic baseline system")
        
        # 自动使用动态模式以获得更好的性能
        return self.run_dynamic(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            data_source=data_source,
            enable_outcome_tracking=True,
            generate_charts=generate_charts,
            generate_baseline_summary=generate_summary,
            generate_outcome_charts=generate_summary
        )


def main():
    """
    主函数 - 提供命令行接口
    
    支持两种运行模式：
    1. 动态模式（推荐）：使用动态基线系统进行增强识别
    2. 传统模式：保持向后兼容性
    """
    parser = argparse.ArgumentParser(
        description='PatternScout - Enhanced Flag Pattern Recognition System',
        epilog='Example: python main.py --symbols AAPL MSFT --start-date 2023-01-01'
    )
    
    # 基本参数
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--symbols', nargs='+', help='Symbols to scan (default: all available)')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--data-source', choices=['csv', 'mongodb'], help='Data source type')
    
    # 形态相关参数
    parser.add_argument('--pattern-types', nargs='+', choices=['flag_pattern', 'flag', 'pennant'], 
                        help='Pattern types (legacy parameter, now uses unified flag_pattern)', default=None)
    parser.add_argument('--min-confidence', type=float, help='Minimum confidence score')
    
    # 输出相关参数
    parser.add_argument('--no-charts', action='store_true', help='Skip chart generation')
    parser.add_argument('--no-summary', action='store_true', help='Skip summary generation')
    parser.add_argument('--export-dataset', choices=['json', 'csv', 'excel'], 
                        help='Export dataset in specified format')
    
    # 模式控制参数
    parser.add_argument('--legacy-mode', action='store_true', 
                        help='Use legacy scanning mode (not recommended)')
    parser.add_argument('--disable-outcome-tracking', action='store_true', 
                        help='Disable outcome tracking in dynamic mode')
    parser.add_argument('--analyze-breakthrough', action='store_true', 
                        help='Enable breakthrough analysis (legacy feature)')
    
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
    
    try:
        # 根据模式选择运行方式
        if args.legacy_mode:
            logger.info("Running in legacy mode")
            result = scout.run_legacy(
                symbols=args.symbols,
                start_date=start_date,
                end_date=end_date,
                data_source=args.data_source,
                pattern_types=args.pattern_types,
                generate_charts=not args.no_charts,
                generate_summary=not args.no_summary
            )
        else:
            logger.info("Running in dynamic baseline mode (recommended)")
            result = scout.run_dynamic(
                symbols=args.symbols,
                start_date=start_date,
                end_date=end_date,
                data_source=args.data_source,
                enable_outcome_tracking=not args.disable_outcome_tracking,
                generate_charts=not args.no_charts,
                generate_baseline_summary=not args.no_summary,
                generate_outcome_charts=not args.no_summary
            )
        
        # 处理结果输出
        _handle_results(result, args.legacy_mode)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"Error: {e}")
        sys.exit(1)


def _handle_results(result, is_legacy_mode):
    """处理并显示结果"""
    if 'error' in result:
        print(f"Error: {result['error']}")
        sys.exit(1)
    
    print("\n" + "="*60)
    mode_text = "Legacy Mode" if is_legacy_mode else "Dynamic Baseline Mode"
    print(f"PatternScout completed successfully in {mode_text}!")
    print("="*60)
    
    if 'summary' in result:
        summary = result['summary']
        
        if is_legacy_mode:
            # 传统模式结果显示
            print(f"\nResults Summary:")
            print(f"  Total patterns detected: {summary.get('total_patterns', 0)}")
            print(f"  Quality distribution: {summary.get('quality_distribution', {})}")
            
        else:
            # 动态模式结果显示
            print(f"\nDynamic Scanning Results:")
            print(f"  Total patterns detected: {summary.get('total_patterns', 0)}")
            print(f"  Successful scans: {summary.get('successful_scans', 0)}/{summary.get('total_symbols_scanned', 0)}")
            
            # 显示性能指标
            perf_metrics = summary.get('performance_metrics', {})
            if perf_metrics:
                print(f"\nSystem Performance:")
                print(f"  Total scans: {perf_metrics.get('total_scans', 0)}")
                print(f"  Regime transitions: {perf_metrics.get('regime_transitions', 0)}")
                print(f"  Average patterns per scan: {perf_metrics.get('avg_patterns_per_scan', 0):.2f}")
            
            # 显示系统状态
            system_status = summary.get('system_status', {})
            if system_status:
                regime_info = system_status.get('regime_detector', {})
                if regime_info:
                    print(f"\nMarket Regime Analysis:")
                    print(f"  Current regime: {regime_info.get('current_regime', 'unknown')}")
                    print(f"  Regime stability: {regime_info.get('is_stable', False)}")
    
    # 显示生成的报告
    if result.get('reports'):
        reports = result['reports']
        print(f"\nGenerated Reports:")
        
        if reports.get('baseline_summary_chart'):
            print(f"  ✓ Baseline summary: {reports['baseline_summary_chart']}")
        
        pattern_charts = reports.get('pattern_charts', [])
        if pattern_charts:
            print(f"  ✓ Pattern charts: {len(pattern_charts)} generated")
        
        data_exports = reports.get('data_exports', [])
        for export_info in data_exports:
            print(f"  ✓ {export_info}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()