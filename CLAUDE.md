# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

PatternScout 是一个采用动态基线系统的高精度旗形形态识别工具。主要功能包括：
- **动态基线系统**: 500K线滚动统计，三层鲁棒统计保护，智能市场状态检测
- **失效信号过滤**: 假突破检测、成交量背离分析、形态变形监控
- **六分类结局追踪**: 强势延续、标准延续、突破停滞、假突破反转、内部瘫解、反向运行
- **多数据源支持**: CSV 或 MongoDB 数据源读取 OHLCV 数据
- **智能形态识别**: 动态阈值调整的旗形 (flag) 和三角旗形 (pennant) 检测
- **增强可视化**: 动态基线图表、失效信号标记、结局分析图表
- **多时间周期**: 自适应检测参数，支持分钟到月线的全周期分析
- **数据归档系统**: 形态数据集管理和统计分析

## 开发环境

### 依赖管理
```bash
# 安装依赖
uv sync

# 运行项目
uv run python main.py

# 运行测试（完整测试套件）
uv run python tests/run_tests.py

# 运行核心测试模块
uv run python -m pytest tests/test_technical_indicators.py -v
uv run python -m pytest tests/test_pattern_detectors.py -v

# 运行动态基线系统测试
uv run python -m pytest tests/test_dynamic_baseline_system.py -v
uv run python -m pytest tests/test_dynamic_flagpole_detector.py -v
uv run python -m pytest tests/test_dynamic_flag_detector.py -v
uv run python -m pytest tests/test_pattern_outcome_tracker.py -v
uv run python -m pytest tests/test_dynamic_pattern_scanner.py -v

# 运行专项算法测试
uv run python -m pytest tests/test_atr_adaptive.py -v
uv run python -m pytest tests/test_ransac.py -v

# 运行算法性能验证
uv run python tests/quick_real_test.py
uv run python tests/simple_performance_analysis.py

# 代码检查
uv run ruff check .
uv run ruff format .
uv run mypy .
```

### 命令行使用
```bash
# 动态基线模式扫描（推荐）
uv run python main.py --symbols RBL8

# 启用结局追踪的完整动态扫描
uv run python main.py --symbols AAPL MSFT --start-date 2023-01-01

# 传统模式（向后兼容）
uv run python main.py --symbols RBL8 --legacy-mode

# 禁用结局追踪的动态扫描
uv run python main.py --symbols RBL8 --disable-outcome-tracking

# 指定配置文件
uv run python main.py --config config_multi_timeframe.yaml

# 时间范围限制
uv run python main.py --start-date 2023-01-01 --end-date 2023-12-31

# 使用MongoDB数据源
uv run python main.py --data-source mongodb

# 生成基线汇总和结局图表
uv run python main.py --symbols RBL8 --no-charts  # 跳过个别图表，只生成汇总

# 向后兼容的形态类型指定（自动转换为动态基线系统）
uv run python main.py --pattern-types flag pennant --min-confidence 0.4
```

## 核心架构

### 动态基线系统架构（最新重大更新）
PatternScout 3.0 引入了革命性的动态基线系统，完全重构了形态识别流程：

#### 六阶段动态识别与数据归档流程
- **阶段0 - 动态基线系统**: 500K线滚动统计，三层鲁棒统计保护，智能市场状态检测
- **阶段1 - 动态旗杆检测**: 基于动态阈值的旗杆识别，斜率评分和成交量爆发验证
- **阶段2 - 动态旗面检测**: 百分位通道构建，失效信号预过滤，几何形态分析
- **阶段3 - 形态结局分析**: 六分类结局监控系统，纯粹的结局分类逻辑
- **阶段4 - 形态数据输出**: 结构化数据记录，形态DNA特征存储，环境快照归档
- **阶段5 - 形态可视化及图表输出**: 标准化PNG格式K线图，形态"可视化指纹"生成

#### 核心技术特性
- **三层鲁棒统计保护**: MAD过滤 + Winsorize + 动态阈值调整，防止异常值污染
- **智能市场状态检测**: 双状态基线管理，防震荡机制，自适应波动率分析
- **失效信号识别**: 假突破检测、成交量背离分析、形态变形监控
- **六分类结局系统**: 强势延续、标准延续、突破停滞、假突破反转、内部瘫解、反向运行

#### 动态基线API接口
```python
# 现代动态基线扫描器
class DynamicPatternScanner:
    def scan(df, enable_outcome_analysis=True, enable_data_export=True, enable_chart_generation=True) -> Dict[str, Any]:
        # 完整的六阶段动态识别与数据归档流程
        # 阶段0: 更新市场状态和动态基线
        # 阶段1: 动态旗杆检测
        # 阶段2: 动态旗面检测和失效信号过滤
        # 阶段3: 形态结局分析（可选）
        # 阶段4: 形态数据输出（可选）
        # 阶段5: 形态可视化及图表输出（可选）

# 统一主程序入口（向后兼容）
class PatternScout(DynamicPatternScout):
    def scan_patterns(symbols, start_date, end_date) -> List[PatternRecord]:
        # 自动使用动态基线系统，向后兼容API
    
    def run_dynamic(**kwargs) -> dict:
        # 完整动态模式，包含增强功能
    
    def run_legacy(**kwargs) -> dict:
        # 传统模式，用于向后兼容
```

### 统一形态检测架构（已更新）
PatternScout 3.0 在动态基线系统基础上保持统一的形态检测架构：

- **统一形态类型**: `PatternType.FLAG_PATTERN` 包含两个子类型：
  - `FlagSubType.FLAG`: 矩形旗（平行通道）
  - `FlagSubType.PENNANT`: 三角旗（收敛三角形）
- **单一检测器**: `FlagDetector` 统一处理所有旗形变体
- **向后兼容**: PatternScanner 提供传统的 `scan_flags()` 和 `scan_pennants()` 方法

### 统一检测接口
```python
# 现代化的统一接口
class FlagDetector(BasePatternDetector):
    def detect(df, timeframe=None) -> List[PatternRecord]:
        # 1. 自动检测时间周期
        # 2. 选择对应策略和参数
        # 3. 数据预处理
        # 4. 检测旗杆（共享逻辑）
        # 5. 统一检测矩形旗和三角旗
        # 6. 重叠检测和置信度优化选择

# PatternScanner 统一调度
class PatternScanner:
    def scan(df, pattern_types=[PatternType.FLAG_PATTERN]) -> Dict[str, List[PatternRecord]]
    def scan_flags(df) -> List[PatternRecord]  # 向后兼容
    def scan_pennants(df) -> List[PatternRecord]  # 向后兼容
```

### 多时间周期架构
- **TimeframeManager**: 自动检测数据时间周期（分析timestamp间隔，不依赖文件名）
- **时间周期分类**:
  - `1m`: 1分钟周期（超高频交易）
  - `5m`: 5分钟周期（高频交易）
  - `15m`: 15分钟周期（短线交易）
  - `1h`: 1小时周期（日内交易）
  - `4h`: 4小时周期（短中线交易）
  - `1d`: 日线周期（中线交易）
  - `1w`: 周线周期（中长线交易）
  - `1M`: 月线周期（长线交易）
- **策略模式**: 不同周期使用专门优化的检测策略
- **自适应参数**: 根据检测到的时间周期自动选择对应参数集

### 数据流
1. **DataConnector** -> 从CSV/MongoDB获取OHLCV数据
2. **TimeframeManager** -> 自动检测时间周期并分类
3. **FlagDetector** -> 使用统一检测器识别所有旗形变体
4. **QualityScorer** -> 多维度质量评分
5. **BreakthroughAnalyzer** -> 分析形态有效性
6. **ChartGenerator** -> 生成TradingView风格图表
7. **DatasetManager** -> 数据集管理和持久化

### 主要组件

#### 检测器层次结构（已重构）
- **BasePatternDetector**: 抽象基类，统一接口
- **FlagDetector**: 统一的旗形检测器（包含矩形旗和三角旗检测逻辑）
- **PatternScanner**: 统一扫描器，提供现代化API和向后兼容API

#### 数据模型（已更新）
使用 Pydantic 和 dataclass 构建严格类型系统：
- **PatternRecord**: 完整形态记录（ID、置信度、质量评级、sub_type字段）
- **Flagpole**: 旗杆模型（方向、高度、成交量比率）
- **TrendLine**: 趋势线模型（起止点、斜率、R²）
- **PatternType**: 主要形态类型枚举（FLAG_PATTERN，为未来扩展预留）
- **FlagSubType**: 旗形子类型枚举（FLAG, PENNANT）

#### 配置系统（已重构）
支持两套配置文件：
- `config.yaml`: 标准配置（已更新为统一架构）
- `config_multi_timeframe.yaml`: 多时间周期配置（已更新为统一架构）

配置结构：
```yaml
# 统一的形态检测参数
pattern_detection:
  flag_pattern:
    flagpole: {...}      # 旗杆检测参数
    flag: {...}          # 矩形旗检测参数
    pennant: {...}       # 三角旗检测参数
    scoring: {...}       # 统一评分参数

# 统一的评分权重
scoring_weights:
  flag_pattern:
    flag:                # 矩形旗评分权重
      slope_direction: 0.25
      parallel_quality: 0.25
      volume_pattern: 0.25
      channel_containment: 0.15
      time_proportion: 0.10
    pennant:             # 三角旗评分权重
      convergence_quality: 0.30
      symmetry: 0.25
      volume_pattern: 0.25
      size_proportion: 0.10
      apex_validity: 0.10
```

## 关键架构模式

### 工厂模式
- **DataConnectorFactory**: 创建CSV/MongoDB连接器
- 根据配置动态选择数据源类型

### 策略模式
- **TimeframeStrategy**:
  - MinuteOneStrategy: 1分钟数据优化
  - MinuteFiveStrategy: 5分钟数据优化
  - MinuteFifteenStrategy: 15分钟数据优化
  - HourOneStrategy: 1小时数据优化
  - HourFourStrategy: 4小时数据优化
  - DayOneStrategy: 日线数据优化
  - WeekOneStrategy: 周线数据优化
  - MonthOneStrategy: 月线数据优化
- 每个策略包含专门的预处理、验证和评分逻辑

### 组件协调
PatternScout主类协调完整工作流：
1. **ConfigManager**: 统一配置管理（YAML + 环境变量）
2. **DataConnector**: 数据访问抽象层
3. **PatternScanner**: 执行统一形态识别
4. **BreakthroughAnalyzer**: 形态成功率分析
5. **ChartGenerator**: 可视化图表生成
6. **DatasetManager**: SQLite + JSON双重存储

## 特殊注意事项

### 数据格式要求
CSV数据必须包含标准OHLCV列：
- `timestamp`/`date`: 时间戳
- `symbol`: 品种代码
- `open`, `high`, `low`, `close`: OHLC价格
- `volume`: 成交量

### TA-Lib依赖
使用特定Windows wheel安装：
```toml
[tool.uv.sources]
ta-lib = { url = "https://github.com/cgohlke/talib-build/releases/download/v0.6.4/ta_lib-0.6.4-cp313-cp313-win_amd64.whl" }
```

### 时间周期检测
系统通过分析数据内容自动检测时间周期：
- 计算timestamp列相邻时间间隔
- 使用中位数匹配标准周期
- 支持±10%误差的模糊匹配
- **不依赖文件名**进行周期判断

### 输出文件组织
```
output/
├── data/
│   ├── patterns/          # JSON形态记录
│   ├── patterns.db        # SQLite数据库
│   └── exports/           # 导出文件
├── charts/
│   ├── flag/              # 旗形图表
│   ├── pennant/           # 三角旗形图表
│   └── summary/           # 汇总图表
└── reports/               # 执行报告
```

### 重要算法细节
- **统一形态检测**: 单一FlagDetector同时处理矩形旗和三角旗，使用重叠检测和置信度优化
- **Pennant收敛检测**: 计算上下边界线交点（apex）和收敛比例
- **旗杆检测**: 基于价格变化率和成交量激增
- **质量评分**: 多维度权重评分（几何特征、成交量模式、技术确认）
- **平行度验证**: 使用线性回归R²值评估旗形通道质量

#### 算法优化（2025年1月实施）
- **ATR自适应参数系统**: 根据市场波动率自动调整检测参数
  - 5级波动率分类（very_low, low, medium, high, very_high）
  - 自动调整最小高度、趋势强度、平行容忍度、置信度阈值
  - 集成到所有检测器中，提供适应性检测能力
- **RANSAC趋势线拟合**: 提供鲁棒的异常值处理
  - 替代传统OLS回归，在存在异常值时R²从0.074提升到0.946
  - 自适应阈值计算，基于数据标准差
  - 支持与传统方法的性能比较和统计分析
- **智能摆动点检测**: 使用基于ATR的突出度过滤，取代简单的局部极值
- **增强成交量分析**: 
  - 趋势线性回归分析
  - 流动性健康度检查
  - 异常峰值检测
  - 多维度成交量健康度评分（0-1）
- **旗形改进**:
  - 通道发散检查
  - 突破准备度评估
  - 摆动点优先拟合
- **三角旗形改进**:
  - 支持三角形分类（对称/上升/下降）
  - 差异化验证逻辑
  - 支撑阻力质量评估

### 日志系统
使用loguru库，配置在config.yaml中：
- 默认输出: `logs/pattern_scout.log`
- 支持文件轮转和级别控制
- 调试时关注`Auto-detected timeframe`日志确认周期检测结果

## 环境配置

### Python版本要求
- Python >= 3.13（严格要求，使用了最新语法特性）
- 使用uv作为包管理器

### 环境变量配置
复制`.env.example`到`.env`并配置MongoDB连接信息（如使用MongoDB数据源）：
```bash
cp .env.example .env
```

环境变量包括：
- `MONGODB_USERNAME`: MongoDB用户名
- `MONGODB_PASSWORD`: MongoDB密码
- `API_KEY`: API密钥（如有需要）

## 项目结构特点

### 六阶段分层架构（PatternScout 3.0完整重构）
```
src/
├── data/           # 数据层
│   ├── connectors/ # CSV/MongoDB连接器
│   └── models/     # Pydantic数据模型（增强版：失效信号、结局分析、市场快照）
├── patterns/       # 形态检测层（阶段0-2）
│   ├── base/       # 基础组件
│   │   ├── robust_statistics.py      # 三层鲁棒统计保护
│   │   ├── market_regime_detector.py # 智能市场状态检测
│   │   └── timeframe_manager.py      # 时间周期管理
│   ├── detectors/  # 检测器实现
│   │   ├── dynamic_flagpole_detector.py # 阶段1: 动态旗杆检测器
│   │   ├── dynamic_flag_detector.py     # 阶段2: 动态旗面检测器
│   │   ├── dynamic_pattern_scanner.py  # 六阶段统一扫描器
│   │   └── pattern_scanner.py          # 传统扫描器（兼容）
│   ├── indicators/ # 技术指标
│   └── strategies/ # 时间周期策略
├── analysis/       # 结局分析层（阶段3）
│   ├── pattern_outcome_analyzer.py    # 纯粹的结局分析器
│   └── pattern_outcome_tracker.py     # 传统结局追踪器（兼容）
├── storage/        # 数据输出层（阶段4）
│   ├── pattern_data_exporter.py       # 形态数据输出器
│   └── dataset_manager.py             # 传统数据集管理器（兼容）
├── visualization/  # 可视化层（阶段5）
│   ├── pattern_chart_generator.py     # 专门的形态图表生成器
│   ├── chart_generator.py             # 传统图表生成器（兼容）
│   └── dynamic_chart_methods.py       # 动态基线图表方法
└── utils/         # 工具类（配置管理）
```

### 测试体系（六阶段系统全覆盖）
完整的unittest测试套件，覆盖所有核心组件：
- **动态基线系统测试**: 三层鲁棒统计、市场状态检测、基线管理（阶段0）
- **动态检测器测试**: 旗杆检测器、旗面检测器、失效信号识别（阶段1-2）
- **结局分析系统测试**: 六分类结局监控、形态有效性验证（阶段3）
- **数据输出系统测试**: 结构化数据存储、形态DNA特征、环境快照（阶段4）
- **可视化系统测试**: 图表生成、标准化可视化、批量处理（阶段5）
- **集成测试**: 完整六阶段流程、系统状态管理、性能指标
- **传统模块测试**: 技术指标、数据连接器、向后兼容性

### 动态基线系统专项测试
- **tests/test_dynamic_baseline_system.py**: 三层鲁棒统计保护和市场状态检测测试
- **tests/test_dynamic_flagpole_detector.py**: 动态旗杆检测器完整功能测试
- **tests/test_dynamic_flag_detector.py**: 动态旗面检测和失效信号识别测试
- **tests/test_pattern_outcome_tracker.py**: 六分类结局追踪系统测试
- **tests/test_dynamic_pattern_scanner.py**: 完整三阶段集成流程测试

### 传统专项测试工具
- **tests/test_atr_adaptive.py**: ATR自适应参数系统测试
- **tests/test_algorithm_improvements.py**: 算法改进综合测试
- **tests/test_ransac.py**: RANSAC算法性能和鲁棒性测试
- **tests/test_real_data.py**: 真实数据算法验证框架
- **tests/quick_real_test.py**: 快速真实数据验证
- **tests/simple_performance_analysis.py**: 性能基准测试和优化分析

### 性能基准
基于真实数据的性能测试结果（RBL8期货15分钟数据）：
- **处理速度**: 250记录/秒，4ms/记录
- **扩展性**: 扩展因子1.04（接近理想值1.0）
- **算法改进**: RANSAC在异常值环境下R²从0.074提升到0.946
- **当前性能水平**: 良好（2000条记录处理时间8.26秒）

### 关键性能组件
- **ATRAdaptiveManager**: 自动波动率分析和参数调整
- **RANSACTrendLineFitter**: 鲁棒趋势线拟合，支持异常值检测
- **PatternComponents**: 增强的形态组件，集成智能检测算法
- **FlagDetector**: 统一的检测器基类，支持ATR自适应和RANSAC拟合，处理所有旗形变体

## 重构说明

### 2025年1月六阶段系统重大重构
**PatternScout 3.0** - 完全重构的六阶段动态基线系统实现：

#### 核心系统重构
- **动态基线系统**: 500K线滚动统计，三层鲁棒统计保护，双状态基线管理
- **六阶段识别与归档流程**: 基线更新 → 动态旗杆检测 → 动态旗面检测 → 结局分析 → 数据输出 → 图表生成
- **失效信号系统**: 假突破检测、成交量背离分析、形态变形监控
- **六分类结局追踪**: 强势延续、标准延续、突破停滞、假突破反转、内部瘫解、反向运行
- **完整数据归档**: 结构化数据存储、形态DNA特征、环境快照、标准化可视化

#### 数据模型扩展
- **新增枚举类型**: PatternOutcome、MarketRegime、IndicatorType
- **增强数据结构**: DynamicBaseline、InvalidationSignal、MarketSnapshot、PatternOutcomeAnalysis
- **向后兼容**: 保持原有PatternRecord、Flagpole、TrendLine结构

#### 六阶段系统重构
- **阶段3 - PatternOutcomeAnalyzer**: 纯粹的结局分析器，专注于结局分类逻辑
- **阶段4 - PatternDataExporter**: 专门的数据输出器，结构化数据存储和导出
- **阶段5 - PatternChartGenerator**: 专门的图表生成器，标准化可视化图表
- **DynamicPatternScanner**: 统一六阶段扫描流程，系统状态管理，性能监控

#### 模块化架构升级
- **独立模块设计**: 每个阶段独立可配置，支持选择性启用/禁用
- **专业化分工**: 结局分析、数据输出、可视化各司其职，提高代码可维护性
- **完整向后兼容**: 保持原有API，用户无感知升级

#### API兼容性设计
- **完全向后兼容**: 原有API自动转换到动态基线系统
- **双模式支持**: run_dynamic()（推荐）和 run_legacy()（兼容）
- **渐进式迁移**: 用户可选择启用或禁用动态功能

#### 测试体系完善
- **全覆盖测试套件**: 六阶段系统所有组件的完整单元测试
- **集成测试**: 六阶段完整流程测试，系统状态验证，跨模块交互
- **性能测试**: 扫描性能、内存使用、系统稳定性测试
- **模块化测试**: 每个阶段独立测试，确保模块间解耦

这次六阶段重构建立了更强大的技术基础，实现了从形态识别到数据归档的完整闭环，为未来扩展更多形态类型（头肩形、双顶双底等）和高级分析功能提供了完善的模块化架构支持。