# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

PatternScout 是一个用于技术分析的自动化旗形和三角旗形形态识别工具。主要功能包括：
- 从 CSV 或 MongoDB 数据源读取 OHLCV 数据
- 自动检测旗形 (flag) 和三角旗形 (pennant) 形态
- 生成可视化图表和形态分析报告
- 支持多时间周期的自适应检测
- 批量数据集管理和突破分析

## 开发环境

### 依赖管理
```bash
# 安装依赖
uv sync

# 运行项目
uv run python main.py

# 运行测试（完整测试套件）
uv run python tests/run_tests.py

# 运行单个测试模块
uv run python -m pytest tests/test_technical_indicators.py -v
uv run python -m pytest tests/test_pattern_detectors.py -v

# 运行专项测试
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
# 基本扫描（所有形态）
uv run python main.py --symbols RBL8

# 指定配置文件
uv run python main.py --config config_multi_timeframe.yaml

# 指定形态类型和置信度
uv run python main.py --pattern-types flag pennant --min-confidence 0.4

# 导出数据集并执行突破分析
uv run python main.py --export-dataset json --analyze-breakthrough

# 时间范围限制
uv run python main.py --start-date 2023-01-01 --end-date 2023-12-31

# 使用MongoDB数据源
uv run python main.py --data-source mongodb
```

## 核心架构

### 多时间周期架构
PatternScout 2.0 采用多时间周期自适应架构：

- **TimeframeManager**: 自动检测数据时间周期（分析timestamp间隔，不依赖文件名）
- **时间周期分类**:
  - `ultra_short`: 1m, 3m, 5m
  - `short`: 15m, 30m, 1h  
  - `medium_long`: 4h, 1d, 1w
- **策略模式**: 不同周期使用专门优化的检测策略
- **自适应参数**: 根据检测到的时间周期自动选择对应参数集

### 统一检测接口
```python
# BasePatternDetector提供统一接口
class BasePatternDetector(ABC):
    def detect(df, timeframe=None) -> List[PatternRecord]:
        # 1. 自动检测时间周期
        # 2. 选择对应策略和参数
        # 3. 数据预处理
        # 4. 检测旗杆（共享逻辑）
        # 5. 检测具体形态（子类实现）
```

### 数据流
1. **DataConnector** -> 从CSV/MongoDB获取OHLCV数据
2. **TimeframeManager** -> 自动检测时间周期并分类
3. **PatternDetector** -> 使用对应策略检测形态
4. **QualityScorer** -> 多维度质量评分
5. **BreakthroughAnalyzer** -> 分析形态有效性
6. **ChartGenerator** -> 生成TradingView风格图表
7. **DatasetManager** -> 数据集管理和持久化

### 主要组件

#### 检测器层次结构
- **BasePatternDetector**: 抽象基类，统一接口
- **FlagDetector**: 旗形检测（平行通道验证）
- **PennantDetector**: 三角旗形检测（收敛三角形，apex计算）
- **PatternScanner**: 统一扫描器，支持多形态检测

#### 数据模型
使用 Pydantic 和 dataclass 构建严格类型系统：
- **PatternRecord**: 完整形态记录（ID、置信度、质量评级）
- **Flagpole**: 旗杆模型（方向、高度、成交量比率）
- **TrendLine**: 趋势线模型（起止点、斜率、R²）
- **PatternType**: 形态类型枚举（FLAG, PENNANT）

#### 配置系统
支持两套配置文件：
- `config.yaml`: 标准配置
- `config_multi_timeframe.yaml`: 多时间周期配置

配置结构：
```yaml
pattern_detection:
  ultra_short:    # 超短周期参数
    flagpole: {...}
    flag: {...}
    pennant: {...}
  short:          # 短周期参数（15分钟数据默认）
  medium_long:    # 中长周期参数
```

## 关键架构模式

### 工厂模式
- **DataConnectorFactory**: 创建CSV/MongoDB连接器
- 根据配置动态选择数据源类型

### 策略模式
- **TimeframeStrategy**: 
  - UltraShortStrategy: 1-5分钟数据优化
  - ShortStrategy: 15分钟-1小时数据优化  
  - MediumLongStrategy: 4小时-周线数据优化
- 每个策略包含专门的预处理、验证和评分逻辑

### 组件协调
PatternScout主类协调完整工作流：
1. **ConfigManager**: 统一配置管理（YAML + 环境变量）
2. **DataConnector**: 数据访问抽象层
3. **PatternDetectors**: 执行多形态识别
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
复制`.env.example`到`.env`并配置MongoDB连接信息（如使用MongoDB数据源）

## 项目结构特点

### 分层架构
```
src/
├── data/           # 数据层
│   ├── connectors/ # CSV/MongoDB连接器
│   └── models/     # Pydantic数据模型
├── patterns/       # 形态检测层
│   ├── base/       # 基础组件（检测器基类、质量评分、时间周期管理）
│   ├── detectors/  # 具体检测器实现
│   ├── indicators/ # 技术指标
│   └── strategies/ # 时间周期策略
├── analysis/       # 分析层（突破分析）
├── visualization/  # 图表生成
├── storage/        # 数据持久化
└── utils/         # 工具类（配置管理）
```

### 测试体系
完整的unittest测试套件，覆盖所有核心组件：
- 技术指标测试
- 数据连接器测试  
- 形态检测器测试（包含时间周期适应性测试）
- 数据集管理器测试
- 向后兼容性测试

### 专项测试工具
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
- **BasePatternDetector**: 统一的检测器基类，支持ATR自适应和RANSAC拟合