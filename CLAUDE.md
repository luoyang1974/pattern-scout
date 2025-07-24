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

### 日志系统
使用loguru库，配置在config.yaml中：
- 默认输出: `logs/pattern_scout.log`
- 支持文件轮转和级别控制
- 调试时关注`Auto-detected timeframe`日志确认周期检测结果