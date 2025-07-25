# PatternScout

PatternScout 是一个用于技术分析的自动化旗形和三角旗形形态识别工具。采用多时间周期自适应检测算法，能够准确识别股票、期货等金融市场中的经典形态。

## 主要功能

- 自动检测旗形 (Flag) 和三角旗形 (Pennant) 形态
- 支持CSV和MongoDB数据源
- 多时间周期自适应检测（超短、短、中长周期）
- 生成高质量的可视化图表和形态分析报告
- 突破分析和形态有效性评估
- 批量数据处理和数据集管理

## 环境要求

### 系统要求
- Python >= 3.13
- Windows/Linux/macOS
- uv 包管理器

### 安装步骤

```bash
# 克隆项目
cd pattern-scout

# 安装依赖
uv sync

# 复制环境配置文件（可选，MongoDB用户需要）
cp .env.example .env
# 编辑 .env 文件配置MongoDB连接信息
```

编辑 `config.yaml` 或 `config_multi_timeframe.yaml` 文件来自定义检测参数。

## 快速开始

### 基础使用

```bash
# 使用默认配置检测所有形态
uv run python main.py

# 指定股票代码
uv run python main.py --symbols AAPL MSFT TSLA

# 指定时间范围
uv run python main.py --start-date 2023-01-01 --end-date 2023-12-31

# 使用MongoDB数据源
uv run python main.py --data-source mongodb

# 禁用图表生成（仅输出检测结果）
uv run python main.py --no-charts
```

### 高级功能

```bash
# 使用多时间周期配置
uv run python main.py --config config_multi_timeframe.yaml

# 指定形态类型和置信度
uv run python main.py --pattern-types flag pennant --min-confidence 0.4

# 导出数据集并执行突破分析
uv run python main.py --export-dataset json --analyze-breakthrough
```

## 数据格式

### CSV数据格式
CSV文件需放置在 `data/csv/` 目录下，包含以下必需列：
- `timestamp`/`date`: 时间戳
- `symbol`: 品种代码  
- `open`: 开盘价
- `high`: 最高价
- `low`: 最低价
- `close`: 收盘价
- `volume`: 成交量

示例文件：`AAPL.csv`、`TSLA.csv`

### MongoDB数据源
在 `config.yaml` 中配置MongoDB连接信息，支持标准OHLCV数据格式。

## 输出结果

### 文件组织
- **output/charts/**: 可视化图表PNG文件
- **output/charts/pattern_summary_*.png**: 汇总图表
- **output/data/patterns/**: JSON格式的形态记录
- **output/data/patterns.db**: SQLite数据库
- **logs/**: 执行日志文件

### 形态评级
- **高质量**: 置信度高，几何特征明显
- **中等质量**: 符合基本条件，有一定可信度
- **低质量**: 边缘案例，需要人工确认

## 技术特性

### 多时间周期架构
- **超短周期** (1m, 3m, 5m): 优化的快速检测策略
- **短周期** (15m, 30m, 1h): 平衡检测精度和速度
- **中长周期** (4h, 1d, 1w): 适合趋势分析的参数

### 智能检测算法
- **自动时间周期识别**: 分析数据间隔，无需依赖文件名
- **旗杆检测**: 基于价格变化率和成交量激增
- **形态验证**: 使用线性回归分析趋势线质量
- **质量评分**: 多维度评分系统（几何特征、成交量模式、技术确认）

### 突破分析
- 形态完成后的价格走势分析
- 成功率统计和有效性评估
- 风险收益比计算

## 开发相关

### 运行测试

```bash
# 运行完整测试套件
uv run python tests/run_tests.py

# 运行单个测试模块
uv run python -m pytest tests/test_technical_indicators.py -v
uv run python -m pytest tests/test_pattern_detectors.py -v
```

### 代码检查

```bash
# 代码格式化和检查
uv run ruff check .
uv run ruff format .
uv run mypy .
```

## 示例输出

成功执行后的输出示例：
```
PatternScout completed successfully!
Total patterns detected: 15
Quality distribution: {'high': 3, 'medium': 8, 'low': 4}
Summary chart: output/charts/pattern_summary_20240721_143022.png
Generated 15 individual charts
```

## 常见问题

### 数据相关
- 确保CSV数据包含所有必需列
- 时间戳格式应为标准datetime格式
- 数据质量直接影响检测效果

### 性能优化
- 大数据集建议使用MongoDB
- 可通过配置文件调整检测参数
- 使用 `--no-charts` 选项可显著提升处理速度

## 技术支持

如有问题或建议，请查看项目日志文件 `logs/pattern_scout.log` 获取详细信息。

---

PatternScout - 专业的形态识别工具