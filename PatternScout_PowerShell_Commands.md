# PatternScout Windows PowerShell 执行命令

本文档包含在Windows PowerShell中运行PatternScout形态检测和图表生成的完整命令序列。

## 前置准备

如果遇到中文编码问题，请先在PowerShell中执行：
```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

## 1. 清空之前的数据（可选）

```powershell
cd "C:\quant\strategy\PatternScout\pattern-scout"
Remove-Item -Path "output\data\patterns.db" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "output\data\patterns\*" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "output\charts\*" -Recurse -Force -ErrorAction SilentlyContinue  
Remove-Item -Path "output\data\exports\*" -Recurse -Force -ErrorAction SilentlyContinue
```

## 2. 生成形态数据

```powershell
cd "C:\quant\strategy\PatternScout\pattern-scout"
uv run python main.py --config config_multi_timeframe.yaml --symbols RBL8 --pattern-types flag --min-confidence 0.2 --export-dataset json
```

**预期结果**：
- 检测约58个旗形形态
- 数据保存到SQLite数据库和JSON文件
- 处理时间约2-4分钟

## 3. 生成所有图表

```powershell
cd "C:\quant\strategy\PatternScout\pattern-scout"
uv run python -c @"
from pathlib import Path
import json
import matplotlib.pyplot as plt
from src.visualization.chart_generator import ChartGenerator
from src.utils.config_manager import ConfigManager

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

config_manager = ConfigManager('config_multi_timeframe.yaml')
config = config_manager.config
chart_generator = ChartGenerator(config)

pattern_files = list(Path('output/data/patterns').glob('*.json'))
print(f'总共找到 {len(pattern_files)} 个形态文件')

success_count = 0
error_count = 0

for i, file in enumerate(pattern_files):
    with open(file, 'r', encoding='utf-8') as f:
        pattern_data = json.load(f)
    
    try:
        chart_path = chart_generator.generate_pattern_chart_from_data(pattern_data)
        if chart_path:
            success_count += 1
            print(f'[{i+1}/{len(pattern_files)}] 生成图表: {Path(chart_path).name}')
    except Exception as e:
        error_count += 1
        print(f'[{i+1}/{len(pattern_files)}] 生成失败 {file.name}: {str(e)[:50]}...')

print(f'\n生成完成! 成功: {success_count}, 失败: {error_count}')
"@
```

**预期结果**：
- 生成约49个TradingView风格的PNG图表
- 成功率约85%
- 处理时间约3-5分钟

## 4. 查看生成结果

### 4.1 查看形态数据统计
```powershell
cd "C:\quant\strategy\PatternScout\pattern-scout"
uv run python -c @"
import sqlite3
conn = sqlite3.connect('output/data/patterns.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM patterns')
count = cursor.fetchone()[0]
print(f'数据库中的形态数量: {count}')
conn.close()
"@
```

### 4.2 查看文件数量统计
```powershell
# 查看生成的图表数量
Write-Host "生成的图表数量: $((Get-ChildItem -Path 'output\charts\flag\*.png' | Measure-Object).Count)"

# 查看JSON文件数量  
Write-Host "JSON文件数量: $((Get-ChildItem -Path 'output\data\patterns\*.json' | Measure-Object).Count)"
```

## 5. 生成统计报告

```powershell
cd "C:\quant\strategy\PatternScout\pattern-scout"
uv run python -c @"
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime

conn = sqlite3.connect('output/data/patterns.db')
total_patterns = pd.read_sql_query('SELECT COUNT(*) as total FROM patterns', conn).iloc[0]['total']
chart_files = list(Path('output/charts/flag').glob('*.png'))
json_files = list(Path('output/data/patterns').glob('*.json'))

print(f'=== PatternScout 生成结果 ===')
print(f'检测形态数量: {total_patterns} 个')
print(f'生成图表数量: {len(chart_files)} 个')
print(f'JSON文件数量: {len(json_files)} 个')

confidence_stats = pd.read_sql_query('SELECT MIN(confidence_score) as min_conf, MAX(confidence_score) as max_conf, AVG(confidence_score) as avg_conf FROM patterns', conn)
print(f'置信度范围: {confidence_stats.iloc[0]["min_conf"]:.3f} - {confidence_stats.iloc[0]["max_conf"]:.3f}')
print(f'平均置信度: {confidence_stats.iloc[0]["avg_conf"]:.3f}')

conn.close()
"@
```

## 6. 快速查看生成的文件（可选）

```powershell
# 查看图表文件列表
Write-Host "=== 生成的图表文件 ==="
Get-ChildItem -Path "output\charts\flag\" -Name

# 查看数据文件列表
Write-Host "`n=== 生成的数据文件 ==="
Get-ChildItem -Path "output\data\patterns\" -Name

# 打开图表文件夹
Write-Host "`n正在打开图表文件夹..."
Invoke-Item "output\charts\flag\"
```

## 7. 详细质量分析（可选）

```powershell
cd "C:\quant\strategy\PatternScout\pattern-scout"
uv run python -c @"
import sqlite3
import pandas as pd
from datetime import datetime

print('# PatternScout 详细分析报告')
print(f'生成时间: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
print()

conn = sqlite3.connect('output/data/patterns.db')

# 质量分布
quality_dist = pd.read_sql_query('SELECT pattern_quality, COUNT(*) as count FROM patterns GROUP BY pattern_quality ORDER BY count DESC', conn)
total_patterns = pd.read_sql_query('SELECT COUNT(*) as total FROM patterns', conn).iloc[0]['total']

print('## 质量等级分布:')
for _, row in quality_dist.iterrows():
    percentage = (row['count'] / total_patterns) * 100
    print(f'- {row[\"pattern_quality\"]}: {row[\"count\"]} 个 ({percentage:.1f}%)')

# 高置信度形态
high_conf = pd.read_sql_query('SELECT symbol, detection_date, confidence_score FROM patterns WHERE confidence_score > 0.75 ORDER BY confidence_score DESC LIMIT 10', conn)
print(f'\\n## 顶级形态 (置信度>0.75):')
for i, row in high_conf.iterrows():
    print(f'{i+1}. 置信度 {row[\"confidence_score\"]:.3f} - {row[\"symbol\"]} - {row[\"detection_date\"][:10]}')

conn.close()
print(f'\\n=== 分析完成 ===')
"@
```

## 输出文件结构

执行完成后，生成的文件结构如下：

```
output/
├── data/
│   ├── patterns.db              # SQLite数据库（58个形态记录）
│   └── patterns/                # JSON格式文件目录
│       ├── RBL8_20240729_1445_上升旗形.json
│       ├── RBL8_20240730_1045_下降旗形.json
│       └── ... (共58个文件)
└── charts/
    └── flag/                    # PNG图表文件目录
        ├── RBL8_20240729_1445_上升旗形.png
        ├── RBL8_20240730_1045_下降旗形.png
        └── ... (约49个文件)
```

## 技术特性

- **ATR自适应参数系统**：根据市场波动率自动调整检测参数
- **RANSAC鲁棒拟合**：提供抗异常值的趋势线拟合
- **多时间周期检测**：15分钟数据专用短周期策略
- **TradingView风格图表**：专业级可视化呈现
- **双重数据存储**：SQLite数据库 + JSON文件格式

## 注意事项

1. **执行顺序**：请按照1-7的顺序执行命令
2. **处理时间**：整个流程需要5-10分钟完成
3. **成功率**：图表生成成功率约85%，少数形态因数据长度问题可能生成失败
4. **中文字体**：如果图表中文显示异常，是正常的matplotlib字体警告
5. **数据完整性**：即使部分图表生成失败，所有形态数据都会完整保存

## 故障排除

如果遇到问题：

1. **Python环境**：确保已安装uv并配置正确
2. **依赖包**：运行`uv sync`确保所有依赖已安装
3. **权限问题**：确保对目录有写入权限
4. **路径问题**：确保在正确的项目根目录执行命令

---

*本文档基于PatternScout项目生成，适用于Windows PowerShell环境*