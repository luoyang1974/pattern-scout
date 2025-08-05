### 量化 / 程序化识别“旗杆”（Flag-Pole）的全流程

> 目标：在任意周期的 OHLCV 数据流中，**实时** 或 **批量** 标注出“短暂而暴力”的推进段，为后续旗面整理与突破交易做准备。

------

## 1  数据与基础指标

| 名称           | 公式                           | 说明                         |
| -------------- | ------------------------------ | ---------------------------- |
| `ATR14`        | `ta.atr(high, low, close, 14)` | 统一“幅度”刻度；跨品种自适应 |
| `VOL20`        | `sma(volume, 20)`              | 统一“量能”刻度               |
| `ΔP`           | `close_t – open_0`             | 旗杆首尾价差（方向保留符号） |
| `ΔT`           | `len(window)`                  | 旗杆根数                     |
| `SlopeScore`   | `(                             | ΔP                           |
| `ImpulseBars%` | `count(                        | body                         |
| `VolumeBurst`  | `avg(volume_window) / VOL20`   | **放量倍数**                 |
| `RetraceRatio` | `maxDrawdown(                  | ΔP                           |



------

## 2  “硬”过滤器 —— 判断是否“短暂且暴力”

| 指标           | 阈值（默认） | 目的             |
| -------------- | ------------ | ---------------- |
| `SlopeScore`   | ≥ 6          | 推进速度足够陡   |
| `ImpulseBars%` | ≥ 70 %       | 大部分是长实体 K |
| `VolumeBurst`  | ≥ 1.5        | 有真资金推动     |
| `RetraceRatio` | ≤ 0.20       | 过程回撤极小     |



> 四项必须全部通过；阈值可用回测调优（见 §5）。

------

## 3  “软”过滤器 —— 根数上限按周期自适应

```
text复制编辑StrictBars = {1m:5, 5m:5, 15m:8, 30m:8, 1h:12, D:12, W:6, M:3}
LooseBars  = {1m:8, 5m:8, 15m:12, 30m:12, 1h:15, D:15, W:8, M:3}
```

- **Strict**：超短/高波动市场，最大限度排除假旗杆
- **Loose**：波段/低波动市场，提高检出率
- 根数超 `LooseBars` → 判为“趋势腿”而非旗杆

------

## 4  核心算法（滑窗扫描伪码，平台无关）

```
python复制编辑def detect_flagpoles(df, tf):
    res = []
    strict_max = StrictBars[tf]; loose_max = LooseBars[tf]

    # 提前算好 ATR14、VOL20，滚动窗口 O(1) 更新
    df["atr14"]  = ta.atr(df["High"], df["Low"], df["Close"], 14)
    df["vol20"]  = df["Volume"].rolling(20).mean()

    for end in range(len(df)):
        for win in range(2, loose_max + 1):
            start = end - win + 1
            if start < 0: break

            window = df.iloc[start:end+1]
            atr   = window["atr14"].iloc[-1]
            vol20 = window["vol20"].iloc[-1]

            dP    = window["Close"].iloc[-1] - window["Open"].iloc[0]
            slope = abs(dP) / win / atr
            impulse = (abs(window["Close"] - window["Open"]) >= 0.8*atr).mean()
            volbst  = window["Volume"].mean() / vol20
            retrace = (window["High"].max() - window["Low"].min()) / abs(dP)

            # ---- 硬过滤 ----
            if slope >= 6 and impulse >= 0.7 and volbst >= 1.5 and retrace <= 0.20:
                tag = "strict" if win <= strict_max else "loose"
                res.append({"start":start, "end":end, "bars":win,
                            "slope":slope, "vol_bst":volbst, "tag":tag})
                break  # 越短越优；找到就跳出
    return res
```

### 性能优化

- **双端队列** 存滑动统计 → O(N·StrictMax)
- **并行多周期扫描**：线程 / 协程并发
- **事件驱动**：只在新 K 收盘触发一次扫描

------

## 5  参数自校准（回测 & 动态调优）

1. **网格回测**：

   ```
   text复制编辑SlopeScore ∈ {5,6,7}, Impulse% ∈ {0.6,0.7,0.8},
   BarsMax ∈ {Strict,Loose}
   ```

   对胜率 / 盈亏比 / Sharpe 作热力图，选 Pareto 前沿。

2. **波动率调节**：
    当 20 日 HV > 平均值 × 2 时：

   - `SlopeScore` 阈值 ↑1
   - `StrictBars`、`LooseBars` 各 +1~2

3. **分品种分桶**：
    按市值、流动性、行业等分组，存独立阈值表。

------

## 结论 · 实战守则

1. **四大硬指标**（Slope、Impulse、VolumeBurst、Retrace）严格把关“暴力”。
2. **根数上限** 作为可调软阈值 —— **≤5** 是超短“剃刀”，**≤8-15** 是通用“渔网”。
3. **回测—调参—分桶** 三连，确保算法在不同品种 / 周期都保持统计优势。
4. 建议把 **strict 与 loose 两条管线并行**：
   - *strict* 捕捉情绪冲刺；
   - *loose* 承接正常趋势首段，整体收益更平滑。

