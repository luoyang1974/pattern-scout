采用更严格的、基于规则的逻辑门控方法，是系统化交易中非常常见且强大的做法。这种方法消除了“灰色地带”，结果只有两种：**“是”** 或 **“否”**。我们将这个框架称为**“序贯过滤框架 (Sequential Filtering Framework)”**。

### 序贯过滤框架 (Sequential Filtering Framework)

**核心思想**：一个潜在的旗杆，必须像通过一系列安检门一样，通过每一个**非黑即白**的逻辑检查。只要在任何一个“门”前失败，它就会被立即丢弃，不再进行后续检查。这极大地提高了计算效率和决策的清晰度。

这个框架由四个“逻辑门”组成，按顺序进行检查：

1.  **火花门 (The Spark Gate)**: 识别潜在的起点。
2.  **结构门 (The Structure Gate)**: 检查形态的基本构成。
3.  **动力门 (The Momentum Gate)**: 验证能量的级别。
4.  **背景门 (The Context Gate)**: 确认其“出生环境”。

---

#### 门1：火花门 (The Spark Gate) - 识别触发K线

这是框架的入口，它的任务是从成千上万根K线中，快速筛选出极少数有潜力的“候选者”。

*   **规则**:
    *   `Rule 1.1 (幅度过滤器)`: **`IF`** `当前K线的真实波幅 (True Range) >= C_atr_mult * ATR(N_atr)`
    *   **`AND`**
    *   `Rule 1.2 (成交量过滤器)`: **`IF`** `当前K线的成交量 >= C_vol_mult * 过去N_vol周期的平均成交量`
*   **输出**: 只有同时满足这两个条件的K线，才能通过“火花门”，成为后续分析的“候选K线0号”。其他99.9%的K线都被直接过滤掉。

---

#### 门2：结构门 (The Structure Gate) - 检查形态构成

这个门检查由“候选K线0号”及其后续K线组成的序列是否具备旗杆的基本“长相”。

*   **规则 (对一个由 `n` 根K线组成的序列进行检查)**:
    *   `Rule 2.1 (长度过滤器)`: **`IF`** `n <= C_max_candles (e.g., 5)`
    *   **`AND`**
    *   `Rule 2.2 (实体过滤器)`: **`IF`** `序列中每一根K线的实体占比 (Body Ratio) >= C_body_ratio (e.g., 0.6)`
    *   **`AND`**
    *   `Rule 2.3 (排列过滤器)`: **`IF`** `序列中没有出现显著的K线重叠或反向回撤`
*   **输出**: 只有通过了所有结构检查的序列，才能进入下一个门。

---

#### 门3：动力门 (The Momentum Gate) - 验证能量级别

这个门量化评估整个序列的“力量感”，确保它不是虚有其表。

*   **规则 (对已通过结构门的 `n` 根K线序列进行检查)**:
    *   `Rule 3.1 (角度过滤器)`: **`IF`** `该序列的线性回归角度 >= C_angle_min (e.g., 55度)`
    *   **`AND`**
    *   `Rule 3.2 (持续成交量过滤器)`: **`IF`** `该序列的平均成交量 >= C_vol_mult * 序列出现前的平均成交量`
*   **输出**: 通过这个门，我们得到了一个在形态和能量上都合格的“候选旗杆”。

---

#### 门4：背景门 (The Context Gate) - 最终确认

这是最后一关，也是最重要的一关。它检查这个看似完美的旗杆，是否诞生于一个理想的“发射台”。

*   **规则 (对已通过动力门的候选旗杆进行检查)**:
    *   `Rule 4.1 (波动率压缩过滤器)`: **`IF`** `旗杆出现前N_lookback周期的平均ATR / 旗杆序列第一根K线的ATR <= C_context_ratio (e.g., 0.6)` (这表示旗杆是从一个低波动状态中爆发出来的)
*   **输出**: **只有通过了全部四个门的候选序列，才被最终认定为一个“有效的、可交易的旗杆”。**

### 框架优势

*   **清晰明确**: 没有分数，没有模糊地带。结果只有“是”或“否”，便于程序化执行和回测。
*   **计算高效**: 大量的无效K线在第一关就被过滤，避免了对它们进行复杂的角度和背景计算，节省了计算资源。
*   **逻辑严谨**: 严格遵循从“点”（火花）到“线”（结构）再到“体”（动力、背景）的分析流程，符合专业交易者的思维习惯。

### 伪代码示例 (Pseudo-Code Example)

```python
function identify_flagpole(candles):
    for i from 0 to len(candles):
        # --- GATE 1: The Spark Gate ---
        if is_spark_candle(candles[i]):
            # Found a potential start, now check sequences
            for n from 1 to C_max_candles:
                candidate = candles[i : i+n]

                # --- GATE 2: The Structure Gate ---
                if not passes_structure_gate(candidate):
                    continue # Try next length or next spark

                # --- GATE 3: The Momentum Gate ---
                if not passes_momentum_gate(candidate):
                    continue # Try next length or next spark

                # --- GATE 4: The Context Gate ---
                if not passes_context_gate(candidate, history_before_it):
                    continue # Try next length or next spark
                
                # --- SUCCESS! ---
                # If it reaches here, it has passed all gates.
                print(f"VALID FLAGPOLE IDENTIFIED at candle {i} with length {n}.")
                # Now you can trigger your trading logic
                # Break the inner loop as we found the best flagpole starting at i
                break 
```



**结论**：对于绝大多数系统化交易者而言，**序贯过滤框架**是在清晰度、可靠性和可实现性之间取得了最佳平衡的非评分方法。