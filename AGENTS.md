## C++ code style

- Prefer compact C++ formatting.
- Do not split expressions, function calls, or parameter lists when they fit within 140 characters.
- Keep short conditions and Eigen expressions on one line.
- Use 4 spaces for indentation.
- 每个逻辑环节之间留一行空行，例如初始化、梯度计算、控制点更新和结果检查之间；同一环节内部保持紧凑。环节有说明注释时，空行放在该注释之前，注释与对应代码紧邻。

## 必要注释

- 新增或修改代码时，补充必要的中文注释，说明关键参数、结构体成员和主要计算步骤的含义。
- 涉及尺寸、坐标或切片时，明确单位、全局/局部坐标、像素索引/分块编号及区间是否包含端点。
- 函数接口应说明输入、输出和用途；注释解释含义和原因，避免逐行复述显而易见的代码。
