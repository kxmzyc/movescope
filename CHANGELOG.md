# 更新日志

## 0.3.0 - 2026-07-26

全栈结构重构。核心数值行为（DTW 路径、评分公式）与 0.2.1 逐点一致，但 API 响应 schema 有破坏性变更。

### 破坏性变更

- `/assess` 与 `/demo` 响应：异常项由拼接字符串 `joint_name`（如 `left_knee:left_hip-left_ankle`）改为结构化字段 `feature_index`/`joint`/`joint_display`/`parent`/`child`；`per_joint_summary` 字典改为 `per_feature_summary` 列表；新增顶层 `segmented` 布尔字段。
- `phases` 不再是写死的单个 `phase_0`：分段对齐成功时输出真实检测到的多个阶段（逐段评分与异常）；回退到全序列对齐时输出单一阶段并标记 `segmented: false`。
- `/health` 新增 `max_upload_bytes` 与 `allowed_extensions`，前端上传校验从此以服务端下发为准。
- `AssessmentEngine.assess()` 拆分为 `assess_features()` / `assess_coords()` / `assess_pose()` 三个入口，`feature_extractor` 参数变为可选；`movescope.llm_advisor.LLMAdvisor` 拆分为 `movescope.advice.RuleBasedAdvisor` 与 `OpenAIAdvisor`；文本摘要与 JSON 落盘移至 `movescope.reporting`。

### 性能

- DTW 代价矩阵与动态规划向量化：300×280 帧加权分段对齐从约 430 ms 降至约 23 ms（约 19 倍），路径与旧实现逐点一致（由随机对拍测试锁定）。

### 结构与工具链

- 新增 `pyproject.toml` 打包：`pip install -e .` 取代 7 处手写 `sys.path` 注入；CLI 注册为 `movescope-*` 命令，`scripts/*.py` 保留为薄转发壳。
- `movescope/alignment.py` 拆分为 `alignment/{dtw,segmentation,aligners}` 子包；API 层拆分为 `api/{main,routes,schemas,services,settings,errors}`，全部端点带 Pydantic 响应模型。
- React 前端由 429 行单组件拆分为 API 客户端、两个 hook 与四个面板组件；修复视频预览 object URL 在组件卸载时的泄漏；新增 20 项 vitest 测试。
- CI 增加 ruff、mypy 与前端 vitest 检查；Python 测试扩展到 69 项（含 25 项 DTW 新旧实现对拍）。
- 新增 `MOVESCOPE_DATA_DIR` 环境变量；评分权重来源显式化为 `score_weights="aligner"|"uniform"`（默认与历史行为一致）。

## 0.2.1 - 2026-07-13

- 将 React、Gradio、FastAPI 错误提示和命令行输出统一为中文。
- 为动作名称、17 个关节、图表字段和合成验证状态增加中文显示映射，同时保持 API 字段兼容。
- 将 README、引用说明、MotionBERT 状态说明与实验 notebook 中文化。
- 修正中文终端输出编码，并更新中文界面截图。

## 0.2.0 - 2026-07-13

- 拒绝非有限特征和退化骨段，避免返回误导性评分。
- 为单样本或低方差专家模板增加 5 度容差下限。
- 分段数量不一致时回退到完整加权对齐，确保 DTW 路径覆盖完整序列。
- 校验特征权重、动作标识、上传扩展名、空文件、文件大小和姿态检测覆盖率。
- 增加本地开发 CORS 配置与确定性合成 `/demo` 评估。
- 为 React 工作台增加模板发现、合成验证、详细问题展示和 JSON 导出。
- 将 Python 回归测试扩展到 40 项。
- 修正 MotionBERT、基准结果、引用方式和数据来源说明。

## 0.1.0 - 2026-06-28

- 首次发布原型，提供 MediaPipe 姿态提取、角度特征、模板评分、DTW 对齐、FastAPI、Gradio 和 React 入口。
