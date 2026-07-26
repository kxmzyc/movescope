# 更新日志

## 0.4.0 - 2026-07-26

主题：「看得见的可解释性」。引擎已经算出的逐帧信息（偏差曲线、2D 骨架、容差与权重）第一次贯通到 API 契约与 Web 工作台。

### 破坏性变更

- 评分口径由「按 DTW 路径行」改为「按唯一测试帧」统计：一对多匹配（如动作停顿）不再重复计入 `anomaly_ratio` 与总分，逐帧偏差以对齐参考曲线（匹配参考帧的均值）为基准。同一视频的分数会与 0.3.0 有小幅漂移。
- `WeightedSegmentedDTWAligner` 的 1/std 特征权重新增极差钳制（`max_weight_ratio`，默认 20）：小样本模板中 std 趋近于零时，单特征不再以 99% 以上权重主导对齐与总分。这同样引入分数漂移。
- 含非有限值的特征列不再整体拒绝（0.3.0 返回 400「只能包含有限值」）：非核心特征列（如手腕出画导致的手肘角 NaN）自动从对齐、权重与评分中剔除并记入 `excluded_features`；双膝双髋核心特征缺失时仍拒绝，并给出点名关节的可读报错。
- 远程建议改为显式开启：`MOVESCOPE_ADVICE_PROVIDER=openai` 时才会把诊断数据发给 OpenAI（此前只要进程存在 `OPENAI_API_KEY` 就会外发且无法关闭）。默认 `rule` 本地规则，`off` 完全关闭。外发载荷剔除 `timeline`/`skeleton` 逐帧数组，只包含聚合诊断，避免长视频下提示词膨胀两个数量级并把逐帧原始数据外发。
- `/actions` 响应新增 `templates` 模板元数据列表（`n_videos`/`feature_dim`/`frames`）。

### 新增

- 响应新增 `timeline`：按测试帧对齐的逐帧关节角时间轴（实测角度、对齐专家角度、容差、逐帧越界标记，超长视频自动降采样），`/demo` 与 `/assess` 均返回。
- 响应新增 `skeleton`：降采样的 2D 骨架关键点、置信度与骨架连接表（`movescope.constants.SKELETON_EDGES` 唯一真源），不可靠关节输出 null。
- `per_feature_summary` 新增 `tolerance_deg` 与 `score_weight`，「为什么是这个分」可以从响应直接读出；新增 `advice_source` 标注建议实际来源（`rule`/`openai`，远程返回空内容而回退本地规则时同样如实标注 `rule`）。
- Web 工作台新增：逐帧关节角时间轴面板（专家容差走廊 + 越界红点 + 阶段边界线 + 点击跳转视频时刻）、视频上的 canvas 骨架叠加（按阶段高亮异常关节）、可点击的阶段时间轴（逐段评分着色）、偏差方向（角度偏大/偏小）、峰值角度、姿态质量（帧数/fps/有效姿态比/伪 3D 来源）与未参与评分关节提示。
- 新增环境变量：`MOVESCOPE_ADVICE_PROVIDER`、`MOVESCOPE_OPENAI_MODEL`、`MOVESCOPE_OPENAI_TIMEOUT_SEC`、`MOVESCOPE_ASSESS_TIMEOUT_SEC`、`MOVESCOPE_CORS_ORIGIN_REGEX`、`MOVESCOPE_MAX_CONCURRENT_ASSESS`。

### 修复

- Windows 上评估超时后删除临时文件抛 `PermissionError`，把预期的 504 变成 500 并泄漏文件；现在并发槽位与临时文件的释放挂在评估线程真正结束时，超时后既不再 500，文件也不再永久残留。
- 损坏视频/伪造扩展名的上传此前返回 500；现在返回可读的 400。
- `/assess` 增加并发上限（默认 2），占满返回 503，保护线程池。超时/断连被遗弃的评估线程结束前，其占用的槽位不会放行新请求，过载场景下上限依然有效。
- OpenAI 建议请求增加超时（默认 20 秒），失败回退本地规则时记录日志。
- Gradio 界面与 API 一致启用双膝双髋核心特征拒绝：此前核心关节未被可靠检测时会被静默剔除并可能给出误导性高分，现在拒绝评估并点名关节；文本摘要同时列出未参与评分的关节。
- 前端上传校验与服务端一致：以扩展名白名单为准，不再放行 MIME 是 video/* 但扩展名不支持的文件。
- API 测试对开发者 shell 的 `MOVESCOPE_*` 环境变量免疫：conftest 在模块导入期与每个测试前后两层隔离，导入时固化的 CORS 与并发配置同样被覆盖。

### 测试

- Python 测试从 69 项扩展到 92 项：新增分段多阶段直测、逐帧口径、timeline 结构与降采样、特征剔除、权重钳制、坏视频 400、超时 504、繁忙 503、超时后槽位保持占用、骨架通道、外发载荷剔除、空建议来源标注、Gradio 核心特征拒绝等覆盖。
- 前端 vitest 从 20 项扩展到 28 项：时间轴面板、阶段时间轴、骨架叠加、方向/质量/建议来源渲染。

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
