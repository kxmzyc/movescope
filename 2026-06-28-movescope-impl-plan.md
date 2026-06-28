# MoveScope 实现计划 — Codex+GPT-5.5 执行手册

**目的：** 为每个阶段提供可直接粘贴给 Codex/GPT-5.5 执行的完整提示词，最大化自动化。
**执行方式：** VSCode 集成终端（Ctrl+`），Codex CLI 或 Claude Code CLI。
**人工干预原则：** 只在「👤 人工操作」标注处介入，其余全部由 agent 完成。

---

## 全局上下文（每次新对话先发这段给 Codex）

```
你是我的高级 AI 开发助手，正在构建 MoveScope——单目视频人体动作质量评估系统。
定位：双非本科毕设 + 北体大AI研究生复试作品集。
技术栈：Python 3.10, MediaPipe, MotionBERT-Lite, FastAPI, Gradio, numpy/scipy, yt-dlp
工作目录：[你的项目路径]
设计文档：docs/superpowers/specs/2026-06-28-movescope-design.md（实现前必读）
核心原则：
1. 只写完成任务最少的代码，不过度工程化
2. 容差值必须从数据统计推导，禁止硬编码魔法数字
3. 核心算法（DTW、特征提取）自实现，不调用第三方AQA库
4. 每个模块写完输出 smoke test 验证
5. 需要人工操作时，明确提示
```

---

## 阶段 0：环境搭建（约 1-2 小时，全自动）

### Step 0.1 — 项目骨架

🤖 **CODEX PROMPT:**
```
读取 docs/superpowers/specs/2026-06-28-movescope-design.md 中的目录结构章节。

创建完整项目目录结构：
  movescope/（Python包）、data/expert/squat/、data/test/squat/
  scripts/、api/、frontend/、notebooks/、tests/、docs/superpowers/specs/

创建以下文件：

1. requirements.txt，固定版本：
   mediapipe==0.10.14
   opencv-python==4.10.0.84
   numpy==1.26.4
   scipy==1.13.1
   fastapi==0.111.0
   uvicorn==0.30.1
   gradio==4.37.1
   yt-dlp==2024.7.9
   pytest==8.2.2
   torch==2.3.1

2. movescope/__init__.py — 加版本注释 __version__ = "0.1.0"
3. README.md — 项目名、一句话描述
4. .gitignore — Python 标准 + data/ 目录排除

运行：pip install -r requirements.txt
输出安装摘要（成功/失败的包）。
```

✅ **验证：** `python -c "import mediapipe; print(mediapipe.__version__)"` 输出版本号


### Step 0.2 — MotionBERT-Lite 权重

🤖 **CODEX PROMPT:**
```
执行以下操作安装 MotionBERT-Lite：

1. git clone https://github.com/Walter0807/MotionBERT.git lib/MotionBERT
2. 创建 lib/__init__.py，内容：
   import sys, os
   sys.path.insert(0, os.path.join(os.path.dirname(__file__), "MotionBERT"))
3. 查看 lib/MotionBERT/README.md，找到预训练权重下载地址
4. 若为 Google Drive 链接，打印出来让用户手动下载
   目标路径：lib/MotionBERT/checkpoint/motionbert_lite.bin
5. 验证命令（权重存在时）：
   python -c "from lib import *; print('MotionBERT path ok')"

若自动下载失败，输出清晰的手动下载步骤。
```

👤 **人工操作（若需要）：** 手动下载权重到 `lib/MotionBERT/checkpoint/motionbert_lite.bin`

---

### Step 0.3 — yt-dlp-mcp 配置

🤖 **CODEX PROMPT:**
```
配置 yt-dlp-mcp 数据采集工具：

1. 验证 Node.js：node --version（若未安装输出安装链接）
2. 验证 yt-dlp：yt-dlp --version
3. 创建 scripts/fetch_videos.py，功能：
   参数：--action（动作名）--mode expert/test --n 数量 --lang zh/en/both --dry-run
   
   内置深蹲搜索词模板：
     zh expert: ["标准深蹲教学", "深蹲正确姿势示范", "深蹲标准动作"]
     en expert: ["perfect squat form tutorial", "squat technique guide", "proper squat depth"]
     zh test:   ["健身深蹲", "深蹲训练", "深蹲错误示例"]
   
   调用 yt-dlp（subprocess）：
     搜索：yt-dlp "ytsearch{n}:{keyword}" --print title --print webpage_url（dry-run）
     下载：yt-dlp --max-downloads {n} --match-filter "duration>15 & duration<120"
            --format "best[height>=480]" -o "data/{mode}/{action}/%(id)s.%(ext)s"
   下载失败记录到 data/download_errors.log，不中断
   完成输出成功/失败统计

4. 测试：python scripts/fetch_videos.py --action squat --mode expert --n 3 --lang zh --dry-run
   输出至少3个视频标题和URL
```

✅ **验证：** dry-run 列出视频标题，时长在 15-120 秒范围内

---

### Step 0.4 — Hello World 端到端验证

🤖 **CODEX PROMPT:**
```
创建 scripts/hello_world.py，端到端验证环境：

参数：--video（视频文件路径，可选，无则尝试摄像头0）

流程：
1. 读取视频（cv2.VideoCapture）
2. 对每帧跑 MediaPipe Pose（static_image_mode=False, model_complexity=1）
3. 在帧上绘制骨架（mp.solutions.drawing_utils.draw_landmarks）
4. 计算左膝关节角度（landmark 23-25-27，向量点积），转换为度数
5. 帧上叠加文字显示膝关节角度
6. 保存输出视频到 data/test/hello_world_output.mp4
7. 打印：处理帧数、平均膝关节角度、最小/最大角度、耗时

运行并展示结果。
```

👤 **人工操作：** 用手机拍 5-10 秒自己做深蹲的视频，传到电脑，提供路径。

✅ **验证：** 输出视频有骨架叠加；膝关节角度在 90-180 度之间变化


---

## 阶段 1：核心引擎 MVP（3-4 周，保底必完成）

### Step 1.1 — L1 姿态提取模块

🤖 **CODEX PROMPT:**
```
读取设计文档"5.2 L1 姿态提取"章节，实现以下文件：

movescope/pose_extractor.py — class PoseExtractor:

  MEDIAPIPE_TO_COCO 常量：33点映射到17关节的索引字典
  关节名称列表 JOINT_NAMES（17个，COCO格式）：
    pelvis, left_hip, right_hip, left_knee, right_knee,
    left_ankle, right_ankle, left_shoulder, right_shoulder,
    left_elbow, right_elbow, left_wrist, right_wrist,
    head, neck, left_eye, right_eye

  方法 extract(video_path: str) -> dict，返回：
    fps: float
    n_frames: int
    joint_names: List[str]
    coords_2d: ndarray shape (T, 17, 2) 归一化到 [0,1]
    confidence: ndarray shape (T, 17)
    coords_3d_pseudo: ndarray shape (T, 17, 3)

  置信度 < 0.3 的帧用相邻帧线性插值填补（前后各找最近高置信度帧）
  若整段视频置信度都低，输出警告

scripts/extract_pose.py — CLI入口：
  python scripts/extract_pose.py --video path --output path.npz
  保存为 np.savez_compressed
  输出：处理帧数、跳过帧数、耗时

tests/test_pose_extractor.py：
  fixture: 用 cv2 生成 10 帧 640x480 纯色视频（白底，中央画黑色人形轮廓）
  test_output_shape: 验证输出 coords_2d shape == (10, 17, 2)
  test_confidence_interpolation: 注入低置信度帧，验证填补逻辑

运行 pytest tests/test_pose_extractor.py，输出结果。
```

✅ **验证：** pytest 通过；对真实深蹲视频跑 extract_pose.py，.npz 文件不为空

---

### Step 1.2 — MotionBERT 3D Lifting 集成

🤖 **CODEX PROMPT:**
```
在 movescope/pose_extractor.py 中添加 3D lifting 功能：

新方法 lift_to_3d(coords_2d: ndarray shape (T,17,2), fps: float) -> ndarray shape (T,17,3)：
  将序列切成 243 帧滑窗（stride=243，不重叠，最后一窗补零填充）
  用 torch.no_grad() + CPU device 运行 MotionBERT-Lite
  拼接各窗结果，截取到原始长度 T
  输出：3D 坐标，单位米，以骨盆为原点

修改 extract() 方法：
  若 lib/MotionBERT/checkpoint/motionbert_lite.bin 存在，
  自动运行 lift_to_3d 并在返回 dict 中添加 coords_3d: ndarray (T,17,3)
  若权重不存在，打印 WARNING，coords_3d 字段置 None，不崩溃

tests/test_pose_extractor.py 新增：
  test_lift_shape: 验证 lift_to_3d 输出 shape (T,17,3)
  test_pelvis_near_origin: 验证骨盆关节 coords_3d[:,0,:] 均值接近 [0,0,0]，误差 < 0.1m

运行完整 pytest，输出结果。
```

✅ **验证：** 处理 10 秒真实视频，coords_3d 有值且 shape 正确；CPU 耗时 < 60 秒

---

### Step 1.3 — L2 视角鲁棒特征提取

🤖 **CODEX PROMPT:**
```
读取设计文档"5.3 L2 特征提取"章节，实现 movescope/features.py — class FeatureExtractor：

JOINT_TRIPLETS 常量，12 个三联组（parent-joint-child），深蹲优先：
  ("left_hip","left_knee","left_ankle"),
  ("right_hip","right_knee","right_ankle"),
  ("left_shoulder","left_hip","left_knee"),
  ("right_shoulder","right_hip","right_knee"),
  ("left_knee","left_hip","right_hip"),
  ("right_knee","right_hip","left_hip"),
  ("left_shoulder","left_elbow","left_wrist"),
  ("right_shoulder","right_elbow","right_wrist"),
  ("neck","left_shoulder","left_elbow"),
  ("neck","right_shoulder","right_elbow"),
  ("pelvis","left_hip","left_knee"),
  ("pelvis","right_hip","right_knee")

compute_angles(coords_3d: ndarray (T,17,3)) -> ndarray (T,12)：
  对每个三联组 (A,B,C)，计算向量 BA 和 BC 的夹角：
    v1 = A - B, v2 = C - B
    cos_angle = dot(v1,v2) / (norm(v1) * norm(v2))
    cos_angle = clip(cos_angle, -1.0, 1.0)  # 防 arccos 域错误
    angle_deg = arccos(cos_angle) * 180 / pi
  返回度数

normalize(angle_seq: ndarray (T,12)) -> ndarray (T,12)：
  逐维度 z-score，std 为 0 时设为 1.0

extract(coords_3d: ndarray) -> ndarray (T,12)：
  调用 compute_angles 再 normalize

tests/test_features.py：
  test_known_angle: 构造已知夹角的三点（如90度），验证误差 < 1度
  test_normalize_stats: 归一化后均值 < 0.01, std 接近 1.0

运行 pytest tests/test_features.py。
```

✅ **验证：** pytest 通过；打印真实深蹲视频的 12 列角度均值，膝关节列应在 100-160 度


---

### Step 1.4 — 专家模板构建

👤 **人工前置（先做这步，再让 Codex 跑 1.4）：**
运行 Step 0.3 下载 10-15 段专家深蹲视频：
```
python scripts/fetch_videos.py --action squat --mode expert --n 15 --lang both
```
目视检查几个视频，确认是正面或侧面的标准深蹲示范，删除明显不合格的。

🤖 **CODEX PROMPT:**
```
读取设计文档"5.4 Step 1 容差矩阵构建"，实现以下文件：

movescope/template.py — class ActionTemplate：

  构造函数：action_name: str
  
  build(expert_dir: str, pose_extractor, feature_extractor) -> None：
    遍历 expert_dir 下所有视频文件（.mp4 .avi .mov .webm）
    对每个视频：extract -> lift_to_3d -> feature_extractor.extract -> 得到 (T,12)
    若 coords_3d 为 None（无权重），改用 coords_3d_pseudo
    对每个视频计算时域均值向量 (12,)
    跨视频统计：mean_vec (12,), std_vec (12,)
    tolerance = std_vec * 1.5（DEFAULT_K=1.5 作为类常量）
    同时保存一段"代表序列"：选与 mean_vec 距离最近的那段视频的特征序列
    保存到 data/templates/{action_name}.npz：
      mean, std, tolerance, representative_seq, action_name, n_videos

  load(action_name: str) -> ActionTemplate（类方法）：
    从 data/templates/{action_name}.npz 加载，返回实例

scripts/build_template.py — CLI：
  python scripts/build_template.py --action squat --expert-dir data/expert/squat/
  完成后用 tabulate 或手动格式化打印每个关节角度的 mean +/- tolerance 表格

tests/test_template.py：
  用3段 mock 特征序列（random ndarray），验证 build() 后 tolerance.shape == (12,) 且全>0

运行 pytest tests/test_template.py，然后运行：
python scripts/build_template.py --action squat --expert-dir data/expert/squat/
打印结果。
```

✅ **验证：** data/templates/squat.npz 存在；表格中膝关节角度均值在 100-155 度之间

---

### Step 1.5 — L3 基线 DTW 对齐与评估

🤖 **CODEX PROMPT:**
```
读取设计文档"5.4 L3 核心算法"章节，实现基线版（标准 DTW，阶段2再升级）：

movescope/alignment.py — class DTWAligner：

  align(query: ndarray (T,D), reference: ndarray (R,D)) -> List[Tuple[int,int]]：
    用 numpy 自实现标准 DTW 动态规划（不调用任何 dtw 第三方库）：
      cost[i,j] = euclidean_distance(query[i], reference[j])
      dp[i,j] = cost[i,j] + min(dp[i-1,j], dp[i,j-1], dp[i-1,j-1])
      边界：dp[0,0]=cost[0,0]，第0行和第0列单向累加
    回溯最优路径（从 dp[T-1,R-1] 逆向贪心），返回 [(i,j),...] 对应关系列表

movescope/assessment.py — class AssessmentEngine：

  构造函数：template: ActionTemplate, aligner: DTWAligner, feature_extractor: FeatureExtractor

  assess(test_coords_3d: ndarray) -> dict：
    1. feature_extractor.extract(test_coords_3d) -> test_features (T,12)
    2. aligner.align(test_features, template.representative_seq) -> path
    3. 沿路径，对每对 (i_test, j_ref)：
         deviation[k] = abs(test_features[i,k] - template.representative_seq[j,k])
         若 deviation[k] > template.tolerance[k]：标记为异常
    4. per_joint_anomaly_rate: ndarray (12,) 各关节异常帧比率
    5. total_score = clamp(100 - mean(per_joint_anomaly_rate)*100, 0, 100)
    6. anomaly_events: List[dict] 连续异常帧合并为事件，含起止时间（用fps换算）和关节索引
    7. 返回完整 dict（见设计文档 diagnosis.json 结构）

tests/test_assessment.py：
  test_perfect_match：query == reference_seq，score == 100
  test_all_wrong：query = reference_seq + large_offset(超过所有tolerance)，score < 50

运行 pytest tests/test_assessment.py，输出结果。
```

✅ **验证：** pytest 通过；对一段"正确深蹲"和一段"故意膝盖内扣"视频各评估，score 差距 > 15 分

---

### Step 1.6 — Gradio MVP Demo

🤖 **CODEX PROMPT:**
```
实现 frontend/gradio_app.py，创建最小可用 Gradio demo：

界面（gr.Blocks）：
  - 左列：gr.Video(label="上传待测视频") 上传组件
  - 中列：gr.Video(label="骨架可视化") 展示处理后视频
  - 右列：
      gr.Number(label="总分") 
      gr.BarPlot(label="各关节偏差") 或 gr.Plot
      gr.Textbox(label="诊断摘要", lines=8)

处理函数（绑定上传事件）：
  1. PoseExtractor().extract(video_path) -> coords_3d
  2. ActionTemplate.load("squat")（若不存在，返回错误提示字符串，不崩溃）
  3. DTWAligner + AssessmentEngine().assess(coords_3d) -> result
  4. 用 cv2 在原视频帧上叠加骨架：
       异常关节画红色圆点（radius=8）
       正常关节画绿色圆点（radius=5）
       帧右上角显示实时分数
  5. 保存叠加视频到 data/test/viz_output.mp4
  6. 诊断摘要文字：列出偏差最大的前3个关节名称和平均偏差度数
  
启动：
  if __name__ == "__main__":
      demo.launch(server_port=7860, share=False)

在 README.md 添加 Quick Start 章节：
  1. 安装依赖：pip install -r requirements.txt
  2. 下载专家视频并构建模板：...
  3. 启动 demo：python frontend/gradio_app.py
  4. 浏览器访问 http://localhost:7860
```

✅ **验证：** 浏览器打开 localhost:7860，上传深蹲视频，能看到骨架叠加视频和分数输出


---

## 阶段 2：创新点实现 + 实验（3-4 周，保底必完成）

### Step 2.1 — 关节加权分段 DTW（升级创新点③）

🤖 **CODEX PROMPT:**
```
升级 movescope/alignment.py，实现加权分段 DTW（在标准 DTW 基础上改进）：

新增 class WeightedSegmentedDTWAligner（继承 DTWAligner）：

compute_joint_weights(template: ActionTemplate) -> ndarray (12,)：
  权重 = 1.0 / (template.std + 1e-6)
  归一化使 sum == 1.0
  方差大的关节权重低（本来就变化多），方差小的权重高（专家都保持一致）

detect_phases(feature_seq: ndarray (T,12), n_phases: int=4) -> List[Tuple[int,int]]：
  用 KMeans（sklearn.cluster.KMeans, n_clusters=n_phases）对帧级特征聚类
  得到每帧的阶段标签序列
  用 run-length encoding 合并连续相同标签为阶段，返回 [(start,end), ...]
  过滤掉时长 < 3 帧的阶段（噪声）

weighted_dtw(query: ndarray (T,D), ref: ndarray (R,D), weights: ndarray (D,)) -> List[Tuple[int,int]]：
  将标准 DTW 的欧氏距离替换为加权距离：
    dist(q,r) = sqrt( sum( weights * (q-r)**2 ) )
  其余 DP 逻辑与标准 DTW 相同

align(query, reference, weights=None, use_segmented=True) -> List[Tuple[int,int]]：
  若 weights 为 None，均匀权重
  若 use_segmented=True：
    检测 reference 中的阶段边界
    检测 query 中的阶段边界（允许阶段数不同，按序匹配）
    各阶段独立调用 weighted_dtw，拼接路径

在 tests/test_alignment.py 中：
  test_weighted_vs_standard：构造一个 query 在某关节有大偏差，
    验证加权后该关节高权重版本的总路径代价 < 标准版本（或至少差别显著）
  test_segmented_path_continuous：验证分段路径是单调递增的

运行 pytest tests/test_alignment.py。
```

✅ **验证：** pytest 通过；用新对齐器替换 Gradio 中的对齐器后，对比评分变化（更精确）

---

### Step 2.2 — 细粒度诊断报告（创新点①）

🤖 **CODEX PROMPT:**
```
升级 movescope/assessment.py，输出完整细粒度诊断报告：

升级 assess() 返回格式（diagnosis.json 结构）：

{
  "action": "squat",
  "total_score": 78.5,
  "phases": [
    {
      "name": "phase_0",            // 后续可配置为"下蹲阶段"等
      "time_range": [0.0, 1.2],
      "phase_score": 72.0,
      "anomalies": [
        {
          "joint_name": "left_knee",
          "joint_idx": 3,
          "direction": "positive",  // 正偏或负偏（相对模板均值）
          "mean_deviation_deg": 11.2,
          "peak_deviation_deg": 18.5,
          "peak_time_sec": 0.8,
          "anomaly_ratio": 0.65     // 该阶段有多少比例帧超过容差
        }
      ]
    }
  ],
  "per_joint_summary": {
    "left_knee": {"mean_dev": 11.2, "anomaly_ratio": 0.4},
    ...
  }
}

新增 save_diagnosis(result: dict, output_path: str)：保存为 JSON 文件

新增 generate_text_summary(result: dict) -> str：
  生成简洁文字摘要，格式：
    总分：78.5/100
    主要问题（按偏差排序前3）：
    1. [0.0-1.2s] 左膝 平均偏差 11.2度
    2. ...
  不调用任何外部API，纯字符串拼接

tests/test_assessment.py 新增：
  test_diagnosis_schema：验证返回 dict 包含所有必需字段
  test_phase_scores_sum：各阶段 phase_score 加权平均应接近 total_score

运行完整 pytest，输出结果。
```

✅ **验证：** 对一段错误深蹲评估，diagnosis.json 中能看到明确的关节异常记录

---

### Step 2.3 — GPT-5.5 自然语言纠错建议

🤖 **CODEX PROMPT:**
```
实现 movescope/llm_advisor.py — class LLMAdvisor：

generate_advice(diagnosis: dict) -> str：
  构造 prompt 并调用 OpenAI API（model="gpt-4o" 或用户配置的模型）：
  
  system: "你是运动姿态分析助手。根据关节偏差诊断数据给出简洁纠错建议。
           建议要口语化、可操作，不做医疗建议，最后加免责声明：本分析仅供参考，非医疗诊断。"
  
  user: f"以下是动作质量评估结果：{json.dumps(diagnosis, ensure_ascii=False)}
         请针对每个标注的异常关节给出1-2句纠错建议。"

  若 OPENAI_API_KEY 环境变量不存在，返回 fallback 文字建议：
    根据 diagnosis 中的关节名称查预置的建议字典：
      "left_knee" / "right_knee" 膝内扣：建议膝盖对准脚尖方向
      "left_hip" / "right_hip" 髋：建议保持髋部稳定
      以此类推（覆盖主要12个关节的通用建议）
  确保无 API key 时仍能产出有意义的文字

在 gradio_app.py 中集成 LLMAdvisor，将建议显示在界面的"纠错建议"文本框中

tests/test_llm_advisor.py：
  test_fallback_no_key：不设环境变量，调用 generate_advice 应返回非空字符串
  test_output_not_medical：输出字符串不含"诊断""治疗""病"等词语（简单关键词检查）
```

✅ **验证：** 无 API key 时仍能输出纠错建议文字；有 API key 时输出更自然的建议

---

### Step 2.4 — 消融实验（论文必需）

🤖 **CODEX PROMPT:**
```
创建 notebooks/ablation_experiment.ipynb，实现完整消融实验：

实验设置：
  test_good_dir = "data/test/good_squat/"    // 动作规范视频
  test_bad_dir  = "data/test/bad_squat/"     // 故意有错误的视频
  template = ActionTemplate.load("squat")

四个对比变体：
  baseline_2d: PoseExtractor (2D coords_2d, 不用3D) + 标准DTW + AssessmentEngine
  ours_3d:     PoseExtractor (coords_3d) + 标准DTW + AssessmentEngine
  ours_weighted: coords_3d + 加权DTW（无分段）+ AssessmentEngine
  ours_full:   coords_3d + 加权分段DTW + AssessmentEngine（本文方法）

对 test_good + test_bad 各视频分别用四个变体评估，收集 total_score

绘制：
  1. 箱线图（每个变体两组的分数分布）
  2. 表格：各变体区分 good/bad 的平均分差（越大越好）
  3. t 检验：各变体 good vs bad 的 p 值

保存图表到 docs/figures/ablation_result.png

notebook 最后一格打印结论字符串：
  "ours_full 相较 baseline_2d，good/bad 分差提升 X 分，p 值 Y"
```

👤 **人工前置：**
1. 运行 fetch_videos.py 下载 20-30 段"正确深蹲"到 data/test/good_squat/
2. 下载或自录 20-30 段"错误深蹲"（搜索"深蹲错误示例"）到 data/test/bad_squat/
3. 对每段视频做二分类标注（创建 data/test/labels.json，格式 {"filename": "good"/"bad"}）

✅ **验证：** notebook 能跑完；ours_full 的 good/bad 分差 >= baseline_2d


### Step 2.5 — 视角鲁棒性实验（创新点④验证）

🤖 **CODEX PROMPT:**
```
创建 notebooks/viewpoint_robustness.ipynb：

实验目标：验证 3D 特征比 2D 特征在不同拍摄角度下评分更一致

数据：使用 data/test/multiview/ 目录下同一段深蹲的多角度录像
（正面、侧面、45度斜前方各一段，由用户自录）

实验流程：
  angles = ["front", "side", "diagonal_45"]
  for each video in multiview/:
    score_2d = evaluate with baseline_2d variant
    score_3d = evaluate with ours_full variant

  计算两种方法在三个角度下的分数标准差（标准差越小=越一致）
  绘制折线图：3个角度 x 2条线（2D vs 3D）
  计算一致性提升比例：(std_2d - std_3d) / std_2d * 100%

保存图表到 docs/figures/viewpoint_robustness.png
打印结论字符串
```

👤 **人工操作：** 用手机在正面、侧面、45度斜前方各录一段自己做深蹲的视频，放到 data/test/multiview/

✅ **验证：** notebook 跑完，3D 方法的分数标准差 < 2D 方法（预期结果）

---

### Step 2.6 — FastAPI 后端

🤖 **CODEX PROMPT:**
```
实现 api/main.py — FastAPI 后端：

端点 POST /assess：
  接受 multipart form：
    video: UploadFile（视频文件）
    action: str = "squat"（动作类型，默认深蹲）
  
  处理流程：
    1. 保存上传文件到临时路径 /tmp/movescope_upload_{uuid}.mp4
    2. PoseExtractor().extract(tmp_path)
    3. ActionTemplate.load(action)（若不存在，返回 422 错误和友好提示）
    4. WeightedSegmentedDTWAligner + AssessmentEngine().assess()
    5. LLMAdvisor().generate_advice(diagnosis)
    6. 清理临时文件
    7. 返回完整 diagnosis dict（含 llm_advice 字段）

  错误处理：
    视频损坏/无法读取：返回 400
    模板不存在：返回 422 with 提示"请先运行 build_template.py"
    处理超时（>5分钟）：返回 504

端点 GET /health：返回 {"status":"ok","version":"0.1.0"}

端点 GET /actions：返回已有模板的动作列表
  扫描 data/templates/*.npz，返回文件名列表

启动：python -m uvicorn api.main:app --port 8000 --reload

tests/test_api.py：
  test_health：GET /health 返回 200
  test_assess_no_template：POST /assess with valid video but no template file，返回 422
  （用 httpx.AsyncClient 或 TestClient）

运行 pytest tests/test_api.py。
```

✅ **验证：** pytest 通过；用 curl 发一段真实视频，返回 JSON 包含 total_score

---

## 阶段 3：完善 + 论文写作（弹性，考研后再推进）

### Step 3.1 — React 完整前端

🤖 **CODEX PROMPT:**
```
创建 React + Tailwind 前端（替换 Gradio，作为作品集展示版）：
工作目录：frontend/web/

初始化：npx create-react-app . --template typescript（在 frontend/web/ 内运行）
安装：npm install tailwindcss @headlessui/react recharts axios

组件结构：
  App.tsx — 主页面
  components/
    VideoUploader.tsx — 拖拽上传 + 进度条
    SkeletonPlayer.tsx — 用 Canvas API 在视频帧上绘制骨架（异常关节红色）
    ScoreDashboard.tsx — 总分仪表盘（圆形进度条，颜色区分红/黄/绿）
    JointDeviationChart.tsx — 各关节偏差条形图（用 recharts BarChart）
    TimelineAnnotation.tsx — 时间轴，标注异常时间段
    DiagnosisReport.tsx — 结构化诊断文字报告 + LLM 建议

API 调用（axios）：
  POST http://localhost:8000/assess（multipart 上传）
  轮询或 SSE 获取进度（若后端支持）

启动：npm start（端口3000，访问 localhost:3000）

先实现 VideoUploader + ScoreDashboard + DiagnosisReport 这三个组件，
其余作为后续迭代。
```

---

### Step 3.2 — 数据集扩充与二次实验

🤖 **CODEX PROMPT:**
```
扩充数据集，提升实验可信度：

1. 运行 fetch_videos.py 将 test_good/test_bad 各扩充到 50 段以上
2. 重新运行 notebooks/ablation_experiment.ipynb，更新图表和结论数字
3. 新增单模板可行性实验（notebooks/template_sensitivity.ipynb）：
   用 1 段 / 3 段 / 5 段 / 10 段专家视频构建模板，
   对比评估结果稳定性（对同一组测试视频，分数的标准差）
   绘制折线图：模板数量 vs 评分稳定性

保存所有图表到 docs/figures/
```

---

### Step 3.3 — GitHub 开源发布

🤖 **CODEX PROMPT:**
```
准备开源发布：

1. 更新 README.md，包含：
   - 项目标题 + 一句话描述（英文）
   - Demo GIF（从 Gradio 或 React 录屏截取，保存为 docs/demo.gif）
   - 功能列表（创新点简述，英文）
   - Installation 章节（Windows + macOS + Linux 三平台命令）
   - Quick Start 章节
   - Architecture 章节（5层流水线文字描述）
   - Citation 章节（毕设论文 BibTeX 占位符）
   - License：Apache 2.0

2. 创建 LICENSE 文件（Apache 2.0 全文）

3. 在 CITATION.md 写论文引用格式占位符：
   @article{zheng2026movescope,
     title={MoveScope: Template-Free, Interpretable Action Quality Assessment via Monocular 3D Pose},
     author={Zheng, Yuchen},
     year={2026}
   }

4. 创建 .github/workflows/ci.yml — GitHub Actions：
   触发条件：push to main
   步骤：pip install、pytest 全量测试
   badge 贴到 README

git add . && git commit -m "feat: initial open-source release v0.1.0"
（不要 push，等用户确认后再 push）
```

---

### Step 3.4 — 毕设论文辅助写作

🤖 **GPT-5.5 PROMPT（论文章节辅助，非 Codex）：**
```
你是学术写作助手。帮我为以下毕业论文章节生成详细草稿，论文题目：
《基于单目视频的免标注可解释人体动作质量评估研究》

请依次生成以下章节，每章节不少于800字，使用学术写作风格：

第一章 绪论：
  1.1 研究背景（运动健康、体育AI的应用价值）
  1.2 研究现状（AQA领域发展，列举MTL-AQA、CoRe、FineDiving、FineParser等工作）
  1.3 现有方法的局限性（标注依赖、黑盒输出、视角敏感）
  1.4 本文研究内容与贡献（四个创新点）
  1.5 论文组织结构

第二章 相关工作：
  2.1 人体姿态估计（2D: MediaPipe/RTMPose；3D: MotionBERT/VideoPose3D）
  2.2 动作质量评估（AQA）综述（按时间线梳理）
  2.3 时序对齐方法（DTW及其变体）
  2.4 可解释AI（XAI）在体育分析中的应用

注意：所有引用格式用 [作者，年份] 占位，我会后续核实具体文献。
```

👤 **你的责任：** 认真阅读生成的草稿，理解每个论点，用自己的话重写关键段落，确保答辩时能流利讲解。这部分不能完全外包给 AI。

---

## 项目全局 Agent 自动化快捷命令

以下命令可以随时在 VSCode 终端直接运行，Codex 会自动处理：

```bash
# 采集数据（expert 示范视频）
python scripts/fetch_videos.py --action squat --mode expert --n 20 --lang both

# 构建/更新专家模板
python scripts/build_template.py --action squat --expert-dir data/expert/squat/

# 评估单个视频
python -c "
from movescope.pose_extractor import PoseExtractor
from movescope.features import FeatureExtractor
from movescope.template import ActionTemplate
from movescope.alignment import WeightedSegmentedDTWAligner
from movescope.assessment import AssessmentEngine
import json

pe = PoseExtractor()
fe = FeatureExtractor()
tmpl = ActionTemplate.load('squat')
engine = AssessmentEngine(tmpl, WeightedSegmentedDTWAligner(), fe)

result = pe.extract('data/test/my_squat.mp4')
coords3d = result.get('coords_3d') or result['coords_3d_pseudo']
diagnosis = engine.assess(coords3d)
print(json.dumps(diagnosis, ensure_ascii=False, indent=2))
"

# 跑全量测试
pytest tests/ -v

# 启动 Gradio demo
python frontend/gradio_app.py

# 启动 API 后端
python -m uvicorn api.main:app --port 8000 --reload

# 运行消融实验
jupyter nbconvert --to notebook --execute notebooks/ablation_experiment.ipynb
```

---

## 阶段完成检查清单

### 保底交付（考研前必须完成）

- [ ] 阶段 0：环境跑通，Hello World 输出骨架视频
- [ ] 阶段 1.1-1.3：姿态提取 + 特征提取 + pytest 全过
- [ ] 阶段 1.4：专家模板构建（需先下载视频）
- [ ] 阶段 1.5：基线 DTW 评估，能区分好/坏动作
- [ ] 阶段 1.6：Gradio demo 可在 localhost 演示
- [ ] 阶段 2.1-2.2：加权分段 DTW + 细粒度诊断
- [ ] 阶段 2.4：消融实验 notebook 有结果图表

### 作品集目标（考研后推进）

- [ ] 阶段 2.3：GPT 建议接入
- [ ] 阶段 2.5：视角鲁棒实验
- [ ] 阶段 3.1：React 前端
- [ ] 阶段 3.3：GitHub 开源发布
- [ ] 阶段 3.4：毕设论文草稿

---

*计划文档结束。将此文档和 2026-06-28-movescope-design.md 一起作为 Codex 每次会话的上下文参考。*
