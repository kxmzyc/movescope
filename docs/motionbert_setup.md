# MotionBERT 集成状态

MoveScope v0.2.1 尚未实现 MotionBERT 推理适配器。

当前默认且经过测试的流程使用 MediaPipe world landmarks 作为伪三维坐标。即使本地存在检查点，`PoseExtractor.lift_to_3d()` 也会主动抛出 `NotImplementedError`。这是因为不同 MotionBERT 版本使用不同的模型入口和预处理约定，单独放置检查点并不能启用真实三维提升。

预留的本地路径为：

```text
lib/MotionBERT/checkpoint/motionbert_lite.bin
```

`lib/MotionBERT/` 默认不会提交到 Git。若要在后续版本完成集成，适配器必须满足以下要求：

1. 固定上游 MotionBERT 提交版本和检查点版本。
2. 将 MoveScope 归一化二维序列转换为上游模型要求的输入格式。
3. 将三维提升结果映射回 MoveScope 自定义 17 关节顺序。
4. 增加确定性的形状校验、有限值校验和真实视频回归测试。
5. 记录检查点来源与开源许可。

完成以上要求前，所有公开输出和文档都必须将当前表示描述为 MediaPipe world-landmark 伪三维坐标。
