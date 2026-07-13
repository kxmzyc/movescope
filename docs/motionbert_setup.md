# MotionBERT Integration Status

MoveScope v0.2.0 does not implement a MotionBERT inference adapter.

The default and tested path uses MediaPipe world landmarks as pseudo-3D coordinates. `PoseExtractor.lift_to_3d()` intentionally raises `NotImplementedError` even if a checkpoint exists, because MotionBERT releases expose different model entry points and preprocessing contracts. A checkpoint alone does not enable true 3D lifting.

The reserved local path is:

```text
lib/MotionBERT/checkpoint/motionbert_lite.bin
```

`lib/MotionBERT/` is git-ignored. To complete this integration in a future release, the adapter must:

1. Pin an upstream MotionBERT commit and checkpoint.
2. Convert MoveScope's normalized 2D sequence to the upstream input contract.
3. Map the lifted output back to MoveScope's custom 17-joint order.
4. Add deterministic shape, finite-value, and real-video regression tests.
5. Document checkpoint provenance and licensing.

Until those requirements are met, public output and documentation must describe the active representation as MediaPipe world-landmark pseudo-3D.
