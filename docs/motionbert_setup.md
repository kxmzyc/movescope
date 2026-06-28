# MotionBERT Setup

MoveScope can run without MotionBERT weights by falling back to MediaPipe world landmarks as pseudo-3D coordinates. To complete the optional MotionBERT-Lite path from the implementation plan, install the upstream repository and checkpoint manually:

```bash
git clone --depth 1 https://github.com/Walter0807/MotionBERT.git lib/MotionBERT
python -c "from lib import *; print('MotionBERT path ok')"
```

Then download the pretrained MotionBERT checkpoint referenced by the upstream project page or README and place it at:

```text
lib/MotionBERT/checkpoint/motionbert_lite.bin
```

Local attempts to clone the repository can fail on restricted networks. In that case, download the repository archive in a browser, extract it to `lib/MotionBERT`, then place the checkpoint at the path above.

After the checkpoint exists, `PoseExtractor.extract()` will expose the `coords_3d` field for the true lifting backend. Until then, `coords_3d` is `None` and downstream code uses `coords_3d_pseudo`.
