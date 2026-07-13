# MoveScope Web 前端

这是 MoveScope FastAPI 服务配套的 React/Vite 评估工作台。

```bash
npm ci
npm run dev
```

默认连接 `http://127.0.0.1:8000`。如需使用其他 API 地址，请在启动 Vite 前设置：

```powershell
$env:VITE_MOVESCOPE_API="http://127.0.0.1:8000"
npm run dev
```

界面通过 `/actions` 读取本地动作模板，通过 `/demo` 运行明确标记的确定性合成验证，通过 `/assess` 提交真实视频，并可将诊断结果导出为 JSON。

构建与检查：

```bash
npm run build
npm run lint
```
