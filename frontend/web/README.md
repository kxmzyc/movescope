# MoveScope Web

React/Vite workspace for the MoveScope FastAPI service.

```bash
npm ci
npm run dev
```

The default API is `http://127.0.0.1:8000`. Override it before starting Vite when needed:

```powershell
$env:VITE_MOVESCOPE_API="http://127.0.0.1:8000"
npm run dev
```

The UI discovers templates through `/actions`, runs a clearly labeled deterministic result through `/demo`, submits real videos through `/assess`, and exports returned diagnoses as JSON.

Checks:

```bash
npm run build
npm run lint
```
