# MoveScope Web

React/Vite frontend for the MoveScope assessment API.

```bash
npm install
npm run dev
```

The app expects the FastAPI backend at `http://127.0.0.1:8000`. Override with:

```bash
set VITE_MOVESCOPE_API=http://127.0.0.1:8000
npm run dev
```

Build check:

```bash
npm run build
npm run lint
```
