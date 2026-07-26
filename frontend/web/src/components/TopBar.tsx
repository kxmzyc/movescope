import { CheckCircle2, Loader2, RefreshCw } from 'lucide-react'

import type { Health } from '../api/types'

type Props = {
  health: Health | null
  checking: boolean
  busy: boolean
  onRefresh: () => void
}

export function TopBar({ health, checking, busy, onRefresh }: Props) {
  return (
    <header className="topbar">
      <div>
        <p className="eyebrow">MoveScope</p>
        <h1>单目深蹲动作评估工作台</h1>
      </div>
      <button className="statusButton" type="button" onClick={onRefresh} disabled={busy}>
        {checking ? <Loader2 className="spin" /> : health ? <CheckCircle2 /> : <RefreshCw />}
        {health ? `API v${health.version} 已连接` : '检查 API'}
      </button>
    </header>
  )
}
