import { useCallback, useEffect, useState } from 'react'

import { fetchActions, fetchHealth, readError } from '../api/client'
import type { Health } from '../api/types'

export function useApiStatus() {
  const [health, setHealth] = useState<Health | null>(null)
  const [availableActions, setAvailableActions] = useState<string[]>([])
  const [checking, setChecking] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    setChecking(true)
    setError(null)
    try {
      const [healthData, actions] = await Promise.all([fetchHealth(), fetchActions()])
      setHealth(healthData)
      setAvailableActions(actions)
    } catch (err) {
      setError(readError(err, '无法连接 API，请确认 FastAPI 已在 8000 端口启动。'))
      setHealth(null)
      setAvailableActions([])
    } finally {
      setChecking(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  return { health, availableActions, checking, error, refresh }
}
