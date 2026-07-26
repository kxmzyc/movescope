import axios from 'axios'

import { API_BASE } from '../constants'
import type { Diagnosis, Health } from './types'

export async function fetchHealth(): Promise<Health> {
  const response = await axios.get<Health>(`${API_BASE}/health`, { timeout: 5000 })
  return response.data
}

export async function fetchActions(): Promise<string[]> {
  const response = await axios.get<{ actions: string[] }>(`${API_BASE}/actions`, { timeout: 5000 })
  return response.data.actions
}

export async function fetchDemo(): Promise<Diagnosis> {
  const response = await axios.get<Diagnosis>(`${API_BASE}/demo`, { timeout: 15_000 })
  return response.data
}

export async function assessVideo(file: File, action: string): Promise<Diagnosis> {
  const body = new FormData()
  body.append('video', file)
  body.append('action', action)
  const response = await axios.post<Diagnosis>(`${API_BASE}/assess`, body, { timeout: 300_000 })
  return response.data
}

export function readError(err: unknown, fallback: string): string {
  if (axios.isAxiosError(err)) {
    const detail = err.response?.data?.detail
    if (typeof detail === 'string') return detail
  }
  return fallback
}
