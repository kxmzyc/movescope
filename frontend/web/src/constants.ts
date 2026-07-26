export const API_BASE = import.meta.env.VITE_MOVESCOPE_API ?? 'http://127.0.0.1:8000'

// 服务端 /health 会下发真实上限与扩展名白名单；以下仅作连接失败时的回退值。
export const FALLBACK_MAX_VIDEO_BYTES = 100 * 1024 * 1024
export const FALLBACK_VIDEO_EXTENSIONS = ['.mp4', '.mov', '.avi', '.webm', '.mkv']

export const ACTION_LABELS: Record<string, string> = { squat: '深蹲' }

export function actionLabel(name: string) {
  return ACTION_LABELS[name] ?? name
}

export function isSupportedVideo(file: File, allowedExtensions: string[]) {
  const lowerName = file.name.toLowerCase()
  return file.type.startsWith('video/') || allowedExtensions.some((extension) => lowerName.endsWith(extension))
}
