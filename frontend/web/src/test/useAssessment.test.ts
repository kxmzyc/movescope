import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useAssessment } from '../hooks/useAssessment'

beforeEach(() => {
  vi.stubGlobal('URL', {
    ...URL,
    createObjectURL: vi.fn(() => 'blob:mock'),
    revokeObjectURL: vi.fn(),
  })
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('useAssessment.selectFile', () => {
  it('拒绝不支持的文件类型', () => {
    const { result } = renderHook(() => useAssessment())

    act(() => {
      result.current.selectFile(new File(['x'], 'notes.txt', { type: 'text/plain' }))
    })

    expect(result.current.error).toContain('格式')
    expect(result.current.file).toBeNull()
  })

  it('拒绝超过上限的文件并提示服务端上限', () => {
    const { result } = renderHook(() => useAssessment({ maxVideoBytes: 10 }))

    act(() => {
      result.current.selectFile(new File(['0'.repeat(20)], 'clip.mp4', { type: 'video/mp4' }))
    })

    expect(result.current.error).toContain('上传上限')
    expect(result.current.file).toBeNull()
  })

  it('接受合法视频并生成预览地址', () => {
    const { result } = renderHook(() => useAssessment())
    const file = new File(['x'], 'clip.mp4', { type: 'video/mp4' })

    act(() => {
      result.current.selectFile(file)
    })

    expect(result.current.error).toBeNull()
    expect(result.current.file).toBe(file)
    expect(result.current.previewUrl).toBe('blob:mock')
  })

  it('组件卸载时回收 object URL', () => {
    const { result, unmount } = renderHook(() => useAssessment())

    act(() => {
      result.current.selectFile(new File(['x'], 'clip.mp4', { type: 'video/mp4' }))
    })
    unmount()

    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:mock')
  })
})

describe('useAssessment.submit', () => {
  it('未选择文件时提示先选择视频', async () => {
    const { result } = renderHook(() => useAssessment())

    await act(async () => {
      await result.current.submit('squat')
    })

    expect(result.current.error).toContain('请先选择视频')
  })
})
