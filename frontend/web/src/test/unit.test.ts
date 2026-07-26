import { describe, expect, it } from 'vitest'

import { readError } from '../api/client'
import { actionLabel, isSupportedVideo } from '../constants'

const EXTENSIONS = ['.mp4', '.mov', '.avi', '.webm', '.mkv']

describe('isSupportedVideo', () => {
  it('以扩展名白名单为准，MIME 是 video/* 但扩展名不在白名单的文件被拒绝', () => {
    const file = new File(['x'], 'clip.flv', { type: 'video/x-flv' })
    expect(isSupportedVideo(file, EXTENSIONS)).toBe(false)
  })

  it('接受白名单扩展名（大小写不敏感）', () => {
    const file = new File(['x'], 'CLIP.MP4', { type: '' })
    expect(isSupportedVideo(file, EXTENSIONS)).toBe(true)
  })

  it('拒绝非视频文件', () => {
    const file = new File(['x'], 'notes.txt', { type: 'text/plain' })
    expect(isSupportedVideo(file, EXTENSIONS)).toBe(false)
  })
})

describe('readError', () => {
  it('提取 axios 错误响应中的 detail 字段', () => {
    const err = {
      isAxiosError: true,
      response: { data: { detail: '未找到动作模板' } },
    }
    expect(readError(err, '默认提示')).toBe('未找到动作模板')
  })

  it('detail 非字符串时回退到默认提示', () => {
    const err = {
      isAxiosError: true,
      response: { data: { detail: { nested: true } } },
    }
    expect(readError(err, '默认提示')).toBe('默认提示')
  })

  it('非 axios 错误回退到默认提示', () => {
    expect(readError(new Error('boom'), '默认提示')).toBe('默认提示')
  })
})

describe('actionLabel', () => {
  it('已知动作显示中文名', () => {
    expect(actionLabel('squat')).toBe('深蹲')
  })

  it('未知动作原样返回', () => {
    expect(actionLabel('lunge')).toBe('lunge')
  })
})
