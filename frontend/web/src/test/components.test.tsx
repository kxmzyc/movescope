import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import type { Diagnosis, Health } from '../api/types'
import { ControlsPanel } from '../components/ControlsPanel'
import { ResultsPanel } from '../components/ResultsPanel'
import { TopBar } from '../components/TopBar'
import { VideoPanel } from '../components/VideoPanel'

const HEALTH: Health = {
  status: 'ok',
  version: '0.2.1',
  max_upload_bytes: 104857600,
  allowed_extensions: ['.mp4', '.mov'],
}

const DIAGNOSIS: Diagnosis = {
  action: 'squat',
  total_score: 84.3,
  segmented: true,
  phases: [
    {
      name: 'phase_2',
      index: 2,
      time_range: [0.6, 0.8],
      phase_score: 75.0,
      anomalies: [
        {
          feature_index: 0,
          joint: 'left_knee',
          joint_display: '左膝',
          parent: 'left_hip',
          child: 'left_ankle',
          direction: 'positive',
          mean_deviation_deg: 12.4,
          peak_deviation_deg: 18.0,
          peak_time_sec: 0.7,
          anomaly_ratio: 0.5,
        },
      ],
    },
  ],
  per_feature_summary: [
    {
      feature_index: 0,
      joint: 'left_knee',
      joint_display: '左膝',
      parent: 'left_hip',
      child: 'left_ankle',
      mean_dev: 8.3,
      anomaly_ratio: 0.5,
    },
  ],
  llm_advice: '动作建议：保持膝盖与脚尖方向一致。',
  metadata: {
    source: 'synthetic',
    label: '确定性深蹲关节角演示',
    disclaimer: '仅用于界面与 API 链路验证。',
    frames: 72,
  },
}

describe('TopBar', () => {
  it('未连接时显示检查按钮', () => {
    render(<TopBar health={null} checking={false} busy={false} onRefresh={() => {}} />)

    expect(screen.getByText('检查 API')).toBeInTheDocument()
  })

  it('已连接时显示 API 版本', () => {
    render(<TopBar health={HEALTH} checking={false} busy={false} onRefresh={() => {}} />)

    expect(screen.getByText('API v0.2.1 已连接')).toBeInTheDocument()
  })
})

describe('ControlsPanel', () => {
  it('已连接但无模板时提示先构建模板', () => {
    render(
      <ControlsPanel
        file={null}
        action="squat"
        availableActions={[]}
        health={HEALTH}
        busy={false}
        status="idle"
        onSelectFile={() => {}}
        onActionChange={() => {}}
        onSubmit={() => {}}
        onDemo={() => {}}
      />,
    )

    expect(screen.getByText(/未找到本地动作模板/)).toBeInTheDocument()
  })
})

describe('VideoPanel', () => {
  it('展示错误提示', () => {
    render(<VideoPanel previewUrl={null} diagnosis={null} error="评估失败" />)

    expect(screen.getByRole('alert')).toHaveTextContent('评估失败')
  })

  it('合成结果显示占位视觉', () => {
    render(<VideoPanel previewUrl={null} diagnosis={DIAGNOSIS} error={null} />)

    expect(screen.getByText('合成角度序列')).toBeInTheDocument()
    expect(screen.getByText(/72 帧/)).toBeInTheDocument()
  })
})

describe('ResultsPanel', () => {
  it('渲染总分、异常关节与建议', () => {
    render(<ResultsPanel diagnosis={DIAGNOSIS} onDownload={() => {}} />)

    expect(screen.getByText('84.3')).toBeInTheDocument()
    expect(screen.getAllByText('左膝').length).toBeGreaterThan(0)
    expect(screen.getByText(/保持膝盖与脚尖方向一致/)).toBeInTheDocument()
    expect(screen.getByText('合成验证')).toBeInTheDocument()
  })

  it('导出按钮触发下载回调', () => {
    const onDownload = vi.fn()
    render(<ResultsPanel diagnosis={DIAGNOSIS} onDownload={onDownload} />)

    screen.getByRole('button', { name: /导出 JSON/ }).click()

    expect(onDownload).toHaveBeenCalledTimes(1)
  })
})
