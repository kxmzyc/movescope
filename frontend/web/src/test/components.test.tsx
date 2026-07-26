import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import type { Diagnosis, Health } from '../api/types'
import { ControlsPanel } from '../components/ControlsPanel'
import { ResultsPanel } from '../components/ResultsPanel'
import { TimelinePanel } from '../components/TimelinePanel'
import { TopBar } from '../components/TopBar'
import { VideoPanel } from '../components/VideoPanel'

const HEALTH: Health = {
  status: 'ok',
  version: '0.4.0',
  max_upload_bytes: 104857600,
  allowed_extensions: ['.mp4', '.mov'],
}

const DIAGNOSIS: Diagnosis = {
  action: 'squat',
  total_score: 84.3,
  segmented: true,
  phases: [
    {
      name: 'phase_0',
      label: '下蹲',
      index: 0,
      time_range: [0.0, 0.5],
      phase_score: 95.0,
      anomalies: [],
    },
    {
      name: 'phase_1',
      label: '蹲底',
      index: 1,
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
      tolerance_deg: 5.0,
      score_weight: 0.0833,
    },
  ],
  excluded_features: [
    {
      feature_index: 6,
      joint: 'left_elbow',
      joint_display: '左肘',
      parent: 'left_shoulder',
      child: 'left_wrist',
      reason: '该关节角在待测视频中的数据不完整（对应关节未被可靠检测）',
    },
  ],
  timeline: {
    fps: 30.0,
    frame_count: 4,
    frame_stride: 1,
    time_sec: [0.0, 0.033, 0.067, 0.1],
    series: [
      {
        feature_index: 0,
        joint: 'left_knee',
        joint_display: '左膝',
        parent: 'left_hip',
        child: 'left_ankle',
        tolerance_deg: [5.0, 5.0, 5.0, 5.0],
        test_deg: [170.1, 160.4, 150.2, 140.9],
        reference_deg: [170.0, 162.0, 155.0, 148.0],
        anomaly: [false, false, false, true],
      },
      {
        feature_index: 1,
        joint: 'right_knee',
        joint_display: '右膝',
        parent: 'right_hip',
        child: 'right_ankle',
        tolerance_deg: [4.0, 4.0, 6.0, 6.0],
        test_deg: [170.0, 162.0, 155.0, 148.0],
        reference_deg: [170.0, 162.0, 155.0, 148.0],
        anomaly: [false, false, false, false],
      },
    ],
  },
  llm_advice: '动作建议：保持膝盖与脚尖方向一致。',
  advice_source: 'rule',
  quality: {
    frames: 120,
    fps: 30.0,
    valid_pose_ratio: 0.96,
    pose_source: 'mediapipe_world',
  },
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

    expect(screen.getByText('API v0.4.0 已连接')).toBeInTheDocument()
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

  it('渲染带语义标签的阶段时间轴且点击可跳转', () => {
    const onSeek = vi.fn()
    render(<VideoPanel previewUrl={null} diagnosis={DIAGNOSIS} error={null} onSeek={onSeek} />)

    const first = screen.getByRole('button', { name: /下蹲/ })
    first.click()

    expect(onSeek).toHaveBeenCalledWith(0.0)
    expect(screen.getByRole('button', { name: /蹲底/ })).toBeInTheDocument()
  })

  it('有视频与骨架数据时渲染叠加画布', () => {
    render(<VideoPanel previewUrl="blob:mock" diagnosis={withSkeleton()} error={null} />)

    expect(screen.getByTestId('skeleton-overlay')).toBeInTheDocument()
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

  it('渲染偏差方向、姿态质量与建议来源', () => {
    render(<ResultsPanel diagnosis={DIAGNOSIS} onDownload={() => {}} />)

    expect(screen.getByText('角度偏大')).toBeInTheDocument()
    expect(screen.getByText(/有效姿态 96%/)).toBeInTheDocument()
    expect(screen.getByText('MediaPipe 伪3D')).toBeInTheDocument()
    expect(screen.getByText('本地规则')).toBeInTheDocument()
  })

  it('展示未参与评分的关节', () => {
    render(<ResultsPanel diagnosis={DIAGNOSIS} onDownload={() => {}} />)

    expect(screen.getByText(/未参与评分：左肘/)).toBeInTheDocument()
  })

  it('点击异常项跳转到峰值时刻', () => {
    const onSeek = vi.fn()
    render(<ResultsPanel diagnosis={DIAGNOSIS} onDownload={() => {}} onSeek={onSeek} />)

    screen.getByRole('button', { name: /左膝/ }).click()

    expect(onSeek).toHaveBeenCalledWith(0.7)
  })

  it('多次往复时渲染逐次评分并可点击跳转', () => {
    const onSeek = vi.fn()
    const diagnosis: Diagnosis = {
      ...DIAGNOSIS,
      reps: [
        { index: 0, time_range: [0.3, 2.1], score: 69.2, knee_min_deg: 72.0 },
        { index: 1, time_range: [2.9, 5.0], score: 72.6, knee_min_deg: 70.5 },
      ],
      rep_detail_index: 1,
    }
    render(<ResultsPanel diagnosis={diagnosis} onDownload={() => {}} onSeek={onSeek} />)

    expect(screen.getByText(/检测到 2 次深蹲/)).toBeInTheDocument()
    screen.getByRole('button', { name: /第 1 次/ }).click()

    expect(onSeek).toHaveBeenCalledWith(0.3)
    expect(screen.getByRole('button', { name: /第 2 次/ })).toHaveClass('active')
  })

  it('导出按钮触发下载回调', () => {
    const onDownload = vi.fn()
    render(<ResultsPanel diagnosis={DIAGNOSIS} onDownload={onDownload} />)

    screen.getByRole('button', { name: /导出 JSON/ }).click()

    expect(onDownload).toHaveBeenCalledTimes(1)
  })
})

describe('TimelinePanel', () => {
  it('默认选中越界最多的关节并显示关节选择器', () => {
    render(<TimelinePanel diagnosis={DIAGNOSIS} />)

    const tabs = screen.getAllByRole('tab')
    expect(tabs).toHaveLength(2)
    expect(tabs[0]).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByText(/越界 1\/4 帧/)).toBeInTheDocument()
  })

  it('切换关节后更新统计信息与容差带范围', () => {
    render(<TimelinePanel diagnosis={DIAGNOSIS} />)

    fireEvent.click(screen.getByRole('tab', { name: /右膝/ }))

    expect(screen.getByText(/越界 0\/4 帧/)).toBeInTheDocument()
    expect(screen.getByText(/容差带 ±4\.0–6\.0°/)).toBeInTheDocument()
  })

  it('容差恒定时显示单一容差值', () => {
    render(<TimelinePanel diagnosis={DIAGNOSIS} />)

    expect(screen.getByText(/容差 ±5\.0°/)).toBeInTheDocument()
  })

  it('无时间轴数据时不渲染', () => {
    const { container } = render(
      <TimelinePanel diagnosis={{ ...DIAGNOSIS, timeline: undefined }} />,
    )

    expect(container).toBeEmptyDOMElement()
  })
})

function withSkeleton(): Diagnosis {
  return {
    ...DIAGNOSIS,
    metadata: undefined,
    skeleton: {
      fps: 30.0,
      frame_count: 2,
      frame_stride: 1,
      time_sec: [0.0, 0.033],
      joint_names: ['pelvis', 'left_hip'],
      edges: [[0, 1]],
      keypoints: [
        [[0.5, 0.5], [0.4, 0.6]],
        [[0.5, 0.5], null],
      ],
      confidence: [
        [0.9, 0.9],
        [0.9, 0.1],
      ],
    },
  }
}
