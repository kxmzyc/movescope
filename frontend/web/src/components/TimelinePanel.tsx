import { useMemo, useState } from 'react'
import type { Key } from 'react'
import { Activity } from 'lucide-react'
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import type { Diagnosis } from '../api/types'

type Props = {
  diagnosis: Diagnosis
  onSeek?: (timeSec: number) => void
}

type ChartRow = {
  t: number
  test: number
  ref: number
  band: [number, number]
  anomaly: boolean
}

function formatToleranceRange(toleranceDeg: number[]): string {
  if (!toleranceDeg.length) return '容差 —'
  let min = toleranceDeg[0]
  let max = toleranceDeg[0]
  for (const value of toleranceDeg) {
    if (value < min) min = value
    if (value > max) max = value
  }
  if (max - min < 0.05) return `容差 ±${max.toFixed(1)}°`
  return `容差带 ±${min.toFixed(1)}–${max.toFixed(1)}°（随动作阶段变化）`
}

export function TimelinePanel({ diagnosis, onSeek }: Props) {
  const timeline = diagnosis.timeline
  const series = useMemo(() => timeline?.series ?? [], [timeline])

  const defaultIndex = useMemo(() => {
    if (!series.length) return 0
    let best = 0
    let bestCount = -1
    series.forEach((item, idx) => {
      const count = item.anomaly.filter(Boolean).length
      if (count > bestCount) {
        best = idx
        bestCount = count
      }
    })
    return best
  }, [series])

  const [selected, setSelected] = useState<number | null>(null)
  const activeIndex = selected !== null && selected < series.length ? selected : defaultIndex
  const active = series[activeIndex]

  const rows: ChartRow[] = useMemo(() => {
    if (!timeline || !active) return []
    return timeline.time_sec.map((t, idx) => {
      const ref = active.reference_deg[idx]
      const tol = active.tolerance_deg[idx]
      return {
        t,
        test: active.test_deg[idx],
        ref,
        band: [ref - tol, ref + tol],
        anomaly: active.anomaly[idx],
      }
    })
  }, [timeline, active])

  if (!timeline || !active) return null

  const phaseStarts = diagnosis.phases.slice(1).map((phase) => phase.time_range[0])

  return (
    <section className="panel timelinePanel">
      <div className="panelTitle">
        <Activity />
        <span>逐帧关节角时间轴</span>
        <small className="panelTitleNote">灰带为专家容差走廊，红点为越界帧，点击曲线跳转到对应时刻</small>
      </div>

      <div className="jointChips" role="tablist" aria-label="选择关节角">
        {series.map((item, idx) => {
          const anomalyCount = item.anomaly.filter(Boolean).length
          return (
            <button
              key={item.feature_index}
              type="button"
              role="tab"
              aria-selected={idx === activeIndex}
              className={idx === activeIndex ? 'chip active' : 'chip'}
              title={`${item.parent} - ${item.joint} - ${item.child}`}
              onClick={() => setSelected(idx)}
            >
              {item.joint_display}
              {anomalyCount > 0 && <i className="chipDot" aria-label="存在越界帧" />}
            </button>
          )
        })}
      </div>

      <div className="timelineChart">
        <ResponsiveContainer width="100%" height={260}>
          <ComposedChart
            data={rows}
            margin={{ left: 4, right: 18, top: 10, bottom: 4 }}
            onClick={(state) => {
              const label = state?.activeLabel
              if (onSeek && label !== undefined && label !== null) onSeek(Number(label))
            }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e2ddd4" />
            <XAxis
              dataKey="t"
              type="number"
              domain={['dataMin', 'dataMax']}
              unit="s"
              tick={{ fontSize: 11 }}
              stroke="#68707d"
            />
            <YAxis unit="°" tick={{ fontSize: 11 }} stroke="#68707d" domain={['auto', 'auto']} width={48} />
            <Tooltip
              formatter={(value, name) => {
                if (name === 'band') return null
                const label = name === 'test' ? '实测角度' : '专家角度'
                return [`${Number(value).toFixed(1)}°`, label]
              }}
              labelFormatter={(label) => `${Number(label).toFixed(2)} 秒`}
            />
            <Area dataKey="band" stroke="none" fill="#8b9aa8" fillOpacity={0.18} isAnimationActive={false} />
            <Line
              dataKey="ref"
              stroke="#5f6d7a"
              strokeDasharray="5 4"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              dataKey="test"
              stroke="#b94331"
              strokeWidth={2}
              isAnimationActive={false}
              dot={(props: { key?: Key | null; cx?: number; cy?: number; payload?: unknown }) => {
                const { key, cx, cy } = props
                const row = props.payload as ChartRow | undefined
                if (!row?.anomaly || cx === undefined || cy === undefined) return <g key={key ?? undefined} />
                return (
                  <circle key={key ?? undefined} cx={cx} cy={cy} r={3} fill="#b94331" stroke="#fff" strokeWidth={1} />
                )
              }}
            />
            {phaseStarts.map((start) => (
              <ReferenceLine key={start} x={start} stroke="#c7a76a" strokeDasharray="4 3" />
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <p className="timelineMeta">
        {formatToleranceRange(active.tolerance_deg)}，越界 {active.anomaly.filter(Boolean).length}/
        {active.anomaly.length} 帧
        {timeline.frame_stride > 1 && `（每 ${timeline.frame_stride} 帧采样）`}
      </p>
    </section>
  )
}
