import { useMemo } from 'react'
import { ArrowDownRight, ArrowUpRight, BarChart3, CheckCircle2, Download, EyeOff } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import type { Diagnosis } from '../api/types'

type Props = {
  diagnosis: Diagnosis | null
  onDownload: () => void
  onSeek?: (timeSec: number) => void
}

export function ResultsPanel({ diagnosis, onDownload, onSeek }: Props) {
  const jointRows = useMemo(() => {
    if (!diagnosis) return []
    return diagnosis.per_feature_summary
      .map((item) => ({
        name: item.joint_display,
        meanDev: Number(item.mean_dev.toFixed(2)),
        anomalyRate: Number((item.anomaly_ratio * 100).toFixed(1)),
      }))
      .sort((a, b) => b.meanDev - a.meanDev)
      .slice(0, 8)
  }, [diagnosis])

  const topAnomalies = useMemo(() => {
    return (
      diagnosis?.phases
        .flatMap((phase) => phase.anomalies.map((anomaly) => ({ ...anomaly, phase })))
        .sort((a, b) => b.mean_deviation_deg - a.mean_deviation_deg) ?? []
    )
  }, [diagnosis])

  const excluded = diagnosis?.excluded_features ?? []
  const quality = diagnosis?.quality

  return (
    <aside className="panel results">
      <div className="scoreHeader">
        <div>
          <p className="eyebrow">总分</p>
          <strong>{diagnosis ? diagnosis.total_score.toFixed(1) : '--'}</strong>
          <span className="scoreUnit">/ 100</span>
        </div>
        <div className={diagnosis ? 'scoreBadge active' : 'scoreBadge'}>
          {diagnosis ? <CheckCircle2 /> : <BarChart3 />}
        </div>
      </div>

      {diagnosis && (
        <div className="resultMeta">
          <span className={diagnosis.metadata ? 'sourceBadge synthetic' : 'sourceBadge'}>
            {diagnosis.metadata ? '合成验证' : '上传视频'}
          </span>
          {!diagnosis.segmented && <span className="sourceBadge">全序列对齐</span>}
          {quality && (
            <span className="sourceBadge neutral" title="默认流程使用 MediaPipe world landmarks 伪三维坐标">
              {quality.pose_source === 'motionbert' ? 'MotionBERT 3D' : 'MediaPipe 伪3D'}
            </span>
          )}
          <button className="iconCommand" type="button" onClick={onDownload} title="下载 JSON 报告">
            <Download />
            导出 JSON
          </button>
        </div>
      )}

      {quality && (
        <p className="qualityRow">
          {quality.frames} 帧 · {quality.fps.toFixed(1)} fps · 有效姿态 {(quality.valid_pose_ratio * 100).toFixed(0)}%
        </p>
      )}

      {excluded.length > 0 && (
        <div className="excludedNotice" role="note">
          <EyeOff />
          <span>
            未参与评分：{excluded.map((item) => item.joint_display).join('、')}（{excluded[0].reason}）
          </span>
        </div>
      )}

      <div className="chartBox">
        {jointRows.length ? (
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={jointRows} layout="vertical" margin={{ left: 8, right: 12, top: 8, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#d7dbe2" />
              <XAxis type="number" unit="°" tick={{ fontSize: 11 }} stroke="#68707d" />
              <YAxis dataKey="name" type="category" width={86} tick={{ fontSize: 11 }} stroke="#68707d" />
              <Tooltip
                cursor={{ fill: '#eef2f6' }}
                formatter={(value) => [`${Number(value).toFixed(2)}°`, '平均偏差']}
                labelFormatter={(label) => `关节：${label}`}
              />
              <Bar dataKey="meanDev" fill="#c14835" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div className="emptyChart">完成评估后将在这里显示各关节偏差。</div>
        )}
      </div>

      <section className="summary">
        <h2>主要问题</h2>
        {topAnomalies.length ? (
          <ul>
            {topAnomalies.slice(0, 4).map((item) => (
              <li key={`${item.phase.name}-${item.feature_index}`}>
                <button
                  type="button"
                  className="anomalyItem"
                  title="点击跳转到峰值时刻"
                  onClick={() => onSeek?.(item.peak_time_sec)}
                >
                  <span>
                    {item.joint_display}
                    <em className={item.direction === 'positive' ? 'dirBadge over' : 'dirBadge under'}>
                      {item.direction === 'positive' ? <ArrowUpRight /> : <ArrowDownRight />}
                      {item.direction === 'positive' ? '角度偏大' : '角度偏小'}
                    </em>
                    <small>
                      {item.phase.label} · {item.peak_time_sec.toFixed(2)} 秒峰值{' '}
                      {item.peak_deviation_deg.toFixed(1)}° · {(item.anomaly_ratio * 100).toFixed(0)}% 异常
                    </small>
                  </span>
                  <strong>{item.mean_deviation_deg.toFixed(1)}°</strong>
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <p>暂无评估结果。</p>
        )}
      </section>

      <section className="advice">
        <h2>
          训练建议
          {diagnosis?.advice_source && (
            <span className="adviceSource">{diagnosis.advice_source === 'openai' ? 'OpenAI 生成' : '本地规则'}</span>
          )}
        </h2>
        <p>{diagnosis?.llm_advice ?? '后端返回诊断结果后，将在这里显示动作纠正建议。'}</p>
        {diagnosis?.metadata && <small className="disclaimer">{diagnosis.metadata.disclaimer}</small>}
      </section>
    </aside>
  )
}
