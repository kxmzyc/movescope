"""根据结构化评估结果生成自然语言训练建议。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from movescope.features import JOINT_DISPLAY_NAMES


DISCLAIMER = "本分析仅供动作训练参考，不能替代专业教练或医疗意见。"

FALLBACK_ADVICE = {
    "knee": "膝盖方向尽量和脚尖保持一致，下蹲和起身时避免向内塌陷。",
    "hip": "保持髋部稳定，先向后坐再下蹲，避免一侧髋部明显偏移。",
    "ankle": "脚掌保持稳定贴地，踝关节活动受限时可以适当缩小深蹲幅度。",
    "shoulder": "肩部保持放松和对称，躯干不要因为手臂摆动而晃动。",
    "elbow": "手臂保持自然稳定，不要用手臂代偿身体重心变化。",
    "wrist": "手腕保持自然位置，避免紧张用力影响上半身稳定。",
    "neck": "头颈保持中立，视线稳定，不要低头或过度仰头。",
}


@dataclass
class LLMAdvisor:
    """通过远程模型或本地规则生成简洁的动作建议。"""

    model: str = "gpt-4o"

    def generate_advice(self, diagnosis: dict[str, Any], *, allow_remote: bool = True) -> str:
        if not allow_remote or not os.getenv("OPENAI_API_KEY"):
            return self._fallback_advice(diagnosis)

        try:
            return self._openai_advice(diagnosis)
        except Exception:
            return self._fallback_advice(diagnosis)

    def _openai_advice(self, diagnosis: dict[str, Any]) -> str:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是运动姿态分析助手。根据关节偏差数据给出简洁、可执行的动作训练建议。"
                        "不要提供医疗判断。结尾加入：本分析仅供动作训练参考，不能替代专业教练或医疗意见。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "以下是动作质量评估结果，请针对异常关节给出每项1到2句建议：\n"
                        f"{json.dumps(diagnosis, ensure_ascii=False)}"
                    ),
                },
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content or ""
        return content.strip() or self._fallback_advice(diagnosis)

    def _fallback_advice(self, diagnosis: dict[str, Any]) -> str:
        anomalies = self._collect_anomalies(diagnosis)
        if not anomalies:
            return f"整体动作较稳定，继续保持均匀下蹲和起身节奏。{DISCLAIMER}"

        lines = ["动作建议："]
        for anomaly in anomalies[:3]:
            joint_name = str(anomaly.get("joint_name", ""))
            advice = self._advice_for_joint(joint_name)
            mean_dev = float(anomaly.get("mean_deviation_deg", 0.0))
            lines.append(f"- {self._display_joint(joint_name)}：平均偏差约 {mean_dev:.1f} 度。{advice}")
        lines.append(DISCLAIMER)
        return "\n".join(lines)

    @staticmethod
    def _collect_anomalies(diagnosis: dict[str, Any]) -> list[dict[str, Any]]:
        anomalies: list[dict[str, Any]] = []
        for phase in diagnosis.get("phases", []):
            anomalies.extend(phase.get("anomalies", []))
        return sorted(anomalies, key=lambda item: float(item.get("mean_deviation_deg", 0.0)), reverse=True)

    @staticmethod
    def _advice_for_joint(joint_name: str) -> str:
        joint = joint_name.split(":", 1)[0].lower()
        for key, advice in FALLBACK_ADVICE.items():
            if key in joint:
                return advice
        return "控制动作速度，保持身体重心稳定，优先保证动作轨迹一致。"

    @staticmethod
    def _display_joint(joint_name: str) -> str:
        key = joint_name.split(":", 1)[0]
        return JOINT_DISPLAY_NAMES.get(key, key)
