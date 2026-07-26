"""根据结构化评估结果生成自然语言训练建议。

RuleBasedAdvisor 是确定性的本地规则实现；OpenAIAdvisor 在配置了
OPENAI_API_KEY 时调用远程模型，任何失败都回退到本地规则。两者共用
AdviceProvider 协议，调用方按需选择。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from movescope.features import JOINT_DISPLAY_NAMES

logger = logging.getLogger(__name__)

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


class AdviceProvider(Protocol):
    def generate_advice(self, diagnosis: dict[str, Any]) -> str: ...


@dataclass
class RuleBasedAdvisor:
    """确定性的本地规则建议，无任何外部依赖。"""

    def generate_advice(self, diagnosis: dict[str, Any]) -> str:
        anomalies = collect_anomalies(diagnosis)
        if not anomalies:
            return f"整体动作较稳定，继续保持均匀下蹲和起身节奏。{DISCLAIMER}"

        lines = ["动作建议："]
        for anomaly in anomalies[:3]:
            joint = str(anomaly.get("joint", ""))
            display = str(anomaly.get("joint_display") or JOINT_DISPLAY_NAMES.get(joint, joint))
            mean_dev = float(anomaly.get("mean_deviation_deg", 0.0))
            lines.append(f"- {display}：平均偏差约 {mean_dev:.1f} 度。{self._advice_for_joint(joint)}")
        lines.append(DISCLAIMER)
        return "\n".join(lines)

    @staticmethod
    def _advice_for_joint(joint: str) -> str:
        joint = joint.lower()
        for key, advice in FALLBACK_ADVICE.items():
            if key in joint:
                return advice
        return "控制动作速度，保持身体重心稳定，优先保证动作轨迹一致。"


@dataclass
class OpenAIAdvisor:
    """远程模型建议；未配置密钥或请求失败时回退到本地规则。"""

    model: str = "gpt-4o"
    timeout_sec: float = 20.0
    fallback: RuleBasedAdvisor = field(default_factory=RuleBasedAdvisor)

    def generate_advice(self, diagnosis: dict[str, Any]) -> str:
        return self.generate_advice_with_source(diagnosis)[0]

    def generate_advice_with_source(self, diagnosis: dict[str, Any]) -> tuple[str, str]:
        """返回 (建议文本, 实际来源)；来源为 "openai" 或回退后的 "rule"。"""
        if not os.getenv("OPENAI_API_KEY"):
            return self.fallback.generate_advice(diagnosis), "rule"
        try:
            content = self._remote_advice(diagnosis)
        except Exception:
            logger.warning("OpenAI 建议生成失败，已回退到本地规则建议", exc_info=True)
            return self.fallback.generate_advice(diagnosis), "rule"
        if content:
            return content, "openai"
        logger.warning("OpenAI 返回了空建议内容，已回退到本地规则建议")
        return self.fallback.generate_advice(diagnosis), "rule"

    def _remote_advice(self, diagnosis: dict[str, Any]) -> str:
        from openai import OpenAI

        # timeline/skeleton 是逐帧数组，长视频下可达上百 KB；远程建议只需要
        # 聚合诊断，剔除后再外发，同时控制提示词体积与数据外发范围。
        payload = {key: value for key, value in diagnosis.items() if key not in ("timeline", "skeleton")}
        client = OpenAI(timeout=self.timeout_sec)
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
                        f"{json.dumps(payload, ensure_ascii=False)}"
                    ),
                },
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content or ""
        return content.strip()


def generate_advice(
    diagnosis: dict[str, Any],
    provider: str = "rule",
    model: str = "gpt-4o",
    timeout_sec: float = 20.0,
) -> tuple[str | None, str | None]:
    """按配置的来源生成建议，返回 (建议文本, 实际来源)。

    provider="off" 返回 (None, None)；"openai" 在未配置密钥或请求失败
    时回退本地规则，实际来源以返回值为准。
    """
    if provider == "off":
        return None, None
    if provider == "openai":
        return OpenAIAdvisor(model=model, timeout_sec=timeout_sec).generate_advice_with_source(diagnosis)
    return RuleBasedAdvisor().generate_advice(diagnosis), "rule"


def collect_anomalies(diagnosis: dict[str, Any]) -> list[dict[str, Any]]:
    anomalies: list[dict[str, Any]] = []
    for phase in diagnosis.get("phases", []):
        anomalies.extend(phase.get("anomalies", []))
    return sorted(anomalies, key=lambda item: float(item.get("mean_deviation_deg", 0.0)), reverse=True)
