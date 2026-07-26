import sys
import types

from movescope.advice import OpenAIAdvisor, RuleBasedAdvisor, generate_advice


def make_diagnosis():
    return {
        "total_score": 78.5,
        "phases": [
            {
                "name": "phase_0",
                "index": 0,
                "time_range": [0.0, 1.2],
                "anomalies": [
                    {
                        "feature_index": 0,
                        "joint": "left_knee",
                        "joint_display": "左膝",
                        "parent": "left_hip",
                        "child": "left_ankle",
                        "mean_deviation_deg": 11.2,
                    }
                ],
            }
        ],
    }


def test_rule_based_advice_mentions_joint():
    advice = RuleBasedAdvisor().generate_advice(make_diagnosis())

    assert advice.strip()
    assert "左膝" in advice


def test_rule_based_advice_not_medical():
    advice = RuleBasedAdvisor().generate_advice(make_diagnosis())

    forbidden = ["诊断", "治疗", "病"]
    assert not any(word in advice for word in forbidden)


def test_openai_advisor_falls_back_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    advisor = OpenAIAdvisor()
    monkeypatch.setattr(
        advisor,
        "_remote_advice",
        lambda _diagnosis: (_ for _ in ()).throw(AssertionError("remote provider called")),
    )

    advice = advisor.generate_advice(make_diagnosis())

    assert "左膝" in advice


def test_openai_advisor_falls_back_on_remote_failure(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "configured")
    advisor = OpenAIAdvisor()
    monkeypatch.setattr(
        advisor,
        "_remote_advice",
        lambda _diagnosis: (_ for _ in ()).throw(RuntimeError("remote unavailable")),
    )

    advice = advisor.generate_advice(make_diagnosis())

    assert "左膝" in advice


def test_no_anomalies_gives_stable_message():
    advice = RuleBasedAdvisor().generate_advice({"total_score": 100.0, "phases": []})

    assert "整体动作较稳定" in advice


def test_generate_advice_off_returns_none():
    assert generate_advice(make_diagnosis(), provider="off") == (None, None)


def test_generate_advice_rule_reports_source():
    advice, source = generate_advice(make_diagnosis(), provider="rule")

    assert source == "rule"
    assert advice and "左膝" in advice


def test_generate_advice_openai_without_key_falls_back_to_rule(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    advice, source = generate_advice(make_diagnosis(), provider="openai")

    assert source == "rule"
    assert advice and "左膝" in advice


def _install_fake_openai(monkeypatch, content, captured=None):
    """替换 openai 模块：记录请求内容并返回固定回复，避免真实外发。"""

    class FakeCompletions:
        def create(self, **kwargs):
            if captured is not None:
                captured["messages"] = kwargs["messages"]
            message = types.SimpleNamespace(content=content)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    class FakeOpenAI:
        def __init__(self, timeout=None):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "configured")


def test_openai_payload_excludes_per_frame_arrays(monkeypatch):
    """外发给 OpenAI 的诊断数据应剔除 timeline/skeleton 逐帧数组。"""
    captured = {}
    _install_fake_openai(monkeypatch, "针对左膝的建议。", captured)
    diagnosis = make_diagnosis() | {
        "timeline": {"series": [{"test_deg": [1.0] * 500}]},
        "skeleton": {"keypoints": [[[0.5, 0.5]]]},
    }

    advice, source = OpenAIAdvisor().generate_advice_with_source(diagnosis)

    assert source == "openai"
    assert advice == "针对左膝的建议。"
    user_content = captured["messages"][1]["content"]
    assert "timeline" not in user_content
    assert "skeleton" not in user_content
    assert "left_knee" in user_content


def test_openai_empty_content_falls_back_and_reports_rule(monkeypatch):
    """远程模型返回空内容时回退本地规则，advice_source 也应如实标注 rule。"""
    _install_fake_openai(monkeypatch, "   ")

    advice, source = OpenAIAdvisor().generate_advice_with_source(make_diagnosis())

    assert source == "rule"
    assert advice and "左膝" in advice
