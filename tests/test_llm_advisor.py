from movescope.llm_advisor import LLMAdvisor


def make_diagnosis():
    return {
        "total_score": 78.5,
        "phases": [
            {
                "name": "phase_0",
                "time_range": [0.0, 1.2],
                "anomalies": [
                    {
                        "joint_name": "left_knee:left_hip-left_ankle",
                        "mean_deviation_deg": 11.2,
                    }
                ],
            }
        ],
    }


def test_fallback_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    advice = LLMAdvisor().generate_advice(make_diagnosis())

    assert advice.strip()
    assert "左膝" in advice


def test_output_not_medical(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    advice = LLMAdvisor().generate_advice(make_diagnosis())

    forbidden = ["诊断", "治疗", "病"]
    assert not any(word in advice for word in forbidden)


def test_synthetic_path_never_calls_remote_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "configured-but-not-used")
    advisor = LLMAdvisor()
    monkeypatch.setattr(
        advisor,
        "_openai_advice",
        lambda _diagnosis: (_ for _ in ()).throw(AssertionError("remote provider called")),
    )

    advice = advisor.generate_advice(make_diagnosis(), allow_remote=False)

    assert "左膝" in advice
