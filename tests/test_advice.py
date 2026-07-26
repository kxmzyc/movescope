from movescope.advice import OpenAIAdvisor, RuleBasedAdvisor


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
