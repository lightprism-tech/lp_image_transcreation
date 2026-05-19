from src.realization.config_loader import load_realization_config, section_value


def test_load_realization_config_merges_user_overrides(monkeypatch):
    monkeypatch.delenv("REALIZATION_ARTIFACT_GATE__MIN_MEAN_ABS_CHANGE", raising=False)
    monkeypatch.delenv("REALIZATION_INPAINT_MASK_PAD_PCT", raising=False)
    config = load_realization_config(
        {
            "artifact_gate": {"min_mean_abs_change": 9.0},
            "use_inpainting": True,
        }
    )
    assert config["artifact_gate"]["min_mean_abs_change"] == 9.0
    assert config["artifact_gate"]["min_p95_channel_change"] == 10.0
    assert config["inpaint_mask_pad_pct"] == 0.05


def test_section_value_reads_required_key():
    config = load_realization_config({})
    assert section_value(config, "artifact_gate", "enabled") is True
