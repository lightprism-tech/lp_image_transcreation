from perception.understanding.scene_summarizer import SceneSummarizer


def test_strip_prompt_echo_removes_prompt_prefix():
    summarizer = SceneSummarizer.__new__(SceneSummarizer)
    prompt = "Describe this scene in one detailed sentence."
    generated = "Describe this scene in one detailed sentence. Japan infographic elements"

    assert summarizer._strip_prompt_echo(generated, prompt) == "Japan infographic elements"


def test_generation_validation_rejects_prompt_echo():
    summarizer = SceneSummarizer.__new__(SceneSummarizer)

    assert not summarizer._is_valid_generation(
        "What is the setting or location type in this scene? Answer briefly.",
        "What is the setting or location type in this scene? Answer briefly.",
    )
    assert summarizer._is_valid_generation("Japan infographic elements")


def test_visual_context_includes_blip_fields_and_ocr_sample():
    summarizer = SceneSummarizer.__new__(SceneSummarizer)
    summarizer.prompt_config = {
        "visual_context_fields": ["description", "style", "layout"],
    }

    context = summarizer._build_visual_context(
        generated_fields={
            "description": "Japan travel infographic with icons",
            "style": "flat illustrated design",
            "layout": "title and icon grid",
        },
        image_type={"type": "infographic"},
        extracted_text=[{"text": "JAPAN"}, {"text": "Travel tips"}],
    )

    assert context["source"] == "blip"
    assert context["image_type_hint"] == "infographic"
    assert "flat illustrated design" in context["prompt_context"]
    assert context["ocr_text_sample"] == "JAPAN Travel tips"
