from src.realization import prompt_refiner


def test_is_culturally_grounded_prompt_requires_culture_or_signal():
    assert not prompt_refiner._is_culturally_grounded_prompt(
        "A photorealistic sushi plate in the same lighting.",
        new_label="sushi",
        target_culture="Japan",
    )
    assert prompt_refiner._is_culturally_grounded_prompt(
        "A photorealistic Japanese sushi plate in the same lighting and perspective.",
        new_label="sushi",
        target_culture="Japan",
    )


def test_refine_inpaint_prompt_returns_fallback_for_weak_llm_prompt(monkeypatch):
    class FakeClient:
        api_key = "x"

        def generate_reasoning(self, _prompt):
            return {"inpaint_prompt": "same as original"}

    monkeypatch.setattr("src.reasoning.llm_client.LLMClient", FakeClient)
    fallback = "photorealistic ramen bowl, same scene and lighting"
    out = prompt_refiner.refine_inpaint_prompt(
        original_label="burger",
        new_label="ramen bowl",
        target_culture="Japan",
        fallback_prompt=fallback,
    )
    assert out == fallback


def test_refine_inpaint_prompt_accepts_grounded_llm_prompt(monkeypatch):
    class FakeClient:
        api_key = "x"

        def generate_reasoning(self, _prompt):
            return {
                "inpaint_prompt": (
                    "A Japanese ramen bowl with culturally authentic garnish, "
                    "matching the same camera angle, lighting direction, and photorealistic texture."
                )
            }

    monkeypatch.setattr("src.reasoning.llm_client.LLMClient", FakeClient)
    fallback = "photorealistic ramen bowl, same scene and lighting"
    out = prompt_refiner.refine_inpaint_prompt(
        original_label="burger",
        new_label="ramen bowl",
        target_culture="Japan",
        fallback_prompt=fallback,
    )
    assert out.startswith("A Japanese ramen bowl")
