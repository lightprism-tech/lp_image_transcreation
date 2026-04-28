"""
Use the project LLM (e.g. Groq/Llama) to refine inpainting prompts.
LLMs are text-only: they do not generate images. They only improve the text
prompt that is sent to the image model (e.g. Stable Diffusion).
"""
import logging
import re

from src.realization.prompt_config import get_prompt

logger = logging.getLogger(__name__)


def _is_culturally_grounded_prompt(prompt: str, new_label: str, target_culture: str) -> bool:
    """
    Ensure generated inpaint prompt explicitly reflects target-culture adaptation.
    """
    text = (prompt or "").strip().lower()
    if not text:
        return False
    if len(text.split()) < 8:
        return False
    # Prompt should mention the replacement concept.
    if (new_label or "").strip() and (new_label or "").strip().lower() not in text:
        return False
    # Prompt should carry explicit cultural adaptation signal.
    culture_key = (target_culture or "").strip().lower()
    if culture_key and culture_key not in text:
        cultural_tokens = ("cultural", "local", "regional", "traditional")
        if not any(token in text for token in cultural_tokens):
            return False
    # Avoid placeholder-like or low-information responses.
    banned = ("unknown", "n/a", "generic object", "same as original")
    return not any(token in text for token in banned)


def refine_inpaint_prompt(
    original_label: str,
    new_label: str,
    target_culture: str,
    fallback_prompt: str,
) -> str:
    """
    Ask the LLM for a detailed inpainting prompt. Uses LLM_PROVIDER and keys from .env
    (OpenAI, Groq, or Azure: AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT; see scripts/test_api.py).
    Returns fallback_prompt if LLM is unavailable or returns invalid JSON.
    """
    try:
        from src.reasoning.llm_client import LLMClient
    except ImportError:
        logger.debug("LLM client not available; using fallback prompt")
        return fallback_prompt

    client = LLMClient()
    if not getattr(client, "api_key", None):
        logger.debug("No LLM API key; using fallback prompt")
        return fallback_prompt

    template = get_prompt(
        "prompt_refiner.request_template",
        (
            "You are writing a single inpainting prompt for cultural image transcreation.\n"
            "Target culture: {target_culture}\n"
            "Object to replace: {original_label}\n"
            "Replacement object: {new_label}\n\n"
            "Goal: produce a culturally authentic replacement for the target culture while preserving scene continuity.\n"
            "Hard constraints:\n"
            "1) Mention the target culture explicitly by name.\n"
            "2) Mention the replacement object explicitly.\n"
            "3) Preserve original composition cues: same camera angle, perspective, lighting direction, shadows, and realism level.\n"
            "4) Keep surroundings untouched; only change the masked region.\n"
            "5) Avoid stereotypes, caricatures, or disrespectful depictions.\n"
            "6) Output exactly one sentence, no line breaks.\n\n"
            'Return ONLY valid JSON with this exact schema: {"inpaint_prompt":"..."}'
        ),
    )
    user_prompt = template.format(
        target_culture=target_culture,
        original_label=original_label,
        new_label=new_label,
        scene_name="existing scene context",
        shape="context-inferred shape",
        color="context-inferred color palette",
        texture="context-inferred texture",
    )
    try:
        out = client.generate_reasoning(user_prompt)
        if isinstance(out, dict) and out.get("inpaint_prompt"):
            candidate = re.sub(r"\s+", " ", str(out["inpaint_prompt"]).strip())
            if _is_culturally_grounded_prompt(candidate, new_label, target_culture):
                return candidate
            logger.debug("LLM prompt refinement returned non-grounded prompt; using fallback")
    except Exception as e:
        logger.debug("LLM prompt refinement failed: %s", e)
    return fallback_prompt
