"""
Use the project LLM (e.g. Groq/Llama) to refine inpainting prompts.
LLMs are text-only: they do not generate images. They only improve the text
prompt that is sent to the image model (e.g. Stable Diffusion).
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


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

    user_prompt = (
        f"We are adapting a photorealistic image for culture: {target_culture}. "
        f"In one region we are replacing the visual of '{original_label}' with '{new_label}'. "
        "Write a single detailed inpainting prompt for a diffusion model (e.g. Stable Diffusion) "
        "to draw this replacement. Keep it one sentence, in English. "
        'Return ONLY valid JSON with one key: "inpaint_prompt" whose value is your prompt string.'
    )
    try:
        out = client.generate_reasoning(user_prompt)
        if isinstance(out, dict) and out.get("inpaint_prompt"):
            return out["inpaint_prompt"].strip()
    except Exception as e:
        logger.debug("LLM prompt refinement failed: %s", e)
    return fallback_prompt
