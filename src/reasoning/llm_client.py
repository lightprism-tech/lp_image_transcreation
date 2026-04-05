import os
import json
import re
import time
import logging
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
logger = logging.getLogger(__name__)

_FALLBACK_RESPONSE = {"action": "preserve", "rationale": "LLM returned invalid or empty JSON", "confidence": 0.0}


def _parse_llm_json(content: Optional[str]) -> Dict[str, Any]:
    """Parse JSON from LLM response; handle empty, markdown code blocks, or surrounding text."""
    if content is None or not isinstance(content, str):
        return _FALLBACK_RESPONSE
    raw = content.strip()
    if not raw:
        return _FALLBACK_RESPONSE
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass
    return _FALLBACK_RESPONSE


def _normalize_provider(value: str) -> str:
    """Normalize provider string (e.g. 'groq?', 'Groq ' -> 'groq')."""
    if not value:
        return "openai"
    normalized = "".join(c for c in value.lower().strip() if c.isalpha())
    return normalized or "openai"


def _build_azure_chat_url(base_url: str, deployment: str) -> str:
    """Build full Azure chat completions URL from base endpoint and deployment (see scripts/test_api.py)."""
    base = base_url.rstrip("/")
    if "/chat/completions" in base or "/openai/" in base:
        return base_url
    api_version = "2024-06-01"
    return f"{base}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"


def _extract_deployment_from_chat_url(chat_url: str) -> Optional[str]:
    """Extract Azure deployment name from chat URL if present."""
    match = re.search(r"/deployments/([^/]+)/chat/completions", chat_url or "")
    if match:
        return match.group(1)
    return None


class LLMClient:
    def __init__(self):
        raw = os.getenv("LLM_PROVIDER", "openai")
        self.provider = _normalize_provider(raw)

        # Azure is intentionally disabled for this pipeline build.
        # Even if LLM_PROVIDER is set to "azure", we fall back to Groq/OpenAI keys.
        if self.provider == "azure":
            logger.warning("Azure provider disabled in llm_client; falling back to groq/openai.")
            self.provider = "groq" if (os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("API_KEY")) else "openai"

        self.api_key = os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("API_KEY")
        self.model = os.getenv(
            "LLM_MODEL",
            "gpt-4o" if self.provider == "openai" else "llama-3.1-8b-instant",
        )
        self.azure_deployment = None
        self._azure_chat_url = None
        self._azure_fallback_deployments = []

        if not self.api_key:
            if self.provider == "azure":
                logger.warning(
                    "AZURE_OPENAI_API_KEY / LLM_API_KEY not found in environment variables."
                )
            else:
                logger.warning("LLM_API_KEY / GROQ_API_KEY not found in environment variables.")
        self._service_unavailable = False
        self._service_unavailable_reason = ""

    def generate_reasoning(self, prompt: str) -> Dict[str, Any]:
        """
        Sends a prompt to the LLM and expects a JSON response.
        """
        logger.info("LLM generate_reasoning: provider=%s model=%s", self.provider, self.model)
        if self.provider == "openai":
            return self._call_openai(prompt)
        if self.provider == "groq":
            return self._call_groq(prompt)
        if self.provider == "azure":
            return _FALLBACK_RESPONSE
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def generate_candidates(
        self,
        obj_label: str,
        obj_type: str,
        target_culture: str,
        context: str,
        avoid_list: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Generate fallback substitution candidates via the same reasoning LLM.
        Used only when KB has no candidates.
        """
        avoid_list = avoid_list or []
        prompt = (
            "Suggest culturally appropriate substitution candidates.\n"
            f"Target culture: {target_culture}\n"
            f"Object label: {obj_label}\n"
            f"Object type: {obj_type}\n"
            f"Scene context: {context}\n"
            f"Avoid list: {avoid_list}\n\n"
            "Return exactly one JSON object with this structure only:\n"
            '{"candidates": ["candidate1", "candidate2", "candidate3"]}\n'
            "Rules: keep candidates short noun phrases, do not include avoided terms."
        )

        if self.provider == "openai":
            logger.info("LLM generate_candidates via openai: obj=%s type=%s", obj_label, obj_type)
            result = self._call_openai(prompt)
        elif self.provider == "groq":
            logger.info("LLM generate_candidates via groq: obj=%s type=%s", obj_label, obj_type)
            result = self._call_groq(prompt)
        elif self.provider == "azure":
            result = _FALLBACK_RESPONSE
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

        candidates = result.get("candidates", [])
        if not isinstance(candidates, list):
            return []
        normalized: List[str] = []
        seen = set()
        for c in candidates:
            if not isinstance(c, str):
                continue
            val = c.strip()
            if not val:
                continue
            key = val.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(val)
        logger.info(
            "LLM candidate normalization complete: obj=%s type=%s input=%d output=%d",
            obj_label,
            obj_type,
            len(candidates),
            len(normalized),
        )
        return normalized

    def _call_openai(self, prompt: str, retries: int = 3) -> Dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a Cultural Reasoning assistant. You help adapt images from one culture to another. Return ONLY valid JSON."
                },
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2
        }

        for attempt in range(retries):
            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30 # Add timeout
                )
                response.raise_for_status()
                result = response.json()
                content = (result.get("choices") or [{}])[0].get("message", {}).get("content")
                return _parse_llm_json(content)

            except requests.exceptions.RequestException as e:
                logger.warning(f"OpenAI API call failed (attempt {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt) # Exponential backoff
                else:
                    logger.error("Max retries reached for OpenAI API.")
                    return {"error": str(e), "action": "preserve", "rationale": "LLM Service Unavailable"}
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM JSON response: {e}")
                return {"error": "Invalid JSON", "action": "preserve", "rationale": "LLM returned invalid JSON"}
            except Exception as e:
                logger.error(f"Unexpected error in LLM client: {e}")
                return {"error": str(e)}

    def _call_azure(self, prompt: str, retries: int = 3) -> Dict[str, Any]:
        """Azure OpenAI chat completions: api-key header, deployment in URL (no model in body)."""
        if self._service_unavailable:
            return {
                "error": "azure_unavailable_cached",
                "action": "preserve",
                "rationale": self._service_unavailable_reason or "LLM Service Unavailable",
            }
        if not self._azure_chat_url:
            logger.error("Azure LLM: set AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_CHAT_URL.")
            self._service_unavailable = True
            self._service_unavailable_reason = "LLM Service Unavailable"
            return {
                "error": "missing_azure_endpoint",
                "action": "preserve",
                "rationale": "LLM Service Unavailable",
            }

        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a Cultural Reasoning assistant. You help adapt images "
                        "from one culture to another. Return ONLY valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            # Keep aligned with scripts/test_api: simple chat payload with max_tokens.
            "max_tokens": 512,
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }

        base_url = self._azure_chat_url
        attempted_urls = []
        deployment_in_base = _extract_deployment_from_chat_url(base_url)
        candidate_urls = [base_url]
        if deployment_in_base:
            prefix = base_url.split("/openai/deployments/")[0]
            for dep in self._azure_fallback_deployments:
                if dep and dep != deployment_in_base:
                    candidate_urls.append(_build_azure_chat_url(prefix, dep))

        for url in candidate_urls:
            for attempt in range(retries):
                try:
                    response = requests.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=30,
                    )
                    response.raise_for_status()
                    result = response.json()
                    content = (result.get("choices") or [{}])[0].get("message", {}).get("content")
                    return _parse_llm_json(content)
                except requests.exceptions.RequestException as e:
                    attempted_urls.append(url)
                    logger.warning(
                        "Azure OpenAI API call failed (attempt %s/%s) on %s: %s",
                        attempt + 1,
                        retries,
                        url,
                        e,
                    )
                    if attempt < retries - 1:
                        time.sleep(2**attempt)
                    else:
                        # If this URL is a 404 and we have fallback URLs, continue to next URL.
                        status = getattr(getattr(e, "response", None), "status_code", None)
                        if status == 404 and url != candidate_urls[-1]:
                            logger.warning(
                                "Azure deployment not found at %s, trying fallback deployment URL.",
                                url,
                            )
                        else:
                            logger.error("Max retries reached for Azure OpenAI API on %s.", url)
                except json.JSONDecodeError as e:
                    logger.error("Failed to parse Azure LLM JSON response: %s", e)
                    return {
                        "error": "Invalid JSON",
                        "action": "preserve",
                        "rationale": "LLM returned invalid JSON",
                    }
                except Exception as e:
                    logger.error("Unexpected error in Azure LLM client: %s", e)
                    return {"error": str(e)}

        attempted = ", ".join(dict.fromkeys(attempted_urls)) if attempted_urls else "none"
        self._service_unavailable = True
        self._service_unavailable_reason = f"LLM Service Unavailable (attempted: {attempted})"
        return {
            "error": "Azure chat request failed",
            "action": "preserve",
            "rationale": self._service_unavailable_reason,
        }

    def _call_groq(self, prompt: str, retries: int = 3) -> Dict[str, Any]:
        """Call Groq API (OpenAI-compatible). Uses GROQ_API_KEY or LLM_API_KEY."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a Cultural Reasoning assistant. You help adapt images from one culture to another. Return ONLY valid JSON, no other text.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }

        for attempt in range(retries):
            try:
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()
                content = (result.get("choices") or [{}])[0].get("message", {}).get("content")
                return _parse_llm_json(content)
            except requests.exceptions.RequestException as e:
                logger.warning("Groq API call failed (attempt %s/%s): %s", attempt + 1, retries, e)
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error("Max retries reached for Groq API.")
                    return {"error": str(e), "action": "preserve", "rationale": "LLM Service Unavailable"}
            except Exception as e:
                logger.error("Unexpected error in LLM client: %s", e)
                return _FALLBACK_RESPONSE
