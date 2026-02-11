import os
import json
import time
import logging
import requests
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "openai")
        self.api_key = os.getenv("LLM_API_KEY")
        self.model = os.getenv("LLM_MODEL", "gpt-4o")
        
        if not self.api_key:
            logger.warning("LLM_API_KEY not found in environment variables.")

    def generate_reasoning(self, prompt: str) -> Dict[str, Any]:
        """
        Sends a prompt to the LLM and expects a JSON response.
        """
        if self.provider == "openai":
            return self._call_openai(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

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
                content = result["choices"][0]["message"]["content"]
                return json.loads(content)
            
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
