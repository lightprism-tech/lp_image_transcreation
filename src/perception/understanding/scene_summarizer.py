"""
Scene Summarizer using BLIP
Generates a global description of the entire scene
"""

import logging
import re

import numpy as np
import torch
from PIL import Image

from perception.config import settings
from perception.understanding.blip_model_manager import BLIPModelManager

logger = logging.getLogger(__name__)

_DEFAULT_SCENE_PROMPTS = {
    "description": {
        "prompts": [
            "Describe this scene in one detailed sentence.",
            "A detailed visual description of this image is",
        ],
        "max_new_tokens": 100,
        "allow_unprompted": True,
    },
    "setting": {
        "prompts": ["Setting:", "The location type is"],
        "max_new_tokens": 24,
    },
    "mood": {
        "prompts": ["Mood:", "The visual atmosphere is"],
        "max_new_tokens": 24,
    },
    "activity": {
        "prompts": ["Main activity:", "The main visible action is"],
        "max_new_tokens": 24,
    },
}


class SceneSummarizer:
    """Generates global scene descriptions using BLIP"""
    
    def __init__(self, model_name=None):
        """Initialize BLIP scene summarization model (uses shared model manager)"""
        self.model_name = model_name or settings.BLIP2_MODEL_NAME
        
        # Use shared model manager instead of loading separate model
        self.model_manager = BLIPModelManager()
        self.model = self.model_manager.get_model()
        self.processor = self.model_manager.get_processor()
        self.device = self.model_manager.get_device()
        self.prompt_config = self._load_prompt_config()
        
        logger.info("SceneSummarizer initialized with shared BLIP model")
    
    def summarize(
        self,
        image: np.ndarray,
        image_type: dict | None = None,
        extracted_text: list | None = None,
    ) -> dict:
        """
        Generate global scene description
        
        Args:
            image: Input image as numpy array (RGB)
            
        Returns:
            Scene summary:
            {
                'description': str,  # Overall scene description
                'setting': str,      # Indoor/outdoor, location type
                'mood': str,         # Overall mood/atmosphere
                'activity': str,     # Main activity happening
                'confidence': float
            }
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("BLIP-2 model not loaded")
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image.astype('uint8'))
        
        generated_fields = {}
        for field_name, field_cfg in self.prompt_config["fields"].items():
            fallback_context = generated_fields.get("description")
            generated_fields[field_name] = self._generate_valid_text(
                pil_image=pil_image,
                prompts=field_cfg["prompts"],
                max_new_tokens=field_cfg["max_new_tokens"],
                allow_unprompted=field_cfg["allow_unprompted"],
                fallback_context=fallback_context,
            )

        description = generated_fields.get("description", "")
        summary = {
            "description": description,
            "setting": generated_fields.get("setting", description),
            "mood": generated_fields.get("mood", description),
            "activity": generated_fields.get("activity", description),
            "confidence": 1.0,
            "visual_context": self._build_visual_context(
                generated_fields=generated_fields,
                image_type=image_type,
                extracted_text=extracted_text,
            ),
        }
        return summary

    def _load_prompt_config(self) -> dict:
        configured = getattr(settings, "PERCEPTION_PROMPTS", {}).get("scene_summarizer", {})
        field_cfg = configured.get("fields") if isinstance(configured, dict) else {}
        fields = {}
        for field_name, defaults in _DEFAULT_SCENE_PROMPTS.items():
            cfg = field_cfg.get(field_name, {}) if isinstance(field_cfg, dict) else {}
            prompts = cfg.get("prompts", defaults["prompts"])
            fields[field_name] = {
                "prompts": [str(prompt) for prompt in prompts if str(prompt).strip()],
                "max_new_tokens": int(cfg.get("max_new_tokens", defaults["max_new_tokens"])),
                "allow_unprompted": bool(cfg.get("allow_unprompted", defaults.get("allow_unprompted", False))),
            }
        if isinstance(field_cfg, dict):
            for field_name, cfg in field_cfg.items():
                if field_name in fields or not isinstance(cfg, dict):
                    continue
                prompts = [str(prompt) for prompt in cfg.get("prompts", []) if str(prompt).strip()]
                if not prompts:
                    continue
                fields[field_name] = {
                    "prompts": prompts,
                    "max_new_tokens": int(cfg.get("max_new_tokens", 32)),
                    "allow_unprompted": bool(cfg.get("allow_unprompted", False)),
                }
        visual_context_fields = configured.get("visual_context_fields", list(fields.keys()))
        return {
            "fields": fields,
            "visual_context_fields": [
                str(field) for field in visual_context_fields if str(field).strip()
            ],
        }

    def _build_visual_context(
        self,
        generated_fields: dict,
        image_type: dict | None,
        extracted_text: list | None,
    ) -> dict:
        context_fields = self.prompt_config.get("visual_context_fields", [])
        context = {
            "source": "blip",
            "generated_fields": {
                field: generated_fields[field]
                for field in context_fields
                if field in generated_fields and generated_fields[field]
            },
            "image_type_hint": (image_type or {}).get("type", ""),
            "ocr_text_sample": self._ocr_text_sample(extracted_text or []),
        }
        context["prompt_context"] = " ".join(
            str(value)
            for value in context["generated_fields"].values()
            if str(value).strip()
        ).strip()
        return context

    @staticmethod
    def _ocr_text_sample(extracted_text: list) -> str:
        words = []
        for item in extracted_text or []:
            text = item.get("text") if isinstance(item, dict) else item
            if text:
                words.append(str(text).strip())
            if len(" ".join(words).split()) >= 40:
                break
        return " ".join(words)
    
    def _generate_valid_text(
        self,
        pil_image: Image.Image,
        prompts: list[str],
        max_new_tokens: int,
        allow_unprompted: bool = False,
        fallback_context: str | None = None,
    ) -> str:
        """Generate a non-empty, non-prompt-echo scene field."""
        for prompt in prompts:
            candidate = self._generate_with_prompt(
                pil_image=pil_image,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
            if self._is_valid_generation(candidate, prompt):
                return candidate

        if allow_unprompted:
            candidate = self._generate_caption(pil_image=pil_image, max_new_tokens=max_new_tokens)
            if self._is_valid_generation(candidate):
                return candidate

        if fallback_context and self._is_valid_generation(fallback_context):
            return fallback_context

        raise RuntimeError("BLIP scene summarization produced only empty or prompt-echo output")

    def _generate_caption(self, pil_image: Image.Image, max_new_tokens: int) -> str:
        """Generate an unprompted caption from BLIP."""
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    def _generate_with_prompt(self, pil_image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        """Generate text from BLIP using an image-conditioned prompt."""
        inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return self._strip_prompt_echo(decoded, prompt)

    def _strip_prompt_echo(self, generated_text: str, prompt: str) -> str:
        """Remove decoded prompt text when BLIP returns prompt plus continuation."""
        generated = generated_text.strip()
        prompt_clean = prompt.strip()
        if generated.lower().startswith(prompt_clean.lower()):
            generated = generated[len(prompt_clean):].strip(" .,:;-")
        return generated.strip()

    def _is_valid_generation(self, generated_text: str, prompt: str | None = None) -> bool:
        """Reject empty output and common prompt-echo artifacts."""
        text = generated_text.strip()
        if not text:
            return False
        normalized = re.sub(r"\W+", " ", text).strip().lower()
        if not normalized:
            return False
        if prompt:
            prompt_normalized = re.sub(r"\W+", " ", prompt).strip().lower()
            if normalized == prompt_normalized or prompt_normalized in normalized:
                return False
        prompt_terms = ("answer briefly", "describe this scene", "what is the", "main activity")
        return not any(term in normalized for term in prompt_terms)
