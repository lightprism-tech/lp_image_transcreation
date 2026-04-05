"""
Test an API endpoint with URL and API key.
Supports OpenAI-compatible chat completions (OpenAI, Groq, Azure).
Azure AI Foundry FLUX (Black Forest Labs): set AZURE_FLUX_IMAGE_URL and use --test-image.
Default: Azure (uses AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT from .env).
"""
import argparse
import base64
import json
import os
import re
import sys
from datetime import datetime
from typing import List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


def test_api(
    url: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    prompt: str = "Reply with exactly: OK",
) -> dict:
    """
    Send a minimal test request to an OpenAI-compatible chat completions API.

    Args:
        url: Full API endpoint URL
        api_key: API key for authorization
        model: Model/deployment name (used for OpenAI/Groq; Azure uses deployment in URL)
        provider: openai | groq | azure (azure uses api-key header instead of Bearer)
        prompt: User message to send

    Returns:
        dict with keys: success (bool), response (dict or None), error (str or None)
    """
    if provider == "azure":
        headers = {
            "Content-Type": "application/json",
            "api-key": api_key,
        }
    else:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
    }
    if provider != "azure":
        payload["model"] = model

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        raw_text = response.text
        response.raise_for_status()
        result = response.json()
        content = (result.get("choices") or [{}])[0].get("message", {}).get("content", "")
        return {"success": True, "response": result, "reply": content}
    except requests.exceptions.RequestException as e:
        return {"success": False, "response": None, "error": str(e)}
    except json.JSONDecodeError as e:
        preview = (raw_text[:500] + "...") if len(raw_text) > 500 else raw_text
        err_detail = f"Invalid JSON: {e}. Response (status={response.status_code}): {repr(preview)}"
        return {"success": False, "response": None, "error": err_detail}


def test_image_generation(
    url: str,
    api_key: str,
    model: str = "dall-e-3",
    provider: str = "openai",
    prompt: str = "A simple red circle on white background",
) -> dict:
    """
    Test if the API key can generate images (DALL-E / image models).

    Args:
        url: Image generations endpoint URL
        api_key: API key for authorization
        model: Model/deployment name (used for OpenAI; Azure uses deployment in URL)
        provider: openai | groq | azure

    Returns:
        dict with keys: success (bool), response (dict or None), error (str or None)
    """
    if provider == "groq":
        return {"success": False, "response": None, "error": "Groq does not support image generation"}

    if provider == "azure":
        headers = {"Content-Type": "application/json", "api-key": api_key}
    else:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    payload = {"prompt": prompt, "n": 1}
    if provider != "azure":
        payload["model"] = model

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        data = result.get("data") or []
        if data and len(data) > 0:
            return {"success": True, "response": result, "image_count": len(data)}
        return {"success": True, "response": result, "image_count": 0}
    except requests.exceptions.RequestException as e:
        return {"success": False, "response": None, "error": str(e)}
    except json.JSONDecodeError as e:
        return {"success": False, "response": None, "error": f"Invalid JSON: {e}"}


def _is_azure_flux_bfl_url(url: str) -> bool:
    """Azure AI Foundry Black Forest Labs FLUX (native BFL API, not OpenAI images)."""
    u = url.lower()
    return "blackforestlabs" in u or "flux-2-pro" in u or "flux-pro-1.1" in u or "flux-kontext" in u


def test_flux_image_generation(
    url: str,
    api_key: str,
    model: str = "FLUX.2-pro",
    prompt: str = "A simple red circle on white background",
    width: int = 1024,
    height: int = 1024,
    output_format: str = "png",
) -> dict:
    """
    Azure AI Foundry FLUX image generation (BFL native JSON body).
    See: providers/blackforestlabs/v1/flux-2-pro?api-version=preview
    """
    headers = {"Content-Type": "application/json", "api-key": api_key}
    payload = {
        "prompt": prompt,
        "n": 1,
        "width": width,
        "height": height,
        "output_format": output_format,
        "model": model,
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        data = result.get("data") or []
        if data and len(data) > 0:
            return {"success": True, "response": result, "image_count": len(data)}
        return {"success": True, "response": result, "image_count": 0}
    except requests.exceptions.RequestException as e:
        return {"success": False, "response": None, "error": str(e)}
    except json.JSONDecodeError as e:
        return {"success": False, "response": None, "error": f"Invalid JSON: {e}"}


def _build_azure_chat_url(base_url: str, deployment: str) -> str:
    """Build full Azure chat completions URL from base endpoint and deployment."""
    base = base_url.rstrip("/")
    if "/chat/completions" in base or "/openai/" in base:
        return base_url
    api_version = "2024-06-01"
    return f"{base}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"


def _get_image_url(
    chat_url: str, provider: str, image_deployment: Optional[str] = None
) -> str:
    """
    Derive image generations URL from chat URL.
    For Azure: use image_deployment (e.g. dall-e-3, gpt-image-1) - must be different from chat deployment.
    """
    if provider == "azure":
        url = chat_url.replace("chat/completions", "images/generations")
        if image_deployment:
            url = re.sub(
                r"/deployments/[^/]+/images/generations",
                f"/deployments/{image_deployment}/images/generations",
                url,
            )
        return url
    if provider == "openai":
        return "https://api.openai.com/v1/images/generations"
    return ""


def save_generated_images(result: dict, output_dir: str) -> List[str]:
    """
    Parse image generation API response and save images to output_dir.
    Handles both url and b64_json formats (OpenAI/Azure compatible).

    Returns:
        List of saved file paths.
    """
    data = result.get("data") or []
    saved_paths = []
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        if "b64_json" in item:
            try:
                raw = base64.b64decode(item["b64_json"])
                ext = "png"
                path = os.path.join(output_dir, f"generated_{timestamp}_{i}.{ext}")
                with open(path, "wb") as f:
                    f.write(raw)
                saved_paths.append(path)
            except (ValueError, OSError) as e:
                print(f"  Failed to save image {i} (base64): {e}")
        elif "url" in item:
            try:
                resp = requests.get(item["url"], timeout=30)
                resp.raise_for_status()
                ext = "png"
                path = os.path.join(output_dir, f"generated_{timestamp}_{i}.{ext}")
                with open(path, "wb") as f:
                    f.write(resp.content)
                saved_paths.append(path)
            except requests.exceptions.RequestException as e:
                print(f"  Failed to download image {i} from URL: {e}")

    return saved_paths


def main():
    parser = argparse.ArgumentParser(
        description="Test API endpoint with URL and key (default: Azure)"
    )
    parser.add_argument(
        "--url",
        default=(
            os.getenv("AZURE_OPENAI_ENDPOINT")
            or os.getenv("AZURE_OPENAI_CHAT_URL")
            or os.getenv("LLM_API_URL")
            or os.getenv("API_URL")
        ),
        help="API URL (default: AZURE_OPENAI_ENDPOINT from .env)",
    )
    parser.add_argument(
        "--key",
        default=(
            os.getenv("AZURE_OPENAI_API_KEY")
            or os.getenv("LLM_API_KEY")
            or os.getenv("API_KEY")
            or os.getenv("GROQ_API_KEY")
        ),
        help="API key (default: AZURE_OPENAI_API_KEY from .env)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        help="Model name for OpenAI/Groq (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--deployment",
        default=(
            os.getenv("AZURE_OPENAI_DEPLOYMENT")
            or os.getenv("AZURE_DEPLOYMENT")
            or "gpt-4o"
        ),
        help="Azure chat deployment (default: gpt-4o or AZURE_OPENAI_DEPLOYMENT)",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: OK",
        help="Text prompt to send for chat test",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "groq", "azure"],
        default=os.getenv("LLM_PROVIDER", "azure"),
        help="Provider preset (default: azure)",
    )
    parser.add_argument(
        "--test-image",
        action="store_true",
        help="Also test if the key can generate images (DALL-E / FLUX / image models)",
    )
    parser.add_argument(
        "--skip-chat",
        action="store_true",
        help="Skip chat test (use with --test-image for image-only, e.g. FLUX)",
    )
    parser.add_argument(
        "--image-url",
        default=(
            os.getenv("AZURE_FLUX_IMAGE_URL")
            or os.getenv("AZURE_OPENAI_IMAGES_URL")
            or os.getenv("IMAGE_API_URL")
        ),
        help="Image endpoint: OpenAI DALL-E, Azure images, or FLUX (AZURE_FLUX_IMAGE_URL)",
    )
    parser.add_argument(
        "--image-model",
        default=os.getenv("IMAGE_MODEL", "dall-e-3"),
        help="Image model for OpenAI/Azure DALL-E (default: dall-e-3)",
    )
    parser.add_argument(
        "--flux-model",
        default=os.getenv("AZURE_FLUX_MODEL", "FLUX.2-pro"),
        help="FLUX model name in request body for Azure BFL (default: FLUX.2-pro)",
    )
    parser.add_argument(
        "--image-prompt",
        default=os.getenv("IMAGE_PROMPT", "A simple red circle on white background"),
        help="Prompt for image generation test",
    )
    parser.add_argument(
        "--flux-width",
        type=int,
        default=int(os.getenv("FLUX_WIDTH", "1024")),
        help="FLUX image width (default: 1024)",
    )
    parser.add_argument(
        "--flux-height",
        type=int,
        default=int(os.getenv("FLUX_HEIGHT", "1024")),
        help="FLUX image height (default: 1024)",
    )
    parser.add_argument(
        "--image-deployment",
        default=(
            os.getenv("AZURE_OPENAI_IMAGES_DEPLOYMENT")
            or os.getenv("AZURE_IMAGES_DEPLOYMENT")
            or "dall-e-3"
        ),
        help="Azure image deployment (default: dall-e-3 or AZURE_OPENAI_IMAGES_DEPLOYMENT)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("IMAGE_OUTPUT_DIR", "outputs/generated_images"),
        help="Directory to save generated images (default: outputs/generated_images)",
    )
    args = parser.parse_args()

    url = args.url
    key = args.key

    # Provider-aware defaults override the argparse defaults, which may point to Azure
    # when AZURE_* env vars are set (even if --provider is groq/openai).
    if args.provider == "groq":
        if not args.url or args.url == os.getenv("AZURE_OPENAI_ENDPOINT") or "cognitiveservices" in (args.url or ""):
            url = "https://api.groq.com/openai/v1/chat/completions"
        # Ignore AZURE_OPENAI_API_KEY for Groq requests.
        key = os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("API_KEY")
    elif args.provider == "openai":
        if not args.url or "cognitiveservices" in (args.url or ""):
            url = "https://api.openai.com/v1/chat/completions"
        key = os.getenv("LLM_API_KEY") or os.getenv("API_KEY")
    elif args.provider == "azure":
        # Keep existing azure defaults
        key = args.key or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("API_KEY")

    if not url and args.provider == "groq":
        url = "https://api.groq.com/openai/v1/chat/completions"
    if not url and args.provider == "openai":
        url = "https://api.openai.com/v1/chat/completions"
    if not url and args.provider == "azure":
        url = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_CHAT_URL")

    if url and args.provider == "azure" and "/chat/completions" not in url:
        url = _build_azure_chat_url(url, args.deployment)

    image_only = args.skip_chat and args.test_image
    if not url and not (image_only and args.image_url):
        env_hint = "API_URL" if args.provider != "azure" else "AZURE_OPENAI_ENDPOINT"
        print(f"Error: No URL provided. Use --url or set {env_hint} in .env")
        sys.exit(1)
    if not key:
        print("Error: No API key provided. Use --key or set AZURE_OPENAI_API_KEY in .env")
        sys.exit(1)

    if args.skip_chat and not args.test_image:
        print("Error: --skip-chat requires --test-image")
        sys.exit(1)

    print(f"Provider: {args.provider}")
    if not args.skip_chat:
        print(f"Testing API: {url}")
        if args.provider != "azure":
            print(f"Model: {args.model}")
        print(f"Prompt: {args.prompt[:60]}{'...' if len(args.prompt) > 60 else ''}")
    print("-" * 50)

    if not args.skip_chat:
        result = test_api(url, key, args.model, args.provider, args.prompt)
        if result["success"]:
            print("Chat: SUCCESS - API responded correctly")
            if result.get("reply"):
                print(f"Reply: {result['reply']}")
        else:
            print("Chat: FAILED -", result.get("error", "Unknown error"))
            sys.exit(1)
    else:
        print("Chat: skipped (--skip-chat)")

    if args.test_image:
        print("-" * 50)
        image_url = args.image_url or _get_image_url(
            url, args.provider, args.image_deployment
        )
        if not image_url:
            print("Image: SKIP - No image URL (set --image-url or AZURE_FLUX_IMAGE_URL / IMAGE_API_URL)")
        else:
            print(f"Testing image generation: {image_url}")
            img_result = None
            if _is_azure_flux_bfl_url(image_url):
                print(f"Mode: Azure FLUX (BFL) model={args.flux_model}")
                img_result = test_flux_image_generation(
                    image_url,
                    key,
                    model=args.flux_model,
                    prompt=args.image_prompt,
                    width=args.flux_width,
                    height=args.flux_height,
                )
            else:
                img_result = test_image_generation(
                    image_url,
                    key,
                    args.image_model,
                    args.provider,
                    prompt=args.image_prompt,
                )
            if img_result["success"]:
                count = img_result.get("image_count", 0)
                print(f"Image: SUCCESS - Key can generate images (got {count} image(s))")
                if count > 0 and img_result.get("response"):
                    saved = save_generated_images(img_result["response"], args.output_dir)
                    if saved:
                        print(f"Saved to: {args.output_dir}")
                        for p in saved:
                            print(f"  - {p}")
            else:
                print(f"Image: FAILED - Key cannot generate images: {img_result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
