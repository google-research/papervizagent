# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utility functions for interacting with Gemini and Claude APIs, image processing, and PDF handling.
"""

import json
import asyncio
import base64
from io import BytesIO
from functools import partial
from ast import literal_eval
from typing import List, Dict, Any

import aiofiles
from PIL import Image
from google import genai
from google.genai import types
from anthropic import AsyncAnthropicVertex
from openai import AsyncOpenAI

import os

import yaml
from pathlib import Path

# Load config
config_path = Path(__file__).parent.parent / "configs" / "model_config.yaml"
model_config = {}
if config_path.exists():
    with open(config_path, "r") as f:
        model_config = yaml.safe_load(f) or {}

def get_config_val(section, key, env_var, default=""):
    val = os.getenv(env_var)
    if not val and section in model_config:
        val = model_config[section].get(key)
    return val or default


def get_openrouter_base_url():
    return get_config_val("openrouter", "base_url", "OPENAI_BASE_URL", "")


def get_openrouter_api_key():
    return (
        get_config_val("api_keys", "openrouter_api_key", "OPENROUTER_API_KEY", "")
        or get_config_val("openrouter", "api_key", "OPENROUTER_API_KEY", "")
    )


def get_openai_api_key():
    return get_config_val("api_keys", "openai_api_key", "OPENAI_API_KEY", "")


def is_openrouter_configured():
    return bool(get_openrouter_api_key() and get_openrouter_base_url())


def should_use_openrouter_backend(model_name: str) -> bool:
    """Route through OpenRouter when configured and model uses an OpenRouter slug."""
    if not is_openrouter_configured():
        return False
    if "/" in model_name:
        return True
    return gemini_client is None


def get_openrouter_headers():
    headers = {}
    site_url = get_config_val("openrouter", "site_url", "OPENROUTER_SITE_URL", "")
    site_name = get_config_val("openrouter", "site_name", "OPENROUTER_SITE_NAME", "")
    if site_url:
        headers["HTTP-Referer"] = site_url
    if site_name:
        headers["X-OpenRouter-Title"] = site_name
    return headers


project_id = ""
gemini_client = None

# Initialize Gemini only when Google credentials are available.
try:
    import google.auth

    creds, _ = google.auth.default()
    if not hasattr(creds, "service_account_email"):
        print(f"DEBUG: Running with credentials: {type(creds)}")
    project_id = get_config_val("google_cloud", "project_id", "GOOGLE_CLOUD_PROJECT", "")
    location = get_config_val("google_cloud", "location", "GOOGLE_CLOUD_LOCATION", "global")
    print(f"DEBUG: Initialized Gemini Client with Project: {project_id}, Location: {location}")
    gemini_client = genai.Client(vertexai=True, project=project_id, location=location)
except Exception:
    api_key = get_config_val("api_keys", "google_api_key", "GOOGLE_API_KEY", "")
    if api_key:
        gemini_client = genai.Client(api_key=api_key)
        print("Initialized Gemini Client with API Key")
    else:
        print("Warning: Could not initialize Gemini Client. Missing credentials.")

anthropic_project_id = get_config_val(
    "anthropic", "project_id", "ANTHROPIC_PROJECT_ID", project_id
)
anthropic_region = get_config_val("anthropic", "region", "ANTHROPIC_REGION", "us-central1")
try:
    anthropic_client = AsyncAnthropicVertex(
        region=anthropic_region, project_id=anthropic_project_id
    )
except Exception as e:
    print(f"Warning: Could not initialize Anthropic Vertex Client: {e}")
    anthropic_client = None

openai_client = None
try:
    openrouter_key = get_openrouter_api_key()
    openai_key = get_openai_api_key()
    api_key = openrouter_key or openai_key
    if api_key:
        client_kwargs = {"api_key": api_key}
        base_url = get_openrouter_base_url()
        if openrouter_key and base_url:
            client_kwargs["base_url"] = base_url
            headers = get_openrouter_headers()
            if headers:
                client_kwargs["default_headers"] = headers
            print(f"Initialized OpenRouter client at {base_url}")
        elif base_url:
            client_kwargs["base_url"] = base_url
            print(f"Initialized OpenAI-compatible client at {base_url}")
        else:
            print("Initialized OpenAI client")
        openai_client = AsyncOpenAI(**client_kwargs)
except Exception as e:
    print(f"Warning: Could not initialize OpenAI Client: {e}")
    openai_client = None



def _convert_to_gemini_parts(contents: List[Dict[str, Any]]) -> List[types.Part]:
    """
    Convert a generic content list to a list of Gemini's genai.types.Part objects.
    """
    gemini_parts = []
    for item in contents:
        if item.get("type") == "text":
            gemini_parts.append(types.Part.from_text(text=item["text"]))
        elif item.get("type") == "image":
            if item.get("image_base64"):
                gemini_parts.append(
                    types.Part.from_bytes(
                        data=base64.b64decode(item["image_base64"]),
                        mime_type="image/jpeg",
                    )
                )
                continue
            source = item.get("source", {})
            if source.get("type") == "base64":
                gemini_parts.append(
                    types.Part.from_bytes(
                        data=base64.b64decode(source["data"]),
                        mime_type=source["media_type"],
                    )
                )
    return gemini_parts


def _convert_to_openai_format(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts the generic content list (Claude format) to OpenAI's API format.
    """
    openai_contents = []
    for item in contents:
        if item.get("type") == "text":
            openai_contents.append({"type": "text", "text": item["text"]})
        elif item.get("type") == "image":
            if item.get("image_base64"):
                data_url = f"data:image/jpeg;base64,{item['image_base64']}"
                openai_contents.append(
                    {"type": "image_url", "image_url": {"url": data_url}}
                )
                continue
            source = item.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/jpeg")
                data = source.get("data", "")
                data_url = f"data:{media_type};base64,{data}"
                openai_contents.append(
                    {"type": "image_url", "image_url": {"url": data_url}}
                )
    return openai_contents


def _extract_image_b64_from_openrouter_message(message) -> str | None:
    images = getattr(message, "images", None)
    if images:
        for image in images:
            image_url = getattr(image, "image_url", None)
            url = getattr(image_url, "url", None) if image_url else None
            if isinstance(image, dict):
                url = image.get("image_url", {}).get("url")
            if url and url.startswith("data:"):
                return url.split(",", 1)[1]

    content = getattr(message, "content", None)
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    return url.split(",", 1)[1]
    return None


def _gemini_config_to_openrouter_kwargs(config, model_name: str) -> Dict[str, Any]:
    system_prompt = getattr(config, "system_instruction", "") or ""
    temperature = getattr(config, "temperature", 1.0)
    max_output_tokens = getattr(config, "max_output_tokens", 50000)
    response_modalities = getattr(config, "response_modalities", None) or []
    image_config = getattr(config, "image_config", None)

    kwargs = {
        "system_prompt": system_prompt,
        "temperature": temperature,
        "max_completion_tokens": max_output_tokens,
    }

    is_image_request = (
        "image" in model_name
        or "nanoviz" in model_name
        or "IMAGE" in response_modalities
    )
    if is_image_request:
        extra_body = {"modalities": ["image", "text"]}
        if image_config is not None:
            image_cfg = build_openrouter_image_config(
                aspect_ratio=getattr(image_config, "aspect_ratio", None),
                image_size=getattr(image_config, "image_size", None),
            )
            if image_cfg:
                extra_body["image_config"] = image_cfg
        kwargs["extra_body"] = extra_body
        kwargs["image_generation"] = True

    return kwargs


async def _call_openrouter_chat_with_retry_async(
    model_name,
    contents,
    system_prompt="",
    temperature=1.0,
    max_completion_tokens=50000,
    candidate_count=1,
    extra_body=None,
    image_generation=False,
    max_attempts=5,
    retry_delay=5,
    error_context="",
):
    if openai_client is None:
        raise RuntimeError("OpenAI-compatible client is not configured.")

    result_list = []
    openai_contents = _convert_to_openai_format(contents)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": openai_contents})

    request_kwargs = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
    }
    if extra_body:
        request_kwargs["extra_body"] = extra_body
    headers = get_openrouter_headers()
    if headers:
        request_kwargs["extra_headers"] = headers

    for attempt in range(max_attempts):
        try:
            response = await openai_client.chat.completions.create(**request_kwargs)
            message = response.choices[0].message

            if image_generation:
                image_b64 = _extract_image_b64_from_openrouter_message(message)
                if image_b64:
                    result_list.append(image_b64)
                else:
                    raise ValueError("No image data returned from OpenRouter response")
            else:
                content = message.content
                if not content:
                    raise ValueError("Empty text response from OpenRouter")
                result_list.append(content)

            while len(result_list) < candidate_count:
                follow_up = await openai_client.chat.completions.create(**request_kwargs)
                follow_message = follow_up.choices[0].message
                if image_generation:
                    image_b64 = _extract_image_b64_from_openrouter_message(follow_message)
                    if image_b64:
                        result_list.append(image_b64)
                elif follow_message.content:
                    result_list.append(follow_message.content)

            return result_list[:candidate_count]

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            current_delay = min(retry_delay * (2 ** attempt), 30)
            print(
                f"Attempt {attempt + 1} for OpenRouter model {model_name} failed{context_msg}: {e}. "
                f"Retrying in {current_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"Error: All {max_attempts} OpenRouter attempts failed{context_msg}")
                return ["Error"] * candidate_count

    return ["Error"] * candidate_count


async def call_gemini_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=5, error_context=""
):
    """
    ASYNC: Call Gemini API with asynchronous retry logic.
    Falls back to OpenRouter via the OpenAI-compatible client when configured.
    """
    if should_use_openrouter_backend(model_name):
        openrouter_kwargs = _gemini_config_to_openrouter_kwargs(config, model_name)
        return await _call_openrouter_chat_with_retry_async(
            model_name=model_name,
            contents=contents,
            candidate_count=getattr(config, "candidate_count", 1),
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
            **openrouter_kwargs,
        )

    result_list = []
    target_candidate_count = config.candidate_count
    # Gemini API max candidate count is 8. We will call multiple times if needed.
    if config.candidate_count > 8:
        config.candidate_count = 8

    current_contents = contents
    for attempt in range(max_attempts):
        try:
            # Use global client
            client = gemini_client
            
            # Convert generic content list to Gemini's format right before the API call
            gemini_contents = _convert_to_gemini_parts(current_contents)
            response = await client.aio.models.generate_content(
                model=model_name, contents=gemini_contents, config=config
            )

            # If we are using Image Generation models to generate images
            if (
                "nanoviz" in model_name
                or "image" in model_name
            ):
                raw_response_list = []
                if not response.candidates or not response.candidates[0].content.parts:
                    print(
                        f"[Warning]: Failed to generate image, retrying in {retry_delay} seconds..."
                    )
                    await asyncio.sleep(retry_delay)
                    continue

                # In this mode, we can only have one candidate
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        # Append base64 encoded image data to raw_response_list
                        raw_response_list.append(
                            base64.b64encode(part.inline_data.data).decode("utf-8")
                        )
                        break

            # Otherwise, for text generation models
            else:
                raw_response_list = [
                    part.text
                    for candidate in response.candidates
                    for part in candidate.content.parts
                ]
            result_list.extend([r for r in raw_response_list if r.strip() != ""])
            if len(result_list) >= target_candidate_count:
                result_list = result_list[:target_candidate_count]
                break

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            
            # Exponential backoff (capped at 30s)
            current_delay = min(retry_delay * (2 ** attempt), 30)
            
            print(
                f"Attempt {attempt + 1} for model {model_name} failed{context_msg}: {e}. Retrying in {current_delay} seconds..."
            )

            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                result_list = ["Error"] * target_candidate_count

    if len(result_list) < target_candidate_count:
        result_list.extend(["Error"] * (target_candidate_count - len(result_list)))
    return result_list

def _convert_to_claude_format(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts the generic content list to Claude's API format.
    Currently, the formats are identical, so this acts as a pass-through
    for architectural consistency and future-proofing.

    Claude API's format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}},
        ...
    ]
    """
    return contents


async def call_claude_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call Claude API with asynchronous retry logic.
    This version efficiently handles input size errors by validating and modifying
    the content list once before generating all candidates.
    """
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_output_tokens = config["max_output_tokens"]
    response_text_list = []

    # --- Preparation Phase ---
    # Convert to the Claude-specific format and perform an initial optimistic resize.
    current_contents = contents

    # --- Validation and Remediation Phase ---
    # We loop until we get a single successful response, proving the input is valid.
    # Note that this check is required because Claude only has 128k / 256k context windows.
    # For Gemini series that support 1M, we do not need this step.
    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            claude_contents = _convert_to_claude_format(current_contents)
            # Attempt to generate the very first candidate.
            first_response = await anthropic_client.messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": claude_contents}],
                system=system_prompt,
            )
            response_text_list.append(first_response.content[0].text)
            is_input_valid = True
            break

        except Exception as e:
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Validation attempt {attempt + 1} failed{context_msg}: {error_str}. Retrying in {retry_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    # --- Sampling Phase ---
    if not is_input_valid:
        print(
            f"Error: All {max_attempts} attempts failed to validate the input{context_msg}. Returning errors."
        )
        return ["Error"] * candidate_num

    # We already have 1 successful candidate, now generate the rest.
    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        print(
            f"Input validated. Now generating remaining {remaining_candidates} candidates..."
        )
        valid_claude_contents = _convert_to_claude_format(current_contents)
        tasks = [
            anthropic_client.messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": valid_claude_contents}
                ],
                system=system_prompt,
            )
            for _ in range(remaining_candidates)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                print(f"Error generating a subsequent candidate: {res}")
                response_text_list.append("Error")
            else:
                response_text_list.append(res.content[0].text)

    return response_text_list

async def call_openai_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call OpenAI-compatible API with asynchronous retry logic.
    """
    return await _call_openrouter_chat_with_retry_async(
        model_name=model_name,
        contents=contents,
        system_prompt=config["system_prompt"],
        temperature=config["temperature"],
        max_completion_tokens=config["max_completion_tokens"],
        candidate_count=config["candidate_num"],
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        error_context=error_context,
    )


# Supported by Gemini image models via OpenRouter (Google AI Studio)
SUPPORTED_ASPECT_RATIOS = [
    "1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5", "5:4",
    "8:1", "9:16", "16:9", "21:9",
]
ASPECT_RATIO_ALIASES = {"32:9": "21:9"}


def coerce_aspect_ratio_value(aspect_ratio) -> str:
    if aspect_ratio is None:
        return ""
    if hasattr(aspect_ratio, "value"):
        return str(aspect_ratio.value).strip()
    return str(aspect_ratio).strip()


def normalize_aspect_ratio(aspect_ratio, fallback: str = "16:9") -> str:
    ratio = coerce_aspect_ratio_value(aspect_ratio)
    if not ratio:
        return fallback
    if ratio in SUPPORTED_ASPECT_RATIOS:
        return ratio
    if ratio in ASPECT_RATIO_ALIASES:
        mapped = ASPECT_RATIO_ALIASES[ratio]
        print(f"Note: aspect ratio {ratio} mapped to supported {mapped}")
        return mapped
    print(f"Warning: aspect ratio {ratio} not supported, using {fallback}")
    return fallback


def build_openrouter_image_config(aspect_ratio=None, image_size=None) -> dict:
    """Build OpenRouter image_config with only Gemini-supported fields."""
    image_cfg = {}
    if aspect_ratio:
        image_cfg["aspect_ratio"] = normalize_aspect_ratio(aspect_ratio)
    # OpenRouter Gemini image models reject lowercase sizes like "1k".
    if image_size:
        size = str(image_size).strip().upper()
        if size in {"1K", "2K", "4K"}:
            image_cfg["image_size"] = size
    return image_cfg


async def call_openrouter_image_generation_with_retry_async(
    model_name,
    contents,
    system_prompt="",
    temperature=1.0,
    max_completion_tokens=50000,
    aspect_ratio=None,
    image_size=None,
    max_attempts=5,
    retry_delay=30,
    error_context="",
):
    extra_body = {"modalities": ["image", "text"]}
    image_cfg = build_openrouter_image_config(
        aspect_ratio=aspect_ratio,
        image_size=image_size,
    )
    if image_cfg:
        extra_body["image_config"] = image_cfg

    return await _call_openrouter_chat_with_retry_async(
        model_name=model_name,
        contents=contents,
        system_prompt=system_prompt,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        candidate_count=1,
        extra_body=extra_body,
        image_generation=True,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        error_context=error_context,
    )


async def call_openai_image_generation_with_retry_async(
    model_name, prompt, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call OpenAI Image Generation API (GPT-Image) with asynchronous retry logic.
    Falls back to OpenRouter chat-based image generation when configured.
    """
    if is_openrouter_configured() or should_use_openrouter_backend(model_name):
        contents = [{"type": "text", "text": prompt}]
        return await call_openrouter_image_generation_with_retry_async(
            model_name=model_name,
            contents=contents,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )

    size = config.get("size", "1536x1024")
    quality = config.get("quality", "high")
    background = config.get("background", "opaque")
    output_format = config.get("output_format", "png")
    
    # Base parameters for all models
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "n": 1,
        "size": size,
    }
    
    # Add GPT-Image specific parameters
    gen_params.update({
        "quality": quality,
        "background": background,
        "output_format": output_format,
    })

    for attempt in range(max_attempts):
        try:
            response = await openai_client.images.generate(**gen_params)
            
            # OpenAI images.generate returns a list of images in response.data
            if response.data and response.data[0].b64_json:
                return [response.data[0].b64_json]
            else:
                print(f"[Warning]: Failed to generate image via OpenAI, no data returned.")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
                continue

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Attempt {attempt + 1} for OpenAI image generation model {model_name} failed{context_msg}: {e}. Retrying in {retry_delay} seconds..."
            )

            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                return ["Error"]

    return ["Error"]
