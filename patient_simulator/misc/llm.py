from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union, List, TypedDict
import asyncio
import copy
import hashlib
import json
import re
from pathlib import Path
import gc
import torch
import logging
import warnings
import sys
import os
import subprocess
import threading
import httpx
from contextlib import contextmanager, nullcontext
from google import genai
from google.genai import types
from anthropic import AsyncAnthropicVertex
from anthropic import RateLimitError as AnthropicRateLimitError
from anthropic import InternalServerError as AnthropicInternalServerError
from anthropic import APIConnectionError as AnthropicConnectionError
from anthropic import APITimeoutError as AnthropicTimeoutError
import pydantic
import outlines
from vllm import LLM as VLLMEngine, SamplingParams
from google.api_core import exceptions as google_exceptions


@contextmanager
def _suppress_stderr():
    """Context manager to suppress stderr output."""
    stderr = sys.stderr
    try:
        sys.stderr = open(os.devnull, "w")
        yield
    finally:
        sys.stderr.close()
        sys.stderr = stderr


class ConversationMessage(TypedDict):
    """Type definition for a conversation message."""

    role: str
    content: str


PromptType = Union[str, List[ConversationMessage]]


def _estimate_tokens(prompt: PromptType) -> int:
    """Word-count based estimator for evaluation."""
    if not prompt:
        return 0

    if isinstance(prompt, str):
        return len(prompt.split())
    elif isinstance(prompt, list):
        total = 0
        for message in prompt:
            if isinstance(message, dict) and "content" in message:
                total += len(message["content"].split())
        return total

    return 0


class LLM(ABC):
    """Abstract base class for LLM implementations."""

    _response_cache: Dict[str, dict] = {}
    _cache_lock = threading.RLock()

    def _model_cache_id(self) -> str:
        return str(getattr(self, "model", self.__class__.__name__))

    @staticmethod
    def _outline_class_id(
        outlines_class: Optional[type[pydantic.BaseModel]],
    ) -> Optional[str]:
        if outlines_class is None:
            return None
        return f"{outlines_class.__module__}.{outlines_class.__qualname__}"

    @staticmethod
    def _normalize_prompt_for_cache(prompt: PromptType) -> Any:
        if isinstance(prompt, str):
            return prompt
        return [
            {
                "role": str(message.get("role", "")),
                "content": str(message.get("content", "")),
            }
            for message in prompt
        ]

    @classmethod
    def _make_cache_key(
        cls,
        model_id: str,
        prompt: PromptType,
        system_instruction: Optional[str],
        outlines_class: Optional[type[pydantic.BaseModel]],
        extra_cache_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload = {
            "model": model_id,
            "system_instruction": system_instruction or "",
            "prompt": cls._normalize_prompt_for_cache(prompt),
            "outlines_class": cls._outline_class_id(outlines_class),
            "extra": extra_cache_context or {},
        }
        serialized = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _cache_lookup(
        self,
        prompt: PromptType,
        system_instruction: Optional[str],
        outlines_class: Optional[type[pydantic.BaseModel]],
        extra_cache_context: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, Optional[dict]]:
        cache_key = self._make_cache_key(
            model_id=self._model_cache_id(),
            prompt=prompt,
            system_instruction=system_instruction,
            outlines_class=outlines_class,
            extra_cache_context=extra_cache_context,
        )
        if not getattr(self, "use_cache", True):
            return cache_key, None
        with self._cache_lock:
            cached = self._response_cache.get(cache_key)
        return cache_key, copy.deepcopy(cached) if cached is not None else None

    def _cache_store(self, cache_key: str, response: dict) -> None:
        with self._cache_lock:
            self._response_cache[cache_key] = copy.deepcopy(response)

    @abstractmethod
    async def generate_response(
        self,
        prompt: PromptType,
        system_instruction: Optional[str] = None,
        outlines_class: Optional[type[pydantic.BaseModel]] = None,
        use_cache: bool = True,
    ) -> dict:
        """Generate a response from the LLM.

        Args:
            prompt: Either a string or a list of conversation messages
            system_instruction: Optional system instruction for the model
            outlines_class: Pydantic model class for structured output schema

        Returns:
            Dictionary containing:
                - response: The generated text response
                - prompt_token_count: Estimated input token count
                - output_token_count: Estimated output token count
        """
        pass


class VLLMModelConflictError(RuntimeError):
    pass


class APILLM(LLM):
    """LLM implementation using Google Gemini API."""

    _RETRYABLE_GOOGLE_EXCEPTIONS = (
        google_exceptions.Aborted,
        google_exceptions.DeadlineExceeded,
        google_exceptions.InternalServerError,
        google_exceptions.ResourceExhausted,
        google_exceptions.ServiceUnavailable,
        google_exceptions.TooManyRequests,
    )

    _RETRYABLE_TRANSPORT_EXCEPTIONS = (
        httpx.NetworkError,
        httpx.TimeoutException,
    )

    _RETRYABLE_ANTHROPIC_EXCEPTIONS = (
        AnthropicRateLimitError,
        AnthropicInternalServerError,
        AnthropicConnectionError,
        AnthropicTimeoutError,
    )

    @staticmethod
    def _normalize_generate_config(config_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(config_kwargs)

        if "stop" in normalized and "stop_sequences" not in normalized:
            normalized["stop_sequences"] = normalized.pop("stop")

        # Compatibility alias: internal callers often use OpenAI/vLLM-style max_tokens.
        # Gemini expects max_output_tokens in GenerateContentConfig.
        if "max_tokens" in normalized and "max_output_tokens" not in normalized:
            normalized["max_output_tokens"] = normalized.pop("max_tokens")

        safety_settings = normalized.get("safety_settings")
        if isinstance(safety_settings, list):
            fixed_safety_settings = []
            for setting in safety_settings:
                if isinstance(setting, dict) and setting.get("threshold") is False:
                    fixed_setting = dict(setting)
                    fixed_setting["threshold"] = "OFF"
                    fixed_safety_settings.append(fixed_setting)
                    continue
                fixed_safety_settings.append(setting)
            normalized["safety_settings"] = fixed_safety_settings

        return normalized

    def __init__(
        self,
        model: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        vertexai: bool = True,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        max_retries: int = 5,
        initial_delay: float = 2.0,
        verbosity: int = 0,
        use_cache: bool = True,
    ):
        self.model = model
        self.use_cache = use_cache
        self.project = project
        self.location = location
        self.vertexai = vertexai
        self.api_key = api_key or None
        self.api_version = api_version
        self.generation_config = dict(generation_config or {})
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.verbosity = verbosity
        self._use_anthropic = model.startswith("claude-")
        if self._use_anthropic:
            self.client = AsyncAnthropicVertex(
                region=location or "global",
                project_id=project,
            )
        else:
            client_kwargs: Dict[str, Any] = {"vertexai": vertexai}
            if api_version is not None:
                client_kwargs["http_options"] = types.HttpOptions(
                    api_version=api_version
                )
            if project is not None:
                client_kwargs["project"] = project
            if location is not None:
                client_kwargs["location"] = location
            if self.api_key is not None:
                client_kwargs["api_key"] = self.api_key
            self.client = genai.Client(**client_kwargs)

    async def generate_response(
        self,
        prompt: PromptType,
        system_instruction: Optional[str] = None,
        outlines_class: Optional[type[pydantic.BaseModel]] = None,
        sampling_kwargs: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> dict:
        """Generate a response using Google Gemini API with retry logic."""

        if sampling_kwargs and self.verbosity >= 1:
            print(
                "NOTE: sampling_kwargs are not natively supported by Google Gemini API and will be ignored in this implementation."
            )

        cache_key: Optional[str] = None
        if use_cache:
            cache_key, cached = self._cache_lookup(
                prompt=prompt,
                system_instruction=system_instruction,
                outlines_class=outlines_class,
            )
            if cached is not None:
                return cached

        if self._use_anthropic:
            messages = (
                [{"role": m["role"], "content": m["content"]} for m in prompt]
                if isinstance(prompt, list)
                else [{"role": "user", "content": prompt}]
            )
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                **self.generation_config,
            }
            if "max_tokens" not in kwargs:
                raise ValueError(
                    "generation_config must include 'max_tokens' for Claude models"
                )
            if system_instruction is not None:
                kwargs["system"] = system_instruction
            if outlines_class is not None:
                kwargs["tools"] = [
                    {
                        "name": "structured_response",
                        "description": "Return the response in the required structured format.",
                        "input_schema": outlines_class.model_json_schema(),
                    }
                ]
                kwargs["tool_choice"] = {"type": "tool", "name": "structured_response"}
            for attempt in range(self.max_retries):
                try:
                    response = await self.client.messages.create(**kwargs)
                    if outlines_class is not None:
                        tool_block = next(
                            b for b in response.content if b.type == "tool_use"
                        )
                        response_value = outlines_class.model_validate(tool_block.input)
                    else:
                        response_value = response.content[0].text
                    result = {
                        "response": response_value,
                        "prompt_token_count": response.usage.input_tokens,
                        "output_token_count": response.usage.output_tokens,
                    }
                    if use_cache and cache_key is not None:
                        self._cache_store(cache_key, result)
                    return result
                except self._RETRYABLE_ANTHROPIC_EXCEPTIONS as e:
                    if attempt == self.max_retries - 1:
                        raise
                    delay = self.initial_delay * (2**attempt)
                    print(
                        f"Transient error {type(e).__name__}. Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})..."
                    )
                    await asyncio.sleep(delay)
            raise RuntimeError("Max retries exceeded")

        if isinstance(prompt, list):
            contents = [
                types.Content(
                    role=item["role"],
                    parts=[types.Part.from_text(text=item["content"])],
                )
                for item in prompt
            ]
        else:
            contents = prompt

        config_kwargs: Dict[str, Any] = {
            **self.generation_config,
        }
        if system_instruction is not None:
            config_kwargs["system_instruction"] = system_instruction
        if "automatic_function_calling" not in config_kwargs:
            config_kwargs["automatic_function_calling"] = (
                types.AutomaticFunctionCallingConfig(disable=True)
            )
        if outlines_class is not None and "response_schema" not in config_kwargs:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = outlines_class

        config_kwargs = self._normalize_generate_config(config_kwargs)
        generate_config = types.GenerateContentConfig(**config_kwargs)

        for attempt in range(self.max_retries):
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=generate_config,
                )
                if outlines_class is None:
                    response_text = response.text
                else:
                    if response.parsed is not None:
                        response_text = outlines_class.model_validate(response.parsed)
                    else:
                        response_text = outlines_class.model_validate_json(
                            response.text
                        )

                result = {
                    "response": response_text,
                    "prompt_token_count": _estimate_tokens(prompt),
                    "output_token_count": _estimate_tokens(response_text),
                }
                if use_cache and cache_key is not None:
                    self._cache_store(cache_key, result)
                return result
            except (
                *self._RETRYABLE_GOOGLE_EXCEPTIONS,
                *self._RETRYABLE_TRANSPORT_EXCEPTIONS,
            ) as e:
                if attempt == self.max_retries - 1:
                    raise
                delay = self.initial_delay * (2**attempt)
                print(
                    f"Transient error {type(e).__name__}. Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})..."
                )
                await asyncio.sleep(delay)
            except google_exceptions.GoogleAPICallError:
                raise

        raise RuntimeError("Max retries exceeded")


class OpenRouterLLM(LLM):
    """LLM implementation using OpenRouter API."""

    def __init__(
        self,
        model: str,
        api_key: str,
        keys_file: Optional[str] = None,
        max_retries: int = 5,
        initial_delay: float = 2.0,
        verbosity: int = 0,
        use_cache: bool = True,
    ):
        self.model = model
        self.use_cache = use_cache
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.verbosity = verbosity
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"

    @staticmethod
    def _enforce_strict_schema(schema: Any) -> Any:
        if isinstance(schema, dict):
            if schema.get("type") == "object" or "properties" in schema:
                schema["additionalProperties"] = False
                if "properties" in schema:
                    schema["required"] = list(schema["properties"].keys())
            for value in schema.values():
                OpenRouterLLM._enforce_strict_schema(value)
        elif isinstance(schema, list):
            for item in schema:
                OpenRouterLLM._enforce_strict_schema(item)
        return schema

    @staticmethod
    def _build_response_format(
        outlines_class: type[pydantic.BaseModel],
    ) -> Dict[str, Any]:
        schema = OpenRouterLLM._enforce_strict_schema(
            outlines_class.model_json_schema()
        )
        return {
            "type": "json_schema",
            "json_schema": {
                "name": outlines_class.__name__,
                "strict": True,
                "schema": schema,
            },
        }

    @staticmethod
    def _extract_message_text(message: Dict[str, Any]) -> str:
        content = message.get("content")
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        elif content is not None:
            text = str(content)

        if not text.strip():
            reasoning = message.get("reasoning") or message.get("reasoning_content")
            if isinstance(reasoning, str):
                text = reasoning
        return text

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped

        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            return "\n".join(lines[1:-1]).strip()
        return stripped

    @staticmethod
    def _strip_reasoning_tags(text: str) -> str:
        return re.sub(
            r"<(think|thinking|reasoning)>.*?</\1>", "", text, flags=re.DOTALL
        ).strip()

    @staticmethod
    def _extract_first_json_object(text: str) -> Optional[str]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return match.group(0) if match else None

    @classmethod
    def _parse_structured_response(
        cls,
        message: Dict[str, Any],
        outlines_class: type[pydantic.BaseModel],
    ) -> pydantic.BaseModel:
        parsed = message.get("parsed")
        if parsed is not None:
            return outlines_class.model_validate(parsed)

        response_text = cls._extract_message_text(message)
        if not response_text or not response_text.strip():
            raise ValueError(f"Empty structured content. Raw message: {message!r}")

        candidates: List[str] = []
        seen: set[str] = set()
        for variant in (
            response_text,
            cls._strip_code_fences(response_text),
            cls._strip_reasoning_tags(response_text),
            cls._strip_code_fences(cls._strip_reasoning_tags(response_text)),
            cls._extract_first_json_object(cls._strip_reasoning_tags(response_text)),
        ):
            if variant and variant.strip() and variant not in seen:
                seen.add(variant)
                candidates.append(variant)

        last_error: Optional[Exception] = None
        for candidate in candidates:
            try:
                return outlines_class.model_validate_json(candidate)
            except (pydantic.ValidationError, ValueError) as e:
                last_error = e
            try:
                return outlines_class.model_validate(json.loads(candidate))
            except (json.JSONDecodeError, pydantic.ValidationError) as e:
                last_error = e

        raise ValueError(
            f"Could not parse structured output as {outlines_class.__name__}. "
            f"Response snippet: {response_text[:300]!r}. Last error: {last_error}"
        )

    async def generate_response(
        self,
        prompt: PromptType,
        system_instruction: Optional[str] = None,
        outlines_class: Optional[type[pydantic.BaseModel]] = None,
        sampling_kwargs: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> dict:
        sampling_kwargs = sampling_kwargs or {}
        cache_key: Optional[str] = None
        if use_cache:
            cache_key, cached = self._cache_lookup(
                prompt=prompt,
                system_instruction=system_instruction,
                outlines_class=outlines_class,
                extra_cache_context={"sampling_kwargs": sampling_kwargs},
            )
            if cached is not None:
                return cached

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        if isinstance(prompt, list):
            messages.extend(prompt)
        else:
            messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            **sampling_kwargs,
        }

        if outlines_class is not None:
            payload["response_format"] = self._build_response_format(outlines_class)
            provider = payload.get("provider")
            if provider is None:
                payload["provider"] = {"require_parameters": True}
            elif isinstance(provider, dict):
                payload["provider"] = {
                    **provider,
                    "require_parameters": provider.get("require_parameters", True),
                }
            else:
                raise TypeError("sampling_kwargs['provider'] must be a dictionary")

            schema_instruction = (
                "Respond with a single JSON object that strictly matches this JSON schema. "
                "Do not include any other text, prose, markdown, or code fences.\n\n"
                f"Schema:\n{json.dumps(outlines_class.model_json_schema(), indent=2)}"
            )
            if messages and messages[0]["role"] == "system":
                messages[0] = {
                    "role": "system",
                    "content": messages[0]["content"] + "\n\n" + schema_instruction,
                }
            else:
                messages.insert(0, {"role": "system", "content": schema_instruction})
            payload["messages"] = messages

        async with httpx.AsyncClient(timeout=120.0) as http_client:
            for attempt in range(self.max_retries):
                try:
                    if self.verbosity >= 1:
                        print(
                            f"Sending request to OpenRouter with model {self.model}..."
                        )

                    response = await http_client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    response.raise_for_status()

                    data = response.json()

                    if self.verbosity >= 1:
                        print(f"Response received: {data}")

                    message = data["choices"][0]["message"]

                    if outlines_class is not None:
                        parsed_response = self._parse_structured_response(
                            message,
                            outlines_class,
                        )
                        response_value = parsed_response
                        output_fallback = parsed_response.model_dump_json()
                    else:
                        response_value = self._extract_message_text(message)
                        output_fallback = str(response_value)

                    usage = data.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", _estimate_tokens(prompt))
                    completion_tokens = usage.get(
                        "completion_tokens", _estimate_tokens(output_fallback)
                    )

                    result = {
                        "response": response_value,
                        "prompt_token_count": prompt_tokens,
                        "output_token_count": completion_tokens,
                    }
                    if use_cache and cache_key is not None:
                        self._cache_store(cache_key, result)
                    return result

                except httpx.HTTPStatusError as e:
                    if attempt == self.max_retries - 1:
                        raise RuntimeError(
                            f"OpenRouter API request failed: {e.response.text}"
                        )
                    delay = self.initial_delay * (2**attempt)
                    print(
                        f"HTTP error {e.response.status_code}. Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})..."
                    )
                    await asyncio.sleep(delay)
                except (
                    pydantic.ValidationError,
                    ValueError,
                    json.JSONDecodeError,
                    KeyError,
                    TypeError,
                ) as e:
                    if attempt == self.max_retries - 1:
                        raise RuntimeError(
                            f"OpenRouter structured parsing failed: {str(e)}"
                        )
                    delay = self.initial_delay * (2**attempt)
                    print(
                        f"Parsing error {type(e).__name__}: {str(e)}. Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})..."
                    )
                    await asyncio.sleep(delay)
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise RuntimeError(f"OpenRouter API request failed: {str(e)}")
                    delay = self.initial_delay * (2**attempt)
                    print(
                        f"Error {type(e).__name__}: {str(e)}. Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})..."
                    )
                    await asyncio.sleep(delay)

        raise RuntimeError("Max retries exceeded")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    def close(self):
        return None

    def __del__(self):
        self.close()


class VLLM(LLM):
    """LLM implementation using vLLM for local inference."""

    _DEFAULT_CACHE_DIR = Path.home() / "models" / "hf"
    _active_instance: Optional["VLLM"] = None
    _active_config: Optional[Dict[str, Any]] = None
    _active_model_path: Optional[str] = None

    @staticmethod
    def _is_engine_startup_failure(error_text: str) -> bool:
        text = error_text.lower()
        return (
            "engine core initialization failed" in text
            or "free memory on device" in text
            or "failed core proc" in text
        )

    @staticmethod
    def _kill_engine_processes() -> None:
        subprocess.run(["pkill", "-f", "VLLM::EngineCore"], check=False)

    def __init__(
        self,
        model: str,
        endpoint_id: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        sampling_kwargs: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = None,
        cache_dir: Optional[str] = None,
        verbosity: int = 0,
        use_cache: bool = True,
    ):
        self.model = model
        self.use_cache = use_cache
        self.default_sampling_kwargs = sampling_kwargs or {}
        if batch_size is not None and int(batch_size) < 1:
            raise ValueError("batch_size must be >= 1 when provided")
        self.batch_size = int(batch_size) if batch_size is not None else None
        self.verbosity = verbosity
        self.remote = endpoint_id is not None

        if self.remote:
            from google.cloud import aiplatform

            aiplatform.init(project=project, location=location)
            self.endpoint = aiplatform.Endpoint(endpoint_id)
            self.engine = None
            self.cache_dir = None
            return

        self.cache_dir = Path(cache_dir) if cache_dir else self._DEFAULT_CACHE_DIR

        if verbosity < 1:
            import os

            # Suppress vLLM's internal logging and progress bars
            os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
            os.environ["VLLM_CONFIGURE_LOGGING"] = "0"

            # Suppress logging from various modules
            logging.getLogger("vllm").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)

            # Suppress warnings from vLLM, transformers, and mistral_common
            warnings.filterwarnings("ignore", category=UserWarning, module="vllm")
            warnings.filterwarnings(
                "ignore", category=UserWarning, module="transformers"
            )
            warnings.filterwarnings(
                "ignore", category=FutureWarning, module="mistral_common"
            )

        model_path = self._resolve_model_path(model)

        engine_kwargs = engine_kwargs or {}
        engine_kwargs = {k: v for k, v in dict(engine_kwargs).items() if v is not None}

        if verbosity < 1 and "disable_log_stats" not in engine_kwargs:
            engine_kwargs["disable_log_stats"] = True

        current_config = {
            "model_path": model_path,
            "engine_kwargs": engine_kwargs,
            "cache_dir": str(self.cache_dir),
        }

        if VLLM._active_instance is not None and VLLM._active_model_path == model_path:
            print("Reusing existing vLLM engine with same model")
            self.engine = VLLM._active_instance.engine
        else:
            if VLLM._active_instance is not None:
                running_model = VLLM._active_model_path or "unknown"
                raise VLLMModelConflictError(
                    f"another model named {running_model} is running please reinitialize the instance to replace it with your requested model"
                )

            print(f"Loading LLM model: {model_path}...")
            try:
                self.engine = VLLMEngine(model=model_path, **engine_kwargs)
            except RuntimeError as e:
                error_text = str(e)
                error_text_lower = error_text.lower()
                if "already" in error_text_lower and (
                    "running" in error_text_lower or "initialized" in error_text_lower
                ):
                    raise VLLMModelConflictError(
                        "another model named unknown is running please reinitialize the instance to replace it with your requested model"
                    ) from e

                if self._is_engine_startup_failure(error_text):
                    print(
                        "vLLM engine startup failed, killing stale engine processes and retrying once..."
                    )
                    self._kill_engine_processes()
                    gc.collect()
                    torch.cuda.empty_cache()
                    try:
                        self.engine = VLLMEngine(model=model_path, **engine_kwargs)
                    except RuntimeError as retry_error:
                        raise RuntimeError(
                            "vLLM failed to start after recovery attempt. "
                            "Killed stale engine processes and retried once, but initialization still failed. "
                            f"Original error: {retry_error}"
                        ) from retry_error
                else:
                    raise
            VLLM._active_instance = self
            VLLM._active_config = current_config
            VLLM._active_model_path = model_path

    def _resolve_model_path(self, model: str) -> str:
        """Resolve model to local path, downloading from HuggingFace if needed."""
        if Path(model).exists():
            return str(Path(model).resolve())

        if "/" in model:
            local_dir = self.cache_dir / model.replace("/", "__")

            if local_dir.exists() and any(local_dir.iterdir()):
                print(f"Using cached model from: {local_dir}")
                return str(local_dir)

            print(f"Downloading model {model} to {local_dir}...")
            local_dir.parent.mkdir(parents=True, exist_ok=True)

            try:
                from huggingface_hub import snapshot_download

                snapshot_download(
                    repo_id=model,
                    local_dir=str(local_dir),
                    local_dir_use_symlinks=False,
                )
                return str(local_dir)
            except ImportError:
                print(
                    "Warning: huggingface_hub not installed. Using model ID directly."
                )
                return model

        return model

    def cleanup(self):
        """Release GPU memory by destroying the engine."""
        if getattr(self, "remote", False):
            return
        if hasattr(self, "engine") and self.engine is not None:
            del self.engine
            gc.collect()
            torch.cuda.empty_cache()
            self.engine = None
            if VLLM._active_instance is self:
                VLLM._active_instance = None
                VLLM._active_config = None
                VLLM._active_model_path = None
            print("GPU memory released")

    def __del__(self):
        """Cleanup on object destruction."""
        self.cleanup()

    def _vllm_prompt_text(
        self, prompt: PromptType, system_instruction: Optional[str]
    ) -> str:
        text = ""
        if isinstance(prompt, list):
            for m in prompt:
                if m["role"] == "user":
                    text += f"Question: {m['content']}\n"
                elif m["role"] == "assistant":
                    text += f"Answer: {m['content']}\n"
        else:
            text = f"Question: {prompt}\n"

        if system_instruction:
            return f"{system_instruction}\n\n{text}\nAnswer:"
        return f"{text}\nAnswer:"

    def _prepare_sampling_params(
        self, sampling_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Prepare sampling parameters with stop tokens."""
        merged_sampling_kwargs = self.default_sampling_kwargs.copy()
        merged_sampling_kwargs.update(sampling_kwargs or {})

        tokenizer = self.engine.get_tokenizer()
        stop_token_ids = []
        if tokenizer.eos_token_id is not None:
            stop_token_ids.append(tokenizer.eos_token_id)

        for stop_token in ["<|end|>", "<|return|>", "<|endoftext|>"]:
            try:
                # Suppress token conversion warnings if verbosity < 1
                if self.verbosity < 1:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        tid = tokenizer.convert_tokens_to_ids(stop_token)
                else:
                    tid = tokenizer.convert_tokens_to_ids(stop_token)
                if (
                    isinstance(tid, int)
                    and tid != tokenizer.unk_token_id
                    and tid not in stop_token_ids
                ):
                    stop_token_ids.append(tid)
            except Exception:
                pass

        merged_sampling_kwargs["stop_token_ids"] = stop_token_ids
        return merged_sampling_kwargs

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped

        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            return "\n".join(lines[1:-1]).strip()
        return stripped

    @staticmethod
    def _coerce_structured_response(
        raw_response: Any,
        outlines_class: Optional[type[pydantic.BaseModel]],
    ) -> Any:
        if outlines_class is None:
            return raw_response

        if isinstance(raw_response, outlines_class):
            return raw_response

        if isinstance(raw_response, pydantic.BaseModel):
            return outlines_class.model_validate(raw_response.model_dump())

        if isinstance(raw_response, dict):
            return outlines_class.model_validate(raw_response)

        if isinstance(raw_response, str):
            cleaned = VLLM._strip_code_fences(raw_response)

            try:
                return outlines_class.model_validate_json(cleaned)
            except Exception:
                pass

            try:
                parsed = json.loads(cleaned)
                return outlines_class.model_validate(parsed)
            except Exception:
                pass

            # Fallback for list-like text often produced by local models.
            model_fields = getattr(outlines_class, "model_fields", {})
            if "fields" in model_fields:
                items = [
                    item.strip()
                    for item in re.split(r"[,;\n]", cleaned)
                    if item.strip()
                ]
                return outlines_class.model_validate({"fields": items})

        raise TypeError(
            f"Failed to coerce structured response into {outlines_class.__name__}. "
            f"Received type: {type(raw_response).__name__}"
        )

    async def generate_response(
        self,
        prompt: PromptType,
        system_instruction: Optional[str] = None,
        sampling_kwargs: Optional[Dict[str, Any]] = None,
        outlines_class: Optional[type[pydantic.BaseModel]] = None,
        use_cache: bool = True,
    ) -> dict:
        """Generate a response using vLLM."""
        if self.remote:
            return await self._remote_generate_response(
                prompt=prompt,
                system_instruction=system_instruction,
                sampling_kwargs=sampling_kwargs,
                outlines_class=outlines_class,
                use_cache=use_cache,
            )

        sampling_kwargs = sampling_kwargs or {}
        cache_key: Optional[str] = None
        if use_cache:
            cache_key, cached = self._cache_lookup(
                prompt=prompt,
                system_instruction=system_instruction,
                outlines_class=outlines_class,
                extra_cache_context={"sampling_kwargs": sampling_kwargs},
            )
            if cached is not None:
                return cached

        if outlines_class is not None:
            prompt_text = self._vllm_prompt_text(prompt, system_instruction)

            if self.verbosity >= 1:
                print("[PROMPT TEXT FOR OUTLINES]")
                print(prompt_text)

            merged_sampling_kwargs = self._prepare_sampling_params(sampling_kwargs)

            def _generate():
                ctx = _suppress_stderr() if self.verbosity < 1 else nullcontext()
                with ctx:
                    outlines_model = outlines.models.from_vllm_offline(self.engine)
                    return outlines_model(
                        prompt_text,
                        output_type=outlines_class,
                        sampling_params=SamplingParams(**merged_sampling_kwargs),
                    )

            raw_response = await asyncio.to_thread(_generate)
            response = self._coerce_structured_response(raw_response, outlines_class)

            result = {
                "response": response,
                "prompt_token_count": _estimate_tokens(prompt),
                "output_token_count": _estimate_tokens(response),
            }
            if use_cache and cache_key is not None:
                self._cache_store(cache_key, result)
            return result

        results = await self.generate_batch(
            [prompt], [system_instruction], sampling_kwargs
        )
        result = results[0]
        if use_cache and cache_key is not None:
            self._cache_store(cache_key, result)
        return result

    async def _remote_generate_response(
        self,
        prompt: PromptType,
        system_instruction: Optional[str] = None,
        sampling_kwargs: Optional[Dict[str, Any]] = None,
        outlines_class: Optional[type[pydantic.BaseModel]] = None,
        use_cache: bool = True,
    ) -> dict:
        merged_sampling_kwargs = {
            **self.default_sampling_kwargs,
            **(sampling_kwargs or {}),
        }

        cache_key: Optional[str] = None
        if use_cache:
            cache_key, cached = self._cache_lookup(
                prompt=prompt,
                system_instruction=system_instruction,
                outlines_class=outlines_class,
                extra_cache_context={"sampling_kwargs": merged_sampling_kwargs},
            )
            if cached is not None:
                return cached

        messages: List[Dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        if isinstance(prompt, list):
            messages.extend(prompt)
        else:
            messages.append({"role": "user", "content": prompt})

        if outlines_class is not None:
            schema_instruction = (
                "Respond with a single JSON object that strictly matches this JSON schema. "
                "Do not include any other text, prose, markdown, or code fences.\n\n"
                f"Schema:\n{json.dumps(outlines_class.model_json_schema(), indent=2)}"
            )
            if messages and messages[0]["role"] == "system":
                messages[0] = {
                    "role": "system",
                    "content": messages[0]["content"] + "\n\n" + schema_instruction,
                }
            else:
                messages.insert(0, {"role": "system", "content": schema_instruction})

        instance = {
            "@requestFormat": "chatCompletions",
            "messages": messages,
            **merged_sampling_kwargs,
        }

        response = await asyncio.to_thread(self.endpoint.predict, instances=[instance])
        predictions = response.predictions
        pred = (
            predictions[0]
            if isinstance(predictions, list) and predictions
            else predictions
        )
        message = pred["choices"][0]["message"]

        if outlines_class is not None:
            response_value = OpenRouterLLM._parse_structured_response(
                message, outlines_class
            )
            output_fallback = response_value.model_dump_json()
        else:
            response_value = OpenRouterLLM._extract_message_text(message)
            output_fallback = str(response_value)

        usage = pred.get("usage", {}) if isinstance(pred, dict) else {}
        prompt_tokens = usage.get("prompt_tokens", _estimate_tokens(prompt))
        completion_tokens = usage.get(
            "completion_tokens", _estimate_tokens(output_fallback)
        )

        result = {
            "response": response_value,
            "prompt_token_count": prompt_tokens,
            "output_token_count": completion_tokens,
        }
        if use_cache and cache_key is not None:
            self._cache_store(cache_key, result)
        return result

    async def generate_batch(
        self,
        prompts: List[PromptType],
        system_instructions: Optional[List[Optional[str]]] = None,
        sampling_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[dict]:
        """Generate responses for multiple prompts in a single batch.

        Args:
            prompts: List of prompts to generate responses for
            system_instructions: Optional list of system instructions (one per prompt)
            sampling_kwargs: Sampling parameters to use for all prompts

        Returns:
            List of response dictionaries
        """
        if not prompts:
            return []

        if system_instructions is None:
            system_instructions = [None] * len(prompts)

        prompt_texts = [
            self._vllm_prompt_text(prompt, sys_inst)
            for prompt, sys_inst in zip(prompts, system_instructions)
        ]

        if self.verbosity >= 1:
            print("[BATCH PROMPT TEXTS]")
            for i, pt in enumerate(prompt_texts):
                print(f"Prompt {i + 1}:\n{pt}\n{'-' * 20}")

        merged_sampling_kwargs = self._prepare_sampling_params(sampling_kwargs)

        def _generate_batch(batch_prompt_texts: List[str]):
            ctx = _suppress_stderr() if self.verbosity < 1 else nullcontext()
            with ctx:
                outputs = self.engine.generate(
                    batch_prompt_texts,
                    sampling_params=SamplingParams(**merged_sampling_kwargs),
                    use_tqdm=False,
                )
                results = []
                for output in outputs:
                    if not output.outputs:
                        results.append("")
                    else:
                        results.append(output.outputs[0].text)
                return results

        if self.batch_size is None:
            responses = await asyncio.to_thread(_generate_batch, prompt_texts)
        else:
            responses = []
            for i in range(0, len(prompt_texts), self.batch_size):
                batch_prompt_texts = prompt_texts[i : i + self.batch_size]
                batch_responses = await asyncio.to_thread(
                    _generate_batch, batch_prompt_texts
                )
                responses.extend(batch_responses)

        return [
            {
                "response": response,
                "prompt_token_count": _estimate_tokens(prompt),
                "output_token_count": _estimate_tokens(response),
            }
            for prompt, response in zip(prompts, responses)
        ]
