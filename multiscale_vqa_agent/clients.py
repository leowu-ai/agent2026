import base64
import io
import json
import mimetypes
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


class OpenAICompatibleClient:
    """Small dependency-free client for local vLLM/OpenAI-compatible servers."""

    def __init__(self, config: Dict[str, Any]):
        self.backend = config.get("backend", "mock")
        self.base_url = config.get("base_url", "").rstrip("/")
        self.model = config.get("model", "")
        self.api_key = config.get("api_key", "EMPTY")
        self.enable_thinking = config.get("enable_thinking")

    @property
    def enabled(self) -> bool:
        return self.backend == "openai_compatible"

    def chat(
        self,
        system: str,
        user: str,
        images: Optional[List[str]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: Optional[Dict[str, Any]] = None,
        retries: int = 2,
        image_max_size: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Optional[str]:
        if not self.enabled:
            return None
        content: Any = user
        if images:
            content = [{"type": "text", "text": user}]
            for image_path in images:
                path = Path(image_path)
                mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
                image_bytes = path.read_bytes()
                if image_max_size:
                    try:
                        from PIL import Image

                        with Image.open(io.BytesIO(image_bytes)) as image:
                            if max(image.size) > image_max_size:
                                image = image.convert("RGB")
                                image.thumbnail((image_max_size, image_max_size))
                                buffer = io.BytesIO()
                                image.save(buffer, format="JPEG", quality=90)
                                image_bytes = buffer.getvalue()
                                mime = "image/jpeg"
                    except Exception:
                        pass
                encoded = base64.b64encode(image_bytes).decode("ascii")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{encoded}"},
                })
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        request_enable_thinking = (
            self.enable_thinking
            if enable_thinking is None
            else enable_thinking
        )
        if request_enable_thinking is not None:
            payload["chat_template_kwargs"] = {
                "enable_thinking": request_enable_thinking
            }
        if top_p is not None:
            payload["top_p"] = top_p
        if top_k is not None:
            payload["top_k"] = top_k
        if response_format:
            payload["response_format"] = response_format
        def make_request() -> urllib.request.Request:
            return urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )

        last_error = None
        for attempt in range(max(1, retries + 1)):
            try:
                with urllib.request.urlopen(make_request(), timeout=300) as response:
                    result = json.loads(response.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"]
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                last_error = error
                if isinstance(error, urllib.error.HTTPError) and error.code in {400, 422} and payload.pop("response_format", None):
                    continue
                if attempt >= retries:
                    break
                time.sleep(1.5 * (attempt + 1))
        if last_error:
            raise last_error
        return None


def parse_json_response(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """Extract the longest valid JSON object without trusting surrounding prose."""
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:])
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    candidates = []
    for match in __import__("re").finditer(r"\{", stripped):
        try:
            value, end = decoder.raw_decode(stripped[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append((end, value))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None
