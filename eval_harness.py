"""Ollama API client for running inference and measuring performance."""

import json
import time
from dataclasses import dataclass

import requests


@dataclass
class InferenceResult:
    """Result from a single inference run."""
    response_text: str
    tokens_per_sec: float
    ttft_ms: float
    eval_count: int
    eval_duration_ns: int
    prompt_eval_duration_ns: int
    total_duration_ns: int


def ollama_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def pull_model(model: str, base_url: str) -> None:
    """Pull a model from the Ollama library. Blocks until complete."""
    print(f"  Pulling model {model}...")
    resp = requests.post(
        f"{base_url}/api/pull",
        json={"model": model, "stream": True},
        stream=True,
        timeout=600,
    )
    resp.raise_for_status()

    for line in resp.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        status = data.get("status", "")
        if "pulling" in status and "completed" in data:
            total = data.get("total", 0)
            completed = data.get("completed", 0)
            if total > 0:
                pct = completed / total * 100
                print(f"\r  {status}: {pct:.0f}%", end="", flush=True)
        elif status:
            print(f"\r  {status}                    ", end="", flush=True)

    print(f"\n  Model {model} ready.")


def check_gpu_fit(model: str, base_url: str) -> bool:
    """Check if a model is fully loaded on GPU (no CPU layer offload).

    Loads the model first by sending a minimal request, then checks /api/ps.
    Returns True if 100% GPU, False otherwise.
    """
    # Ensure model is loaded by sending a tiny request
    try:
        resp = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "options": {"num_predict": 1},
            },
            timeout=120,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  WARNING: Could not load model {model} for GPU check: {e}")
        return False

    # Now check /api/ps
    try:
        resp = requests.get(f"{base_url}/api/ps", timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"  WARNING: Could not check /api/ps: {e}")
        return False

    for m in data.get("models", []):
        if model in m.get("name", ""):
            size_vram = m.get("size_vram", 0)
            size = m.get("size", 0)
            if size > 0 and size_vram < size:
                gpu_pct = size_vram / size * 100
                print(f"  Model {model}: {gpu_pct:.0f}% GPU — SKIPPING (CPU offload detected)")
                return False
            print(f"  Model {model}: 100% GPU")
            return True

    print(f"  WARNING: Model {model} not found in /api/ps after loading")
    return False


def warmup(model: str, base_url: str) -> None:
    """Send a throwaway prompt to warm up the model before timing."""
    try:
        requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Say hello."}],
                "stream": False,
                "options": {"num_predict": 10},
            },
            timeout=60,
        )
    except requests.RequestException:
        pass  # Best effort


def run_inference(
    model: str,
    prompt: str,
    options: dict,
    base_url: str,
    system_prompt: str | None = None,
) -> InferenceResult:
    """Run inference against Ollama /api/chat with streaming to measure TTFT.

    Args:
        model: Ollama model name
        prompt: User prompt text
        options: Per-request parameters (num_ctx, temperature, etc.)
        base_url: Ollama API base URL
        system_prompt: Optional system prompt

    Returns:
        InferenceResult with response text and performance metrics
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    request_body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": options,
    }

    start_time = time.perf_counter()
    first_token_time = None
    response_chunks = []

    # Final metrics from the done=true chunk
    eval_count = 0
    eval_duration_ns = 0
    prompt_eval_duration_ns = 0
    total_duration_ns = 0

    resp = requests.post(
        f"{base_url}/api/chat",
        json=request_body,
        stream=True,
        timeout=300,
    )
    resp.raise_for_status()

    for line in resp.iter_lines():
        if not line:
            continue

        data = json.loads(line)

        if data.get("done", False):
            # Final chunk contains metrics
            eval_count = data.get("eval_count", 0)
            eval_duration_ns = data.get("eval_duration", 0)
            prompt_eval_duration_ns = data.get("prompt_eval_duration", 0)
            total_duration_ns = data.get("total_duration", 0)
        else:
            # Content chunk
            msg = data.get("message", {})
            content = msg.get("content", "")
            if content and first_token_time is None:
                first_token_time = time.perf_counter()
            response_chunks.append(content)

    # Calculate metrics
    if first_token_time is not None:
        ttft_ms = (first_token_time - start_time) * 1000
    else:
        ttft_ms = (time.perf_counter() - start_time) * 1000

    if eval_duration_ns > 0:
        tokens_per_sec = eval_count / (eval_duration_ns / 1e9)
    else:
        tokens_per_sec = 0.0

    return InferenceResult(
        response_text="".join(response_chunks),
        tokens_per_sec=tokens_per_sec,
        ttft_ms=ttft_ms,
        eval_count=eval_count,
        eval_duration_ns=eval_duration_ns,
        prompt_eval_duration_ns=prompt_eval_duration_ns,
        total_duration_ns=total_duration_ns,
    )


def wait_for_api(base_url: str, timeout: int = 60) -> bool:
    """Poll Ollama API until it's ready or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False
