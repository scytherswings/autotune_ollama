"""Ollama API client for running inference and measuring performance."""

import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field

import requests


class OllamaOomError(RuntimeError):
    """Raised when Ollama returns a 500 that is confirmed to be an OOM/allocation failure."""


_OOM_KEYWORDS = (
    "out of memory", "cuda error", "cudamalloc", "failed to allocate",
    "cannot allocate", "kv cache", "ggml_cuda", "cublas",
)


def _raise_for_status_with_oom_check(resp: requests.Response) -> None:
    """Like raise_for_status() but raises OllamaOomError for confirmed OOM 500s."""
    if resp.status_code == 500:
        try:
            body = resp.text.lower()
        except Exception:
            body = ""
        if any(kw in body for kw in _OOM_KEYWORDS):
            raise OllamaOomError(f"Ollama OOM (500): {resp.text[:200]}")
        resp.raise_for_status()  # non-OOM 500 — raise normally
    else:
        resp.raise_for_status()


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


@dataclass
class ToolCallResult:
    """Result from a tool-calling inference run."""
    tool_calls: list          # [{"function": {"name": str, "arguments": dict}}]
    response_text: str        # any text the model produced alongside tool calls
    tokens_per_sec: float
    ttft_ms: float
    total_duration_ns: int    # wall-clock time from request to last token (ns)
    eval_count: int
    used_native_tools: bool   # True = Ollama tools param worked; False = text fallback


@dataclass
class ObjectiveResult:
    """Deterministic (free) checks on a model response."""
    json_valid: bool          # tool call is parseable / well-formed
    correct_tool: bool        # called the expected tool (or correctly called nothing)
    fields_present: bool      # all required_args present in the call
    no_spurious_call: bool    # didn't call a tool when none was expected
    objective_score: float    # fraction of applicable checks passed (0.0–1.0)
    details: dict = field(default_factory=dict)


def ollama_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def pull_model(model: str, base_url: str) -> None:
    """Pull a model from the Ollama library. Blocks until complete.

    Raises RuntimeError if the model name is invalid or the pull fails.
    """
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

        # Ollama sends {"error": "..."} in the stream body for bad model names
        if "error" in data:
            raise RuntimeError(f"Ollama pull failed for '{model}': {data['error']}")
        if data.get("status") == "error":
            raise RuntimeError(f"Ollama pull error for '{model}': {data}")

        status = data.get("status", "")
        if "pulling" in status and "completed" in data:
            total = data.get("total", 0)
            completed = data.get("completed", 0)
            if total > 0:
                pct = completed / total * 100
                if sys.stdout.isatty():
                    print(f"\r  {status}: {pct:.0f}%", end="", flush=True)
                else:
                    # Log-friendly: print milestones at every 10%
                    milestone = int(pct / 10) * 10
                    if not hasattr(pull_model, "_last_milestone") or pull_model._last_milestone != milestone:
                        pull_model._last_milestone = milestone
                        if milestone % 10 == 0:
                            print(f"  {status}: {milestone}%", flush=True)
        elif status:
            if sys.stdout.isatty():
                print(f"\r  {status}                    ", end="", flush=True)
            else:
                print(f"  {status}", flush=True)

    pull_model._last_milestone = -1  # reset for next call
    print(f"  Verifying {model} in local model list...")
    try:
        tags_resp = requests.get(f"{base_url}/api/tags", timeout=10)
        tags_resp.raise_for_status()
        local_models = [m["name"] for m in tags_resp.json().get("models", [])]
        # Ollama may add a digest suffix; check for a prefix match on the model name
        if not any(m == model or m.startswith(model + ":") or model.split(":")[0] in m
                   for m in local_models):
            raise RuntimeError(
                f"Model '{model}' not found in /api/tags after pull. "
                f"Available models: {local_models or '(none)'}"
            )
    except requests.RequestException as e:
        print(f"  WARNING: Could not verify model in /api/tags: {e}")

    print(f"  Model {model} ready.")


def get_ollama_allocation(container: str, since: str) -> dict:
    """Parse Ollama container logs since a given ISO timestamp to extract
    GPU/CPU layer offload and KV cache allocation for the most recent model load.

    Returns a dict with keys: gpu_layers, total_layers, kv_gpu_mb, kv_cpu_mb, flash_attn.
    Returns an empty dict if logs can't be read or parsed.
    """
    try:
        out = subprocess.check_output(
            ["docker", "logs", container, "--since", since],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError:
        return {}

    result = {}

    # e.g. "load_tensors: offloaded 27/49 layers to GPU"
    m = re.search(r"offloaded (\d+)/(\d+) layers to GPU", out)
    if m:
        result["gpu_layers"] = int(m.group(1))
        result["total_layers"] = int(m.group(2))

    # e.g. "llama_kv_cache:      CUDA0 KV buffer size =  3456.00 MiB"
    #      "llama_kv_cache:        CPU KV buffer size =  2688.00 MiB"
    kv_gpu = re.findall(r"CUDA\d+ KV buffer size =\s+([\d.]+) MiB", out)
    kv_cpu = re.findall(r"CPU KV buffer size =\s+([\d.]+) MiB", out)
    if kv_gpu:
        result["kv_gpu_mb"] = float(kv_gpu[-1])
    if kv_cpu:
        result["kv_cpu_mb"] = float(kv_cpu[-1])

    # e.g. "flash_attn    = enabled"
    if "flash_attn    = enabled" in out:
        result["flash_attn"] = True
    elif "flash_attn    = disabled" in out:
        result["flash_attn"] = False

    return result


def unload_model(model: str, base_url: str) -> None:
    """Force Ollama to unload a model from memory so the next load can allocate cleanly."""
    try:
        requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": ""}],
                "stream": False,
                "keep_alive": 0,
            },
            timeout=30,
        )
    except requests.RequestException:
        pass  # Best-effort — if it fails the model may already be unloaded


def run_tool_inference(
    model: str,
    system_prompt: str,
    user_message: str,
    tools: list,
    options: dict,
    base_url: str,
) -> ToolCallResult:
    """Run inference with Ollama's native tools parameter and measure performance.

    Falls back to parsing JSON from text if the model doesn't emit native tool calls.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    request_body = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": True,
        "options": options,
    }

    start_time = time.perf_counter()
    first_token_time = None
    response_chunks = []
    tool_calls = []
    eval_count = 0
    eval_duration_ns = 0
    total_duration_ns = 0

    resp = requests.post(
        f"{base_url}/api/chat",
        json=request_body,
        stream=True,
        timeout=300,
    )

    # If the model doesn't support the tools parameter (400), retry as plain chat
    # and rely on text-based JSON parsing for tool call extraction.
    if resp.status_code == 400 and tools:
        text_body = {k: v for k, v in request_body.items() if k != "tools"}
        resp = requests.post(
            f"{base_url}/api/chat",
            json=text_body,
            stream=True,
            timeout=300,
        )

    _raise_for_status_with_oom_check(resp)

    for line in resp.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        msg = data.get("message", {})

        if data.get("done", False):
            eval_count = data.get("eval_count", 0)
            eval_duration_ns = data.get("eval_duration", 0)
            total_duration_ns = data.get("total_duration", 0)
            if msg.get("tool_calls"):
                tool_calls = msg["tool_calls"]
        else:
            content = msg.get("content", "")
            if content and first_token_time is None:
                first_token_time = time.perf_counter()
            response_chunks.append(content)
            # Some models stream tool_calls in intermediate chunks
            if msg.get("tool_calls"):
                tool_calls = msg["tool_calls"]

    ttft_ms = ((first_token_time or time.perf_counter()) - start_time) * 1000
    raw_text = "".join(response_chunks)
    tokens_per_sec = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0.0
    used_native = bool(tool_calls)

    # Fallback: try to parse a JSON tool call from text response
    if not tool_calls and raw_text.strip():
        tool_calls = _parse_tool_call_from_text(raw_text)

    return ToolCallResult(
        tool_calls=tool_calls,
        response_text=raw_text,
        tokens_per_sec=tokens_per_sec,
        ttft_ms=ttft_ms,
        total_duration_ns=total_duration_ns,
        eval_count=eval_count,
        used_native_tools=used_native,
    )


def _parse_tool_call_from_text(text: str) -> list:
    """Try to extract a tool call JSON from a plain-text response (non-native fallback)."""
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return []
        obj = json.loads(text[start:end])
        # Accept {"name": ..., "arguments": ...} or {"function": {"name": ..., "arguments": ...}}
        if "name" in obj and "arguments" in obj:
            return [{"function": obj}]
        if "function" in obj and "name" in obj.get("function", {}):
            return [obj]
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def check_objective_criteria(
    prompt_entry: dict,
    tool_calls: list,
    response_text: str,
) -> ObjectiveResult:
    """Run free deterministic checks against expected outcomes in the prompt definition.

    For tool_call prompts: checks JSON validity, correct tool name, required fields.
    For rag prompts: checks for hallucination trap phrases.
    """
    prompt_type = prompt_entry.get("category", "tool_call")
    checks = {}

    if prompt_type == "tool_call":
        expected_tool = prompt_entry.get("expected_tool")   # None = should not call a tool
        required_args = set(prompt_entry.get("required_args", []))

        if expected_tool is None:
            # Model should NOT call any tool
            checks["no_spurious_call"] = len(tool_calls) == 0
            checks["json_valid"] = True       # N/A
            checks["correct_tool"] = True     # N/A
            checks["fields_present"] = True   # N/A
        else:
            checks["no_spurious_call"] = True  # N/A (tool call was expected)

            if tool_calls:
                fn = tool_calls[0].get("function", {})
                tool_name = fn.get("name", "")
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        args = {}

                checks["json_valid"] = isinstance(args, dict)
                checks["correct_tool"] = (tool_name == expected_tool)
                checks["fields_present"] = required_args.issubset(set(args.keys())) if isinstance(args, dict) else False
            else:
                checks["json_valid"] = False
                checks["correct_tool"] = False
                checks["fields_present"] = False

    elif prompt_type == "rag":
        traps = prompt_entry.get("hallucination_traps", [])
        text_lower = response_text.lower()
        trapped = any(trap.lower() in text_lower for trap in traps)
        checks["no_spurious_call"] = True   # N/A
        checks["json_valid"] = True          # N/A
        checks["correct_tool"] = True        # N/A
        checks["fields_present"] = not trapped  # repurposed: True = no hallucination trap hit

    passed = sum(1 for v in checks.values() if v)
    objective_score = passed / len(checks) if checks else 0.0

    return ObjectiveResult(
        json_valid=checks.get("json_valid", True),
        correct_tool=checks.get("correct_tool", True),
        fields_present=checks.get("fields_present", True),
        no_spurious_call=checks.get("no_spurious_call", True),
        objective_score=objective_score,
        details=checks,
    )


def check_gpu_fit(model: str, base_url: str) -> bool:
    """Log GPU vs CPU memory split for a loaded model. Always returns True.

    We no longer gate on GPU fraction — MoE models legitimately spill inactive
    experts to CPU without major TPS impact. The min_tokens_per_sec guard in
    autotune.py is the real arbiter of whether a model is fast enough.
    """
    try:
        resp = requests.get(f"{base_url}/api/ps", timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return True  # Can't check — let it proceed, TPS guard will catch slow models

    loaded = data.get("models", [])
    match = next((m for m in loaded if m.get("name", "") == model), None)
    if match is None:
        match = next((m for m in loaded if model in m.get("name", "")), None)

    if match:
        size_vram = match.get("size_vram", 0)
        size = match.get("size", 0)
        if size > 0:
            gpu_pct = size_vram / size * 100
            cpu_pct = 100 - gpu_pct
            if cpu_pct > 5:
                print(f"  Model {model}: {gpu_pct:.0f}% GPU / {cpu_pct:.0f}% CPU (TPS guard will validate)")
            else:
                print(f"  Model {model}: fully in GPU VRAM")

    return True


def _ctx_probe(model: str, base_url: str, num_ctx: int, params: dict, min_tps: float) -> bool:
    """Return True if the model loads and runs above min_tps at num_ctx.

    Ollama pre-allocates the full KV cache on load, so any short inference at
    a given num_ctx is a sufficient VRAM test — no need for a long prompt.
    """
    unload_model(model, base_url)  # force reload at new ctx
    probe_params = {**params, "num_ctx": num_ctx, "num_predict": 40}
    try:
        result = run_inference(
            model=model,
            prompt="List three benefits of regular massage therapy.",
            options=probe_params,
            base_url=base_url,
        )
        return result.tokens_per_sec >= min_tps
    except (OllamaOomError, requests.RequestException):
        return False


def detect_max_ctx(
    model: str,
    base_url: str,
    params: dict,
    min_tps: float = 0,
    ctx_min: int = 4096,
    ctx_max: int = 32768,
    precision: int = 1024,
) -> int:
    """Binary search for the largest num_ctx that keeps TPS above min_tps.

    Ollama pre-allocates the full KV cache on model load, so a short inference
    at a given num_ctx is sufficient to detect whether that context size fits
    in VRAM. Sweeping num_ctx as a quality parameter is wrong — bigger is always
    better for chat/RAG use, up to the VRAM cliff. This finds that cliff.

    Returns the largest passing ctx value (a multiple of precision).
    Falls back to ctx_min if even that value doesn't pass.
    """
    def snap(v: int) -> int:
        """Round v down to the nearest multiple of precision."""
        return max(ctx_min, (v // precision) * precision)

    print(f"  Detecting max viable num_ctx (binary search {ctx_min}–{ctx_max}, precision {precision})...")

    # Fast path: max works outright
    if _ctx_probe(model, base_url, ctx_max, params, min_tps):
        print(f"  → max ctx {ctx_max} passes — using it")
        return ctx_max

    # Fast path: min fails too — model is unusably slow
    if not _ctx_probe(model, base_url, ctx_min, params, min_tps):
        print(f"  → model fails TPS check even at ctx_min={ctx_min}")
        return ctx_min  # caller / TPS guard will handle this

    lo, hi = ctx_min, ctx_max
    while hi - lo > precision:
        mid = snap((lo + hi) // 2)
        passes = _ctx_probe(model, base_url, mid, params, min_tps)
        print(f"    ctx={mid}: {'pass' if passes else 'FAIL'}")
        if passes:
            lo = mid
        else:
            hi = mid

    print(f"  → max viable num_ctx: {lo}")
    return lo


def warmup(model: str, base_url: str, options: dict | None = None) -> None:
    """Send a throwaway prompt to warm up the model before timing.

    Pass the same options (especially num_ctx) that the eval will use so Ollama
    loads the model at the correct context size. Without this, the first real eval
    prompt triggers a reload if its num_ctx differs from Ollama's default.
    """
    opts = {"num_predict": 10}
    if options:
        opts.update({k: v for k, v in options.items() if k != "num_predict"})
        opts["num_predict"] = 10
    try:
        requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Say hello."}],
                "stream": False,
                "options": opts,
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
    _raise_for_status_with_oom_check(resp)

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

    response_text = "".join(response_chunks)

    if not response_text and eval_count == 0:
        raise RuntimeError(
            f"Inference returned empty response for model '{model}'. "
            "The model may have failed to load or returned an error."
        )

    if eval_duration_ns > 0:
        tokens_per_sec = eval_count / (eval_duration_ns / 1e9)
    else:
        tokens_per_sec = 0.0

    return InferenceResult(
        response_text=response_text,
        tokens_per_sec=tokens_per_sec,
        ttft_ms=ttft_ms,
        eval_count=eval_count,
        eval_duration_ns=eval_duration_ns,
        prompt_eval_duration_ns=prompt_eval_duration_ns,
        total_duration_ns=total_duration_ns,
    )


def run_chat_inference(
    model: str,
    turns: list[str],
    options: dict,
    base_url: str,
    system_prompt: str | None = None,
) -> InferenceResult:
    """Run a multi-turn chat conversation against Ollama /api/chat.

    Sends each user turn sequentially, threading assistant responses back
    into the message history. TTFT is from the first turn. TPS and
    total_duration_ns are aggregated across all turns.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    total_eval_count = 0
    total_eval_duration_ns = 0
    total_prompt_eval_duration_ns = 0
    total_duration_ns_sum = 0
    first_ttft_ms: float | None = None
    final_response = ""

    for user_message in turns:
        messages.append({"role": "user", "content": user_message})

        request_body = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": options,
        }

        start_time = time.perf_counter()
        first_token_time = None
        response_chunks = []
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
        _raise_for_status_with_oom_check(resp)

        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if data.get("done", False):
                eval_count = data.get("eval_count", 0)
                eval_duration_ns = data.get("eval_duration", 0)
                prompt_eval_duration_ns = data.get("prompt_eval_duration", 0)
                total_duration_ns = data.get("total_duration", 0)
            else:
                content = data.get("message", {}).get("content", "")
                if content and first_token_time is None:
                    first_token_time = time.perf_counter()
                response_chunks.append(content)

        turn_response = "".join(response_chunks)
        final_response = turn_response

        if first_ttft_ms is None:
            ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else (time.perf_counter() - start_time) * 1000
            first_ttft_ms = ttft_ms

        total_eval_count += eval_count
        total_eval_duration_ns += eval_duration_ns
        total_prompt_eval_duration_ns += prompt_eval_duration_ns
        total_duration_ns_sum += total_duration_ns

        messages.append({"role": "assistant", "content": turn_response})

    if not final_response and total_eval_count == 0:
        raise RuntimeError(
            f"Chat inference returned empty response for model '{model}'."
        )

    tokens_per_sec = total_eval_count / (total_eval_duration_ns / 1e9) if total_eval_duration_ns > 0 else 0.0

    return InferenceResult(
        response_text=final_response,
        tokens_per_sec=tokens_per_sec,
        ttft_ms=first_ttft_ms or 0.0,
        eval_count=total_eval_count,
        eval_duration_ns=total_eval_duration_ns,
        prompt_eval_duration_ns=total_prompt_eval_duration_ns,
        total_duration_ns=total_duration_ns_sum,
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
