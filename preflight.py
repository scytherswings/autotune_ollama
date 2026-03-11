"""Pre-flight environment checks for autotune-ollama.

Run directly to verify your setup before a full autotune run:

    python preflight.py

Imported and called automatically by autotune.py at startup.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import anthropic
import yaml
from dotenv import load_dotenv

load_dotenv()  # Load .env before anything touches os.environ


def load_config(path: str = "config.yaml") -> dict:
    if not Path(path).exists():
        print(f"ERROR: {path} not found.")
        print("  Copy config.yaml.example to config.yaml and fill in your settings.")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


# (quant prefix → bytes per parameter, most-specific first)
_QUANT_BYTES_PER_PARAM: list[tuple[str, float]] = [
    ("fp16", 2.0),
    ("f16",  2.0),
    ("q8",   1.0),
    ("q6",   0.75),
    ("q5",   0.625),
    ("q4",   0.5),
    ("q3",   0.375),
    ("q2",   0.25),
]


def _estimate_model_gb(model_name: str) -> float:
    """Estimate download size in GB from a model name string.

    Parses the parameter count (e.g. '7b', '13b') and quantization level
    (e.g. 'q8_0', 'q4_K_M') embedded in the name, then applies 10% overhead
    for metadata/tokenizer files.  Returns 10.0 GB when the name cannot be
    parsed — a conservative fallback that errs on the side of caution.
    """
    name = model_name.lower()

    param_match = re.search(r"(\d+(?:\.\d+)?)b", name)
    if not param_match:
        return 10.0  # unparseable → assume large

    params_b = float(param_match.group(1))

    bytes_per_param = 0.75  # ~q6 equivalent when quantization isn't detected
    for quant, bpp in _QUANT_BYTES_PER_PARAM:
        if quant in name:
            bytes_per_param = bpp
            break

    return params_b * bytes_per_param * 1.1  # +10 % metadata overhead


def preflight_check(config: dict) -> None:
    """Run all pre-flight checks before touching Docker or the API.

    Exits with a clear, actionable message on any failure.
    """
    ok = True

    def fail(msg: str, fix: str = "") -> None:
        nonlocal ok
        ok = False
        print(f"  FAIL  {msg}")
        if fix:
            print(f"        fix: {fix}")

    print("Running pre-flight checks...")

    # 1. Anthropic API key — check it's set and actually works (credits, auth)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        fail(
            "ANTHROPIC_API_KEY is not set",
            "add it to .env or export it in your shell",
        )
    else:
        try:
            client = anthropic.Anthropic(api_key=api_key)
            client.messages.create(
                model=config.get("judge", {}).get("model", "claude-sonnet-4-20250514"),
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
            print("  ok    Anthropic API key valid and account has credits")
        except anthropic.AuthenticationError:
            fail(
                "ANTHROPIC_API_KEY is invalid or revoked",
                "check your key at console.anthropic.com and update .env",
            )
        except anthropic.PermissionDeniedError:
            fail(
                "Anthropic API key lacks permission to use the judge model",
                "check your API key permissions at console.anthropic.com",
            )
        except anthropic.BadRequestError as e:
            if "credit" in str(e).lower() or "balance" in str(e).lower():
                fail(
                    "Anthropic account has insufficient credits",
                    "top up at console.anthropic.com/settings/billing",
                )
            else:
                fail(f"Anthropic API returned an unexpected error: {e}")
        except Exception as e:
            fail(f"Could not reach Anthropic API: {e}")

    # 2. Docker daemon accessible
    result = subprocess.run(
        ["docker", "info"],
        capture_output=True,
        timeout=10,
    )
    if result.returncode != 0:
        fail(
            "Docker daemon is not running or current user lacks access",
            "start Docker, or run: sudo usermod -aG docker $USER && newgrp docker",
        )
    else:
        print("  ok    Docker daemon reachable")

    # 3. Docker Compose v2 plugin available
    result = subprocess.run(
        ["docker", "compose", "version"],
        capture_output=True,
        timeout=10,
    )
    if result.returncode != 0:
        fail(
            "Docker Compose v2 plugin not found ('docker compose version' failed)",
            "install it: mkdir -p ~/.docker/cli-plugins && "
            "curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 "
            "-o ~/.docker/cli-plugins/docker-compose && "
            "chmod +x ~/.docker/cli-plugins/docker-compose",
        )
    else:
        version = result.stdout.decode().strip().split("\n")[0]
        print(f"  ok    {version}")

    # 5. Configured Ollama volume exists
    volume_name = config.get("infra", {}).get("ollama_volume", "ollama_ollama")
    result = subprocess.run(
        ["docker", "volume", "inspect", volume_name],
        capture_output=True,
        timeout=10,
    )
    if result.returncode != 0:
        fail(
            f"Docker volume '{volume_name}' does not exist",
            f"docker volume create {volume_name}",
        )
    else:
        print(f"  ok    Docker volume '{volume_name}' exists")

    # 6. compose_dir exists
    compose_dir = Path(config["infra"]["compose_dir"])
    if not compose_dir.is_dir():
        fail(
            f"compose_dir '{compose_dir}' not found (run from the project root?)",
            f"cd to the directory containing '{compose_dir}/' and try again",
        )
    else:
        print(f"  ok    compose_dir '{compose_dir}' found")

    # 7. All compose files exist
    for infra_config in config.get("infra_configs", []):
        compose_file = compose_dir / f"docker-compose.{infra_config}.yml"
        if not compose_file.exists():
            fail(
                f"Compose file missing: {compose_file}",
                f"ensure configs/docker-compose.{infra_config}.yml is present",
            )
        else:
            print(f"  ok    {compose_file}")

    # 8. Disk space — estimate total model download size and compare to free space
    #    on the filesystem that backs the configured Docker volume.
    models = config.get("models", [])
    if models:
        model_sizes_gb = {m: _estimate_model_gb(m) for m in models}
        total_gb = sum(model_sizes_gb.values())
        largest_gb = max(model_sizes_gb.values())

        # Ask Docker for the on-disk path of the volume so we check the right
        # filesystem (Docker volumes may live on a separate mount).
        mountpoint: str | None = None
        try:
            mp = subprocess.run(
                ["docker", "volume", "inspect", volume_name,
                 "--format", "{{.Mountpoint}}"],
                capture_output=True, text=True, timeout=10,
            )
            candidate = mp.stdout.strip()
            if mp.returncode == 0 and candidate:
                mountpoint = candidate
        except Exception:
            pass

        if mountpoint:
            try:
                free_gb = shutil.disk_usage(mountpoint).free / (1024 ** 3)
                if free_gb >= total_gb:
                    print(
                        f"  ok    Disk space: {free_gb:.0f} GB free "
                        f"(~{total_gb:.0f} GB estimated for all {len(models)} models)"
                    )
                elif free_gb >= largest_gb:
                    # Might have room for some but not all — warn, don't abort.
                    # Models already present in the volume won't be re-downloaded.
                    print(
                        f"  WARN  Disk space: {free_gb:.0f} GB free, "
                        f"~{total_gb:.0f} GB estimated to download all "
                        f"{len(models)} models from scratch"
                    )
                    print(
                        "        Already-cached models won't be re-downloaded. "
                        "Downloads may fail if space runs out mid-run."
                    )
                else:
                    # Less free than the single largest model — almost certainly
                    # can't make progress; treat this as a hard failure.
                    fail(
                        f"Disk space critically low: {free_gb:.1f} GB free, "
                        f"~{largest_gb:.1f} GB needed for the largest model alone",
                        "free up space or shorten the model list in config.yaml",
                    )
            except Exception as e:
                print(f"  WARN  Could not read disk usage at {mountpoint}: {e}")
        else:
            # Docker call to get mountpoint failed — still print the estimate so
            # the user knows what to expect.
            print(f"  WARN  Could not determine mountpoint for volume '{volume_name}'")
            print(
                f"        Estimated model download: ~{total_gb:.0f} GB total "
                f"across {len(models)} models"
            )

    if not ok:
        print("\nPre-flight failed. Fix the issues above and re-run.")
        sys.exit(1)

    print("Pre-flight checks passed.\n")


if __name__ == "__main__":
    preflight_check(load_config())
