# autotune-ollama Setup Guide

## Requirements

- Linux host with an NVIDIA GPU (tested on RTX 5070 / Blackwell)
- Docker + Docker Compose v2
- NVIDIA Container Toolkit (for GPU passthrough into Docker)
- Python 3.10+ and [uv](https://github.com/astral-sh/uv)
- Anthropic API key (for Claude judging calls)
- Internet access (for model downloads and Claude API)

> **Runs locally on the same machine as Docker.** No SSH, no remote access required.

---

## 1. Install Docker

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Log out and back in for the group change to take effect
docker run hello-world  # verify
```

## 2. Install NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU is visible inside Docker:
docker run --rm --gpus all ubuntu nvidia-smi
```

> **Note for RTX 5070 (Blackwell):** Requires Ollama v0.17+ which bundles CUDA 12.8.
> Pull the latest image before first run: `docker pull ollama/ollama`

## 3. Clone the repo

```bash
git clone <repo-url> autotune-ollama
cd autotune-ollama
```

## 4. Ollama data volume

Models are stored in the `ollama_ollama` Docker volume (shared with the Portainer
Ollama stack). If you already have Ollama running via Portainer, this volume exists.
Otherwise create it:

```bash
docker volume create ollama_ollama
```

## 5. Set your Anthropic API key

```bash
cp .env.example .env
# Edit .env and replace the placeholder with your real key:
#   ANTHROPIC_API_KEY=sk-ant-...
```

`.env` is gitignored — your key will never be committed.

## 6. Install Python dependencies

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## 7. Create your config.yaml

```bash
cp config.yaml.example config.yaml
# Edit config.yaml if needed — defaults are tuned for a 12GB VRAM GPU.
```

`config.yaml` is gitignored. `config.yaml.example` is the committed template.
No VM IP or SSH key needed — everything runs locally.

## 8. Test the Docker setup

```bash
# Start Ollama with the baseline config
docker compose -p ollama-autotune -f configs/docker-compose.baseline.yml up -d

# Verify API is up
curl http://localhost:11434/api/tags

# Verify GPU is being used
docker exec $(docker ps -qf name=ollama) nvidia-smi

# Tear down (autotune.py manages the container lifecycle from here)
docker compose -p ollama-autotune -f configs/docker-compose.baseline.yml down
```

## 9. Generate reference answers

Once you've finalized your eval prompts in `eval_prompts.json`, generate Opus reference
answers (one-time, ~15 API calls):

```bash
python generate_references.py
```

This writes `references.json` (gitignored). Review it to confirm the answers look correct.

## 10. Run

```bash
python autotune.py
```

autotune.py manages the full lifecycle:
- Starts/restarts the Ollama Docker container for each infra config
- Pulls models as needed (models persist in `ollama_ollama` volume)
- Runs eval suite, judges output quality, logs results to `results.tsv` and `details.jsonl`
- Resume after interruption by running again — reads completed experiment keys from `results.tsv`

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `docker: permission denied` | Run `sudo usermod -aG docker $USER` and re-login |
| `nvidia-smi` not found in container | Install NVIDIA Container Toolkit (step 2) |
| Ollama API not responding | Check `docker ps`; check `docker logs $(docker ps -qf name=ollama)` |
| Model doesn't fit in VRAM | Auto-detected and skipped via `/api/ps`; remove from `config.yaml` if always failing |
| `ANTHROPIC_API_KEY not set` | Export the key before running |
| Models downloading slowly | First run pulls all models into `ollama_ollama` volume; subsequent runs reuse them |
