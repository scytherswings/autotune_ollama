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

## 4. Create the Ollama data volume

Models are stored in a named Docker volume that persists across container restarts
and config switches:

```bash
docker volume create ollama-data
```

## 5. Set your Anthropic API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# Add to ~/.bashrc or ~/.zshrc to persist across sessions
```

## 6. Install Python dependencies

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## 7. Edit config.yaml

Set your GPU-specific preferences if needed. The defaults are tuned for a 12GB VRAM GPU.
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
answers (one-time, ~8 API calls):

```bash
python generate_references.py
```

Review `eval_prompts.json` to confirm the references look correct.

## 10. Run

```bash
python autotune.py
```

autotune.py manages the full lifecycle:
- Starts/restarts the Ollama Docker container for each infra config
- Pulls models as needed (models persist in `ollama-data` volume)
- Runs eval suite, judges output quality, logs results to `results.tsv`
- Resume after interruption by running again — reads existing `results.tsv`

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `docker: permission denied` | Run `sudo usermod -aG docker $USER` and re-login |
| `nvidia-smi` not found in container | Install NVIDIA Container Toolkit (step 2) |
| Ollama API not responding | Check `docker ps`; check `docker logs $(docker ps -qf name=ollama)` |
| Model doesn't fit in VRAM | Auto-detected and skipped via `/api/ps`; remove from `config.yaml` if always failing |
| `ANTHROPIC_API_KEY not set` | Export the key before running |
| Models downloading slowly | First run pulls all models into `ollama-data` volume; subsequent runs reuse them |
