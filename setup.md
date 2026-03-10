# autotune-ollama Setup Guide

## Prerequisites

- Mac with Python 3.10+ and `uv`
- Ubuntu VM in Proxmox with NVIDIA RTX 5070 GPU passthrough
- Docker + Docker Compose installed on the VM
- Anthropic API key

## VM Setup

### 1. Create the autotune user

```bash
# On the Ubuntu VM (as root/sudo user):
sudo useradd -m -s /bin/bash autotune
```

### 2. SSH key auth (Mac to VM)

```bash
# On Mac:
ssh-keygen -t rsa -b 4096 -f ~/.ssh/autotune_rsa -N ""
ssh-copy-id -i ~/.ssh/autotune_rsa autotune@<VM_IP>

# Test:
ssh -i ~/.ssh/autotune_rsa autotune@<VM_IP> whoami
```

### 3. Deploy VM-side files

```bash
# On Mac, from this repo:
scp -i ~/.ssh/autotune_rsa -r vm/* autotune@<VM_IP>:/home/autotune/ollama-configs/
scp -i ~/.ssh/autotune_rsa vm/ollama-reconfig autotune@<VM_IP>:/home/autotune/bin/ollama-reconfig

# On VM:
ssh -i ~/.ssh/autotune_rsa autotune@<VM_IP>
chmod +x ~/bin/ollama-reconfig
mkdir -p ~/bin
```

### 4. Sudoers rule (scoped docker compose access)

```bash
# On VM as root:
sudo visudo -f /etc/sudoers.d/autotune-docker

# Add this line:
autotune ALL=(ALL) NOPASSWD: /usr/bin/docker compose -f /home/autotune/ollama-configs/*
```

### 5. Create Docker volume

```bash
# On VM:
sudo docker volume create ollama-data
```

### 6. Test gatekeeper

```bash
# From Mac:
ssh -i ~/.ssh/autotune_rsa autotune@<VM_IP> /home/autotune/bin/ollama-reconfig baseline
```

## Mac Setup

### 1. Update config.yaml

Edit `config.yaml` and set `vm.host` to your VM's IP address.

### 2. Set API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Install Python dependencies

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 4. Generate reference answers

```bash
python generate_references.py
```

This calls Claude Opus once per eval prompt to create gold-standard reference answers.
Review `eval_prompts.json` to verify the references look good.

### 5. Run autotune

```bash
python autotune.py
```

Results are logged to `results.tsv`. The process can be stopped and restarted — it will resume from where it left off.

## Troubleshooting

- **SSH connection refused**: Check VM firewall, ensure sshd is running
- **Ollama API unreachable**: Verify port 11434 is exposed, check `docker ps` on VM
- **Model doesn't fit in GPU**: Will be auto-detected and skipped via `/api/ps` check
- **Rate limited by Anthropic**: Judge calls have exponential backoff built in
- **Budget exhausted**: Increase `budget.max_api_calls` in config.yaml
