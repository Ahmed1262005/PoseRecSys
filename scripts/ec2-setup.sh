#!/usr/bin/env bash
# ==============================================================================
# OutfitTransformer - EC2 One-Time Setup Script
# ==============================================================================
# Run this on a fresh Ubuntu 22.04 EC2 instance (g6.4xlarge with NVIDIA L4)
#
# Usage:
#   chmod +x scripts/ec2-setup.sh
#   sudo ./scripts/ec2-setup.sh
# ==============================================================================

set -euo pipefail

echo "============================================"
echo " OutfitTransformer EC2 Setup"
echo "============================================"

# --- 1. System updates ---
echo ""
echo "[1/6] Updating system packages..."
apt-get update && apt-get upgrade -y

# --- 2. Install Docker ---
echo ""
echo "[2/6] Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sh
    usermod -aG docker ubuntu
    systemctl enable docker
    systemctl start docker
    echo "Docker installed: $(docker --version)"
else
    echo "Docker already installed: $(docker --version)"
fi

# --- 3. Install Docker Compose plugin ---
echo ""
echo "[3/6] Installing Docker Compose..."
if ! docker compose version &> /dev/null; then
    apt-get install -y docker-compose-plugin
    echo "Docker Compose installed: $(docker compose version)"
else
    echo "Docker Compose already installed: $(docker compose version)"
fi

# --- 4. Install NVIDIA drivers + Container Toolkit ---
echo ""
echo "[4/6] Installing NVIDIA drivers and Container Toolkit..."
if ! command -v nvidia-smi &> /dev/null; then
    # Install NVIDIA drivers
    apt-get install -y linux-headers-$(uname -r)
    apt-get install -y nvidia-driver-550
    echo "NVIDIA driver installed (reboot may be required)"
else
    echo "NVIDIA driver already installed:"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
fi

# Install NVIDIA Container Toolkit (for --gpus flag)
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update
    apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
    echo "NVIDIA Container Toolkit installed"
else
    echo "NVIDIA Container Toolkit already installed"
fi

# --- 5. Create app directory ---
echo ""
echo "[5/6] Setting up application directory..."
APP_DIR="/opt/outfit-transformer"
mkdir -p "$APP_DIR"
mkdir -p "$APP_DIR/ssl"
mkdir -p "$APP_DIR/models"
mkdir -p "$APP_DIR/data/recbole"
chown -R ubuntu:ubuntu "$APP_DIR"

# --- 6. Summary ---
echo ""
echo "============================================"
echo " Setup Complete!"
echo "============================================"
echo ""
echo "Next steps (run as 'ubuntu' user, not root):"
echo ""
echo "  1. Clone the repo:"
echo "     cd /opt/outfit-transformer"
echo "     git clone git@github.com:YOUR_ORG/recommendationSystem.git ."
echo ""
echo "  2. Copy your .env file:"
echo "     cp .env.example .env"
echo "     nano .env  # fill in real values"
echo ""
echo "  3. Add Cloudflare origin certs:"
echo "     nano ssl/origin.pem      # paste certificate"
echo "     nano ssl/origin-key.pem  # paste private key"
echo "     chmod 600 ssl/origin-key.pem"
echo ""
echo "  4. Copy model files:"
echo "     # SASRec checkpoint -> models/"
echo "     # RecBole dataset   -> data/recbole/"
echo ""
echo "  5. Build and start:"
echo "     docker compose up -d --build"
echo ""
echo "  6. Verify:"
echo "     docker compose ps"
echo "     curl -s http://localhost:8000/health"
echo ""
echo "If NVIDIA driver was just installed, REBOOT first:"
echo "  sudo reboot"
echo ""
