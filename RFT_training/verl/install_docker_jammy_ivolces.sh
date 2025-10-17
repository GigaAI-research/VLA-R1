#!/usr/bin/env bash
set -euo pipefail

# 1)（可选）把 APT 源换到火山内网（已换过就跳过）
cp /etc/apt/sources.list /etc/apt/sources.list.bak.$(date +%F) || true
cat > /etc/apt/sources.list <<'EOF'
deb http://mirrors.ivolces.com/ubuntu/ jammy main restricted universe multiverse
deb http://mirrors.ivolces.com/ubuntu/ jammy-updates main restricted universe multiverse
deb http://mirrors.ivolces.com/ubuntu/ jammy-backports main restricted universe multiverse
deb http://mirrors.ivolces.com/ubuntu/ jammy-security main restricted universe multiverse
EOF
apt-get clean
apt-get update

# 2) 配置 Docker APT 仓库（火山镜像）
apt-get install -y ca-certificates curl gnupg lsb-release
mkdir -p /etc/apt/keyrings
curl -fsSL https://mirrors.ivolces.com/docker/linux/ubuntu/gpg \
  | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] \
https://mirrors.ivolces.com/docker/linux/ubuntu jammy stable" \
  > /etc/apt/sources.list.d/docker.list

# 3) 安装 Docker
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# 4) 启动 & 开机自启（若系统没 systemd，请看下方“无 systemd 启动”）
systemctl enable --now docker || true

# 5) 基本检查
docker --version || true
docker info       || true
echo "[OK] Docker installed."
