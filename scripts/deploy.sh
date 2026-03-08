#!/usr/bin/env bash
# ==============================================================================
# OutfitTransformer - Deploy Script
# ==============================================================================
# Run on EC2 to pull latest code and redeploy.
# This is what GitHub Actions does automatically on push to main.
#
# Usage:
#   ./scripts/deploy.sh
# ==============================================================================

set -euo pipefail

APP_DIR="/opt/outfit-transformer"
cd "$APP_DIR"

echo "=== Pulling latest code ==="
git pull origin main

echo ""
echo "=== Building Docker image ==="
docker compose build

echo ""
echo "=== Deploying ==="
docker compose up -d

echo ""
echo "=== Waiting for health check ==="
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "Health check passed!"
        echo ""
        docker compose ps
        exit 0
    fi
    echo "Waiting for API... ($i/30)"
    sleep 5
done

echo ""
echo "ERROR: Health check failed after 150 seconds"
echo ""
echo "=== Last 50 lines of API logs ==="
docker compose logs api --tail 50
exit 1
