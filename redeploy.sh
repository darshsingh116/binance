#!/bin/bash

# Configuration (change as per your setup)
REPO_DIR="/home/ec2-user/binance"
DOCKER_IMAGE_NAME="binance-dashboard"
DOCKER_CONTAINER_NAME="binance-ui"
PORT=8501

echo "🌀 Pulling latest changes from GitHub..."
cd "$REPO_DIR" || { echo "❌ Failed to change directory to $REPO_DIR"; exit 1; }
git pull origin main || { echo "❌ Git pull failed"; exit 1; }

echo "🔨 Rebuilding Docker image: $DOCKER_IMAGE_NAME"
docker build -t "$DOCKER_IMAGE_NAME" . || { echo "❌ Docker build failed"; exit 1; }

echo "🛑 Stopping old container: $DOCKER_CONTAINER_NAME (if exists)"
docker stop "$DOCKER_CONTAINER_NAME" 2>/dev/null
docker rm "$DOCKER_CONTAINER_NAME" 2>/dev/null

echo "🚀 Starting new container: $DOCKER_CONTAINER_NAME"
docker run -d \
  --name "$DOCKER_CONTAINER_NAME" \
  -p $PORT:8501 \
  --restart unless-stopped \
  "$DOCKER_IMAGE_NAME" || { echo "❌ Failed to run Docker container"; exit 1; }

echo "✅ Deployment complete. App should be live at: http://<your-ec2-ip>:$PORT"
