#!/bin/bash
set -e

echo "==============================================================="
echo " Deploying Diabetic Retinopathy CDSS to Production "
echo "==============================================================="

# 1. Check if .env.production exists
if [ ! -f .env.production ]; then
    echo "❌ ERROR: .env.production file not found!"
    echo "Please copy .env.production.example to .env.production and configure it."
    exit 1
fi

# 2. Pull latest code (if using git)
echo "📦 Pulling latest code..."
git pull origin main || echo "Git pull failed or not a git repository. Continuing..."

# 3. Build the Docker images
echo "🔨 Building Docker images..."
docker-compose -f docker-compose.prod.yml build

# 4. Apply Database Migrations & Restart Containers
echo "🚀 Starting services..."
docker-compose -f docker-compose.prod.yml up -d

# 5. Clean up unused images to free space
echo "🧹 Cleaning up old Docker images..."
docker image prune -f

echo "==============================================================="
echo "✅ Deployment successful!"
echo "The application should now be running on port 3000."
echo "View logs with: docker-compose -f docker-compose.prod.yml logs -f"
echo "==============================================================="
