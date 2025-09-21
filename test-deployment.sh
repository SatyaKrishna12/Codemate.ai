#!/bin/bash
# Vercel Deployment Test Script

echo "🚀 Testing Deep Researcher Agent Deployment"
echo "=========================================="

# Replace with your actual Vercel URL
VERCEL_URL="https://your-app-name.vercel.app"

echo "1. Testing Health Check..."
curl -s "$VERCEL_URL/api/v1/health" | jq .

echo -e "\n2. Testing Main Page..."
curl -s -o /dev/null -w "%{http_code}" "$VERCEL_URL/"

echo -e "\n3. Testing Chat Interface..."
curl -s -o /dev/null -w "%{http_code}" "$VERCEL_URL/chat"

echo -e "\n4. Testing Session Start..."
curl -X POST -s "$VERCEL_URL/api/v1/researcher/start" | jq .

echo -e "\n✅ Testing complete!"
echo "Visit your app at: $VERCEL_URL"