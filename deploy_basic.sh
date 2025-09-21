#!/bin/bash
# Simple deployment helper script

echo "🚀 Preparing for deployment..."

# Backup current requirements
if [ -f "requirements.txt" ]; then
    cp requirements.txt requirements_full.txt
    echo "✅ Backed up full requirements to requirements_full.txt"
fi

# Use basic requirements for deployment
cp requirements_basic.txt requirements.txt
echo "✅ Using basic requirements for deployment"

echo "📋 Current requirements.txt contents:"
cat requirements.txt

echo ""
echo "🎯 Ready for deployment! Commit and push to GitHub."
echo "💡 If deployment succeeds, you can gradually add back features."
