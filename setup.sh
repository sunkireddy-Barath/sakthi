#!/bin/bash

# Setup script for Automated Invoice Compliance Tracker
# This script initializes the project and creates necessary directories

echo "🚀 Setting up Automated Invoice Compliance Tracker..."

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p incoming_invoices
mkdir -p processed_invoices/approved
mkdir -p processed_invoices/flagged
mkdir -p processed_invoices/pending
mkdir -p processed_invoices/error
mkdir -p data
mkdir -p logs
mkdir -p exports

# Create .env file from template if it doesn't exist
if [ ! -f .env ]; then
    echo "📄 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file and add your API keys!"
else
    echo "✅ .env file already exists"
fi

# Set permissions
echo "🔧 Setting permissions..."
chmod +x run.py
chmod +x setup.sh

# Create placeholder files to maintain directory structure in git
touch incoming_invoices/.gitkeep
touch processed_invoices/approved/.gitkeep
touch processed_invoices/flagged/.gitkeep
touch processed_invoices/pending/.gitkeep
touch processed_invoices/error/.gitkeep
touch data/.gitkeep
touch logs/.gitkeep
touch exports/.gitkeep

echo "✅ Directory structure created successfully!"
echo ""
echo "📋 Next steps:"
echo "1. pip install -r requirements.txt"
echo "2. Edit .env file with your API keys"
echo "3. python run.py (or streamlit run app.py)"
echo ""
echo "🌐 The dashboard will be available at: http://localhost:8501"
echo "📁 Drop PDF invoices in the 'incoming_invoices' folder"
echo ""
echo "🎉 Setup complete! Happy coding!"
