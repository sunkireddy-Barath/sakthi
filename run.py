"""
Startup script for the Invoice Compliance Tracker

This script handles system initialization and starts the necessary services.
"""

import os
import sys
import subprocess
import threading
import time
from pathlib import Path

def setup_environment():
    """Set up the environment and install dependencies"""
    print("🚀 Setting up Invoice Compliance Tracker...")
    
    # Check if virtual environment is recommended
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Recommendation: Create a virtual environment first:")
        print("   python -m venv invoice_tracker_env")
        print("   source invoice_tracker_env/bin/activate  # On Linux/Mac")
        print("   # or")
        print("   invoice_tracker_env\\Scripts\\activate     # On Windows")
        print()
    
    # Install dependencies
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("💡 Try installing manually: pip install -r requirements.txt")
        return False
    
    return True

def initialize_data():
    """Initialize the system with sample data"""
    print("📊 Initializing system data...")
    
    try:
        # Import after dependencies are installed
        from utils import initialize_system
        initialize_system()
        print("✅ System data initialized!")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize data: {e}")
        return False

def start_file_watcher():
    """Start the file watcher in a separate thread"""
    try:
        from file_watcher import run_watcher_service
        
        def run_watcher():
            print("👁️  Starting file watcher...")
            run_watcher_service()
        
        watcher_thread = threading.Thread(target=run_watcher, daemon=True)
        watcher_thread.start()
        print("✅ File watcher started!")
        return True
    except Exception as e:
        print(f"❌ Failed to start file watcher: {e}")
        return False

def start_streamlit_app():
    """Start the Streamlit application"""
    print("🌐 Starting Streamlit application...")
    
    try:
        # Start Streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port=8501"])
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Failed to start Streamlit: {e}")

def check_configuration():
    """Check if the system is properly configured"""
    print("🔧 Checking configuration...")
    
    issues = []
    
    # Check .env file
    if not os.path.exists('.env'):
        issues.append("Missing .env file")
    else:
        # Check if critical environment variables are set
        from dotenv import load_dotenv
        load_dotenv()
        
        if not os.getenv('OPENAI_API_KEY'):
            issues.append("OPENAI_API_KEY not set in .env file")
    
    # Check required directories
    required_dirs = [
        'incoming_invoices',
        'processed_invoices',
        'data',
        'logs',
        'exports'
    ]
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            issues.append(f"Missing directory: {directory}")
    
    if issues:
        print("⚠️  Configuration issues found:")
        for issue in issues:
            print(f"   • {issue}")
        print("\n💡 Please fix these issues before proceeding.")
        return False
    else:
        print("✅ Configuration looks good!")
        return True

def main():
    """Main startup function"""
    print("=" * 60)
    print("📋 AUTOMATED INVOICE COMPLIANCE TRACKER")
    print("=" * 60)
    print()
    
    # Step 1: Setup environment
    if not setup_environment():
        print("❌ Setup failed. Please fix the issues and try again.")
        return
    
    # Step 2: Check configuration
    if not check_configuration():
        print("❌ Configuration check failed. Please fix the issues and try again.")
        print("\n🔧 Quick fixes:")
        print("   1. Copy .env to your project directory and fill in API keys")
        print("   2. Make sure all required directories exist")
        return
    
    # Step 3: Initialize data
    if not initialize_data():
        print("⚠️  Data initialization failed, but continuing anyway...")
    
    # Step 4: Start file watcher
    if not start_file_watcher():
        print("⚠️  File watcher failed to start, but continuing anyway...")
    
    print("\n🎉 System startup complete!")
    print("\n📱 Access the dashboard at: http://localhost:8501")
    print("📁 Drop PDF invoices in the 'incoming_invoices' folder for automatic processing")
    print("🛑 Press Ctrl+C to stop the application")
    print("\n" + "=" * 60)
    
    # Step 5: Start Streamlit app (this will block)
    start_streamlit_app()

if __name__ == "__main__":
    main()
