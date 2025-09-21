#!/usr/bin/env python3
"""
Simple import test to isolate the libGL.so.1 issue
"""

import sys
import os

# Configure environment before any imports
os.environ['MPLBACKEND'] = 'Agg'
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

def test_import(module_name, package_name=None):
    """Test importing a specific module"""
    try:
        if package_name:
            exec(f"from {package_name} import {module_name}")
        else:
            exec(f"import {module_name}")
        print(f"✅ {module_name} imported successfully")
        return True
    except Exception as e:
        print(f"❌ {module_name} failed: {e}")
        return False

def main():
    print("🔍 Testing imports to isolate libGL.so.1 issue...\n")
    
    # Test basic imports first
    test_import("os")
    test_import("sys")
    
    # Test core dependencies
    test_import("streamlit")
    test_import("pandas")
    test_import("numpy")
    
    # Test potentially problematic imports
    print("\n📊 Testing plotting libraries...")
    
    # Configure matplotlib first
    try:
        import matplotlib
        matplotlib.use('Agg')
        print("✅ Matplotlib backend configured")
    except Exception as e:
        print(f"❌ Matplotlib configuration failed: {e}")
    
    test_import("matplotlib.pyplot")
    test_import("seaborn")
    test_import("plotly.express")
    
    # Test ML libraries
    print("\n🤖 Testing ML libraries...")
    test_import("sklearn")
    test_import("IsolationForest", "sklearn.ensemble")
    
    # Test OpenCV
    print("\n📷 Testing OpenCV...")
    test_import("cv2")
    
    # Test our backend
    print("\n🏗️ Testing backend modules...")
    test_import("backend")
    
    print("\n🎯 Import test completed!")

if __name__ == "__main__":
    main()
