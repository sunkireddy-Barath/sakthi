"""
Simple deployment test script to verify all dependencies work correctly
"""
import sys
import os

def test_imports():
    """Test all critical imports"""
    try:
        print("Testing imports...")
        
        # Core
        import streamlit as st
        print("✅ Streamlit imported successfully")
        
        import pandas as pd
        print("✅ Pandas imported successfully")
        
        import numpy as np
        print("✅ Numpy imported successfully")
        
        # PDF Processing
        import PyPDF2
        print("✅ PyPDF2 imported successfully")
        
        # ML Libraries
        import matplotlib
        matplotlib.use('Agg')  # Set backend before importing pyplot
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")
        
        from sklearn.ensemble import IsolationForest
        print("✅ Scikit-learn imported successfully")
        
        # Vector Database
        import chromadb
        print("✅ ChromaDB imported successfully")
        
        # OpenAI
        from openai import OpenAI
        print("✅ OpenAI imported successfully")
        
        # Optional imports
        try:
            import cv2
            print("✅ OpenCV imported successfully")
        except ImportError:
            print("⚠️ OpenCV not available (this is OK for deployment)")
        
        print("\n🎉 All critical imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

if __name__ == "__main__":
    test_imports()
