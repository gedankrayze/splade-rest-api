#!/usr/bin/env python3
"""
Test script for the automatic model download functionality
"""

import os
import sys
import shutil

# Add parent directory to path so we can import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.config import settings
from app.core.model_downloader import download_model_if_needed

def test_model_download():
    """Test the model download functionality"""
    print("=" * 50)
    print("MODEL DOWNLOAD TEST")
    print("=" * 50)
    
    # Get the model directory from settings
    model_name = settings.MODEL_NAME
    model_dir = settings.MODEL_DIR.format(model_name=model_name)
    model_id = settings.MODEL_HF_ID
    
    print(f"Model name: {model_name}")
    print(f"Model directory: {model_dir}")
    print(f"Model ID: {model_id}")
    
    # Check if model directory exists
    dir_exists = os.path.exists(model_dir)
    print(f"Directory exists: {dir_exists}")
    
    # Check if model files exist
    model_files = []
    if dir_exists:
        model_files = os.listdir(model_dir)
    
    has_model = any(f.endswith('.bin') or f.endswith('.safetensors') for f in model_files)
    has_config = "config.json" in model_files
    
    print(f"Has model files: {has_model}")
    print(f"Has config: {has_config}")
    
    # Ask user if they want to test download
    if has_model and has_config:
        print("\nModel already exists. Do you want to rename the directory to test download? (y/n)")
        response = input().strip().lower()
        
        if response == 'y':
            # Rename the directory to simulate missing model
            backup_dir = f"{model_dir}_backup"
            print(f"Renaming {model_dir} to {backup_dir}")
            
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
                
            os.rename(model_dir, backup_dir)
            
            try:
                # Test download
                print(f"\nAttempting to download model {model_id}...")
                success = download_model_if_needed(model_dir, model_id)
                
                if success:
                    print(f"✓ Model downloaded successfully to {model_dir}")
                    # Check files
                    model_files = os.listdir(model_dir)
                    print(f"Downloaded files: {', '.join(model_files[:5])}...")
                    print(f"Total files: {len(model_files)}")
                else:
                    print("✗ Model download failed")
            finally:
                # Clean up - restore original directory
                print("\nCleaning up - removing downloaded model and restoring backup...")
                if os.path.exists(model_dir):
                    shutil.rmtree(model_dir)
                os.rename(backup_dir, model_dir)
        else:
            print("Download test skipped.")
    else:
        # Model doesn't exist, test download directly
        print(f"\nAttempting to download model {model_id}...")
        success = download_model_if_needed(model_dir, model_id)
        
        if success:
            print(f"✓ Model downloaded successfully to {model_dir}")
            # Check files
            model_files = os.listdir(model_dir)
            print(f"Downloaded files: {', '.join(model_files[:5])}...")
            print(f"Total files: {len(model_files)}")
        else:
            print("✗ Model download failed")
    
    print("\nTest completed.")

if __name__ == "__main__":
    test_model_download()
