#!/usr/bin/env python3
"""
Installation Script for Complete S2ST Pipeline Dependencies
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages for S2ST pipeline"""
    print("Installing Complete S2ST Pipeline Dependencies")
    print("=" * 50)
    
    requirements_file = "requirements_s2st.txt"
    
    if not os.path.exists(requirements_file):
        print(f"Error: {requirements_file} not found!")
        return False
    
    try:
        # Install requirements
        print("Installing packages from requirements file...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", requirements_file
        ], check=True, capture_output=True, text=True)
        
        print("Installation completed successfully!")
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error during installation: {e}")
        print(f"Error output: {e.stderr}")
        return False

def verify_installation():
    """Verify that key packages are installed"""
    print("\nVerifying Installation")
    print("-" * 30)
    
    packages_to_check = [
        "torch",
        "torchaudio", 
        "transformers",
        "librosa",
        "soundfile",
        "whisper"
    ]
    
    for package in packages_to_check:
        try:
            __import__(package)
            print(f"✓ {package} - OK")
        except ImportError:
            print(f"✗ {package} - MISSING")

def main():
    """Main installation function"""
    print("S2ST Pipeline Dependency Installation")
    print("=" * 40)
    
    # Install requirements
    if install_requirements():
        print("\n" + "=" * 50)
        print("INSTALLATION SUCCESSFUL")
        print("=" * 50)
        
        # Verify installation
        verify_installation()
        
        print("\nNext steps:")
        print("1. Test the ASR component: python spanish_asr_component.py")
        print("2. Test the translation component: python spanish_english_translation.py")
        print("3. Test the complete pipeline: python complete_s2st_pipeline.py")
        
    else:
        print("\n" + "=" * 50)
        print("INSTALLATION FAILED")
        print("=" * 50)
        print("Please check the error messages above and try again.")

if __name__ == "__main__":
    main()

