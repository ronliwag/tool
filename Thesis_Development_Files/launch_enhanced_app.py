#!/usr/bin/env python3
"""
LAUNCH ENHANCED DESKTOP APP
Fixed launcher that handles path issues
"""

import os
import sys
import subprocess

def main():
    """Launch the enhanced desktop app with proper paths"""
    print("Launching Enhanced Desktop App...")
    
    # Change to the correct directory
    app_dir = "Original Streamspeech/Modified Streamspeech/demo"
    
    if os.path.exists(app_dir):
        print(f"Changing to directory: {app_dir}")
        os.chdir(app_dir)
        
        # Check if enhanced_desktop_app.py exists
        app_file = "enhanced_desktop_app.py"
        if os.path.exists(app_file):
            print(f"Found {app_file}")
            
            # Check if config files exist
            config_files = ["config.json", "config_modified.json"]
            for config_file in config_files:
                if os.path.exists(config_file):
                    print(f"[OK] Found config: {config_file}")
                else:
                    print(f"[WARNING] Missing config: {config_file}")
            
            print("Launching enhanced desktop app...")
            try:
                # Launch the app
                subprocess.run([sys.executable, app_file], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error launching app: {e}")
            except KeyboardInterrupt:
                print("App closed by user")
        else:
            print(f"Error: {app_file} not found")
    else:
        print(f"Error: Directory {app_dir} not found")

if __name__ == "__main__":
    main()
