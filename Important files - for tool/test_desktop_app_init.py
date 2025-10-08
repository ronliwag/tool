"""
Test Desktop App Initialization
==============================

Test script to verify the desktop app initialization works correctly.
"""

import os
import sys
import numpy as np
import soundfile as sf

# Add paths
sys.path.append("Original Streamspeech/Modified Streamspeech/integration")
sys.path.append("Original Streamspeech/Modified Streamspeech/demo")

def test_desktop_app_initialization():
    """Test desktop app initialization"""
    try:
        print("=" * 50)
        print("DESKTOP APP INITIALIZATION TEST")
        print("=" * 50)
        
        # Test 1: Import desktop app
        print("\n1. Importing desktop app...")
        from enhanced_desktop_app import StreamSpeechComparisonApp
        print("[OK] Desktop app imported successfully")
        
        # Test 2: Test defense pipeline import
        print("\n2. Testing defense pipeline import...")
        from defense_streamspeech_pipeline import DefenseStreamSpeechPipeline
        print("[OK] Defense pipeline imported successfully")
        
        # Test 3: Create defense pipeline
        print("\n3. Creating defense pipeline...")
        pipeline = DefenseStreamSpeechPipeline()
        print("[OK] Defense pipeline created")
        
        # Test 4: Check initialization status
        print("\n4. Checking initialization status...")
        if hasattr(pipeline, 'is_initialized'):
            is_init = pipeline.is_initialized()
            print(f"[STATUS] Pipeline initialized: {is_init}")
        else:
            print("[STATUS] No is_initialized method")
        
        # Test 5: Test initialize_models
        print("\n5. Testing initialize_models...")
        result = pipeline.initialize_models()
        print(f"[RESULT] initialize_models returned: {result}")
        
        # Test 6: Create mock desktop app instance
        print("\n6. Testing desktop app initialization...")
        import tkinter as tk
        
        # Create a mock root window (hidden)
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        try:
            # Create desktop app instance
            app = StreamSpeechComparisonApp(root)
            print("[OK] Desktop app created successfully")
            
            # Check if modified_streamspeech is initialized
            if hasattr(app, 'modified_streamspeech'):
                if app.modified_streamspeech is None:
                    print("[ERROR] modified_streamspeech is None")
                    return False
                else:
                    print(f"[OK] modified_streamspeech type: {type(app.modified_streamspeech)}")
                    
                    # Check if it has is_initialized method
                    if hasattr(app.modified_streamspeech, 'is_initialized'):
                        is_init = app.modified_streamspeech.is_initialized()
                        print(f"[STATUS] modified_streamspeech initialized: {is_init}")
                    else:
                        print("[INFO] modified_streamspeech has no is_initialized method")
            else:
                print("[ERROR] Desktop app has no modified_streamspeech attribute")
                return False
            
        except Exception as e:
            print(f"[ERROR] Desktop app creation failed: {e}")
            return False
        finally:
            # Clean up
            root.destroy()
        
        print("\n" + "=" * 50)
        print("DESKTOP APP INITIALIZATION TEST COMPLETED")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Desktop app initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting Desktop App Initialization Test...")
    
    success = test_desktop_app_initialization()
    
    if success:
        print("\n[SUCCESS] Desktop app initialization working!")
        print("The desktop app should now work correctly.")
    else:
        print("\n[FAILED] Desktop app initialization needs attention.")

