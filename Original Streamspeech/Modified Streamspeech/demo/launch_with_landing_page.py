#!/usr/bin/env python3
"""
Launch StreamSpeech with Landing Page
====================================

This script launches the StreamSpeech desktop application with the integrated landing page.
"""

import sys
import os

# Add the demo directory to the path
demo_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, demo_dir)

# Import and run the main application
if __name__ == "__main__":
    try:
        from enhanced_desktop_app_streamlit_ui1 import StreamSpeechComparisonApp
        from PySide6.QtWidgets import QApplication
        
        # Create Qt application
        app = QApplication(sys.argv)
        app.setApplicationName("StreamSpeech with Landing Page")
        app.setApplicationVersion("1.0")
        
        # Create and show the main window
        window = StreamSpeechComparisonApp()
        window.show()
        
        # Run the application
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"Error launching StreamSpeech: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
