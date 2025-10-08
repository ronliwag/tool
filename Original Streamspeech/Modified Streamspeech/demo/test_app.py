"""
Test script to verify the desktop application works correctly
"""

import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'fairseq'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'demo'))

try:
    print("Testing imports...")
    from app import StreamSpeechS2STAgent, OnlineFeatureExtractor, reset, run, SAMPLE_RATE
    import app
    print("✓ All imports successful")
    
    print("Testing global variables...")
    print(f"ASR: {app.ASR}")
    print(f"ST: {app.ST}")
    print(f"S2ST: {app.S2ST}")
    print(f"SAMPLE_RATE: {SAMPLE_RATE}")
    print("✓ Global variables accessible")
    
    print("Testing agent creation...")
    import json
    import argparse
    
    with open('config.json', 'r') as f:
        args_dict = json.load(f)
    
    parser = argparse.ArgumentParser()
    StreamSpeechS2STAgent.add_args(parser)
    
    args_list = []
    for key, value in args_dict.items():
        if isinstance(value, bool):
            if value:
                args_list.append(f'--{key}')
        else:
            args_list.append(f'--{key}')
            args_list.append(str(value))
    
    args = parser.parse_args(args_list)
    agent = StreamSpeechS2STAgent(args)
    print("✓ Agent created successfully")
    
    print("\n🎉 All tests passed! Desktop application should work correctly.")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()







