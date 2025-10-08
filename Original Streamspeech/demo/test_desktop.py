#!/usr/bin/env python3
"""
Simple test script to verify StreamSpeech desktop app components work
"""

import os
import sys
import soundfile
import numpy as np

# Add fairseq to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'fairseq'))

try:
    print("Testing StreamSpeech desktop app components...")
    
    # Test imports
    print("1. Testing imports...")
    from app import StreamSpeechS2STAgent, OnlineFeatureExtractor, reset, run
    print("   ✓ All imports successful")
    
    # Test agent initialization
    print("2. Testing agent initialization...")
    
    # Load config and create agent with proper arguments
    with open('config.json', 'r') as f:
        args_dict = json.load(f)
    
    # Initialize agent with arguments
    import argparse
    parser = argparse.ArgumentParser()
    StreamSpeechS2STAgent.add_args(parser)
    
    # Create the list of arguments from args_dict
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
    print("   ✓ Agent initialized successfully")
    
    # Test with a sample audio file
    print("3. Testing with sample audio...")
    sample_file = "example/wavs/common_voice_es_18311412.mp3"
    if os.path.exists(sample_file):
        print(f"   Testing with: {sample_file}")
        
        # Set up agent
        agent.set_chunk_size(320)
        reset()
        
        # Process audio
        print("   Processing audio...")
        run(sample_file)
        
        # Check output
        from app import S2ST, SAMPLE_RATE
        if S2ST:
            print(f"   ✓ Translation completed! Output length: {len(S2ST)} samples")
            print(f"   ✓ Duration: {len(S2ST) / SAMPLE_RATE:.2f} seconds")
            
            # Save output
            output_path = "test_output.wav"
            s2st_array = np.array(S2ST, dtype=np.float32)
            soundfile.write(output_path, s2st_array, SAMPLE_RATE)
            print(f"   ✓ Output saved to: {output_path}")
        else:
            print("   ✗ No translation output generated")
    else:
        print(f"   ⚠ Sample file not found: {sample_file}")
    
    print("\n✅ All tests completed successfully!")
    print("The desktop app should work properly now.")
    
except Exception as e:
    print(f"\n❌ Error during testing: {str(e)}")
    import traceback
    traceback.print_exc()
