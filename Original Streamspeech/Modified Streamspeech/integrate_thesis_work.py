"""
Integration Script for Thesis Work
=================================

This script helps integrate your existing thesis work from D:\Thesis - Tool
into the Modified Streamspeech folder for comparison with the original.

Usage:
    python integrate_thesis_work.py

This will:
1. Copy your trained models and modifications
2. Set up the proper directory structure
3. Ensure compatibility with the comparison tool
"""

import os
import shutil
import sys
from pathlib import Path

def main():
    """Main integration function."""
    print("StreamSpeech Thesis Work Integration")
    print("=" * 40)
    
    # Source and destination paths
    thesis_source = r"D:\Thesis - Tool"
    modified_dest = os.path.dirname(__file__)
    
    print(f"Source: {thesis_source}")
    print(f"Destination: {modified_dest}")
    
    # Check if source exists
    if not os.path.exists(thesis_source):
        print(f"ERROR: Source directory not found: {thesis_source}")
        return
    
    # Create necessary directories
    directories_to_create = [
        "models",
        "evaluation", 
        "configs",
        "integration",
        "trained_models",
        "pretrained_models",
        "examples",
        "logs"
    ]
    
    print("\nCreating directory structure...")
    for dir_name in directories_to_create:
        dir_path = os.path.join(modified_dest, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        print(f"  ✓ Created: {dir_name}/")
    
    # Copy key files and directories
    items_to_copy = [
        ("trained_models", "trained_models"),
        ("pretrained_models", "pretrained_models"), 
        ("evaluation", "evaluation"),
        ("configs", "configs"),
        ("integration_scripts", "integration"),
        ("examples", "examples"),
        ("logs", "logs"),
        ("THESIS_EVALUATION_SUMMARY.md", "THESIS_EVALUATION_SUMMARY.md"),
        ("THESIS_SYSTEM_STATUS_REPORT.md", "THESIS_SYSTEM_STATUS_REPORT.md"),
        ("run_model_comparison.py", "run_model_comparison.py"),
        ("run_streamspeech_comparison.py", "run_streamspeech_comparison.py")
    ]
    
    print("\nCopying thesis work...")
    for source_item, dest_item in items_to_copy:
        source_path = os.path.join(thesis_source, source_item)
        dest_path = os.path.join(modified_dest, dest_item)
        
        try:
            if os.path.exists(source_path):
                if os.path.isdir(source_path):
                    if os.path.exists(dest_path):
                        shutil.rmtree(dest_path)
                    shutil.copytree(source_path, dest_path)
                    print(f"  ✓ Copied directory: {source_item}")
                else:
                    shutil.copy2(source_path, dest_path)
                    print(f"  ✓ Copied file: {source_item}")
            else:
                print(f"  ⚠ Not found: {source_item}")
        except Exception as e:
            print(f"  ✗ Error copying {source_item}: {str(e)}")
    
    # Create integration status file
    status_file = os.path.join(modified_dest, "INTEGRATION_STATUS.md")
    with open(status_file, 'w') as f:
        f.write("""# Integration Status

## Thesis Work Integration
- Source: D:\\Thesis - Tool
- Destination: Modified Streamspeech folder
- Status: Completed

## Available Components
- Trained models with ODConv, GRC, and LoRA modifications
- Evaluation metrics and statistical analysis
- Configuration files for modified system
- Integration scripts for StreamSpeech compatibility
- Thesis evaluation results and reports

## Usage
1. Run the comparison tool: `demo/run_comparison_tool.bat`
2. Switch between Original and Modified modes
3. Process audio samples for comparison
4. Review evaluation metrics for thesis defense

## Next Steps
1. Test the comparison tool with your Spanish audio samples
2. Verify that both Original and Modified modes work correctly
3. Prepare demonstration materials for thesis defense
4. Document any additional findings or improvements

---
*Integration completed for thesis defense preparation*
""")
    
    print(f"\n✓ Integration status saved to: INTEGRATION_STATUS.md")
    
    print("\n" + "=" * 40)
    print("Integration completed successfully!")
    print("\nNext steps:")
    print("1. Run: demo/run_comparison_tool.bat")
    print("2. Test with your Spanish audio samples")
    print("3. Compare Original vs Modified performance")
    print("4. Prepare for thesis defense")
    print("=" * 40)

if __name__ == "__main__":
    main()







