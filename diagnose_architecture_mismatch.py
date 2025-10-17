#!/usr/bin/env python3
"""
Architecture Mismatch Diagnostic Tool
====================================

This script performs deep inspection of:
1. Checkpoint structure (what keys exist in trained weights)
2. Current model structure (what keys the code creates)
3. GRC block architecture
4. Weight shapes and dimensions
5. Potential remapping strategies
"""

import os
import sys
import torch
import json
from pathlib import Path
from collections import OrderedDict

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'Important files - for tool'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Original Streamspeech', 'Modified Streamspeech', 'models'))

print("=" * 80)
print("ARCHITECTURE MISMATCH DIAGNOSTIC TOOL")
print("=" * 80)
print()

class ArchitectureDiagnostic:
    """Diagnose the exact architecture mismatch"""
    
    def __init__(self):
        self.checkpoint_path = "trained_models/hifigan_checkpoints/best_model.pth"
        self.config_path = "trained_models/model_config.json"
        self.results = {
            'checkpoint_keys': [],
            'model_keys': [],
            'mismatched_keys': [],
            'checkpoint_shapes': {},
            'model_shapes': {},
            'grc_structure': {}
        }
    
    def inspect_checkpoint(self):
        """Test 1: Inspect checkpoint structure"""
        print("\n" + "=" * 80)
        print("TEST 1: CHECKPOINT STRUCTURE INSPECTION")
        print("=" * 80)
        
        try:
            if not os.path.exists(self.checkpoint_path):
                print(f"✗ Checkpoint not found: {self.checkpoint_path}")
                return None
            
            print(f"Loading checkpoint from: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            
            # Extract state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("✓ Found 'model_state_dict' in checkpoint")
                
                # Show training info
                if 'epoch' in checkpoint:
                    print(f"  Training epoch: {checkpoint['epoch']}")
                if 'loss' in checkpoint:
                    print(f"  Training loss: {checkpoint['loss']:.6f}")
            else:
                state_dict = checkpoint
                print("✓ Checkpoint is direct state dict")
            
            # Remove 'generator.' prefix if present
            cleaned_state_dict = OrderedDict()
            for key, value in state_dict.items():
                new_key = key.replace('generator.', '')
                cleaned_state_dict[new_key] = value
                self.results['checkpoint_keys'].append(new_key)
                self.results['checkpoint_shapes'][new_key] = list(value.shape)
            
            print(f"\n✓ Checkpoint has {len(cleaned_state_dict)} parameters")
            print(f"✓ Total parameters: {sum(v.numel() for v in cleaned_state_dict.values()):,}")
            
            # Analyze structure patterns
            print("\n📊 KEY PATTERN ANALYSIS:")
            self.analyze_key_patterns(list(cleaned_state_dict.keys()))
            
            # Focus on MRF layers
            print("\n🔍 MRF LAYER STRUCTURE (Multi-Receptive Field Fusion):")
            mrf_keys = [k for k in cleaned_state_dict.keys() if k.startswith('mrfs.')]
            if mrf_keys:
                self.analyze_mrf_structure(mrf_keys, cleaned_state_dict)
            else:
                print("  ✗ No MRF keys found!")
            
            return cleaned_state_dict
            
        except Exception as e:
            print(f"✗ Error inspecting checkpoint: {e}")
            import traceback
            print(traceback.format_exc())
            return None
    
    def analyze_key_patterns(self, keys):
        """Analyze patterns in checkpoint keys"""
        patterns = {}
        for key in keys:
            # Get the top-level module
            top_level = key.split('.')[0]
            if top_level not in patterns:
                patterns[top_level] = 0
            patterns[top_level] += 1
        
        for module, count in sorted(patterns.items()):
            print(f"  {module:.<30} {count:>4} parameters")
    
    def analyze_mrf_structure(self, mrf_keys, state_dict):
        """Deep analysis of MRF layer structure"""
        # Group by MRF index
        mrf_groups = {}
        for key in mrf_keys:
            parts = key.split('.')
            mrf_idx = parts[0] + '.' + parts[1]  # e.g., "mrfs.0"
            if mrf_idx not in mrf_groups:
                mrf_groups[mrf_idx] = []
            mrf_groups[mrf_idx].append(key)
        
        for mrf_idx in sorted(mrf_groups.keys()):
            print(f"\n  {mrf_idx}:")
            keys_in_group = sorted(mrf_groups[mrf_idx])
            
            # Check for nested structure
            has_convs = any('convs.' in k for k in keys_in_group)
            has_conv1_conv2 = any('conv1' in k for k in keys_in_group) and any('conv2' in k for k in keys_in_group)
            
            print(f"    Keys: {len(keys_in_group)}")
            print(f"    Has nested 'convs': {has_convs}")
            print(f"    Has conv1/conv2: {has_conv1_conv2}")
            
            if has_convs:
                # Extract convs structure
                convs_pattern = {}
                for key in keys_in_group:
                    if 'convs.' in key:
                        # Extract convs.X pattern
                        parts = key.split('.')
                        convs_idx = None
                        for i, part in enumerate(parts):
                            if part == 'convs' and i+1 < len(parts):
                                convs_idx = parts[i+1]
                                break
                        if convs_idx is not None:
                            if convs_idx not in convs_pattern:
                                convs_pattern[convs_idx] = []
                            convs_pattern[convs_idx].append(key)
                
                print(f"    Number of GRC blocks (convs): {len(convs_pattern)}")
                for convs_idx in sorted(convs_pattern.keys()):
                    print(f"      convs.{convs_idx}: {len(convs_pattern[convs_idx])} params")
                    # Show first few
                    for key in sorted(convs_pattern[convs_idx])[:4]:
                        shape = state_dict[key].shape
                        print(f"        {key.split(mrf_idx + '.')[1]:.<40} {str(list(shape)):>20}")
                    if len(convs_pattern[convs_idx]) > 4:
                        print(f"        ... and {len(convs_pattern[convs_idx]) - 4} more")
            else:
                # Show all keys
                for key in keys_in_group[:5]:
                    shape = state_dict[key].shape
                    print(f"    {key.split(mrf_idx + '.')[1]:.<40} {str(list(shape)):>20}")
                if len(keys_in_group) > 5:
                    print(f"    ... and {len(keys_in_group) - 5} more")
            
            # Store structure info
            self.results['grc_structure'][mrf_idx] = {
                'total_params': len(keys_in_group),
                'has_nested_convs': has_convs,
                'has_conv1_conv2': has_conv1_conv2,
                'num_grc_blocks': len(convs_pattern) if has_convs else 0
            }
    
    def inspect_current_model(self):
        """Test 2: Inspect current model structure"""
        print("\n" + "=" * 80)
        print("TEST 2: CURRENT MODEL STRUCTURE")
        print("=" * 80)
        
        try:
            from integrate_trained_model import TrainedModelLoader
            
            print("Creating model with current code...")
            loader = TrainedModelLoader()
            
            # Just create architecture, don't load weights
            if not loader.load_model_config():
                print("✗ Failed to load model config")
                return None
            
            if not loader.create_model_architecture():
                print("✗ Failed to create model architecture")
                return None
            
            print("✓ Model architecture created")
            
            # Get state dict
            model_state_dict = loader.trained_model.state_dict()
            self.results['model_keys'] = list(model_state_dict.keys())
            
            for key, value in model_state_dict.items():
                self.results['model_shapes'][key] = list(value.shape)
            
            print(f"✓ Model has {len(model_state_dict)} parameters")
            print(f"✓ Total parameters: {sum(v.numel() for v in model_state_dict.values()):,}")
            
            # Analyze structure
            print("\n📊 KEY PATTERN ANALYSIS:")
            self.analyze_key_patterns(list(model_state_dict.keys()))
            
            # Focus on MRF layers
            print("\n🔍 MRF LAYER STRUCTURE:")
            mrf_keys = [k for k in model_state_dict.keys() if k.startswith('mrfs.')]
            if mrf_keys:
                self.analyze_mrf_structure(mrf_keys, model_state_dict)
            else:
                print("  ✗ No MRF keys found!")
            
            return model_state_dict
            
        except Exception as e:
            print(f"✗ Error inspecting model: {e}")
            import traceback
            print(traceback.format_exc())
            return None
    
    def compare_structures(self, checkpoint_dict, model_dict):
        """Test 3: Compare checkpoint vs model structure"""
        print("\n" + "=" * 80)
        print("TEST 3: STRUCTURE COMPARISON")
        print("=" * 80)
        
        if checkpoint_dict is None or model_dict is None:
            print("✗ Cannot compare - one or both dicts are None")
            return
        
        checkpoint_keys = set(checkpoint_dict.keys())
        model_keys = set(model_dict.keys())
        
        # Find mismatches
        missing_in_model = checkpoint_keys - model_keys
        missing_in_checkpoint = model_keys - checkpoint_keys
        common_keys = checkpoint_keys & model_keys
        
        print(f"\n📊 KEY STATISTICS:")
        print(f"  Checkpoint keys: {len(checkpoint_keys)}")
        print(f"  Model keys: {len(model_keys)}")
        print(f"  Common keys: {len(common_keys)}")
        print(f"  Missing in model: {len(missing_in_model)}")
        print(f"  Missing in checkpoint: {len(missing_in_checkpoint)}")
        
        if len(common_keys) == len(checkpoint_keys) == len(model_keys):
            print("\n✅ PERFECT MATCH! All keys align!")
        else:
            print("\n❌ MISMATCH DETECTED!")
        
        # Show missing keys in model (these are in checkpoint but not in model)
        if missing_in_model:
            print(f"\n🔴 KEYS IN CHECKPOINT BUT NOT IN MODEL ({len(missing_in_model)}):")
            # Group by pattern
            missing_patterns = {}
            for key in missing_in_model:
                pattern = '.'.join(key.split('.')[:3])  # First 3 levels
                if pattern not in missing_patterns:
                    missing_patterns[pattern] = []
                missing_patterns[pattern].append(key)
            
            for pattern in sorted(missing_patterns.keys())[:10]:
                print(f"  {pattern}.*")
                for key in sorted(missing_patterns[pattern])[:3]:
                    shape = checkpoint_dict[key].shape
                    print(f"    {key:.<60} {str(list(shape)):>20}")
                if len(missing_patterns[pattern]) > 3:
                    print(f"    ... and {len(missing_patterns[pattern]) - 3} more")
            
            if len(missing_patterns) > 10:
                print(f"  ... and {len(missing_patterns) - 10} more patterns")
        
        # Show missing keys in checkpoint (these are in model but not in checkpoint)
        if missing_in_checkpoint:
            print(f"\n🔴 KEYS IN MODEL BUT NOT IN CHECKPOINT ({len(missing_in_checkpoint)}):")
            # Group by pattern
            missing_patterns = {}
            for key in missing_in_checkpoint:
                pattern = '.'.join(key.split('.')[:3])
                if pattern not in missing_patterns:
                    missing_patterns[pattern] = []
                missing_patterns[pattern].append(key)
            
            for pattern in sorted(missing_patterns.keys())[:10]:
                print(f"  {pattern}.*")
                for key in sorted(missing_patterns[pattern])[:3]:
                    shape = model_dict[key].shape
                    print(f"    {key:.<60} {str(list(shape)):>20}")
                if len(missing_patterns[pattern]) > 3:
                    print(f"    ... and {len(missing_patterns[pattern]) - 3} more")
            
            if len(missing_patterns) > 10:
                print(f"  ... and {len(missing_patterns) - 10} more patterns")
        
        # Check shape mismatches for common keys
        shape_mismatches = []
        for key in common_keys:
            if list(checkpoint_dict[key].shape) != list(model_dict[key].shape):
                shape_mismatches.append(key)
        
        if shape_mismatches:
            print(f"\n⚠️  SHAPE MISMATCHES ({len(shape_mismatches)}):")
            for key in shape_mismatches[:10]:
                ckpt_shape = checkpoint_dict[key].shape
                model_shape = model_dict[key].shape
                print(f"  {key}")
                print(f"    Checkpoint: {list(ckpt_shape)}")
                print(f"    Model:      {list(model_shape)}")
            if len(shape_mismatches) > 10:
                print(f"  ... and {len(shape_mismatches) - 10} more")
        
        self.results['mismatched_keys'] = {
            'missing_in_model': list(missing_in_model),
            'missing_in_checkpoint': list(missing_in_checkpoint),
            'shape_mismatches': shape_mismatches
        }
    
    def analyze_grc_implementation(self):
        """Test 4: Analyze GRC implementation"""
        print("\n" + "=" * 80)
        print("TEST 4: GRC IMPLEMENTATION ANALYSIS")
        print("=" * 80)
        
        try:
            from grc_lora import GroupedResidualConvolution
            
            print("Testing GroupedResidualConvolution class...")
            
            # Create a test instance
            test_grc = GroupedResidualConvolution(
                channels=256,
                kernel_size=3,
                dilation=1,
                groups=4,
                lora_rank=4
            )
            
            print(f"✓ GRC instance created")
            print(f"\n📊 GRC STRUCTURE:")
            
            # Inspect what it creates
            state_dict = test_grc.state_dict()
            print(f"  Total parameters: {len(state_dict)}")
            
            for key, value in state_dict.items():
                print(f"    {key:.<40} {str(list(value.shape)):>20}")
            
            # Test forward pass
            print(f"\n🧪 TESTING FORWARD PASS:")
            test_input = torch.randn(1, 256, 100)  # [batch, channels, time]
            print(f"  Input shape: {list(test_input.shape)}")
            
            output = test_grc(test_input)
            print(f"  Output shape: {list(output.shape)}")
            
            if list(output.shape) == list(test_input.shape):
                print(f"  ✅ Output shape matches input (correct!)")
            else:
                print(f"  ❌ Output shape mismatch!")
            
            return test_grc
            
        except Exception as e:
            print(f"✗ Error analyzing GRC: {e}")
            import traceback
            print(traceback.format_exc())
            return None
    
    def suggest_fixes(self):
        """Test 5: Suggest specific fixes"""
        print("\n" + "=" * 80)
        print("TEST 5: FIX RECOMMENDATIONS")
        print("=" * 80)
        
        # Analyze the mismatch pattern
        checkpoint_has_nested = any('convs.' in k for k in self.results['checkpoint_keys'])
        model_has_nested = any('convs.' in k for k in self.results['model_keys'])
        
        print(f"\n🔍 MISMATCH ANALYSIS:")
        print(f"  Checkpoint has nested convs structure: {checkpoint_has_nested}")
        print(f"  Model has nested convs structure: {model_has_nested}")
        
        if checkpoint_has_nested and not model_has_nested:
            print(f"\n✅ DIAGNOSIS: Checkpoint expects nested GRC blocks, model creates flat structure")
            print(f"\n💡 FIX STRATEGY:")
            print(f"  1. In integrate_trained_model.py, the MultiReceptiveFieldFusion should create:")
            print(f"     self.convs = nn.ModuleList([")
            print(f"         GroupedResidualConvolution(...),")
            print(f"         GroupedResidualConvolution(...),")
            print(f"         GroupedResidualConvolution(...)")
            print(f"     ])")
            print(f"\n  2. Each GroupedResidualConvolution creates conv1 and conv2")
            print(f"     This matches checkpoint structure: mrfs.X.convs.Y.conv1.weight")
            print(f"\n  3. Current code creates: self.conv = nn.Conv1d(...)")
            print(f"     This creates: mrfs.X.conv.weight")
            print(f"     Which doesn't match checkpoint!")
        
        elif model_has_nested and not checkpoint_has_nested:
            print(f"\n❌ DIAGNOSIS: Model is MORE complex than checkpoint (unexpected!)")
            print(f"\n💡 FIX STRATEGY:")
            print(f"  Model should be simplified to match checkpoint structure")
        
        elif checkpoint_has_nested and model_has_nested:
            print(f"\n⚠️  Both have nested structure but keys still don't match")
            print(f"\n💡 FIX STRATEGY:")
            print(f"  Check the exact nesting structure - number of GRC blocks might differ")
        
        else:
            print(f"\n❓ Both are flat - mismatch is elsewhere")
        
        # Check if we already applied the fix
        print(f"\n📝 CHECKING CURRENT STATE:")
        if model_has_nested:
            print(f"  ✅ Good! Model already creates nested convs structure")
            print(f"  Check if number of GRC blocks matches checkpoint")
        else:
            print(f"  ❌ Model still creates flat structure")
            print(f"  The fix in integrate_trained_model.py needs to be applied")
    
    def generate_report(self):
        """Generate final report"""
        print("\n" + "=" * 80)
        print("DIAGNOSTIC REPORT")
        print("=" * 80)
        
        print(f"\n📊 SUMMARY:")
        print(f"  Checkpoint parameters: {len(self.results['checkpoint_keys'])}")
        print(f"  Model parameters: {len(self.results['model_keys'])}")
        
        if self.results['mismatched_keys']:
            missing_in_model = len(self.results['mismatched_keys']['missing_in_model'])
            missing_in_checkpoint = len(self.results['mismatched_keys']['missing_in_checkpoint'])
            
            print(f"\n❌ MISMATCH STATUS:")
            print(f"  Keys in checkpoint but not model: {missing_in_model}")
            print(f"  Keys in model but not checkpoint: {missing_in_checkpoint}")
            
            if missing_in_model > 0 or missing_in_checkpoint > 0:
                print(f"\n🔧 ACTION REQUIRED:")
                print(f"  1. Fix the model architecture in integrate_trained_model.py")
                print(f"  2. Ensure GRC blocks create nested convs structure")
                print(f"  3. Re-run this diagnostic to verify fix")
                print(f"  4. Then try loading weights again")
        else:
            print(f"\n✅ No mismatch data available (diagnostic incomplete)")
        
        # Save report
        report_path = Path("architecture_diagnostic_report.json")
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Report saved to: {report_path}")
        
        print("\n" + "=" * 80)

def main():
    """Run full architecture diagnostic"""
    
    diag = ArchitectureDiagnostic()
    
    # Run diagnostics
    checkpoint_dict = diag.inspect_checkpoint()
    model_dict = diag.inspect_current_model()
    
    if checkpoint_dict and model_dict:
        diag.compare_structures(checkpoint_dict, model_dict)
    
    diag.analyze_grc_implementation()
    diag.suggest_fixes()
    diag.generate_report()
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review the output above")
    print("  2. Check architecture_diagnostic_report.json for details")
    print("  3. Apply the suggested fixes")
    print("  4. Re-run this diagnostic to verify")
    print("  5. Re-run diagnose_chipmunk_issue.py to test audio output")
    print()

if __name__ == "__main__":
    main()

