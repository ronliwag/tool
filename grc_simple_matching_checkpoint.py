"""
Simplified GRC - Matches Training Checkpoint Exactly
===================================================

This is the ACTUAL GRC structure that was used during training.
NOT the complex one in grc_lora.py!

The checkpoint shows:
- Simple conv1 and conv2 only
- Full channel convolutions (no grouping)
- Different kernel sizes (3, 5, 7) not dilations
- No attention, no LoRA, no normalization
"""

import torch
import torch.nn as nn

class SimpleGRC(nn.Module):
    """
    Simplified Grouped Residual Convolution
    Matches the training checkpoint structure EXACTLY
    
    Each GRC block has ONLY:
    - conv1.weight
    - conv1.bias
    - conv2.weight
    - conv2.bias
    
    Total: 4 parameters per block
    """
    
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        
        self.channels = channels
        self.kernel_size = kernel_size
        
        # Calculate padding to maintain size
        padding = (kernel_size - 1) // 2 * dilation
        
        # Simple full-channel convolutions (NO grouping!)
        self.conv1 = nn.Conv1d(
            channels, 
            channels, 
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        
        self.conv2 = nn.Conv1d(
            channels, 
            channels, 
            kernel_size,
            padding=padding,
            dilation=dilation
        )
    
    def forward(self, x):
        """
        Simple residual forward pass:
        out = conv2(relu(conv1(x))) + x
        """
        residual = x
        
        # First convolution + activation
        out = self.conv1(x)
        out = torch.relu(out)
        
        # Second convolution
        out = self.conv2(out)
        
        # Residual connection
        out = out + residual
        
        return out


class MultiReceptiveFieldFusion(nn.Module):
    """
    MRF layer using Simple GRC blocks
    Matches checkpoint structure with 3 blocks at different kernel sizes
    """
    
    def __init__(self, channels):
        super().__init__()
        
        # Create 3 GRC blocks with different kernel sizes
        # This matches the checkpoint pattern:
        # convs.0: kernel_size=3
        # convs.1: kernel_size=5
        # convs.2: kernel_size=7
        self.convs = nn.ModuleList([
            SimpleGRC(channels, kernel_size=3, dilation=1),
            SimpleGRC(channels, kernel_size=5, dilation=1),
            SimpleGRC(channels, kernel_size=7, dilation=1)
        ])
    
    def forward(self, x):
        """Sum outputs from all GRC blocks"""
        return sum(conv(x) for conv in self.convs)


# Test the implementation
if __name__ == "__main__":
    print("Testing Simplified GRC Implementation")
    print("=" * 50)
    
    # Create test instance
    test_grc = SimpleGRC(channels=512, kernel_size=3)
    
    print(f"\nSimple GRC Structure:")
    print(f"Total parameters: {len(list(test_grc.state_dict().keys()))}")
    for name, param in test_grc.state_dict().items():
        print(f"  {name:.<30} {list(param.shape)}")
    
    print(f"\nExpected checkpoint keys:")
    print(f"  conv1.weight                   [512, 512, 3]")
    print(f"  conv1.bias                     [512]")
    print(f"  conv2.weight                   [512, 512, 3]")
    print(f"  conv2.bias                     [512]")
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    test_input = torch.randn(1, 512, 100)
    output = test_grc(test_input)
    print(f"  Input shape: {list(test_input.shape)}")
    print(f"  Output shape: {list(output.shape)}")
    
    if list(output.shape) == list(test_input.shape):
        print(f"  ✅ Output matches input shape!")
    
    # Test MRF
    print(f"\nTesting MultiReceptiveFieldFusion...")
    mrf = MultiReceptiveFieldFusion(channels=512)
    print(f"Total MRF parameters: {len(list(mrf.state_dict().keys()))}")
    print(f"Expected: 12 (4 params × 3 GRC blocks)")
    
    mrf_output = mrf(test_input)
    print(f"  MRF output shape: {list(mrf_output.shape)}")
    
    if len(list(mrf.state_dict().keys())) == 12:
        print(f"  ✅ MRF matches checkpoint structure!")
    else:
        print(f"  ❌ MRF has {len(list(mrf.state_dict().keys()))} params, expected 12")

