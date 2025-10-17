# Next Diagnostic Steps - Detailed Action Plan

## 🎯 Objective
Precisely identify the architecture mismatch between trained checkpoint and current model loading code, then fix it to enable proper weight loading.

---

## 📋 Diagnostic Steps (In Order)

### Step 1: Deep Architecture Inspection
**Run:** `python diagnose_architecture_mismatch.py`

**What it does:**
1. ✅ Loads your trained checkpoint (`best_model.pth`)
2. ✅ Extracts all parameter keys and shapes
3. ✅ Analyzes the MRF (Multi-Receptive Field) structure
4. ✅ Creates the current model architecture
5. ✅ Compares checkpoint vs model keys
6. ✅ Identifies exact mismatches
7. ✅ Tests GRC implementation
8. ✅ Provides specific fix recommendations

**What you'll learn:**
- Exact checkpoint structure (what keys exist)
- Exact model structure (what keys current code creates)
- Which keys are missing where
- How many GRC blocks the checkpoint expects
- Whether conv1/conv2 structure exists
- Specific line-by-line fixes needed

**Output files:**
- Console output with detailed analysis
- `architecture_diagnostic_report.json` (detailed JSON report)

---

### Step 2: Inspect Training Configuration
**Check:** `trained_models/model_config.json`

**What to look for:**
```json
{
  "model_architecture": {
    "mrf_layers": [
      {
        "channels": 256,
        "num_blocks": 3  ← How many GRC blocks per MRF?
      }
    ]
  },
  "training_config": {
    "sampling_rate": 22050,  ← Verify this is correct
    ...
  }
}
```

**Key questions:**
1. How many GRC blocks per MRF layer?
2. What were the GRC parameters (groups, dilation)?
3. Was LoRA used during training?
4. What sample rate was used?

---

### Step 3: Verify GRC Implementation
**Check:** `Important files - for tool/grc_lora.py`

**Verify the GroupedResidualConvolution class creates:**
```python
class GroupedResidualConvolution(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.conv1 = nn.Conv1d(...)  # ← Must be named conv1
        self.conv2 = nn.Conv1d(...)  # ← Must be named conv2
        # NOT self.conv = nn.Conv1d(...)
```

**Test it:**
```python
from grc_lora import GroupedResidualConvolution
grc = GroupedResidualConvolution(256, 3, 1, 4, 4)
print(list(grc.state_dict().keys()))
# Should see: conv1.weight, conv1.bias, conv2.weight, conv2.bias
```

---

### Step 4: Check Current Model Creation
**Check:** `Important files - for tool/integrate_trained_model.py`

**Line ~70-96: Look at MultiReceptiveFieldFusion**

**What it SHOULD be:**
```python
class MultiReceptiveFieldFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Multiple GRC blocks at different dilations
        self.convs = nn.ModuleList([
            GroupedResidualConvolution(channels, kernel_size=3, dilation=1),
            GroupedResidualConvolution(channels, kernel_size=3, dilation=3),
            GroupedResidualConvolution(channels, kernel_size=3, dilation=5)
        ])
    
    def forward(self, x):
        return sum(conv(x) for conv in self.convs)
```

**What it might currently be:**
```python
class MultiReceptiveFieldFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=1)  # ← WRONG!
    
    def forward(self, x):
        return self.conv(x)
```

---

### Step 5: Manual Weight Inspection
**Create a test script:**

```python
import torch

# Load checkpoint
ckpt = torch.load('trained_models/hifigan_checkpoints/best_model.pth')
state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

# Show MRF structure
mrf_keys = [k for k in state_dict.keys() if 'mrfs.' in k]
print(f"Total MRF keys: {len(mrf_keys)}")

# Show first MRF layer structure
mrf0_keys = [k for k in mrf_keys if k.startswith('mrfs.0.')]
print(f"\nmrfs.0 structure ({len(mrf0_keys)} keys):")
for key in sorted(mrf0_keys):
    print(f"  {key}: {list(state_dict[key].shape)}")

# Check for nested structure
has_convs = any('convs.' in k for k in mrf0_keys)
print(f"\nHas nested 'convs': {has_convs}")

if has_convs:
    # Count GRC blocks
    convs_indices = set()
    for key in mrf0_keys:
        if 'convs.' in key:
            idx = key.split('convs.')[1].split('.')[0]
            convs_indices.add(idx)
    print(f"Number of GRC blocks: {len(convs_indices)}")
    print(f"Indices: {sorted(convs_indices)}")
```

---

### Step 6: Test Weight Loading With Remapping
**If structure is correct but keys differ, try manual remapping:**

```python
import torch
import torch.nn as nn
from integrate_trained_model import TrainedModelLoader

# Load checkpoint
checkpoint = torch.load('trained_models/hifigan_checkpoints/best_model.pth')
checkpoint_state = checkpoint['model_state_dict']

# Create model
loader = TrainedModelLoader()
loader.load_model_config()
loader.create_model_architecture()
model = loader.trained_model

# Get model state dict
model_state = model.state_dict()

# Compare
print("Checkpoint keys:", len(checkpoint_state))
print("Model keys:", len(model_state))

# Try loading
try:
    model.load_state_dict(checkpoint_state, strict=False)
    print("✅ Loaded (with strict=False)")
    
    # Check which keys were missing
    missing, unexpected = [], []
    for key in checkpoint_state.keys():
        if key not in model_state:
            missing.append(key)
    for key in model_state.keys():
        if key not in checkpoint_state:
            unexpected.append(key)
    
    print(f"Missing in model: {len(missing)}")
    print(f"Unexpected in model: {len(unexpected)}")
    
except Exception as e:
    print(f"❌ Failed: {e}")
```

---

### Step 7: Verify Fix Works
**After applying fixes, test:**

1. **Run architecture diagnostic again:**
   ```bash
   python diagnose_architecture_mismatch.py
   ```
   - Should show 0 mismatched keys
   - All checkpoint keys should exist in model

2. **Test weight loading:**
   ```python
   loader = TrainedModelLoader()
   if loader.initialize_full_system():
       print("✅ SUCCESS!")
   ```

3. **Test audio generation:**
   ```bash
   python diagnose_chipmunk_issue.py
   ```
   - Modified mode mel-spectrogram should show vertical variations
   - Output should be speech, not static

---

## 🔍 What Each Diagnostic Reveals

### Architecture Diagnostic (`diagnose_architecture_mismatch.py`):
- ✅ Exact key differences
- ✅ Structure patterns (nested vs flat)
- ✅ Number of GRC blocks needed
- ✅ Parameter shapes
- ✅ Specific code changes required

### Training Config (`model_config.json`):
- ✅ How model was trained
- ✅ Architecture parameters
- ✅ Sample rates used
- ✅ Expected dimensions

### GRC Implementation (`grc_lora.py`):
- ✅ What structure GRC creates
- ✅ Parameter names (conv1/conv2)
- ✅ Forward pass correctness

### Model Creation (`integrate_trained_model.py`):
- ✅ Current MRF structure
- ✅ Whether it matches checkpoint
- ✅ What needs to change

---

## 📊 Expected Diagnostic Outputs

### If Architecture is Correct:
```
✅ PERFECT MATCH! All keys align!
  Checkpoint keys: 150
  Model keys: 150
  Common keys: 150
  Missing in model: 0
  Missing in checkpoint: 0
```

### If Architecture is Wrong (Current State):
```
❌ MISMATCH DETECTED!
  Checkpoint keys: 180
  Model keys: 120
  Common keys: 100
  Missing in model: 80  ← These need to be created
  Missing in checkpoint: 20  ← These shouldn't exist

🔴 KEYS IN CHECKPOINT BUT NOT IN MODEL:
  mrfs.0.convs.0.conv1.weight
  mrfs.0.convs.0.conv1.bias
  mrfs.0.convs.0.conv2.weight
  mrfs.0.convs.0.conv2.bias
  ... (GRC block structure)
```

---

## 🛠️ Recommended Fix Workflow

### 1. Run Initial Diagnostic
```bash
python diagnose_architecture_mismatch.py > architecture_diagnosis.txt
```

### 2. Review Output
- Read the "STRUCTURE COMPARISON" section
- Note which keys are missing
- Check the "FIX RECOMMENDATIONS" section

### 3. Apply Fix
Based on diagnostic output, modify `integrate_trained_model.py`:
```python
# Around line 74-96
from grc_lora import GroupedResidualConvolution

class MultiReceptiveFieldFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Create nested GRC structure to match checkpoint
        self.convs = nn.ModuleList([
            GroupedResidualConvolution(channels, kernel_size=3, dilation=1),
            GroupedResidualConvolution(channels, kernel_size=3, dilation=3),
            GroupedResidualConvolution(channels, kernel_size=3, dilation=5)
        ])
    
    def forward(self, x):
        # Sum outputs from all GRC blocks
        return sum(conv(x) for conv in self.convs)
```

### 4. Re-run Diagnostic
```bash
python diagnose_architecture_mismatch.py
```
Should now show: "✅ PERFECT MATCH!"

### 5. Re-enable Trained Model
In `streamspeech_modifications.py`, remove the bypass:
```python
def initialize_trained_model_loader(self):
    # Remove the early return
    from integrate_trained_model import TrainedModelLoader
    self.trained_model_loader = TrainedModelLoader()
    # This should now work!
```

### 6. Test Full System
```bash
python diagnose_chipmunk_issue.py
```
Modified mode should now generate proper speech!

---

## 📈 Success Criteria

### ✅ Architecture Fix is Complete When:
1. Architecture diagnostic shows 0 mismatched keys
2. Weight loading succeeds without errors
3. Model has same number of parameters as checkpoint
4. All shapes match between checkpoint and model

### ✅ Audio Generation is Fixed When:
1. Modified mode mel-spectrogram shows vertical variations
2. Output sounds like speech (not static)
3. Mel-spectrogram looks similar to Original mode
4. Translations are correct AND audio is speech

---

## 🚨 Common Issues & Solutions

### Issue: "Missing keys" even after fix
**Solution:** Check if GRC implementation creates conv1/conv2 correctly

### Issue: "Unexpected keys" after fix
**Solution:** Model creates extra parameters - simplify architecture

### Issue: Shape mismatches
**Solution:** Check channel dimensions, kernel sizes in GRC blocks

### Issue: Weights load but output still static
**Solution:** Verify forward pass uses loaded weights, check normalization

---

## 💡 Pro Tips

1. **Save intermediate results:** Each diagnostic run saves JSON reports
2. **Compare before/after:** Run diagnostic before and after each fix
3. **Test incrementally:** Fix one MRF layer at a time if needed
4. **Check training logs:** May contain architecture details
5. **Use strict=False first:** Test if weights load at all before enforcing strict matching

---

## 📝 Summary

**Run in this order:**
1. `python diagnose_architecture_mismatch.py` - Identify exact mismatch
2. Review `architecture_diagnostic_report.json` - Understand structure
3. Check `trained_models/model_config.json` - Verify training params
4. Modify `integrate_trained_model.py` - Apply fix
5. Re-run diagnostic - Verify fix
6. Re-enable trained model in `streamspeech_modifications.py`
7. `python diagnose_chipmunk_issue.py` - Test audio output

The architecture diagnostic tool will tell you **exactly** what needs to change!

