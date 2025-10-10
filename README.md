# StreamSpeech Tool – Setup and Usage

This repository contains our Speech‑to‑Speech (S2ST) tool built on a modified HiFi‑GAN vocoder with ODConv and GRC+LoRA enhancements. The code here matches the repository at [ronliwag/tool](https://github.com/ronliwag/tool).

## Contents
- Enhanced desktop applications (Qt/Streamlit)
- Modified HiFi‑GAN and utilities
- Original StreamSpeech reference code
- Diagnostics and training scripts

## Supported OS and Python
- Windows 10/11, Linux, macOS
- Python 3.9–3.11 recommended
- Optional: NVIDIA GPU with a compatible CUDA build of PyTorch

## 1) Clone the repo
```bash
git clone https://github.com/ronliwag/tool.git
cd tool
```

Keep the folder names as‑is. Several modules use relative imports and expect the root layout to remain unchanged.

## 2) Create a virtual environment
Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3) Install PyTorch first (GPU or CPU)
- GPU: choose the exact torch/torchaudio command for your CUDA from the PyTorch website, for example:
```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchaudio
```
- CPU only:
```bash
pip install torch torchaudio
```

## 4) Install the rest of the dependencies
From the repository root:
```bash
pip install -r requirements.txt
```
If `pyaudio` fails on Windows, install a prebuilt wheel, or skip it if you won’t use microphone features.

## 5) Download and place model files
Download trained vocoder checkpoints from the link in the repo home: models folder on Drive. Place files here:
```
trained_models/
└── hifigan_checkpoints/
    ├── best_model.pth            (or)
    ├── checkpoint_epoch_19.pth   (etc.)
```
If a `model_config.json` is provided, keep it in `trained_models/`.

## 6) Recommended path setup (prevents relative‑path breakage)
Run all commands from the repository root and set `PYTHONPATH` to the root for the current shell:
Windows PowerShell (current session):
```powershell
$env:PYTHONPATH = (Get-Location).Path
```
Linux/macOS (current session):
```bash
export PYTHONPATH="$(pwd)"
```
Avoid renaming top‑level directories (e.g., “Important files - for tool”, “Original Streamspeech”).

## 7) Launch
Desktop app (Qt, comparison tool):
```bash
python "Original Streamspeech/Modified Streamspeech/demo/enhanced_desktop_app_old1.py"
```
Alternative desktop app at repo root (if present):
```bash
python enhanced_desktop_app.py
```
Streamlit UI (optional):
```bash
streamlit run streamlit_app.py
```

## 8) Quick functional test
Process a sample WAV via the comparison app. Outputs are saved under `Original Streamspeech/example/outputs/` or a similar outputs directory shown in logs.

## Project structure (high level)
```
tool/
├── Important files - for tool/        # Core vocoder + integration modules
├── Original Streamspeech/             # Upstream reference + demo code
├── Thesis_Development_Files/          # Experiments/training utilities
├── diagnostics/                       # Reports, test artifacts
├── trained_models/                    # Place checkpoints here
├── requirements.txt                   # Consolidated deps
└── README.md
```

## Notes on features
- ODConv and GRC+LoRA are integrated in the runtime path used by the desktop app.
- FiLM components exist in the codebase; the desktop app path may not use full FiLMLayer conditioning unless you switch to the generator variant that applies FiLM per stage and pass real speaker/emotion embeddings.

## Troubleshooting
- Module import errors: confirm you are running from the repo root and `PYTHONPATH` points to the root.
- Torch/CUDA mismatch: reinstall torch/torchaudio for your CUDA version, or use CPU wheels.
- PyAudio failures on Windows: install a prebuilt wheel or comment out microphone features; file processing will still work.
- Qt plugin errors: ensure `PySide6` is installed inside the active virtual environment.

## License and attribution
Academic research project; built on StreamSpeech, HiFi‑GAN, Hugging Face Transformers, and Whisper.
