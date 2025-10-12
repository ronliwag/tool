import os
import sys
import glob
import numpy as np


def _append_paths():
    here = os.path.dirname(os.path.abspath(__file__))
    demo_path = os.path.join(here, "..", "Original Streamspeech", "Modified Streamspeech", "demo")
    sys.path.append(here)
    sys.path.append(demo_path)


def evaluate_directory(input_dir: str, max_files: int = 50):
    """
    Batch evaluate ASR-BLEU for Original and Modified on the same set of files.
    - Original uses StreamSpeech baseline (reset/run) to produce output audio
    - Modified uses WorkingODConvIntegration to generate ODConv-enhanced output
    - Reference text is taken from StreamSpeech (app.ST) after processing

    Returns: dict with per-file scores and mean/Δ statistics
    """
    _append_paths()

    # Lazy imports after paths
    from simple_metrics_calculator import simple_metrics_calculator

    # Collect files
    exts = ("*.wav", "*.mp3", "*.flac")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
    files = sorted(files)[:max_files]
    if not files:
        raise RuntimeError(f"No audio files in {input_dir}")

    # Import StreamSpeech
    from demo import app as demo_app
    from demo.app import reset, run

    # Import Modified ODConv integration
    from working_real_odconv_integration import WorkingODConvIntegration
    odconv = WorkingODConvIntegration()

    original_scores = []
    modified_scores = []
    results = []

    for path in files:
        # ORIGINAL
        reset()
        run(path)
        # Get generated audio and reference from StreamSpeech
        if hasattr(demo_app, 'S2ST') and demo_app.S2ST:
            orig_audio = np.asarray(demo_app.S2ST, dtype=np.float32)
        else:
            orig_audio = None
        # Reference English text from StreamSpeech
        ref_text = ""
        if hasattr(demo_app, 'ST') and demo_app.ST:
            st_val = demo_app.ST
            if isinstance(st_val, dict) and len(st_val) > 0:
                try:
                    ref_text = str(st_val[max(st_val.keys())])
                except Exception:
                    ref_text = str(next(iter(st_val.values())))
            elif isinstance(st_val, str):
                ref_text = st_val
            else:
                ref_text = str(st_val)

        orig_bleu = 0.0
        if orig_audio is not None and ref_text:
            bleu_res = simple_metrics_calculator.calculate_asr_bleu(orig_audio, ref_text)
            orig_bleu = float(bleu_res.get('asr_bleu_score', 0.0)) * 100.0
        original_scores.append(orig_bleu)

        # MODIFIED (ODConv)
        enhanced_audio, _ = odconv.process_audio_with_odconv(audio_path=path)
        mod_bleu = 0.0
        if enhanced_audio is not None and ref_text:
            bleu_res = simple_metrics_calculator.calculate_asr_bleu(enhanced_audio, ref_text)
            mod_bleu = float(bleu_res.get('asr_bleu_score', 0.0)) * 100.0
        modified_scores.append(mod_bleu)

        results.append({
            'file': os.path.basename(path),
            'original_bleu': orig_bleu,
            'modified_bleu': mod_bleu,
            'delta_bleu': (mod_bleu - orig_bleu)
        })

    # Aggregate
    orig_mean = float(np.mean(original_scores)) if original_scores else 0.0
    mod_mean = float(np.mean(modified_scores)) if modified_scores else 0.0
    delta_mean = mod_mean - orig_mean

    summary = {
        'count': len(files),
        'original_mean_bleu': orig_mean,
        'modified_mean_bleu': mod_mean,
        'delta_mean_bleu': delta_mean,
        'per_file': results,
    }
    return summary


def main():
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Directory with audio files')
    parser.add_argument('--max_files', type=int, default=50)
    args = parser.parse_args()

    summary = evaluate_directory(args.input_dir, args.max_files)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()


