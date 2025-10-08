#!/usr/bin/env python3
"""
TRAINING MONITOR
Monitor the progress of professional training
"""

import os
import time
import psutil
from pathlib import Path

def monitor_training():
    """Monitor training progress"""
    print("MONITORING PROFESSIONAL TRAINING")
    print("=" * 40)
    
    # Check for Python processes
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'python.exe':
                cmdline = ' '.join(proc.info['cmdline'])
                if 'professional_training_real_cvss.py' in cmdline:
                    python_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if python_processes:
        print(f"Found {len(python_processes)} training processes running")
        for proc in python_processes:
            print(f"  - PID: {proc.pid}")
            print(f"  - CPU: {proc.cpu_percent():.1f}%")
            print(f"  - Memory: {proc.memory_info().rss / 1024 / 1024:.1f} MB")
    else:
        print("No training processes found")
    
    # Check for checkpoints
    checkpoint_dir = Path("professional_cvss_checkpoints")
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        print(f"\nFound {len(checkpoints)} checkpoint files:")
        for ckpt in checkpoints:
            stat = ckpt.stat()
            print(f"  - {ckpt.name}: {stat.st_size / 1024 / 1024:.1f} MB, {time.ctime(stat.st_mtime)}")
    else:
        print("\nNo checkpoint directory found yet")
    
    # Check for log files
    log_files = list(Path(".").glob("*log*.txt"))
    if log_files:
        print(f"\nFound {len(log_files)} log files:")
        for log in log_files:
            stat = log.stat()
            print(f"  - {log.name}: {stat.st_size / 1024:.1f} KB, {time.ctime(stat.st_mtime)}")
    
    # Check GPU usage
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\nGPU Status:")
            print(f"  - CUDA Available: {torch.cuda.is_available()}")
            print(f"  - GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  - GPU {i}: {props.name}")
                print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024:.1f} GB")
    except ImportError:
        print("\nPyTorch not available for GPU monitoring")

if __name__ == "__main__":
    monitor_training()
