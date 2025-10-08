#!/usr/bin/env python3
"""
StreamSpeech Desktop Application
A simplified desktop version that processes audio files directly without web interface
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import soundfile
import numpy as np
import torch
import json
import argparse
import pygame
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from pathlib import Path

# Add fairseq to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'fairseq'))

# Import StreamSpeech components
from app import StreamSpeechS2STAgent, OnlineFeatureExtractor, reset, run

class StreamSpeechDesktopApp:
    def __init__(self, root):
        self.root = root
        self.root.title("StreamSpeech Desktop - Spanish to English Translation")
        self.root.geometry("1200x800")
        
        # Initialize agent
        self.agent = None
        self.is_processing = False
        
        # Initialize pygame for audio playback
        pygame.mixer.init(frequency=16000, size=-16, channels=1, buffer=1024)
        
        # Initialize visualization data
        self.input_waveform = None
        self.output_waveform = None
        self.spanish_text = ""
        self.english_text = ""
        self.current_latency = 320
        
        self.setup_ui()
        self.initialize_agent()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="StreamSpeech Desktop - Spanish to English Translation", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # File selection
        file_frame = ttk.LabelFrame(main_frame, text="Audio File Selection", padding="10")
        file_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=60)
        file_entry.grid(row=0, column=0, padx=(0, 10), sticky=(tk.W, tk.E))
        
        browse_btn = ttk.Button(file_frame, text="Browse", command=self.browse_file)
        browse_btn.grid(row=0, column=1)
        
        # Processing options
        options_frame = ttk.LabelFrame(main_frame, text="Processing Options", padding="10")
        options_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(options_frame, text="Latency (ms):").grid(row=0, column=0, sticky=tk.W)
        self.latency_var = tk.StringVar(value="320")
        latency_entry = ttk.Entry(options_frame, textvariable=self.latency_var, width=10)
        latency_entry.grid(row=0, column=1, padx=(10, 0), sticky=tk.W)
        
        # Process button
        self.process_btn = ttk.Button(main_frame, text="Process Audio", command=self.process_audio, style="Accent.TButton")
        self.process_btn.grid(row=3, column=0, pady=20, padx=(0, 10))
        
        # Play output button
        self.play_btn = ttk.Button(main_frame, text="Play Last Output", command=self.play_last_output)
        self.play_btn.grid(row=3, column=1, pady=20)
        self.last_output_path = None
        
        # Waveform visualization frame
        viz_frame = ttk.LabelFrame(main_frame, text="Audio Visualization", padding="10")
        viz_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)
        viz_frame.rowconfigure(1, weight=1)
        
        # Input waveform
        input_wave_frame = ttk.Frame(viz_frame)
        input_wave_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        input_wave_frame.columnconfigure(0, weight=1)
        
        ttk.Label(input_wave_frame, text="Input Audio (Spanish)", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W)
        self.input_fig = Figure(figsize=(10, 2), dpi=80)
        self.input_ax = self.input_fig.add_subplot(111)
        self.input_ax.set_ylabel("Amplitude")
        self.input_ax.set_xlabel("Time (s)")
        self.input_canvas = FigureCanvasTkAgg(self.input_fig, input_wave_frame)
        self.input_canvas.get_tk_widget().grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Output waveform
        output_wave_frame = ttk.Frame(viz_frame)
        output_wave_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        output_wave_frame.columnconfigure(0, weight=1)
        
        ttk.Label(output_wave_frame, text="Output Audio (English)", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W)
        self.output_fig = Figure(figsize=(10, 2), dpi=80)
        self.output_ax = self.output_fig.add_subplot(111)
        self.output_ax.set_ylabel("Amplitude")
        self.output_ax.set_xlabel("Time (s)")
        self.output_canvas = FigureCanvasTkAgg(self.output_fig, output_wave_frame)
        self.output_canvas.get_tk_widget().grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Text display frame
        text_frame = ttk.LabelFrame(main_frame, text="Recognition & Translation", padding="10")
        text_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        text_frame.rowconfigure(1, weight=1)
        
        # Spanish text
        spanish_frame = ttk.Frame(text_frame)
        spanish_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        spanish_frame.columnconfigure(0, weight=1)
        
        ttk.Label(spanish_frame, text="Spanish Recognition:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W)
        self.spanish_text_var = tk.StringVar(value="No audio processed yet...")
        spanish_display = ttk.Label(spanish_frame, textvariable=self.spanish_text_var, 
                                   wraplength=1000, justify="left", foreground="blue", font=("Arial", 11))
        spanish_display.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # English text
        english_frame = ttk.Frame(text_frame)
        english_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        english_frame.columnconfigure(0, weight=1)
        
        ttk.Label(english_frame, text="English Translation:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W)
        self.english_text_var = tk.StringVar(value="No translation available yet...")
        english_display = ttk.Label(english_frame, textvariable=self.english_text_var, 
                                   wraplength=1000, justify="left", foreground="orange", font=("Arial", 11))
        english_display.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Progress bar
        self.progress_var = tk.StringVar(value="Ready")
        progress_label = ttk.Label(main_frame, textvariable=self.progress_var)
        progress_label.grid(row=6, column=0, columnspan=2, pady=(0, 10))
        
        self.progress_bar = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress_bar.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Output area
        output_frame = ttk.LabelFrame(main_frame, text="Output Log", padding="10")
        output_frame.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.output_text = tk.Text(output_frame, height=15, width=80)
        scrollbar = ttk.Scrollbar(output_frame, orient="vertical", command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=scrollbar.set)
        
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(8, weight=1)
        file_frame.columnconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
    
    def initialize_agent(self):
        """Initialize the StreamSpeech agent"""
        try:
            self.log("Initializing StreamSpeech agent...")
            
            # Load config and create agent with proper arguments
            with open('config.json', 'r') as f:
                args_dict = json.load(f)
            
            # Initialize agent with arguments
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
            self.agent = StreamSpeechS2STAgent(args)
            
            self.log("Agent initialized successfully!")
        except Exception as e:
            self.log(f"Error initializing agent: {str(e)}")
            messagebox.showerror("Initialization Error", f"Failed to initialize StreamSpeech agent:\n{str(e)}")
    
    def browse_file(self):
        """Open file dialog to select audio file"""
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.m4a"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.file_path_var.set(file_path)
            self.log(f"Selected file: {file_path}")
    
    def log(self, message):
        """Add message to output text area"""
        self.output_text.insert(tk.END, f"{message}\n")
        self.output_text.see(tk.END)
        self.root.update_idletasks()
    
    def plot_waveform(self, audio_data, sample_rate, is_input=True):
        """Plot waveform for input or output audio"""
        try:
            # Convert to numpy array if needed
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.cpu().numpy()
            elif isinstance(audio_data, list):
                audio_data = np.array(audio_data)
            
            # Create time axis
            duration = len(audio_data) / sample_rate
            time_axis = np.linspace(0, duration, len(audio_data))
            
            # Clear previous plot
            if is_input:
                self.input_ax.clear()
                ax = self.input_ax
                color = 'blue'
                title = 'Input Audio (Spanish)'
            else:
                self.output_ax.clear()
                ax = self.output_ax
                color = 'orange'
                title = 'Output Audio (English)'
            
            # Plot waveform
            ax.plot(time_axis, audio_data, color=color, linewidth=0.5)
            ax.set_ylabel("Amplitude")
            ax.set_xlabel("Time (s)")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            # Refresh canvas
            if is_input:
                self.input_canvas.draw()
            else:
                self.output_canvas.draw()
                
        except Exception as e:
            self.log(f"Error plotting waveform: {str(e)}")
    
    def update_text_display(self, spanish_text="", english_text=""):
        """Update the text display for recognition and translation"""
        if spanish_text:
            self.spanish_text_var.set(spanish_text)
        if english_text:
            self.english_text_var.set(english_text)
    
    def process_audio(self):
        """Process the selected audio file"""
        if self.is_processing:
            return
        
        file_path = self.file_path_var.get()
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("Error", "Please select a valid audio file")
            return
        
        if not self.agent:
            messagebox.showerror("Error", "StreamSpeech agent not initialized")
            return
        
        # Start processing in a separate thread
        self.is_processing = True
        self.process_btn.config(state="disabled")
        self.progress_bar.start()
        self.progress_var.set("Processing...")
        
        thread = threading.Thread(target=self._process_audio_thread, args=(file_path,))
        thread.daemon = True
        thread.start()
    
    def _process_audio_thread(self, file_path):
        """Process audio in a separate thread"""
        try:
            self.log(f"Starting processing of: {os.path.basename(file_path)}")
            
            # Set latency
            latency = int(self.latency_var.get())
            self.agent.set_chunk_size(latency)
            self.log(f"Set latency to: {latency}ms")
            
            # Load and display input audio waveform
            samples, sr = soundfile.read(file_path, dtype="float32")
            self.log(f"Loaded audio: {len(samples)} samples at {sr} Hz")
            
            # Plot input waveform
            self.root.after(0, lambda: self.plot_waveform(samples, sr, is_input=True))
            self.root.after(0, lambda: self.update_text_display(spanish_text="Processing Spanish audio..."))
            
            # Reset and process
            reset()
            self.log("Processing audio...")
            run(file_path)
            
            # Get output path - save to example/outputs folder
            outputs_dir = os.path.join(os.path.dirname(__file__), "..", "example", "outputs")
            os.makedirs(outputs_dir, exist_ok=True)
            output_filename = f"output_{os.path.basename(file_path)}"
            output_path = os.path.join(outputs_dir, output_filename)
            
            # Convert S2ST list to numpy array and save
            from app import S2ST, SAMPLE_RATE
            if S2ST:
                s2st_array = np.array(S2ST, dtype=np.float32)
                soundfile.write(output_path, s2st_array, SAMPLE_RATE)
                self.log(f"Translation completed! Output saved to: {output_path}")
                self.log(f"Output duration: {len(S2ST) / SAMPLE_RATE:.2f} seconds")
                
                # Plot output waveform
                self.root.after(0, lambda: self.plot_waveform(s2st_array, SAMPLE_RATE, is_input=False))
                
                # Get the real ASR and ST text from the global variables
                from app import ASR, ST
                
                # Get the latest ASR text (Spanish transcription)
                spanish_text = ""
                if ASR:
                    # Get the text from the largest key (most recent)
                    max_key = max(ASR.keys())
                    spanish_text = ASR[max_key]
                
                # Get the latest ST text (English translation)
                english_text = ""
                if ST:
                    # Get the text from the largest key (most recent)
                    max_key = max(ST.keys())
                    english_text = ST[max_key]
                
                # Update text display with real transcription and translation
                self.root.after(0, lambda: self.update_text_display(
                    spanish_text=spanish_text if spanish_text else "No Spanish transcription available",
                    english_text=english_text if english_text else "No English translation available"
                ))
                
                # Store the output path for replay
                self.last_output_path = output_path
                
                # Play the translated audio
                self.log("Playing translated audio...")
                self.play_audio(output_path)
            else:
                self.log("No translation output generated")
                self.root.after(0, lambda: self.update_text_display(
                    spanish_text="No Spanish audio processed",
                    english_text="No English translation generated"
                ))
            
        except Exception as e:
            self.log(f"Error during processing: {str(e)}")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")
        finally:
            # Reset UI state
            self.root.after(0, self._processing_complete)
    
    def _processing_complete(self):
        """Called when processing is complete"""
        self.is_processing = False
        self.process_btn.config(state="normal")
        self.progress_bar.stop()
        self.progress_var.set("Ready")
    
    def play_audio(self, audio_path):
        """Play audio file using pygame"""
        try:
            # Stop any currently playing audio
            pygame.mixer.music.stop()
            
            # Load and play the audio file
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                self.root.update_idletasks()
            
            self.log("Audio playback completed!")
            
        except Exception as e:
            self.log(f"Error playing audio: {str(e)}")
            # Fallback: try to open with default system player
            try:
                import subprocess
                import platform
                if platform.system() == "Windows":
                    os.startfile(audio_path)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", audio_path])
                else:  # Linux
                    subprocess.run(["xdg-open", audio_path])
                self.log("Opened audio with system default player")
            except Exception as e2:
                self.log(f"Could not play audio: {str(e2)}")
    
    def play_last_output(self):
        """Play the last generated output audio"""
        if self.last_output_path and os.path.exists(self.last_output_path):
            self.log(f"Playing: {os.path.basename(self.last_output_path)}")
            self.play_audio(self.last_output_path)
        else:
            messagebox.showwarning("No Output", "No output audio available. Please process an audio file first.")

def main():
    root = tk.Tk()
    app = StreamSpeechDesktopApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
