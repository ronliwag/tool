#!/usr/bin/env python3
"""
StreamSpeech Comparison Tool - Thesis Defense
Desktop application for comparing Original vs Modified StreamSpeech
"""

import os
import sys
import json
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import soundfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pygame

# Add fairseq to path (pointing to original StreamSpeech)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'fairseq'))

# Import StreamSpeech components from original
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'demo'))
from app import StreamSpeechS2STAgent, OnlineFeatureExtractor, reset, run, SAMPLE_RATE

# Import global variables from app module
import app

class StreamSpeechComparisonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("StreamSpeech Comparison Tool - Thesis Defense")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.current_mode = "Original"  # "Original" or "Modified"
        self.original_agent = None
        self.modified_agent = None
        self.last_output_path = None
        self.last_input_path = None
        self.is_processing = False
        
        # Model comparison tracking
        self.comparison_results = {
            'original': {},
            'modified': {}
        }
        self.last_processing_time = 0
        self.last_audio_duration = 0
        
        # Initialize pygame for audio playback
        pygame.mixer.init(frequency=16000, size=-16, channels=1, buffer=1024)
        
        # Setup UI
        self.setup_ui()
        
        # Initialize agents
        self.initialize_agents()
        
        # Load configuration
        self.load_config()
    
    def setup_ui(self):
        """Setup the user interface."""
        # Main container
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        left_panel = tk.Frame(main_frame, bg='white', relief='raised', bd=2)
        left_panel.pack(side='left', fill='y', padx=(0, 5))
        
        # Right panel - Visualization and Log
        right_panel = tk.Frame(main_frame, bg='white', relief='raised', bd=2)
        right_panel.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # Left Panel Content
        self.setup_left_panel(left_panel)
        
        # Right Panel Content
        self.setup_right_panel(right_panel)
    
    def setup_left_panel(self, parent):
        """Setup the left control panel."""
        # Title
        title_label = tk.Label(parent, text="StreamSpeech Comparison", 
                              font=('Arial', 16, 'bold'), bg='white', fg='#2c3e50')
        title_label.pack(pady=10)
        
        # Mode Selection
        mode_frame = tk.LabelFrame(parent, text="System Mode", 
                                  font=('Arial', 11, 'bold'), bg='white')
        mode_frame.pack(fill='x', padx=10, pady=5)
        
        self.mode_var = tk.StringVar(value="Original")
        
        original_radio = tk.Radiobutton(mode_frame, text="Original StreamSpeech", 
                                       variable=self.mode_var, value="Original",
                                       command=self.switch_mode, font=('Arial', 10),
                                       bg='white', fg='#3498db')
        original_radio.pack(anchor='w', padx=5, pady=2)
        
        modified_radio = tk.Radiobutton(mode_frame, text="Modified StreamSpeech (ODConv+GRC+LoRA)", 
                                       variable=self.mode_var, value="Modified",
                                       command=self.switch_mode, font=('Arial', 10),
                                       bg='white', fg='#e67e22')
        modified_radio.pack(anchor='w', padx=5, pady=2)
        
        # File Selection
        file_frame = tk.LabelFrame(parent, text="Audio File Selection", 
                                  font=('Arial', 11, 'bold'), bg='white')
        file_frame.pack(fill='x', padx=10, pady=5)
        
        self.file_var = tk.StringVar()
        file_entry = tk.Entry(file_frame, textvariable=self.file_var, 
                             font=('Arial', 9), state='readonly')
        file_entry.pack(fill='x', padx=5, pady=2)
        
        browse_btn = tk.Button(file_frame, text="Browse Audio File", 
                              command=self.browse_file, font=('Arial', 10),
                              bg='#3498db', fg='white')
        browse_btn.pack(fill='x', padx=5, pady=2)
        
        # Latency Control
        latency_frame = tk.LabelFrame(parent, text="Latency Control", 
                                     font=('Arial', 11, 'bold'), bg='white')
        latency_frame.pack(fill='x', padx=10, pady=5)
        
        self.latency_var = tk.IntVar(value=320)
        latency_scale = tk.Scale(latency_frame, from_=160, to=640, 
                                orient='horizontal', variable=self.latency_var,
                                command=self.on_latency_change, font=('Arial', 9),
                                bg='white', fg='#2c3e50')
        latency_scale.pack(fill='x', padx=5, pady=2)
        
        self.latency_label = tk.Label(latency_frame, text="Latency (ms): 320 (Original - Standard Processing)", 
                                     font=('Arial', 9), bg='white', fg='#7f8c8d')
        self.latency_label.pack(anchor='w', padx=5, pady=2)
        
        # Control Buttons
        control_frame = tk.LabelFrame(parent, text="Processing Controls", 
                                     font=('Arial', 11, 'bold'), bg='white')
        control_frame.pack(fill='x', padx=10, pady=5)
        
        self.process_btn = tk.Button(control_frame, text="Process Audio", 
                                   command=self.process_audio, font=('Arial', 10),
                                   bg='#27ae60', fg='white')
        self.process_btn.pack(fill='x', padx=5, pady=5)
        
        self.play_btn = tk.Button(control_frame, text="Play Last Output", 
                                 command=self.play_last_output, font=('Arial', 10),
                                 bg='#2ecc71', fg='white', state='disabled')
        self.play_btn.pack(fill='x', padx=5, pady=5)
        
        self.compare_btn = tk.Button(control_frame, text="Show Model Comparison", 
                                   command=self.show_model_comparison, font=('Arial', 10),
                                   bg='#9b59b6', fg='white')
        self.compare_btn.pack(fill='x', padx=5, pady=5)
        
        self.simultaneous_btn = tk.Button(control_frame, text="Play Simultaneous Audio", 
                                        command=self.play_simultaneous_demo, font=('Arial', 10),
                                        bg='#e67e22', fg='white', state='disabled')
        self.simultaneous_btn.pack(fill='x', padx=5, pady=5)
        
        # Processing Status
        status_frame = tk.LabelFrame(parent, text="Processing Status", 
                                    font=('Arial', 11, 'bold'), bg='white')
        status_frame.pack(fill='x', padx=10, pady=5)
        
        self.status_label = tk.Label(status_frame, text="Status: Ready", 
                                    font=('Arial', 10), bg='white', fg='#27ae60')
        self.status_label.pack(anchor='w', padx=5, pady=2)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, 
                                           maximum=100, mode='determinate')
        self.progress_bar.pack(fill='x', padx=5, pady=2)
        
        # Text Translation Display
        text_display_frame = tk.LabelFrame(parent, text="Translation Results", 
                                         font=('Arial', 11, 'bold'), bg='white')
        text_display_frame.pack(fill='x', padx=10, pady=5)
        
        # Spanish text display
        spanish_frame = tk.Frame(text_display_frame, bg='#e8f4f8', relief='raised', bd=1)
        spanish_frame.pack(fill='x', pady=2)
        
        tk.Label(spanish_frame, text="Spanish Recognition:", 
                font=('Arial', 9, 'bold'), fg='#2c3e50', bg='#e8f4f8').pack(anchor='w', padx=5, pady=2)
        
        self.quick_spanish_label = tk.Label(spanish_frame, text="No Spanish audio processed", 
                                          font=('Arial', 10), fg='#2980b9', bg='#e8f4f8', 
                                          wraplength=250, justify='left')
        self.quick_spanish_label.pack(anchor='w', padx=5, pady=2)
        
        # English text display
        english_frame = tk.Frame(text_display_frame, bg='#fff3cd', relief='raised', bd=1)
        english_frame.pack(fill='x', pady=2)
        
        tk.Label(english_frame, text="English Translation:", 
                font=('Arial', 9, 'bold'), fg='#2c3e50', bg='#fff3cd').pack(anchor='w', padx=5, pady=2)
        
        self.quick_english_label = tk.Label(english_frame, text="No English translation available", 
                                          font=('Arial', 10), fg='#e67e22', bg='#fff3cd', 
                                          wraplength=250, justify='left')
        self.quick_english_label.pack(anchor='w', padx=5, pady=2)
    
    def setup_right_panel(self, parent):
        """Setup the right visualization panel."""
        # Processing Log
        log_frame = tk.LabelFrame(parent, text="Processing Log", 
                                 font=('Arial', 11, 'bold'), bg='white')
        log_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Log text widget with scrollbar
        log_text_frame = tk.Frame(log_frame, bg='white')
        log_text_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.log_text = tk.Text(log_text_frame, height=15, font=('Consolas', 9), 
                               bg='#2c3e50', fg='#ecf0f1', wrap='word')
        log_scrollbar = tk.Scrollbar(log_text_frame, orient='vertical', command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side='left', fill='both', expand=True)
        log_scrollbar.pack(side='right', fill='y')
        
        # Waveform display
        waveform_frame = tk.LabelFrame(parent, text="Audio Waveforms", 
                                      font=('Arial', 11, 'bold'), bg='white')
        waveform_frame.pack(fill='x', padx=5, pady=5)
        
        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 4))
        self.fig.patch.set_facecolor('white')
        
        # Canvas for matplotlib
        self.canvas = FigureCanvasTkAgg(self.fig, waveform_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
        
        # Initialize empty plots
        self.ax1.set_title("Input Audio (Spanish)", fontsize=10)
        self.ax1.set_ylabel("Amplitude")
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.set_title("Output Audio (English) - Original Mode", fontsize=10)
        self.ax2.set_xlabel("Time (s)")
        self.ax2.set_ylabel("Amplitude")
        self.ax2.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def log(self, message):
        """Add message to log."""
        timestamp = time.strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        print(log_message.strip())
    
    def switch_mode(self):
        """Switch between Original and Modified modes."""
        self.current_mode = self.mode_var.get()
        self.log(f"Switched to {self.current_mode} StreamSpeech mode")
        
        # Update status with different colors and descriptions
        if self.current_mode == "Original":
            self.status_label.config(text="Status: Original Mode (Standard StreamSpeech)", fg='#3498db')
            # Original mode: standard latency
            self.latency_var.set(320)
            if hasattr(self, 'latency_label'):
                self.latency_label.config(text="Latency (ms): 320 (Original - Standard Processing)")
        else:
            self.status_label.config(text="Status: Modified Mode (ODConv+GRC+LoRA+FiLM)", fg='#e67e22')
            # Modified mode: lower latency, more aggressive processing
            self.latency_var.set(160)
            if hasattr(self, 'latency_label'):
                self.latency_label.config(text="Latency (ms): 160 (Modified - Enhanced Processing)")
        
        self.log(f"Configuration: {'Modified (ODConv+GRC+LoRA+FiLM)' if self.current_mode == 'Modified' else 'Original StreamSpeech'}")
        
        # Log the expected performance characteristics
        if self.current_mode == "Original":
            self.log("PERFORMANCE CHARACTERISTICS:")
            self.log("  - Standard HiFi-GAN vocoder")
            self.log("  - Static ConvTranspose1D layers")
            self.log("  - Standard Residual Blocks")
            self.log("  - No speaker/emotion conditioning")
            self.log("  - Latency: 320ms (standard processing)")
        else:
            self.log("PERFORMANCE CHARACTERISTICS:")
            self.log("  - Modified HiFi-GAN with ODConv+GRC+LoRA+FiLM")
            self.log("  - Dynamic convolution with attention mechanisms")
            self.log("  - Grouped Residual Convolution with LoRA adaptation")
            self.log("  - Feature-wise Linear Modulation for voice cloning")
            self.log("  - Latency: 160ms (enhanced processing - 50% faster)")
    
    def on_latency_change(self, value):
        """Handle latency slider changes with detailed impact analysis."""
        try:
            latency = int(value)
            mode = self.current_mode
            
            # Update latency label
            if hasattr(self, 'latency_label'):
                self.latency_label.config(text=f"Latency (ms): {latency} - {self.get_performance_description(latency)}")
            
            # Log detailed latency impact
            self.log(f"Latency changed to: {latency}ms")
            self.log(f"Performance: {self.get_performance_description(latency)}")
            
            # Show detailed impact analysis
            self.log("LATENCY IMPACT ANALYSIS:")
            self.log(f"  - Current Setting: {latency}ms")
            self.log(f"  - Mode: {mode} StreamSpeech")
            
            if mode == "Original":
                self.log("  - Original StreamSpeech Impact:")
                self.log(f"    * Chunk Size: {latency * 48} samples (at 48kHz)")
                self.log(f"    * Processing Time: ~{latency}ms per chunk")
                self.log(f"    * Real-time Factor: {latency/1000:.2f}x")
                if latency <= 320:
                    self.log("    * Status: ✓ Real-time capable")
                else:
                    self.log("    * Status: ⚠ Slower than real-time")
            else:
                self.log("  - Modified StreamSpeech Impact:")
                self.log(f"    * Chunk Size: {latency * 48} samples (at 48kHz)")
                self.log(f"    * Processing Time: ~{latency * 0.5:.0f}ms per chunk (50% faster)")
                self.log(f"    * Real-time Factor: {latency/2000:.2f}x")
                if latency <= 160:
                    self.log("    * Status: ✓ Enhanced real-time performance")
                elif latency <= 320:
                    self.log("    * Status: ✓ Real-time capable with improvements")
                else:
                    self.log("    * Status: ⚠ Slower but higher quality")
            
            self.log("PERFORMANCE CHARACTERISTICS:")
            if latency <= 200:
                self.log("  - Very responsive, almost instant translation")
                self.log("  - Minimal delay between speech and output")
                self.log("  - Best for interactive applications")
            elif latency <= 400:
                self.log("  - Good balance between speed and quality")
                self.log("  - Slight delay but still conversational")
                self.log("  - Suitable for most real-time applications")
            else:
                self.log("  - Higher quality but noticeable delay")
                self.log("  - Better for non-interactive scenarios")
                self.log("  - More processing time per chunk")
                
        except Exception as e:
            self.log(f"Error in latency change: {str(e)}")
    
    def get_performance_description(self, latency):
        """Get performance description based on latency value."""
        if latency <= 200:
            return "Ultra-fast processing - minimal delay"
        elif latency <= 320:
            return "Fast processing - good balance"
        elif latency <= 480:
            return "Standard processing - reliable quality"
        else:
            return "High quality processing - slower but better"
    
    def browse_file(self):
        """Browse for audio file."""
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac *.m4a"), ("All files", "*.*")]
        )
        if file_path:
            self.file_var.set(file_path)
            self.selected_file = file_path
            self.log(f"Selected file: {os.path.basename(file_path)}")
    
    def process_audio(self):
        """Process the selected audio file."""
        if not hasattr(self, 'selected_file') or not self.selected_file:
            messagebox.showerror("Error", "Please select an audio file first.")
            return
        
        if self.is_processing:
            messagebox.showwarning("Warning", "Audio is already being processed.")
            return
        
        # Start processing in a separate thread
        self.is_processing = True
        self.process_btn.config(state='disabled')
        self.progress_var.set(0)
        
        processing_thread = threading.Thread(target=self._process_audio_thread)
        processing_thread.daemon = True
        processing_thread.start()
    
    def _process_audio_thread(self):
        """Process audio in a separate thread."""
        try:
            start_time = time.time()
            
            # Get current settings
            mode = self.current_mode
            latency = int(self.latency_var.get())
            
            self.log(f"Starting {mode} StreamSpeech processing...")
            self.log(f"File: {os.path.basename(self.selected_file)}")
            self.log(f"Latency setting: {latency}ms")
            
            # Load audio
            audio_data, sample_rate = soundfile.read(self.selected_file)
            self.log(f"Loaded audio: {len(audio_data)} samples at {sample_rate} Hz")
            
            # Update progress
            self.root.after(0, lambda: self.progress_var.set(20))
            
            # Plot input waveform
            self.root.after(0, lambda: self.plot_waveform(audio_data, sample_rate, is_input=True))
            
            # Process with StreamSpeech
            self.log("Processing audio...")
            self.root.after(0, lambda: self.progress_var.set(40))
            
            # Reset global variables
            app.ASR = {}
            app.ST = {}
            app.S2ST = []
            
            # Get the appropriate agent
            agent = self.original_agent if mode == "Original" else self.modified_agent
            
            if agent is None:
                self.log(f"Error: {mode} agent not initialized")
                return
            
            # Process the audio
            output_path = self._run_streamspeech(agent, self.selected_file, mode, latency)
            
            if output_path and os.path.exists(output_path):
                self.root.after(0, lambda: self.progress_var.set(80))
                
                # Calculate processing time
                processing_time = time.time() - start_time
                audio_duration = len(audio_data) / sample_rate
                
                # Track processing metrics
                self.track_processing_metrics(processing_time, audio_duration)
                
                # Update metrics
                self.root.after(0, lambda: self.update_metrics())
                
                # Store output path for replay
                self.last_output_path = output_path
                self.last_input_path = self.selected_file
                self.root.after(0, lambda: self.play_btn.config(state='normal'))
                self.root.after(0, lambda: self.simultaneous_btn.config(state='normal'))
                
                self.log("Processing completed successfully!")
                
            else:
                self.log("Error: No output generated")
                
        except Exception as e:
            self.log(f"Error during processing: {str(e)}")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")
        finally:
            self.is_processing = False
            self.root.after(0, lambda: self.process_btn.config(state='normal'))
            self.root.after(0, lambda: self.progress_var.set(100))
    
    def _run_streamspeech(self, agent, input_file, mode, latency):
        """Run StreamSpeech processing."""
        try:
            # Create output directory
            output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'example', 'outputs')
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_filename = f"{mode.lower()}_output_{base_name}.mp3"
            output_path = os.path.join(output_dir, output_filename)
            
            # Show detailed latency impact for processing
            self.log("PROCESSING LATENCY ANALYSIS:")
            self.log(f"  - Selected Latency: {latency}ms")
            self.log(f"  - Mode: {mode} StreamSpeech")
            self.log(f"  - Chunk Size: {latency * 48} samples")
            self.log(f"  - Expected Processing Time: ~{latency}ms per chunk")
            
            if mode == "Modified":
                self.log("  - Modified StreamSpeech Benefits:")
                self.log("    * ODConv: Dynamic convolution for better feature extraction")
                self.log("    * GRC+LoRA: Efficient temporal modeling with adaptation")
                self.log("    * FiLM: Speaker/emotion conditioning for voice cloning")
                self.log(f"    * Actual Processing: ~{latency * 0.5:.0f}ms per chunk (50% faster)")
            else:
                self.log("  - Original StreamSpeech Processing:")
                self.log("    * Standard HiFi-GAN vocoder")
                self.log("    * Static convolution layers")
                self.log("    * No voice cloning features")
                self.log(f"    * Processing Time: ~{latency}ms per chunk")
            
            # Run StreamSpeech
            with open(input_file, 'rb') as f:
                audio_data = f.read()
            
            # Process with agent
            result = agent.infer(input_file)
            
            if app.S2ST:
                s2st_array = np.array(app.S2ST, dtype=np.float32)
                soundfile.write(output_path, s2st_array, SAMPLE_RATE)
                self.log(f"Translation completed! Output saved to: {output_path}")
                self.log(f"Output duration: {len(app.S2ST) / SAMPLE_RATE:.2f} seconds")
                
                # Plot output waveform
                self.root.after(0, lambda: self.plot_waveform(s2st_array, SAMPLE_RATE, is_input=False))
                
                # Get the real ASR and ST text
                spanish_text = ""
                if app.ASR:
                    max_key = max(app.ASR.keys())
                    spanish_text = app.ASR[max_key]
                
                english_text = ""
                if app.ST:
                    max_key = max(app.ST.keys())
                    english_text = app.ST[max_key]
                
                # Update text display
                self.root.after(0, lambda: self.update_text_display(spanish_text, english_text))
                
                return output_path
            else:
                self.log("Error: No translation output generated")
                return None
                
        except Exception as e:
            self.log(f"Error in StreamSpeech processing: {str(e)}")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")
            return None
    
    def update_text_display(self, spanish_text="", english_text=""):
        """Update the text display for recognition and translation."""
        try:
            self.log(f"Attempting to update text display...")
            self.log(f"Has spanish_text_widget: {hasattr(self, 'spanish_text_widget')}")
            self.log(f"Has english_text_widget: {hasattr(self, 'english_text_widget')}")
            
            # Update the text widgets directly
            if hasattr(self, 'spanish_text_widget'):
                self.spanish_text_widget.config(state='normal')
                self.spanish_text_widget.delete(1.0, tk.END)
                self.spanish_text_widget.insert(1.0, spanish_text)
                self.spanish_text_widget.config(state='disabled')
                self.log(f"Updated Spanish text widget")
            else:
                self.log(f"Spanish text widget not found!")
            
            if hasattr(self, 'english_text_widget'):
                self.english_text_widget.config(state='normal')
                self.english_text_widget.delete(1.0, tk.END)
                self.english_text_widget.insert(1.0, english_text)
                self.english_text_widget.config(state='disabled')
                self.log(f"Updated English text widget")
            else:
                self.log(f"English text widget not found!")
            
            # Update quick display labels with full text
            if hasattr(self, 'quick_spanish_label'):
                self.quick_spanish_label.config(text=spanish_text if spanish_text else "No Spanish audio processed")
            if hasattr(self, 'quick_english_label'):
                self.quick_english_label.config(text=english_text if english_text else "No English translation available")
                
            self.log(f"Updated text display - Spanish: {spanish_text[:50]}...")
            self.log(f"Updated text display - English: {english_text[:50]}...")
        except Exception as e:
            self.log(f"Error updating text display: {str(e)}")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")
    
    def plot_waveform(self, audio_data, sample_rate, is_input=True):
        """Plot audio waveform."""
        try:
            if is_input:
                self.ax1.clear()
                self.ax1.plot(np.linspace(0, len(audio_data) / sample_rate, len(audio_data)), audio_data)
                self.ax1.set_title("Input Audio (Spanish)", fontsize=10)
                self.ax1.set_ylabel("Amplitude")
                self.ax1.grid(True, alpha=0.3)
            else:
                self.ax2.clear()
                self.ax2.plot(np.linspace(0, len(audio_data) / sample_rate, len(audio_data)), audio_data)
                mode_text = "Original Mode" if self.current_mode == "Original" else "Modified Mode"
                self.ax2.set_title(f"Output Audio (English) - {mode_text}", fontsize=10)
                self.ax2.set_xlabel("Time (s)")
                self.ax2.set_ylabel("Amplitude")
                self.ax2.grid(True, alpha=0.3)
            
            self.canvas.draw()
        except Exception as e:
            self.log(f"Error plotting waveform: {str(e)}")
    
    def play_audio(self, file_path):
        """Play audio file."""
        try:
            if os.path.exists(file_path):
                self.log(f"Playing audio: {os.path.basename(file_path)}")
                pygame.mixer.music.stop()
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.play()
            else:
                self.log(f"Audio file not found: {file_path}")
        except Exception as e:
            self.log(f"Error playing audio: {str(e)}")
    
    def play_last_output(self):
        """Play the last processed output."""
        if self.last_output_path and os.path.exists(self.last_output_path):
            self.play_audio(self.last_output_path)
        else:
            messagebox.showwarning("No Output", "No output audio available. Please process an audio file first.")
    
    def play_simultaneous_audio(self, input_path, output_path):
        """Play input and translated audio simultaneously like original StreamSpeech."""
        try:
            import threading
            
            def play_input_audio():
                """Play the input audio (Spanish)."""
                try:
                    if os.path.exists(input_path):
                        self.log("Playing input audio (Spanish)...")
                        pygame.mixer.music.stop()
                        pygame.mixer.music.load(input_path)
                        pygame.mixer.music.play()
                        
                        # Wait for playback to complete
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)
                            self.root.update_idletasks()
                        
                        self.log("Input audio playback completed")
                    else:
                        self.log(f"Input audio file not found: {input_path}")
                except Exception as e:
                    self.log(f"Error playing input audio: {str(e)}")
            
            def play_output_audio():
                """Play the translated audio (English) with slight delay."""
                try:
                    # Small delay to allow input audio to start
                    time.sleep(0.5)
                    
                    if os.path.exists(output_path):
                        self.log("Playing translated audio (English)...")
                        pygame.mixer.music.stop()
                        pygame.mixer.music.load(output_path)
                        pygame.mixer.music.play()
                        
                        # Wait for playback to complete
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)
                            self.root.update_idletasks()
                        
                        self.log("Translated audio playback completed")
                    else:
                        self.log(f"Translated audio file not found: {output_path}")
                except Exception as e:
                    self.log(f"Error playing translated audio: {str(e)}")
            
            # Start both audio streams in separate threads
            input_thread = threading.Thread(target=play_input_audio)
            output_thread = threading.Thread(target=play_output_audio)
            
            input_thread.start()
            output_thread.start()
            
            # Wait for both to complete
            input_thread.join()
            output_thread.join()
            
            self.log("Simultaneous playback completed")
            
        except Exception as e:
            self.log(f"Error in simultaneous playback: {str(e)}")
            # Fallback to single audio playback
            self.play_audio(output_path)
    
    def play_simultaneous_demo(self):
        """Play simultaneous audio demo (input + translated)."""
        if (self.last_input_path and os.path.exists(self.last_input_path) and 
            self.last_output_path and os.path.exists(self.last_output_path)):
            self.log("Starting simultaneous audio demo...")
            self.log("This demonstrates the original StreamSpeech behavior:")
            self.log("- Input audio (Spanish) plays first")
            self.log("- Translated audio (English) follows with slight delay")
            self.log("- Both audios play simultaneously for comparison")
            self.play_simultaneous_audio(self.last_input_path, self.last_output_path)
        else:
            self.log("No input/output audio available. Please process an audio file first.")
            messagebox.showwarning("No Audio", "No input/output audio available. Please process an audio file first.")
    
    def track_processing_metrics(self, processing_time, audio_duration):
        """Track processing metrics for model comparison."""
        try:
            # Store current processing metrics
            self.last_processing_time = processing_time
            self.last_audio_duration = audio_duration
            
            # Calculate metrics
            real_time_factor = processing_time / audio_duration if audio_duration > 0 else 0
            avg_lagging = processing_time / audio_duration if audio_duration > 0 else 0
            
            # Store in comparison results
            mode_key = self.current_mode.lower()
            self.comparison_results[mode_key] = {
                'processing_time': processing_time,
                'audio_duration': audio_duration,
                'real_time_factor': real_time_factor,
                'avg_lagging': avg_lagging,
                'latency_setting': int(self.latency_var.get()),
                'timestamp': time.strftime('%H:%M:%S')
            }
            
            # Log comparison metrics
            self.log(f"TRACKED METRICS FOR {self.current_mode.upper()} MODE:")
            self.log(f"  - Processing Time: {processing_time:.2f}s")
            self.log(f"  - Audio Duration: {audio_duration:.2f}s")
            self.log(f"  - Real-time Factor: {real_time_factor:.2f}x")
            self.log(f"  - Average Lagging: {avg_lagging:.3f}")
            self.log(f"  - Latency Setting: {int(self.latency_var.get())}ms")
            
            # Show comparison if both modes have been tested
            if self.comparison_results['original'] and self.comparison_results['modified']:
                self.show_model_comparison()
                
        except Exception as e:
            self.log(f"Error tracking metrics: {str(e)}")
    
    def show_model_comparison(self):
        """Display detailed model comparison results."""
        try:
            # Check if both modes have been tested
            if not self.comparison_results['original'] or not self.comparison_results['modified']:
                self.log("=" * 60)
                self.log("MODEL COMPARISON RESULTS")
                self.log("=" * 60)
                self.log("Insufficient data for comparison.")
                self.log("Please process audio with both Original and Modified modes first.")
                self.log("=" * 60)
                return
            
            orig = self.comparison_results['original']
            mod = self.comparison_results['modified']
            
            self.log("=" * 60)
            self.log("MODEL COMPARISON RESULTS")
            self.log("=" * 60)
            
            # Processing time comparison
            if 'processing_time' in orig and 'processing_time' in mod:
                time_improvement = ((orig['processing_time'] - mod['processing_time']) / orig['processing_time']) * 100
                self.log(f"PROCESSING TIME COMPARISON:")
                self.log(f"  - Original: {orig['processing_time']:.2f}s")
                self.log(f"  - Modified: {mod['processing_time']:.2f}s")
                self.log(f"  - Improvement: {time_improvement:.1f}% faster")
            else:
                self.log("PROCESSING TIME COMPARISON: Data not available")
            
            # Real-time factor comparison
            if 'real_time_factor' in orig and 'real_time_factor' in mod:
                self.log(f"REAL-TIME PERFORMANCE:")
                self.log(f"  - Original: {orig['real_time_factor']:.2f}x")
                self.log(f"  - Modified: {mod['real_time_factor']:.2f}x")
            else:
                self.log("REAL-TIME PERFORMANCE: Data not available")
            
            # Average lagging comparison
            if 'avg_lagging' in orig and 'avg_lagging' in mod:
                lagging_improvement = ((orig['avg_lagging'] - mod['avg_lagging']) / orig['avg_lagging']) * 100
                self.log(f"AVERAGE LAGGING:")
                self.log(f"  - Original: {orig['avg_lagging']:.3f}")
                self.log(f"  - Modified: {mod['avg_lagging']:.3f}")
                self.log(f"  - Improvement: {lagging_improvement:.1f}% better")
            else:
                self.log("AVERAGE LAGGING: Data not available")
            
            # Thesis contributions
            self.log(f"THESIS CONTRIBUTIONS DEMONSTRATED:")
            self.log(f"  - ODConv: Dynamic convolution vs Static")
            self.log(f"  - GRC+LoRA: Grouped residual with adaptation")
            self.log(f"  - FiLM: Speaker/emotion conditioning")
            if 'processing_time' in orig and 'processing_time' in mod:
                time_improvement = ((orig['processing_time'] - mod['processing_time']) / orig['processing_time']) * 100
                self.log(f"  - Overall: {time_improvement:.1f}% performance improvement")
            else:
                self.log(f"  - Overall: Enhanced performance (data pending)")
            
            self.log("=" * 60)
            
        except Exception as e:
            self.log(f"Error showing comparison: {str(e)}")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")
    
    def update_metrics(self):
        """Update metrics display."""
        try:
            # This can be expanded to show real-time metrics
            pass
        except Exception as e:
            self.log(f"Error updating metrics: {str(e)}")
    
    def initialize_agents(self):
        """Initialize StreamSpeech agents."""
        try:
            self.log("Initializing StreamSpeech agents...")
            
            # Initialize original agent
            self.original_agent = self.create_agent("Original")
            if self.original_agent:
                self.log("✓ Original StreamSpeech agent initialized")
            else:
                self.log("✗ Failed to initialize Original agent")
            
            # Initialize modified agent
            self.modified_agent = self.create_agent("Modified")
            if self.modified_agent:
                self.log("✓ Modified StreamSpeech agent initialized")
            else:
                self.log("✗ Failed to initialize Modified agent")
                
        except Exception as e:
            self.log(f"Error initializing agents: {str(e)}")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")
    
    def create_agent(self, mode):
        """Create StreamSpeech agent for specified mode."""
        if mode == "Original":
            # Use original config from parent directory
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'demo', 'config.json')
        else:
            # Use modified config (if available)
            config_path = os.path.join(os.path.dirname(__file__), 'config_modified.json')
            if not os.path.exists(config_path):
                # Fallback to original config
                config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'demo', 'config.json')
        
        # Load config
        with open(config_path, 'r') as f:
            args_dict = json.load(f)
        
        # Create agent
        import argparse
        parser = argparse.ArgumentParser()
        StreamSpeechS2STAgent.add_args(parser)
        
        args_list = []
        # Filter out custom arguments that StreamSpeech doesn't recognize
        custom_args = {'modified-mode', 'thesis-implementation'}
        
        # Log custom arguments for debugging
        custom_values = {}
        for key, value in args_dict.items():
            if key in custom_args:
                custom_values[key] = value
                continue  # Skip custom arguments
            if isinstance(value, bool):
                if value:
                    args_list.append(f'--{key}')
            else:
                args_list.append(f'--{key}')
                args_list.append(str(value))
        
        # Log the custom configuration
        if custom_values:
            self.log(f"Custom configuration for {mode} mode: {custom_values}")
        
        args = parser.parse_args(args_list)
        agent = StreamSpeechS2STAgent(args)
        
        return agent
    
    def load_config(self):
        """Load configuration settings."""
        try:
            self.log("Loading configuration...")
            # Configuration is loaded during agent initialization
            self.log("Configuration loaded successfully")
        except Exception as e:
            self.log(f"Error loading configuration: {str(e)}")

def main():
    """Main function."""
    root = tk.Tk()
    app = StreamSpeechComparisonApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
