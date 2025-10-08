#!/usr/bin/env python3
"""
Test script to verify text display functionality
"""

import sys
import os
import tkinter as tk
from tkinter import ttk

# Add the current directory to path
sys.path.append(os.path.dirname(__file__))

def test_text_display():
    """Test the text display functionality"""
    root = tk.Tk()
    root.title("Text Display Test")
    root.geometry("600x400")
    
    # Create a simple text display
    frame = ttk.Frame(root)
    frame.pack(fill='both', expand=True, padx=10, pady=10)
    
    # Spanish text
    spanish_frame = ttk.LabelFrame(frame, text="Spanish Recognition")
    spanish_frame.pack(fill='x', pady=5)
    
    spanish_text = tk.Text(spanish_frame, height=4, font=('Arial', 11), 
                          wrap='word', state='disabled')
    spanish_text.pack(fill='x', padx=5, pady=5)
    
    # English text
    english_frame = ttk.LabelFrame(frame, text="English Translation")
    english_frame.pack(fill='x', pady=5)
    
    english_text = tk.Text(english_frame, height=4, font=('Arial', 11), 
                          wrap='word', state='disabled')
    english_text.pack(fill='x', padx=5, pady=5)
    
    # Test button
    def update_text():
        spanish_text.config(state='normal')
        spanish_text.delete(1.0, tk.END)
        spanish_text.insert(1.0, "Hola, este es un texto de prueba en español")
        spanish_text.config(state='disabled')
        
        english_text.config(state='normal')
        english_text.delete(1.0, tk.END)
        english_text.insert(1.0, "Hello, this is a test text in English")
        english_text.config(state='disabled')
    
    test_btn = tk.Button(frame, text="Test Text Update", command=update_text)
    test_btn.pack(pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    test_text_display()







