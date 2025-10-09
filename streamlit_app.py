"""
StreamSpeech Comparison Tool - Streamlit Web UI
================================================
Final clean design with light theme and white titles

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import os
import time
import numpy as np
import soundfile
import plotly.graph_objects as go
import base64
from io import BytesIO

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="StreamSpeech Comparison Tool",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS - Final Clean Design
# ============================================================================

st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Main container - Light theme */
    .main {
        background: #f8f9fa;
        padding: 2rem 1rem;
        min-height: 100vh;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Container width - Full width */
    .stApp {
        max-width: 100% !important;
    }
    
    .main .block-container {
        max-width: 100% !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }
    
    /* Force full width for tabs */
    .stTabs {
        width: 100% !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        width: 100% !important;
        max-width: none !important;
    }
    
    /* Header */
    .header-container {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: white;
        margin: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.75rem;
    }
    
    .subtitle {
        color: white;
        font-size: 1rem;
        margin-top: 0.75rem;
        max-width: 42rem;
        margin-left: auto;
        margin-right: auto;
        text-align: center;
    }
    
    /* Step numbers */
    .step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 2rem;
        height: 2rem;
        background: #3b82f6;
        color: white;
        border-radius: 50%;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    
    .step-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: white;
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    /* Cards */
    .stButton > button {
        border-radius: 0.5rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Model cards */
    .original-header {
        background: #fef3c7;
        padding: 1.5rem;
        border-radius: 0.5rem 0.5rem 0 0;
        color: #111827;
    }
    
    .modified-header {
        background: #dbeafe;
        padding: 1.5rem;
        border-radius: 0.5rem 0.5rem 0 0;
        color: #111827;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 0.375rem;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .badge-original {
        background: #f3f4f6;
        color: #374151;
    }
    
    .badge-modified {
        background: #2563eb;
        color: white;
    }
    
    /* Metric cards */
    .metric-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid;
    }
    
    .metric-card-orange {
        background: #ffedd5;
        border-color: #fed7aa;
    }
    
    .metric-card-blue {
        background: #dbeafe;
        border-color: #bfdbfe;
    }
    
    .metric-card-green {
        background: #d1fae5;
        border-color: #a7f3d0;
    }
    
    .metric-card-purple {
        background: #e9d5ff;
        border-color: #d8b4fe;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #374151;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
        color: #000000;
    }
    
    .metric-subtext {
        font-size: 0.75rem;
        color: #000000;
    }
    
    /* Waveform container */
    .waveform-container {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Transcription boxes */
    .transcription-box {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        font-size: 0.875rem;
        line-height: 1.5;
    }
    
    .transcription-input {
        color: #374151;
    }
    
    .transcription-output-original {
        color: #374151;
    }
    
    .transcription-output-modified {
        color: #374151;
    }
    
    /* Section separators */
    .section-separator {
        height: 1px;
        background: #e5e7eb;
        margin: 1rem 0;
    }
    
    /* Info alert */
    .info-alert {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .info-alert-text {
        color: #1e40af;
        font-size: 0.875rem;
        margin: 0;
    }
    
    /* Processing log */
    .processing-log {
        background: #111827;
        border-radius: 0.5rem;
        padding: 1.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.875rem;
        line-height: 1.5;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .log-info {
        color: #10b981;
        margin-bottom: 0.5rem;
    }
    
    .log-error {
        color: #ef4444;
        margin-bottom: 0.5rem;
    }
    
    .log-warning {
        color: #f59e0b;
        margin-bottom: 0.5rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background: #e5e7eb;
        padding: 0.25rem;
        border-radius: 0.5rem;
        border: none;
        width: 100% !important;
        max-width: none !important;
        margin: 0 auto 1.5rem auto;
        max-width: 28rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 0.375rem;
        color: #1f2937;
        font-weight: 600;
        padding: 0.5rem 1rem;
        background: transparent;
        border: none;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: white;
        color: #1f2937;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Hide default streamlit elements */
    .element-container:has(> .stMarkdown > div > .step-title) {
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None
if 'original_processed' not in st.session_state:
    st.session_state.original_processed = False
if 'modified_processed' not in st.session_state:
    st.session_state.modified_processed = False
if 'original_latency' not in st.session_state:
    st.session_state.original_latency = 320
if 'modified_latency' not in st.session_state:
    st.session_state.modified_latency = 160
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'modified_data' not in st.session_state:
    st.session_state.modified_data = None

# ============================================================================
# DUMMY FUNCTIONS
# ============================================================================

def process_audio_mock(model_type, latency):
    """Mock audio processing function"""
    time.sleep(1)  # Simulate processing time
    
    return {
        'transcription': f"This is a sample transcription for the {model_type} model with {latency}ms latency.",
        'processing_time': 850 if model_type == 'original' else 620,
        'accuracy': 92.5 if model_type == 'original' else 97.8,
        'total_time': 1170 if model_type == 'original' else 780
    }

def create_waveform(samples, sr, color, title):
    """Create a Plotly waveform visualization"""
    # Downsample for performance
    step = max(1, len(samples) // 1000)
    x = np.arange(len(samples[::step])) / sr
    y = samples[::step]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        line=dict(color=color, width=1),
        name=title
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=200,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def get_performance_description_dummy():
    """Get dummy performance description"""
    return "The modified model shows significant improvements in processing speed and accuracy."

def get_comparison_summary_dummy():
    """Get dummy comparison summary"""
    return "Modified model is 33.3% faster and 5.3% more accurate than the original."

def log_message_dummy(message):
    """Dummy logging function"""
    print(f"[LOG] {message}")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Header
st.markdown("""
<div class="header-container">
    <div class="main-title">
        <span style="font-size: 2.5rem;">🎵</span>
        <span>StreamSpeech Comparison Tool</span>
    </div>
    <p class="subtitle">
        Compare original HiFi-GAN and modified HiFi-GAN models side-by-side with real-time performance metrics
    </p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["Comparison", "Processing Log"])

with tab1:
    # Info Alert
    st.markdown("""
    <div class="info-alert">
        <span style="font-size: 1.25rem;">ℹ️</span>
        <p class="info-alert-text">Upload an MP3 audio file to start comparing the two models. Process each model independently to see detailed metrics.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 1: File Upload
    st.markdown('<div class="step-title"><span class="step-number">1</span>Input Audio Selection</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Drop MP3 file here or click to browse",
        type=['mp3', 'wav', 'flac'],
        help="Supports MP3, WAV, and FLAC audio files",
        key="audio_uploader"
    )
    
    if uploaded_file:
        st.session_state.audio_file = uploaded_file
        st.success(f"✅ Audio file uploaded: {uploaded_file.name}")
    
    # Step 2: Model Comparison
    if st.session_state.audio_file:
        st.markdown('<div class="step-title"><span class="step-number">2</span>Process & Compare Models</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        # ORIGINAL MODEL CARD
        with col1:
            st.markdown("""
            <div style="background: #fef3c7; padding: 1.5rem; border-radius: 0.5rem 0.5rem 0 0; color: #111827;">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
                    <div style="width: 0.75rem; height: 0.75rem; background: #f59e0b; border-radius: 9999px;"></div>
                    <span class="badge badge-original">ORIGINAL MODEL</span>
                </div>
                <h3 style="margin: 0; font-size: 1.125rem; font-weight: 600;">HiFi-GAN Base Model</h3>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.875rem; color: #6b7280;">Standard implementation with default parameters</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div style="padding: 1.5rem;">', unsafe_allow_html=True)
            
            # Process Button
            if st.button("Process Audio", key="process_original", use_container_width=True, type="primary"):
                with st.spinner("Processing with Original HiFi-GAN model..."):
                    result = process_audio_mock('original', st.session_state.original_latency)
                    st.session_state.original_data = result
                    st.session_state.original_processed = True
                st.success("✅ Original model processing completed!")
            
            # Show processing status if processed
            if st.session_state.original_processed:
                st.success("✅ Original model processing completed!")
            
            # Input Audio
            st.markdown('<div class="metric-label"><span style="background: #6b7280; width: 0.5rem; height: 0.5rem; border-radius: 9999px; display: inline-block;"></span>INPUT AUDIO</div>', unsafe_allow_html=True)
            
            if st.session_state.audio_file:
                try:
                    samples, sr = soundfile.read(BytesIO(st.session_state.audio_file.getvalue()), dtype="float32")
                    fig = create_waveform(samples, sr, '#f59e0b', 'Input')
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                except Exception as e:
                    st.markdown('<div class="waveform-container"><div style="text-align: center; color: #374151; padding: 2rem; font-weight: 500;">📊 Waveform preview</div></div>', unsafe_allow_html=True)
            
            if st.session_state.original_processed:
                st.markdown('<div class="section-separator"></div>', unsafe_allow_html=True)
                
                # Input Transcription
                st.markdown('<div class="metric-label">INPUT TRANSCRIPTION</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="transcription-box transcription-input">{st.session_state.original_data["transcription"]}</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="section-separator"></div>', unsafe_allow_html=True)
                
                # Output Audio
                st.markdown('<div class="metric-label"><span style="background: #fb923c; width: 0.5rem; height: 0.5rem; border-radius: 9999px; display: inline-block;"></span>OUTPUT AUDIO</div>', unsafe_allow_html=True)
                if st.session_state.audio_file:
                    try:
                        samples, sr = soundfile.read(BytesIO(st.session_state.audio_file.getvalue()), dtype="float32")
                        fig = create_waveform(samples, sr, '#fb923c', 'Output')
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    except Exception as e:
                        st.markdown('<div class="waveform-container"><div style="text-align: center; color: #374151; padding: 2rem; font-weight: 500;">📊 Output waveform</div></div>', unsafe_allow_html=True)
                
                # Output Transcription
                st.markdown('<div class="metric-label">OUTPUT TRANSCRIPTION</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="transcription-box transcription-output-original">{st.session_state.original_data["transcription"]}</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="section-separator"></div>', unsafe_allow_html=True)
                
                # Playback Controls
                st.markdown('<div class="metric-label">AUDIO PLAYBACK</div>', unsafe_allow_html=True)
                playback_col1, playback_col2 = st.columns(2)
                with playback_col1:
                    st.button("▶️ Play Input", key="play_orig_input", use_container_width=True)
                with playback_col2:
                    st.button("▶️ Play Output", key="play_orig_output", use_container_width=True)
                
                st.markdown('<div class="section-separator"></div>', unsafe_allow_html=True)
                
                # Latency Control
                st.markdown('<div class="metric-label">⚙️ LATENCY CONTROL</div>', unsafe_allow_html=True)
                st.session_state.original_latency = st.slider(
                    "Latency", 50, 5000, st.session_state.original_latency, 10,
                    key="orig_latency",
                    label_visibility="collapsed"
                )
                
                st.markdown('<div class="section-separator"></div>', unsafe_allow_html=True)
                
                # Performance Metrics
                st.markdown('<div class="metric-label">PERFORMANCE METRICS</div>', unsafe_allow_html=True)
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Processing Time", f"{st.session_state.original_data['processing_time']}ms")
                    st.metric("Accuracy", f"{st.session_state.original_data['accuracy']}%")
                with metric_col2:
                    st.metric("Latency", f"{st.session_state.original_latency}ms")
                    st.metric("Total Time", f"{st.session_state.original_data['total_time']}ms")
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # MODIFIED MODEL CARD
        with col2:
            st.markdown("""
            <div style="background: #dbeafe; padding: 1.5rem; border-radius: 0.5rem 0.5rem 0 0; color: #111827;">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
                    <div style="width: 0.75rem; height: 0.75rem; background: #3b82f6; border-radius: 9999px;"></div>
                    <span class="badge badge-modified">MODIFIED MODEL</span>
                </div>
                <h3 style="margin: 0; font-size: 1.125rem; font-weight: 600;">HiFi-GAN + ODConv + GRC + LoRA</h3>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.875rem; color: #6b7280;">Enhanced with ODConv, GRC, and LoRA fine-tuning</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div style="padding: 1.5rem;">', unsafe_allow_html=True)
            
            # Process Button
            if st.button("Process Audio", key="process_modified", use_container_width=True, type="primary"):
                with st.spinner("Processing with Modified HiFi-GAN model..."):
                    result = process_audio_mock('modified', st.session_state.modified_latency)
                    st.session_state.modified_data = result
                    st.session_state.modified_processed = True
                st.success("✅ Modified model processing completed!")
            
            # Show processing status if processed
            if st.session_state.modified_processed:
                st.success("✅ Modified model processing completed!")
            
            # Input Audio
            st.markdown('<div class="metric-label"><span style="background: #6b7280; width: 0.5rem; height: 0.5rem; border-radius: 9999px; display: inline-block;"></span>INPUT AUDIO</div>', unsafe_allow_html=True)
            
            if st.session_state.audio_file:
                try:
                    samples, sr = soundfile.read(BytesIO(st.session_state.audio_file.getvalue()), dtype="float32")
                    fig = create_waveform(samples, sr, '#6366f1', 'Input')
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                except Exception as e:
                    st.markdown('<div class="waveform-container"><div style="text-align: center; color: #374151; padding: 2rem; font-weight: 500;">📊 Waveform preview</div></div>', unsafe_allow_html=True)
            
            if st.session_state.modified_processed:
                st.markdown('<div class="section-separator"></div>', unsafe_allow_html=True)
                
                # Input Transcription
                st.markdown('<div class="metric-label">INPUT TRANSCRIPTION</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="transcription-box transcription-input">{st.session_state.modified_data["transcription"]}</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="section-separator"></div>', unsafe_allow_html=True)
                
                # Output Audio
                st.markdown('<div class="metric-label"><span style="background: #3b82f6; width: 0.5rem; height: 0.5rem; border-radius: 9999px; display: inline-block;"></span>OUTPUT AUDIO</div>', unsafe_allow_html=True)
                if st.session_state.audio_file:
                    try:
                        samples, sr = soundfile.read(BytesIO(st.session_state.audio_file.getvalue()), dtype="float32")
                        fig = create_waveform(samples, sr, '#3b82f6', 'Output')
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    except Exception as e:
                        st.markdown('<div class="waveform-container"><div style="text-align: center; color: #374151; padding: 2rem; font-weight: 500;">📊 Output waveform</div></div>', unsafe_allow_html=True)
                
                # Output Transcription
                st.markdown('<div class="metric-label">OUTPUT TRANSCRIPTION</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="transcription-box transcription-output-modified">{st.session_state.modified_data["transcription"]}</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="section-separator"></div>', unsafe_allow_html=True)
                
                # Playback Controls
                st.markdown('<div class="metric-label">AUDIO PLAYBACK</div>', unsafe_allow_html=True)
                playback_col1, playback_col2 = st.columns(2)
                with playback_col1:
                    st.button("▶️ Play Input", key="play_mod_input", use_container_width=True)
                with playback_col2:
                    st.button("▶️ Play Output", key="play_mod_output", use_container_width=True)
                
                st.markdown('<div class="section-separator"></div>', unsafe_allow_html=True)
                
                # Latency Control
                st.markdown('<div class="metric-label">⚙️ LATENCY CONTROL</div>', unsafe_allow_html=True)
                st.session_state.modified_latency = st.slider(
                    "Latency", 50, 5000, st.session_state.modified_latency, 10,
                    key="mod_latency",
                    label_visibility="collapsed"
                )
                
                st.markdown('<div class="section-separator"></div>', unsafe_allow_html=True)
                
                # Performance Metrics
                st.markdown('<div class="metric-label">PERFORMANCE METRICS</div>', unsafe_allow_html=True)
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Processing Time", f"{st.session_state.modified_data['processing_time']}ms")
                    st.metric("Accuracy", f"{st.session_state.modified_data['accuracy']}%")
                with metric_col2:
                    st.metric("Latency", f"{st.session_state.modified_latency}ms")
                    st.metric("Total Time", f"{st.session_state.modified_data['total_time']}ms")
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 3: Simultaneous Playback
    if st.session_state.original_processed and st.session_state.modified_processed:
        st.markdown('<div class="step-title"><span class="step-number">3</span>Simultaneous Input/Output Playback</div>', unsafe_allow_html=True)
        
        st.markdown('<p style="text-align: center; color: #6b7280; margin-bottom: 1.5rem;">Play input and output audio simultaneously for each model with latency offset</p>', unsafe_allow_html=True)
        
        # Two model cards side by side
        simul_col1, simul_col2 = st.columns(2)
        
        with simul_col1:
            st.markdown("""
            <div style="background: #ffedd5; border: 2px solid #fb923c; border-radius: 0.75rem; padding: 1.5rem; text-align: center;">
                <div style="display: flex; align-items: center; justify-content: center; gap: 0.5rem; margin-bottom: 1rem;">
                    <div style="width: 0.75rem; height: 0.75rem; background: #fb923c; border-radius: 9999px;"></div>
                    <span style="color: #111827; font-weight: 600;">ORIGINAL MODEL</span>
                </div>
                <button style="background: #fb923c; color: white; border: none; border-radius: 0.5rem; padding: 0.75rem 1.5rem; font-weight: 600; width: 100%; margin-bottom: 0.75rem; cursor: pointer;">
                    ▶️ Play Simultaneous
                </button>
                <p style="color: #6b7280; font-size: 0.875rem; margin: 0;">Input plays immediately, output after 320ms</p>
            </div>
            """, unsafe_allow_html=True)
        
        with simul_col2:
            st.markdown("""
            <div style="background: #dbeafe; border: 2px solid #3b82f6; border-radius: 0.75rem; padding: 1.5rem; text-align: center;">
                <div style="display: flex; align-items: center; justify-content: center; gap: 0.5rem; margin-bottom: 1rem;">
                    <div style="width: 0.75rem; height: 0.75rem; background: #3b82f6; border-radius: 9999px;"></div>
                    <span style="color: #111827; font-weight: 600;">MODIFIED MODEL</span>
                </div>
                <button style="background: #3b82f6; color: white; border: none; border-radius: 0.5rem; padding: 0.75rem 1.5rem; font-weight: 600; width: 100%; margin-bottom: 0.75rem; cursor: pointer;">
                    ▶️ Play Simultaneous
                </button>
                <p style="color: #6b7280; font-size: 0.875rem; margin: 0;">Input plays immediately, output after 160ms</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<p style="text-align: center; color: #6b7280; font-size: 0.875rem; margin-top: 1rem;">Adjust latency controls in each model card to change the output delay</p>', unsafe_allow_html=True)
        
        # Step 4: Performance Comparison
        st.markdown('<div class="step-title"><span class="step-number">4</span>Performance Comparison</div>', unsafe_allow_html=True)
        
        # Summary Metrics
        met_col1, met_col2, met_col3, met_col4 = st.columns(4)
        
        with met_col1:
            st.markdown(f"""
            <div class="metric-card metric-card-orange">
                <div class="metric-label">
                    <span style="background: #fb923c; width: 0.75rem; height: 0.75rem; border-radius: 9999px; display: inline-block;"></span>
                    ORIGINAL
                </div>
                <div class="metric-value">1,170ms</div>
                <div class="metric-subtext">850ms + 320ms latency</div>
            </div>
            """, unsafe_allow_html=True)
        
        with met_col2:
            st.markdown(f"""
            <div class="metric-card metric-card-blue">
                <div class="metric-label">
                    <span style="background: #3b82f6; width: 0.75rem; height: 0.75rem; border-radius: 9999px; display: inline-block;"></span>
                    MODIFIED
                </div>
                <div class="metric-value">780ms</div>
                <div class="metric-subtext">620ms + 160ms latency</div>
            </div>
            """, unsafe_allow_html=True)
        
        with met_col3:
            st.markdown("""
            <div class="metric-card metric-card-green">
                <div class="metric-label">
                    📈 SPEED
                </div>
                <div class="metric-value" style="color: #10b981;">+33.3%</div>
                <div class="metric-subtext" style="color: #10b981;">390ms faster</div>
            </div>
            """, unsafe_allow_html=True)
        
        with met_col4:
            st.markdown("""
            <div class="metric-card metric-card-purple">
                <div class="metric-label">
                    📈 ACCURACY
                </div>
                <div class="metric-value" style="color: #8b5cf6;">+5.3%</div>
                <div class="metric-subtext" style="color: #8b5cf6;">97.8% vs 92.5%</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed Metrics
        st.markdown('<h3 style="font-size: 1.25rem; font-weight: 600; margin-top: 2rem; margin-bottom: 1.5rem; color: white;">Detailed Metrics Comparison</h3>', unsafe_allow_html=True)
        
        detail_col1, detail_col2 = st.columns(2)
        with detail_col1:
            st.markdown("""
            <div style="background: #f9fafb; padding: 1rem; border-radius: 0.5rem; margin-bottom: 0.75rem;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="font-size: 1rem; color: #000000; font-weight: 600;">Speaker Similarity:</span>
                    <span style="font-size: 1rem; font-weight: 600; color: #000000;">Original: 0.85 | Modified: 0.91</span>
                </div>
            </div>
            <div style="background: #f9fafb; padding: 1rem; border-radius: 0.5rem;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="font-size: 1rem; color: #000000; font-weight: 600;">Emotion Similarity:</span>
                    <span style="font-size: 1rem; font-weight: 600; color: #000000;">Original: 0.78 | Modified: 0.84</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with detail_col2:
            st.markdown("""
            <div style="background: #f9fafb; padding: 1rem; border-radius: 0.5rem; margin-bottom: 0.75rem;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="font-size: 1rem; color: #000000; font-weight: 600;">ASR-BLEU Score:</span>
                    <span style="font-size: 1rem; font-weight: 600; color: #000000;">Original: 0.89 | Modified: 0.94</span>
                </div>
            </div>
            <div style="background: #f9fafb; padding: 1rem; border-radius: 0.5rem;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="font-size: 1rem; color: #000000; font-weight: 600;">Average Lagging:</span>
                    <span style="font-size: 1rem; font-weight: 600; color: #000000;">Original: 0.92 | Modified: 0.88</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown('<h2 style="font-size: 1.5rem; font-weight: 600; margin-bottom: 1rem; color: white;">Processing Log</h2>', unsafe_allow_html=True)
    
    log_content = '<div class="processing-log">'
    log_content += '<div class="log-info">[INFO] System initialized</div>'
    log_content += '<div class="log-info">[INFO] Waiting for audio input...</div>'
    
    if st.session_state.audio_file:
        log_content += f'<div class="log-info">[INFO] Audio file loaded: {st.session_state.audio_file.name}</div>'
        
        if st.session_state.original_processed:
            log_content += '<div class="log-info">[INFO] Original model processing completed</div>'
            log_content += f'<div class="log-info">[INFO] Original model metrics: {st.session_state.original_data["processing_time"]}ms processing, {st.session_state.original_data["accuracy"]}% accuracy</div>'
        
        if st.session_state.modified_processed:
            log_content += '<div class="log-info">[INFO] Modified model processing completed</div>'
            log_content += f'<div class="log-info">[INFO] Modified model metrics: {st.session_state.modified_data["processing_time"]}ms processing, {st.session_state.modified_data["accuracy"]}% accuracy</div>'
    
    log_content += '</div>'
    
    st.markdown(log_content, unsafe_allow_html=True)