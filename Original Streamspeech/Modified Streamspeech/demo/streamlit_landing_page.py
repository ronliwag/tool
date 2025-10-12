"""
StreamSpeech Landing Page - Streamlit Version
Converted from LandingPage.tsx
"""

import streamlit as st
import time

def show_landing_page():
    """Display the StreamSpeech landing page"""
    
    # Custom CSS for the landing page
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        padding-left: 0rem;
        padding-right: 0rem;
    }
    
    .landing-container {
        min-height: 100vh;
        background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 50%, #ddeeff 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .background-orb {
        position: absolute;
        border-radius: 50%;
        filter: blur(60px);
        opacity: 0.4;
        animation: pulse 4s ease-in-out infinite;
    }
    
    .orb-1 {
        width: 384px;
        height: 384px;
        background: linear-gradient(135deg, #87ceeb, #87ceeb);
        top: 5rem;
        right: 8rem;
        animation-delay: 0s;
    }
    
    .orb-2 {
        width: 256px;
        height: 256px;
        background: linear-gradient(135deg, #93c5fd, #60a5fa);
        top: 2.5rem;
        right: 24rem;
        animation-delay: 1s;
    }
    
    .orb-3 {
        width: 320px;
        height: 320px;
        background: linear-gradient(135deg, #cffafe, #a5f3fc);
        top: 12rem;
        right: 12rem;
        animation-delay: 2s;
    }
    
    .orb-4 {
        width: 224px;
        height: 224px;
        background: linear-gradient(135deg, #c7d2fe, #93c5fd);
        top: 20rem;
        right: 24rem;
        animation-delay: 2.5s;
    }
    
    .orb-5 {
        width: 288px;
        height: 288px;
        background: linear-gradient(135deg, #e0f2fe, #bae6fd);
        bottom: 8rem;
        right: 16rem;
        animation-delay: 1.5s;
    }
    
    .orb-6 {
        width: 192px;
        height: 192px;
        background: linear-gradient(135deg, #dbeafe, #bfdbfe);
        top: 4rem;
        right: 4rem;
        animation-delay: 3s;
    }
    
    .orb-7 {
        width: 160px;
        height: 160px;
        background: linear-gradient(135deg, #a5f3fc, #87ceeb);
        bottom: 12rem;
        right: 8rem;
        animation-delay: 2.8s;
    }
    
    .orb-8 {
        width: 256px;
        height: 256px;
        background: linear-gradient(135deg, #bfdbfe, #c7d2fe);
        top: 33%;
        left: 8rem;
        animation-delay: 3.5s;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.4; transform: scale(1); }
        50% { opacity: 0.6; transform: scale(1.05); }
    }
    
    .main-content {
        max-width: 72rem;
        width: 100%;
        text-align: center;
        position: relative;
        z-index: 10;
        padding: 2rem;
    }
    
    .logo-section {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        margin-bottom: 3rem;
    }
    
    .logo-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
    }
    
    .dot-1 { background-color: #0284c7; }
    .dot-2 { background-color: #2563eb; }
    .dot-3 { background-color: #4f46e5; }
    
    .logo-text {
        color: #111827;
        font-size: 1.25rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    
    .main-heading {
        color: #111827;
        font-weight: bold;
        font-size: 2.5rem;
        line-height: 1.2;
        margin-bottom: 2rem;
    }
    
    .description {
        color: #374151;
        font-size: 1.125rem;
        line-height: 1.75;
        margin-bottom: 2rem;
        max-width: 48rem;
        margin-left: auto;
        margin-right: auto;
    }
    
    .cta-button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 1rem 2rem;
        font-size: 1.125rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.3);
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .cta-button:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px 0 rgba(37, 99, 235, 0.4);
    }
    
    .copyright {
        margin-top: 4rem;
        color: #6b7280;
        font-size: 0.75rem;
    }
    
    .arrow-icon {
        width: 20px;
        height: 20px;
        transition: transform 0.3s ease;
    }
    
    .cta-button:hover .arrow-icon {
        transform: translateX(4px);
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp > header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
    
    # Landing page HTML structure
    st.markdown("""
    <div class="landing-container">
        <!-- Background Orbs -->
        <div class="background-orb orb-1"></div>
        <div class="background-orb orb-2"></div>
        <div class="background-orb orb-3"></div>
        <div class="background-orb orb-4"></div>
        <div class="background-orb orb-5"></div>
        <div class="background-orb orb-6"></div>
        <div class="background-orb orb-7"></div>
        <div class="background-orb orb-8"></div>
        
        <!-- Main Content -->
        <div class="main-content">
            <!-- Logo Section -->
            <div class="logo-section">
                <div class="logo-dot dot-1"></div>
                <div class="logo-dot dot-2"></div>
                <div class="logo-dot dot-3"></div>
                <div class="logo-text">StreamSpeech</div>
            </div>
            
            <!-- Main Heading -->
            <h1 class="main-heading">
                A MODIFIED HIFI-GAN VOCODER USING ODCONV AND GRC FOR EXPRESSIVE VOICE CLONING IN STREAMSPEECH'S REAL-TIME TRANSLATION
            </h1>
            
            <!-- Description -->
            <p class="description">
                A simultaneous translation that finally preserves your expressive voice and unique identity by leveraging an enhanced HiFi-GAN vocoder architecture for seamless voice cloning.
            </p>
            
            <!-- CTA Button -->
            <button class="cta-button" onclick="window.parent.postMessage('start_app', '*')">
                Try to test it out
                <svg class="arrow-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 8l4 4m0 0l-4 4m4-4H3"></path>
                </svg>
            </button>
            
            <!-- Copyright -->
            <div class="copyright">
                <p>Copyright 2025</p>
            </div>
        </div>
    </div>
    
    <script>
        // Listen for button clicks and communicate with Streamlit
        document.addEventListener('click', function(e) {
            if (e.target.classList.contains('cta-button') || e.target.closest('.cta-button')) {
                // Use Streamlit's session state to trigger navigation
                const button = document.createElement('button');
                button.style.display = 'none';
                button.onclick = function() {
                    // This will be handled by the parent Streamlit app
                    window.parent.postMessage('start_app', '*');
                };
                document.body.appendChild(button);
                button.click();
            }
        });
    </script>
    """, unsafe_allow_html=True)

def create_landing_page_component():
    """Create a reusable landing page component"""
    show_landing_page()

if __name__ == "__main__":
    # If run directly, show the landing page
    st.set_page_config(
        page_title="StreamSpeech - Landing Page",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    create_landing_page_component()
