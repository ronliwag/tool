"""
Enhanced Configuration System
===========================

Runtime configuration for switching between full and enhanced HiFi-GAN
for thesis enhanced demonstration.
"""

import os
import json
from pathlib import Path

class EnhancedConfig:
    """Configuration manager for enhanced demonstration"""
    
    def __init__(self, config_path=None):
        self.config_path = config_path or "enhanced_config.json"
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from file or create default"""
        default_config = {
            "enhanced_mode": "enhanced",  # "full" or "enhanced"
            "use_original_baseline": True,
            "save_enhanced_outputs": True,
            "output_directory": "enhanced_outputs",
            "enable_conditioning": False,  # Disable for stability
            "enable_odconv": False,        # Disable for stability
            "enable_grc_lora": False,      # Disable for stability
            "enable_film": False,          # Disable for stability
            "force_english_audio": True,   # Ensure English output
            "max_audio_amplitude": 0.95,   # Prevent clipping
            "sample_rate": 22050,
            "mel_channels": 80
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                # Merge with defaults
                default_config.update(loaded_config)
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")
        
        return default_config
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"Configuration saved to: {self.config_path}")
        except Exception as e:
            print(f"Warning: Could not save config file: {e}")
    
    def set_enhanced_mode(self, mode):
        """Set enhanced mode: 'full' or 'enhanced'"""
        if mode in ["full", "enhanced"]:
            self.config["enhanced_mode"] = mode
            if mode == "enhanced":
                # Disable unstable features for enhanced mode
                self.config["enable_conditioning"] = False
                self.config["enable_odconv"] = False
                self.config["enable_grc_lora"] = False
                self.config["enable_film"] = False
            print(f"Enhanced mode set to: {mode}")
        else:
            print(f"Invalid mode: {mode}. Use 'full' or 'enhanced'")
    
    def is_enhanced_mode(self):
        """Check if in enhanced mode"""
        return self.config["enhanced_mode"] == "enhanced"
    
    def should_use_conditioning(self):
        """Check if conditioning should be enabled"""
        return self.config["enable_conditioning"] and not self.is_enhanced_mode()
    
    def should_save_outputs(self):
        """Check if outputs should be saved"""
        return self.config["save_enhanced_outputs"]
    
    def get_output_directory(self):
        """Get output directory path"""
        output_dir = self.config["output_directory"]
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def get_max_amplitude(self):
        """Get maximum audio amplitude"""
        return self.config["max_audio_amplitude"]
    
    def get_sample_rate(self):
        """Get sample rate"""
        return self.config["sample_rate"]
    
    def get_mel_channels(self):
        """Get number of mel channels"""
        return self.config["mel_channels"]

# Global configuration instance
enhanced_config = EnhancedConfig()

def set_enhanced_mode(mode):
    """Set enhanced mode globally"""
    enhanced_config.set_enhanced_mode(mode)

def is_enhanced_mode():
    """Check if in enhanced mode globally"""
    return enhanced_config.is_enhanced_mode()

def should_use_conditioning():
    """Check if conditioning should be used globally"""
    return enhanced_config.should_use_conditioning()

def get_enhanced_output_dir():
    """Get enhanced output directory"""
    return enhanced_config.get_output_directory()

def save_enhanced_audio(audio_data, filename, sample_rate=22050):
    """Save audio for enhanced demonstration"""
    if enhanced_config.should_save_outputs():
        output_dir = get_enhanced_output_dir()
        filepath = os.path.join(output_dir, filename)
        
        try:
            import soundfile as sf
            sf.write(filepath, audio_data, sample_rate, subtype='PCM_16')
            print(f"Enhanced audio saved: {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving enhanced audio: {e}")
            return None
    return None

def save_enhanced_results(results, filename="enhanced_results.json"):
    """Save enhanced results"""
    if enhanced_config.should_save_outputs():
        output_dir = get_enhanced_output_dir()
        filepath = os.path.join(output_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Enhanced results saved: {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving enhanced results: {e}")
            return None
    return None

if __name__ == "__main__":
    # Test configuration system
    print("Testing Enhanced Configuration System...")
    
    # Test default configuration
    config = EnhancedConfig()
    print(f"Default mode: {config.config['enhanced_mode']}")
    print(f"Enhanced mode: {config.is_enhanced_mode()}")
    print(f"Use conditioning: {config.should_use_conditioning()}")
    print(f"Output directory: {config.get_output_directory()}")
    
    # Test mode switching
    config.set_enhanced_mode("full")
    print(f"After switching to full: conditioning={config.should_use_conditioning()}")
    
    config.set_enhanced_mode("enhanced")
    print(f"After switching to enhanced: conditioning={config.should_use_conditioning()}")
    
    # Test saving
    config.save_config()
    
    print("✅ Enhanced configuration system test completed!")
