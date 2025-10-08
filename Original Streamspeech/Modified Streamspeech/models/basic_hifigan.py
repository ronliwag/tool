"""
Basic HiFi-GAN Generator that matches the trained checkpoint
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicHiFiGAN(nn.Module):
    """
    Basic HiFi-GAN generator that matches the trained checkpoint architecture
    """
    def __init__(self, config):
        super(BasicHiFiGAN, self).__init__()
        
        self.mel_channels = config.get('mel_channels', 80)
        self.audio_channels = config.get('audio_channels', 1)
        self.upsample_rates = config.get('upsample_rates', [8, 8, 2, 2])
        self.upsample_kernel_sizes = config.get('upsample_kernel_sizes', [16, 16, 4, 4])
        self.upsample_initial_channel = config.get('upsample_initial_channel', 512)
        self.speaker_embed_dim = config.get('speaker_embed_dim', 192)
        self.emotion_embed_dim = config.get('emotion_embed_dim', 256)
        
        # Initial convolution
        self.initial_conv = nn.Conv1d(
            self.mel_channels, 
            self.upsample_initial_channel, 
            kernel_size=7, 
            padding=3
        )
        
        # Upsample layers
        self.upsample_layers = nn.ModuleList()
        in_channels = self.upsample_initial_channel
        
        for i, (upsample_rate, kernel_size) in enumerate(zip(self.upsample_rates, self.upsample_kernel_sizes)):
            out_channels = in_channels // 2
            self.upsample_layers.append(nn.ConvTranspose1d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                stride=upsample_rate, 
                padding=(kernel_size - upsample_rate) // 2
            ))
            in_channels = out_channels
        
        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(len(self.upsample_rates)):
            self.res_blocks.append(nn.ModuleList([
                nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1),
                nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
            ]))
        
        # FiLM layers for conditioning
        self.film_layers = nn.ModuleList()
        for i in range(len(self.upsample_rates)):
            self.film_layers.append(nn.ModuleList([
                nn.Linear(self.speaker_embed_dim + self.emotion_embed_dim, in_channels),
                nn.LeakyReLU(0.1),
                nn.Linear(in_channels, in_channels)
            ]))
        
        # Final convolution
        self.final_conv = nn.Conv1d(
            in_channels, 
            self.audio_channels, 
            kernel_size=7, 
            padding=3
        )
    
    def apply_film(self, x, speaker_embed, emotion_embed, layer_idx):
        """Apply FiLM conditioning"""
        if speaker_embed is None or emotion_embed is None:
            return x
        
        # Combine embeddings
        combined_embed = torch.cat([speaker_embed, emotion_embed], dim=-1)
        
        # Apply FiLM
        gamma = self.film_layers[layer_idx][0](combined_embed)
        beta = self.film_layers[layer_idx][2](self.film_layers[layer_idx][1](combined_embed))
        
        # Reshape for broadcasting
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        
        return gamma * x + beta
    
    def forward(self, mel, speaker_embed=None, emotion_embed=None):
        """
        Forward pass
        Args:
            mel: Mel spectrogram [B, mel_channels, T]
            speaker_embed: Speaker embedding [B, speaker_embed_dim]
            emotion_embed: Emotion embedding [B, emotion_embed_dim]
        """
        # Initial convolution
        x = self.initial_conv(mel)
        
        # Upsample layers with residual blocks
        for i, (upsample_layer, res_block) in enumerate(zip(self.upsample_layers, self.res_blocks)):
            # Upsample
            x = upsample_layer(x)
            
            # Apply FiLM conditioning
            x = self.apply_film(x, speaker_embed, emotion_embed, i)
            
            # Residual block
            residual = x
            x = res_block[0](x)
            x = res_block[1](x)
            x = res_block[2](x)
            x = x + residual
        
        # Final convolution
        audio = self.final_conv(x)
        
        return torch.tanh(audio)  # Output in [-1, 1] range

class BasicHiFiGANVocoder:
    """
    Wrapper class for BasicHiFiGAN to match ModifiedStreamSpeechVocoder interface
    """
    def __init__(self, config):
        self.config = config
        self.generator = BasicHiFiGAN(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator.to(self.device)
        
    def load_checkpoint(self, checkpoint_path):
        """Load trained checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.generator.load_state_dict(checkpoint)
            print(f"✅ Loaded trained BasicHiFiGAN from {checkpoint_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            return False
    
    def __call__(self, mel, speaker_embed=None, emotion_embed=None):
        """Generate audio from mel spectrogram"""
        with torch.no_grad():
            return self.generator(mel, speaker_embed, emotion_embed)
    
    def eval(self):
        """Set to evaluation mode"""
        self.generator.eval()








