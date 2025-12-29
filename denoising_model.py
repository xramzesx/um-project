import torch
import torch.nn as nn
import torchaudio
import os
import glob
import random
import soundfile as sf
import math
from torch.utils.data import Dataset

class SpectrogramDataset(Dataset):
    def __init__(self, clean_dir, noise_dir=None, n_fft=512, hop_length=256, sample_rate=16000):
        self.clean_dir = clean_dir
        self.noise_dir = noise_dir 
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        
        self.clean_files = sorted(glob.glob(os.path.join(clean_dir, '*.wav')))
        
        if self.noise_dir:
            self.noise_files = sorted(glob.glob(os.path.join(noise_dir, '*.wav')))
            if not self.noise_files:
                print(f"Warning: No noise files found in {noise_dir}")
        else:
            self.noise_files = []
            
    def __len__(self):
        return len(self.clean_files)

    def _load_and_process(self, path, fixed_length):
        data, sr = sf.read(path)
        waveform = torch.from_numpy(data).float()
        
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=-1)
            
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        # Pad or Crop
        if len(waveform) < fixed_length:
            pad_amount = fixed_length - len(waveform)
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        else:
            # Random crop for training variety could be good, but simple crop is safer for now
            # Let's do random start if it's much longer
            if len(waveform) > fixed_length:
                start = random.randint(0, len(waveform) - fixed_length)
                waveform = waveform[start:start+fixed_length]
            else:
                waveform = waveform[:fixed_length]
                
        return waveform

    def __getitem__(self, idx):
        clean_path = self.clean_files[idx]
        fixed_length = self.sample_rate * 2  # 2 second clips
        
        clean_waveform = self._load_and_process(clean_path, fixed_length)
        
        # Dynamic Mixing
        if self.noise_files:
            noise_path = random.choice(self.noise_files)
            noise_waveform = self._load_and_process(noise_path, fixed_length)
            
            clean_rms = torch.sqrt(torch.mean(clean_waveform ** 2))
            noise_rms = torch.sqrt(torch.mean(noise_waveform ** 2))
            
            # Avoid division by zero
            if clean_rms == 0: clean_rms = 1e-6
            if noise_rms == 0: noise_rms = 1e-6

            # Random SNR between 0 and 30 dB
            snr_db = random.uniform(0, 30)
            snr_linear = 10 ** (snr_db / 20)
            
            # Target noise RMS = Clean RMS / SNR
            # We scale noise
            target_noise_rms = clean_rms / snr_linear
            scale_factor = target_noise_rms / noise_rms
            
            scaled_noise = noise_waveform * scale_factor
            noisy_waveform = clean_waveform + scaled_noise
            
            # Optional: Normalize to avoid clipping? 
            # Simple max norm:
            max_val = torch.max(torch.abs(noisy_waveform))
            if max_val > 1.0:
                noisy_waveform = noisy_waveform / max_val
                clean_waveform = clean_waveform / max_val
                
        else:
            # Fallback for testing or missing noise (should not happen in this prompt context)
            noisy_waveform = clean_waveform.clone()
        
        # STFT
        window = torch.hann_window(self.n_fft)
        noisy_stft = torch.stft(noisy_waveform, n_fft=self.n_fft, hop_length=self.hop_length, 
                                return_complex=True, normalized=False, window=window)
        clean_stft = torch.stft(clean_waveform, n_fft=self.n_fft, hop_length=self.hop_length, 
                                return_complex=True, normalized=False, window=window)
        
        noisy_mag = torch.abs(noisy_stft)
        clean_mag = torch.abs(clean_stft)
        
        # Add channel dimension
        noisy_mag = noisy_mag.unsqueeze(0)
        clean_mag = clean_mag.unsqueeze(0)
        
        return noisy_mag, clean_mag


class SpectrogramUNet(nn.Module):
    def __init__(self):
        super(SpectrogramUNet, self).__init__()
        
        self.enc1 = self._conv_block(1, 16)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self._conv_block(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = self._conv_block(32, 64)
        self.pool3 = nn.MaxPool2d(2)
        
        self.bottleneck = self._conv_block(64, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(128, 64)
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(64, 32)
        
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(32, 16)
        
        self.out = nn.Conv2d(16, 1, kernel_size=1)
        self.relu = nn.ReLU()
        
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        b = self.bottleneck(p3)
        
        u3 = self.up3(b)
        if u3.shape != e3.shape:
            u3 = nn.functional.interpolate(u3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(d3)
        
        u2 = self.up2(d3)
        if u2.shape != e2.shape:
            u2 = nn.functional.interpolate(u2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(d2)
        
        u1 = self.up1(d2)
        if u1.shape != e1.shape:
            u1 = nn.functional.interpolate(u1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(d1)

        mask = torch.sigmoid(self.out(d1))
        out = mask * x 

        return out
