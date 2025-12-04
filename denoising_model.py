import torch
import torch.nn as nn
import torchaudio
import os
import glob
import soundfile as sf
from torch.utils.data import Dataset

class SpectrogramDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, n_fft=512, hop_length=256, sample_rate=16000):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        
        self.noisy_files = sorted(glob.glob(os.path.join(noisy_dir, '*.wav')))
        self.clean_files = sorted(glob.glob(os.path.join(clean_dir, '*.wav')))
        
        self.clean_file_map = {os.path.basename(f): f for f in self.clean_files}
        
        self.data_pairs = []
        for noisy_path in self.noisy_files:
            noisy_filename = os.path.basename(noisy_path)
            parts = noisy_filename.split('_')
            if len(parts) > 1:
                potential_clean_name = parts[-1]
                if potential_clean_name in self.clean_file_map:
                    self.data_pairs.append((noisy_path, self.clean_file_map[potential_clean_name]))
            
    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        noisy_path, clean_path = self.data_pairs[idx]
        
        noisy_data, noisy_sr = sf.read(noisy_path)
        clean_data, clean_sr = sf.read(clean_path)
        
        noisy_waveform = torch.from_numpy(noisy_data).float()
        clean_waveform = torch.from_numpy(clean_data).float()
        
        if noisy_waveform.dim() > 1:
            noisy_waveform = noisy_waveform.mean(dim=-1)
        if clean_waveform.dim() > 1:
            clean_waveform = clean_waveform.mean(dim=-1)
            
        if noisy_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(noisy_sr, self.sample_rate)
            noisy_waveform = resampler(noisy_waveform)
        if clean_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(clean_sr, self.sample_rate)
            clean_waveform = resampler(clean_waveform)
            
        min_len = min(len(noisy_waveform), len(clean_waveform))
        noisy_waveform = noisy_waveform[:min_len]
        clean_waveform = clean_waveform[:min_len]
        
        fixed_length = self.sample_rate * 2
        if len(noisy_waveform) < fixed_length:
            pad_amount = fixed_length - len(noisy_waveform)
            noisy_waveform = torch.nn.functional.pad(noisy_waveform, (0, pad_amount))
            clean_waveform = torch.nn.functional.pad(clean_waveform, (0, pad_amount))
        else:
            noisy_waveform = noisy_waveform[:fixed_length]
            clean_waveform = clean_waveform[:fixed_length]
        
        noisy_stft = torch.stft(noisy_waveform, n_fft=self.n_fft, hop_length=self.hop_length, 
                                return_complex=True, normalized=False, window=torch.hann_window(self.n_fft))
        clean_stft = torch.stft(clean_waveform, n_fft=self.n_fft, hop_length=self.hop_length, 
                                return_complex=True, normalized=False, window=torch.hann_window(self.n_fft))
        
        noisy_mag = torch.abs(noisy_stft)
        clean_mag = torch.abs(clean_stft)
        
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
        
        out = self.out(d1)
        out = self.relu(out) 

        return out
