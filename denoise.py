import torch
import torchaudio
import soundfile as sf
import os
import glob
import random
from denoising_model import SpectrogramUNet
from metrics import calculate_snr

def test():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    NOISY_DIR = os.path.join(BASE_DIR, 'NoisySpeech_training')
    CLEAN_DIR = os.path.join(BASE_DIR, 'CleanSpeech_training')
    MODEL_PATH = os.path.join(BASE_DIR, 'denoiser.pth')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    
    N_FFT = 512
    HOP_LENGTH = 256
    SAMPLE_RATE = 16000
    
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    model = SpectrogramUNet().to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded.")
    
    noisy_files = glob.glob(os.path.join(NOISY_DIR, '*.wav'))
    if not noisy_files:
        print("No noisy files found.")
        return
        
    test_file = random.choice(noisy_files)
    print(f"Testing on file: {test_file}")
    
    # Try to find corresponding clean file
    # Convention seems to be: noisy10_SNRdb_0.0_clnsp10.wav -> clnsp10.wav
    # Split by underscore and take the last part
    filename = os.path.basename(test_file)
    parts = filename.split('_')
    clean_filename = parts[-1] if len(parts) > 1 else None
    
    clean_audio_path = None
    if clean_filename:
        potential_path = os.path.join(CLEAN_DIR, clean_filename)
        if os.path.exists(potential_path):
            clean_audio_path = potential_path
            print(f"Found corresponding clean file: {clean_audio_path}")
        else:
            print(f"Warning: Clean file not found at {potential_path}")
    else:
        print("Warning: Could not deduce clean filename from noisy filename.")

    audio_data, sr = sf.read(test_file)
    waveform = torch.from_numpy(audio_data).float()
    
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=-1)
        
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)
        sr = SAMPLE_RATE

    # Load clean audio if available for metric comparison
    clean_waveform = None
    if clean_audio_path:
        clean_data, clean_sr = sf.read(clean_audio_path)
        clean_waveform = torch.from_numpy(clean_data).float()
        if clean_waveform.dim() > 1:
            clean_waveform = clean_waveform.mean(dim=-1)
        if clean_sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(clean_sr, SAMPLE_RATE)
            clean_waveform = resampler(clean_waveform)
    
    # Use Hann window to match training logic (likely) or just to be correct signal processing wise
    window = torch.hann_window(N_FFT).to(DEVICE)
    
    # Move waveform to device for STFT
    waveform = waveform.to(DEVICE)
    
    noisy_stft = torch.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, 
                            return_complex=True, normalized=False, window=window)
    
    noisy_mag = torch.abs(noisy_stft)
    noisy_phase = torch.angle(noisy_stft)
    
    input_mag = noisy_mag.unsqueeze(0).unsqueeze(0) # Already on DEVICE
    
    with torch.no_grad():
        denoised_mag = model(input_mag)
        
    denoised_mag = denoised_mag.squeeze(0).squeeze(0)
    
    denoised_stft = denoised_mag * torch.exp(1j * noisy_phase)
    
    denoised_waveform = torch.istft(denoised_stft, n_fft=N_FFT, hop_length=HOP_LENGTH, 
                                     normalized=False, length=len(waveform), window=window)
    
    # Move back to CPU for saving
    waveform = waveform.cpu()
    denoised_waveform = denoised_waveform.cpu()
    
    noisy_save_path = os.path.join(RESULTS_DIR, 'test_noisy.wav')
    denoised_save_path = os.path.join(RESULTS_DIR, 'test_denoised.wav')
    
    sf.write(noisy_save_path, waveform.numpy(), sr)
    sf.write(denoised_save_path, denoised_waveform.numpy(), sr)
    
    print(f"Saved original noisy audio to: {noisy_save_path}")
    print(f"Saved denoised audio to: {denoised_save_path}")

    # Calculate Metrics if clean audio is available
    if clean_waveform is not None:
        input_snr = calculate_snr(clean_waveform, waveform)
        output_snr = calculate_snr(clean_waveform, denoised_waveform)
        
        print("\n--- Metrics ---")
        print(f"Input SNR:  {input_snr:.2f} dB")
        print(f"Output SNR: {output_snr:.2f} dB")
        print(f"Improvement: {output_snr - input_snr:.2f} dB")

if __name__ == "__main__":
    test()
