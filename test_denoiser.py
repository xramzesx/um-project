import torch
import torchaudio
import soundfile as sf
import os
import glob
import random
from denoising_model import SpectrogramUNet

def test():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    NOISY_DIR = os.path.join(BASE_DIR, 'NoisySpeech_training')
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
    
    audio_data, sr = sf.read(test_file)
    waveform = torch.from_numpy(audio_data).float()
    
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=-1)
        
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)
        sr = SAMPLE_RATE
    
    noisy_stft = torch.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, 
                            return_complex=True, normalized=False)
    
    noisy_mag = torch.abs(noisy_stft)
    noisy_phase = torch.angle(noisy_stft)
    
    input_mag = noisy_mag.unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        denoised_mag = model(input_mag)
        
    denoised_mag = denoised_mag.squeeze(0).squeeze(0).cpu()
    
    denoised_stft = denoised_mag * torch.exp(1j * noisy_phase)
    
    denoised_waveform = torch.istft(denoised_stft, n_fft=N_FFT, hop_length=HOP_LENGTH, 
                                     normalized=False, length=len(waveform))
    
    noisy_save_path = os.path.join(RESULTS_DIR, 'test_noisy.wav')
    denoised_save_path = os.path.join(RESULTS_DIR, 'test_denoised.wav')
    
    sf.write(noisy_save_path, waveform.numpy(), sr)
    sf.write(denoised_save_path, denoised_waveform.numpy(), sr)
    
    print(f"Saved original noisy audio to: {noisy_save_path}")
    print(f"Saved denoised audio to: {denoised_save_path}")

if __name__ == "__main__":
    test()
