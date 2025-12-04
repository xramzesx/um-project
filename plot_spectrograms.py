import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os
import glob

def plot_spectrograms():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    CLEAN_DIR = os.path.join(BASE_DIR, 'CleanSpeech_training')
    
    noisy_path = os.path.join(RESULTS_DIR, 'test_noisy.wav')
    denoised_path = os.path.join(RESULTS_DIR, 'test_denoised.wav')
    
    if not os.path.exists(noisy_path) or not os.path.exists(denoised_path):
        print("Please run test_denoiser.py first to generate audio files.")
        return
    
    noisy, sr = librosa.load(noisy_path, sr=None)
    denoised, _ = librosa.load(denoised_path, sr=None)
    
    _ = os.path.basename(noisy_path)

    clean_files = sorted(glob.glob(os.path.join(CLEAN_DIR, '*.wav')))
    if clean_files:
        clean_path = clean_files[0]
        clean, _ = librosa.load(clean_path, sr=None)
        
        min_len = min(len(noisy), len(denoised), len(clean))
        noisy = noisy[:min_len]
        denoised = denoised[:min_len]
        clean = clean[:min_len]
    else:
        clean = None
    
    noisy_spec = librosa.stft(noisy)
    denoised_spec = librosa.stft(denoised)
    
    noisy_db = librosa.amplitude_to_db(np.abs(noisy_spec), ref=np.max)
    denoised_db = librosa.amplitude_to_db(np.abs(denoised_spec), ref=np.max)
    
    if clean is not None:
        clean_spec = librosa.stft(clean)
        clean_db = librosa.amplitude_to_db(np.abs(clean_spec), ref=np.max)
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        
        img1 = librosa.display.specshow(noisy_db, sr=sr, x_axis='time', y_axis='hz', 
                                         ax=axes[0], cmap='viridis')
        axes[0].set_title('Noisy Audio Spectrogram')
        axes[0].set_ylabel('Frequency (Hz)')
        fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')
        
        img2 = librosa.display.specshow(denoised_db, sr=sr, x_axis='time', y_axis='hz', 
                                         ax=axes[1], cmap='viridis')
        axes[1].set_title('Denoised Audio Spectrogram (Model Output)')
        axes[1].set_ylabel('Frequency (Hz)')
        fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')
        
        img3 = librosa.display.specshow(clean_db, sr=sr, x_axis='time', y_axis='hz', 
                                         ax=axes[2], cmap='viridis')
        axes[2].set_title('Clean Audio Spectrogram (Ground Truth)')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Frequency (Hz)')
        fig.colorbar(img3, ax=axes[2], format='%+2.0f dB')
    else:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        img1 = librosa.display.specshow(noisy_db, sr=sr, x_axis='time', y_axis='hz', 
                                         ax=axes[0], cmap='viridis')
        axes[0].set_title('Noisy Audio Spectrogram')
        axes[0].set_ylabel('Frequency (Hz)')
        fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')
        
        img2 = librosa.display.specshow(denoised_db, sr=sr, x_axis='time', y_axis='hz', 
                                         ax=axes[1], cmap='viridis')
        axes[1].set_title('Denoised Audio Spectrogram')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Frequency (Hz)')
        fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')
    
    plt.tight_layout()
    
    output_path = os.path.join(RESULTS_DIR, 'spectrograms.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Spectrograms saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    plot_spectrograms()
