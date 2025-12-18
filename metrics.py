import torch

def calculate_snr(clean, estimated):
    """
    Calculate Signal-to-Noise Ratio (SNR).
    
    Args:
        clean (torch.Tensor): Clean signal waveform (1D).
        estimated (torch.Tensor): Estimated/Denoised signal waveform (1D).
        
    Returns:
        float: SNR in dB.
    """
    # Ensure they are the same length
    min_len = min(len(clean), len(estimated))
    clean = clean[:min_len]
    estimated = estimated[:min_len]
    
    noise = clean - estimated
    
    signal_power = torch.mean(clean ** 2)
    noise_power = torch.mean(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
        
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()
