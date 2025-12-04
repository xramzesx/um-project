import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from denoising_model import SpectrogramDataset, SpectrogramUNet
import os

def train():
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001
    EPOCHS = 100
    VALIDATION_SPLIT = 0.2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    NOISY_DIR = os.path.join(BASE_DIR, 'NoisySpeech_training')
    CLEAN_DIR = os.path.join(BASE_DIR, 'CleanSpeech_training')
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'denoiser.pth')
    
    print(f"Looking for data in:\nNoisy: {NOISY_DIR}\nClean: {CLEAN_DIR}")
    
    full_dataset = SpectrogramDataset(NOISY_DIR, CLEAN_DIR)
    if len(full_dataset) == 0:
        print("Error: No paired data found. Please check directory structure and file names.")
        return
        
    print(f"Found {len(full_dataset)} pairs of audio files.")
    
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Split: {train_size} training, {val_size} validation")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = SpectrogramUNet().to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (noisy, clean) in enumerate(train_loader):
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(noisy)
            loss = criterion(outputs, clean)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        avg_train_loss = running_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
                outputs = model(noisy)
                loss = criterion(outputs, clean)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  → Best model saved (Val Loss: {avg_val_loss:.4f})")
    
    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
