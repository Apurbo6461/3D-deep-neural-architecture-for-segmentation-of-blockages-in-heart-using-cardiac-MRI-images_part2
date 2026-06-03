import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import pandas as pd
import json

from data.dataset import MedicalDataset
from models.unet3d_standard import UNet3D
from models.vnet import VNet3D
from models.res_att_unet3d import ResAttUNet3D
from utils import DiceLoss, calculate_dice_score, calculate_iou, calculate_accuracy
import matplotlib.pyplot as plt

def train_model(model, model_name, train_loader, val_loader, device, epochs=20, lr=1e-4):
    """Train a single model and return training history."""
    model = model.to(device)
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_dice = 0.0
    history = {
        'epoch': [],
        'train_loss': [],
        'val_dice': [],
        'val_iou': [],
        'val_accuracy': []
    }
    
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, gts in pbar:
            imgs = imgs.to(device)
            gts = gts.to(device)
            gts_binary = (gts > 0).float()
            optimizer.zero_grad()
            preds = model(imgs)
            
            # Applied sigmoid for binary output
            if preds.shape[1] == 1:
                loss = criterion(preds, gts_binary)
            else:
                loss = criterion(preds[:, 0:1], gts_binary)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        # Validation phase
        model.eval()
        val_dice = 0
        val_iou = 0
        val_acc = 0
        
        with torch.no_grad():
            for imgs, gts in val_loader:
                imgs = imgs.to(device)
                gts_binary = (gts > 0).float().to(device)
                
                preds = model(imgs)
                
                # Applied sigmoid and threshold
                if preds.shape[1] == 1:
                    preds_sigmoid = torch.sigmoid(preds)
                    preds_binary = (preds_sigmoid > 0.5).float()
                else:
                    preds_sigmoid = torch.softmax(preds, dim=1)
                    preds_binary = (preds_sigmoid[:, 0:1] < 0.5).float()
                
                val_dice += calculate_dice_score(preds_binary, gts_binary)
                val_iou += calculate_iou(preds_binary, gts_binary)
                val_acc += calculate_accuracy(preds_binary, gts_binary)
        
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)
        val_acc /= len(val_loader)
        
        avg_loss = epoch_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Dice={val_dice:.4f}, IoU={val_iou:.4f}, Acc={val_acc:.4f}")
        
        # Log history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)
        history['val_accuracy'].append(val_acc)
        
        scheduler.step(avg_loss)
        
        # Saving best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            model_path = f"best_{model_name.lower().replace(' ', '-')}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"✓ Saved best model (Dice: {best_val_dice:.4f}) to {model_path}")
    
    return history, model

def main():
    # Hyperparameters
    BATCH_SIZE = 1  # 3D models consume lots of VRAM
    LR = 1e-4
    EPOCHS = 20
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cpu" and torch.backends.mps.is_available():
        DEVICE = "mps"  
    
    ROOT_DIR = "E:/Thesis Dataset 2/training"
    TARGET_SHAPE = (128, 128, 64)
    NUM_WORKERS = 0
    
    print(f"Using device: {DEVICE}")
    print(f"Target shape: {TARGET_SHAPE}")
    
    # Loading dataset
    full_dataset = MedicalDataset(root_dir=ROOT_DIR, target_shape=TARGET_SHAPE)
    
    # Splitting Train/Val
    val_percent = 0.2
    n_val = int(len(full_dataset) * val_percent)
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(full_dataset, [n_train, n_val], 
                                     generator=torch.Generator().manual_seed(42))
    
    print(f"Training samples: {len(train_set)}")
    print(f"Validation samples: {len(val_set)}")
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Defining models to train
    models_to_train = {
        '3D-U-Net':        UNet3D(in_channels=1, out_channels=1),
        'V-Net':           VNet3D(in_channels=1, out_channels=1),
        'ResAtt-3D-U-Net': ResAttUNet3D(in_channels=1, out_channels=1, base_filters=16),
    }
    
    # Training all models
    all_histories = {}
    trained_models = {}
    
    for model_name, model in models_to_train.items():
        # Checking if model already exists
        model_path = f"best_{model_name.lower().replace(' ', '-')}.pth"
        if os.path.exists(model_path):
            print(f"⚠ Model {model_name} already exists at {model_path}")
            print(f"  Skipping training. Delete the file if you want to retrain.")
            continue
            
        history, trained_model = train_model(
            model, model_name, train_loader, val_loader, DEVICE, epochs=EPOCHS, lr=LR
        )
        all_histories[model_name] = history
        trained_models[model_name] = trained_model
    
    # Saving training histories
    for model_name, history in all_histories.items():
        df = pd.DataFrame(history)
        filename = f"training_log_{model_name.lower().replace(' ', '_').replace('-', '_')}.csv"
        df.to_csv(filename, index=False)
        print(f"Saved training log to {filename}")
    
    # Plot comparison
    plot_training_comparison(all_histories)
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)

def plot_training_comparison(all_histories):
    """Plot training curves for all models."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    metrics = [
        ('train_loss', 'Training Loss', axes[0, 0]),
        ('val_dice', 'Validation Dice Score', axes[0, 1]),
        ('val_iou', 'Validation IoU', axes[1, 0]),
        ('val_accuracy', 'Validation Accuracy', axes[1, 1])
    ]
    
    for metric_key, title, ax in metrics:
        for model_name, history in all_histories.items():
            if metric_key in history:
                ax.plot(history['epoch'], history[metric_key], 
                       label=model_name, marker='o', markersize=3)
        
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title.replace('Validation ', '').replace('Training ', ''))
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("training_comparison.png", dpi=300, bbox_inches='tight')
    print("Saved training comparison plot to training_comparison.png")

if __name__ == "__main__":
    main()
