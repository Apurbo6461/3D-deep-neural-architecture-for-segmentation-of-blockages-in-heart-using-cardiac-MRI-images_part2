import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import time
import pandas as pd

# Dataset and Loss imports
from data.dataset import MedicalDataset
from utils import DiceLoss

# Model imports based on your models directory
from models.res_att_unet3d import ResAttUNet3D
from models.unet3d_standard import UNet3D 
from models.vnet import VNet3D

def train_model(model_name, model_obj, train_loader, val_loader, device, epochs=20):
    print(f"\n{'='*30}\nStarting Training: {model_name}\n{'='*30}")
    
    # Create directory for model results
    os.makedirs(f"results/{model_name}", exist_ok=True)
    
    criterion = DiceLoss()
    optimizer = optim.Adam(model_obj.parameters(), lr=1e-4)
    
    best_val_dice = 0.0
    history = {
        'epoch': [], 'train_loss': [], 'val_dice': [], 
        'val_iou': [], 'val_sens': [], 'val_spec': [], 'epoch_time': []
    }

    for epoch in range(epochs):
        start_time = time.time()
        model_obj.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"{model_name} - Epoch {epoch+1}/{epochs}")
        for imgs, gts in pbar:
            imgs = imgs.to(device)
            # Heart structure labels: RV, Myo, LV[cite: 1]
            gts_binary = (gts > 0).float().to(device)

            optimizer.zero_grad()
            preds = model_obj(imgs)
            loss = criterion(preds, gts_binary)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        # Validation Phase
        model_obj.eval()
        total_dice, total_iou, total_sens, total_spec = 0, 0, 0, 0
        
        with torch.no_grad():
            for imgs, gts in val_loader:
                imgs = imgs.to(device)
                gts = (gts > 0).float().to(device)
                
                preds = torch.sigmoid(model_obj(imgs))
                preds = (preds > 0.5).float()
                
                # Metric Calculations
                tp = (preds * gts).sum()
                fp = (preds * (1 - gts)).sum()
                fn = ((1 - preds) * gts).sum()
                tn = ((1 - preds) * (1 - gts)).sum()
                
                total_dice += (2 * tp) / (2 * tp + fp + fn + 1e-7)
                total_iou += tp / (tp + fp + fn + 1e-7)
                total_sens += tp / (tp + fn + 1e-7)
                total_spec += tn / (tn + fp + 1e-7)
        
        n_val = len(val_loader)
        avg_dice = (total_dice / n_val).item()
        epoch_duration = time.time() - start_time
        
        print(f"Time: {epoch_duration:.2f}s | Dice: {avg_dice:.4f} | IoU: {(total_iou/n_val):.4f}")
        
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(epoch_loss / len(train_loader))
        history['val_dice'].append(avg_dice)
        history['val_iou'].append((total_iou / n_val).item())
        history['val_sens'].append((total_sens / n_val).item())
        history['val_spec'].append((total_spec / n_val).item())
        history['epoch_time'].append(epoch_duration)

        if avg_dice > best_val_dice:
            best_val_dice = avg_dice
            torch.save(model_obj.state_dict(), f"results/{model_name}/best_model.pth")

    pd.DataFrame(history).to_csv(f"results/{model_name}/training_log.csv", index=False)

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ROOT_DIR = r"E:\Thesis Dataset 2\training"
    TARGET_SHAPE = (128, 128, 64)
    
    full_dataset = MedicalDataset(root_dir=ROOT_DIR, target_shape=TARGET_SHAPE)
    
    if len(full_dataset) == 0:
        print("Dataset is empty. Check your folder structure and file extensions.")
        return

    n_val = int(len(full_dataset) * 0.2)
    train_set, val_set = random_split(full_dataset, [len(full_dataset)-n_val, n_val])
    
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # Standardizing arguments to out_channels
    models_to_run = {
        "ResAttUNet3D": ResAttUNet3D(in_channels=1, out_channels=1, base_filters=16).to(DEVICE),
        "UNet3D": UNet3D(in_channels=1, out_channels=1).to(DEVICE),
        "VNet3D": VNet3D(out_channels=1).to(DEVICE) # Changed from num_classes to out_channels
    }

    for name, model in models_to_run.items():
        train_model(name, model, train_loader, val_loader, DEVICE, epochs=20)

if __name__ == "__main__":
    main()