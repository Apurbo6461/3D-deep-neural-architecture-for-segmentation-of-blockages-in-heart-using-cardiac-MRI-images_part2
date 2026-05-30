import os
import torch
from torch.utils.data import DataLoader
from data.dataset import MedicalDataset
from models.vnet import VNet3D
from models.unet3d_standard import UNet3D
from utils import calculate_dice_score, calculate_iou

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = MedicalDataset(root_dir="E:/Thesis Dataset 2/testing", target_shape=(128, 128, 64))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    vnet = VNet3D(in_channels=1, out_channels=1, base_filters=16).to(device)
    vnet.load_state_dict(torch.load("E:/Thesis Dataset 2/src/best_v-net.pth", map_location=device))
    vnet.eval()

    resatt = UNet3D(in_channels=1, out_channels=1, base_filters=32).to(device)
    resatt.load_state_dict(torch.load("E:/Thesis Dataset 2/src/best_3d-u-net.pth", map_location=device))
    resatt.eval()

    # test thresholds
    thresholds = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999, 0.9999]
    vnet_res = {t: [] for t in thresholds}
    resatt_res = {t: [] for t in thresholds}

    with torch.no_grad():
        for i, (imgs, gts) in enumerate(loader):
            if i >= 5:
                break
            imgs = imgs.to(device)
            gts_binary = (gts > 0).float().to(device)

            v_preds = torch.sigmoid(vnet(imgs))
            r_preds = torch.sigmoid(resatt(imgs))

            for t in thresholds:
                v_bin = (v_preds > t).float()
                r_bin = (r_preds > t).float()
                vnet_res[t].append(calculate_dice_score(v_bin, gts_binary))
                resatt_res[t].append(calculate_dice_score(r_bin, gts_binary))
                
    for t in thresholds:
        print(f"Threshold {t}:")
        print(f"  V-Net Dice: {sum(vnet_res[t])/5:.4f}")
        print(f"  ResAtt Dice: {sum(resatt_res[t])/5:.4f}")

if __name__ == "__main__":
    test()
