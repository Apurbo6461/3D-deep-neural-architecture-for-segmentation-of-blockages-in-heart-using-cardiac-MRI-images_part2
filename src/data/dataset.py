import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from scipy.ndimage import zoom

class MedicalDataset(Dataset):
    def __init__(self, root_dir, phase='train', target_shape=(128, 128, 64)):
        """
        Args:
            root_dir (str): Path to the dataset directory (e.g., .../training)
            phase (str): 'train' or 'val'. Currently uses correct naming convention from exploration.
            target_shape (tuple): Desired output shape (H, W, D).
        """
        self.root_dir = root_dir
        self.phase = phase
        self.target_shape = target_shape
        self.samples = self._find_samples()

    def _find_samples(self):
        # Expected structure: root_dir/patientXXX/patientXXX_frameXX.nii
        # and ground truth: patientXXX_frameXX_gt.nii
        samples = []
        patient_dirs = sorted(glob.glob(os.path.join(self.root_dir, 'patient*')))
        
        for p_dir in patient_dirs:
            # Find all frame files that are NOT ground truth
            # Pattern: patientXXX_frameXX.nii (and ignore _gt.nii and _4d.nii)
            nii_files = sorted(glob.glob(os.path.join(p_dir, '*.nii.gz')))
            for f_path in nii_files:
                f_name = os.path.basename(f_path)
                if '_gt' in f_name or '_4d' in f_name:
                    continue
                
                # Construct corresponding GT path
                # e.g. patient001_frame01.nii -> patient001_frame01_gt.nii
                gt_name = f_name.replace('.nii.gz', '_gt.nii.gz')
                gt_path = os.path.join(p_dir, gt_name)
                
                if os.path.exists(gt_path):
                    samples.append((f_path, gt_path))
        
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt_path = self.samples[idx]
        
        # Load NIfTI files
        img_nii = nib.load(img_path)
        gt_nii = nib.load(gt_path)
        
        # Get data as numpy arrays (float32 for image, uint8/long for mask)
        img_data = img_nii.get_fdata().astype(np.float32)
        gt_data = gt_nii.get_fdata().astype(np.float32) # Will convert to long later
        
        # Preprocessing
        img_data = self._normalize(img_data)
        
        # Resize to target shape
        img_data = self._resize_volume(img_data, self.target_shape, order=1) # Linear
        gt_data = self._resize_volume(gt_data, self.target_shape, order=0)   # Nearest
        
        # Add channel dimension: (D, H, W) -> (C, D, H, W) or (C, H, W, D)
        # PyTorch 3D Conv expects (N, C, D, H, W) usually, or (N, C, H, W, D).
        # Let's standardize to (C, H, W, D) for now, matching target_shape order.
        # Actually standard 3D Grid is often (D, H, W). 
        # But target_shape passed as (128, 128, 64) likely implies (H, W, D).
        # Let's expand dim 0.
        img_tensor = torch.from_numpy(img_data).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_data).unsqueeze(0)
        
        return img_tensor, gt_tensor

    def _normalize(self, volume):
        """
        Normalize volume intensities to [0, 1] based on percentiles.
        """
        min_val = np.percentile(volume, 1)
        max_val = np.percentile(volume, 99)
        volume = np.clip(volume, min_val, max_val)
        
        if max_val - min_val > 0:
            volume = (volume - min_val) / (max_val - min_val)
        return volume

    def _resize_volume(self, volume, target_shape, order=1):
        """
        Resize volume to target shape using scipy.ndimage.zoom.
        """
        current_shape = volume.shape
        zoom_factors = [t / c for t, c in zip(target_shape, current_shape)]
        resize_vol = zoom(volume, zoom_factors, order=order)
        return resize_vol
