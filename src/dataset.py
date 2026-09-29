import os
import torch
from torch.utils.data import Dataset
import numpy as np
import rasterio
from rasterio.enums import Resampling

class CropFusionDataset(Dataset):
    def __init__(self, sat_geotiff_path, drone_geotiff_path, patch_size=256, num_patches=200):
        """
        sat_geotiff_path: Path to coarse satellite GeoTIFF (e.g., Sentinel-2 10m)
        drone_geotiff_path: Path to fine drone GeoTIFF (e.g., Drone 0.1m)
        patch_size: Target resolution patch size (256x256)
        """
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches

        # Open spatial rasters
        with rasterio.open(sat_geotiff_path) as sat_src:
            # Read and normalize Satellite channels (e.g., RGB + NIR) to [0, 1]
            sat_data = sat_src.read().astype(np.float32)
            self.sat_img = sat_data / (np.percentile(sat_data, 99) + 1e-6)
            self.sat_img = np.clip(self.sat_img, 0.0, 1.0)

        with rasterio.open(drone_geotiff_path) as drone_src:
            drone_data = drone_src.read().astype(np.float32)
            self.drone_img = drone_data / (np.percentile(drone_data, 99) + 1e-6)
            self.drone_img = np.clip(self.drone_img, 0.0, 1.0)

        # Match spatial channels (Ensure equal channel count, e.g., 3 or 4)
        ch_count = min(self.sat_img.shape[0], self.drone_img.shape[0])
        self.sat_img = self.sat_img[:ch_count]
        self.drone_img = self.drone_img[:ch_count]

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        # Random spatial crop coordinates
        _, h_sat, w_sat = self.sat_img.shape
        _, h_dr, w_dr = self.drone_img.shape

        # Sample low-res patch
        lr_w, lr_h = self.patch_size // 4, self.patch_size // 4
        x_lr = np.random.randint(0, max(1, w_sat - lr_w))
        y_lr = np.random.randint(0, max(1, h_sat - lr_h))
        
        sat_patch = self.sat_img[:, y_lr : y_lr + lr_h, x_lr : x_lr + lr_w]

        # Calculate matching drone high-res patch coordinates
        scale_x = w_dr / w_sat
        scale_y = h_dr / h_sat
        x_hr, y_hr = int(x_lr * scale_x), int(y_lr * scale_y)
        
        drone_patch = self.drone_img[:, y_hr : y_hr + self.patch_size, x_hr : x_hr + self.patch_size]

        # Pad if edge patches fall out of bounds
        if drone_patch.shape[1] != self.patch_size or drone_patch.shape[2] != self.patch_size:
            drone_patch = np.pad(
                drone_patch, 
                ((0, 0), (0, max(0, self.patch_size - drone_patch.shape[1])), 
                 (0, max(0, self.patch_size - drone_patch.shape[2]))),
                mode='reflect'
            )

        # Convert arrays to Tensors
        sat_tensor = torch.from_numpy(sat_patch).float()
        drone_tensor = torch.from_numpy(drone_patch).float()

        # Upsample Satellite input using bicubic interpolation to fit target 256x256 size
        sat_upsampled = torch.nn.functional.interpolate(
            sat_tensor.unsqueeze(0), size=(self.patch_size, self.patch_size), mode='bicubic', align_corners=False
        ).squeeze(0)

        return sat_upsampled, drone_tensor