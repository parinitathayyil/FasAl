import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

class MetricSuite:
    def __init__(self, device="cpu"):
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.device = device

    def compute_all(self, pred_tensor, target_tensor):
        """
        pred_tensor: Generated High-Res imagery batch [B, C, H, W]
        target_tensor: Ground Truth Drone imagery batch [B, C, H, W]
        """
        psnr_val = self.psnr(pred_tensor, target_tensor).item()
        ssim_val = self.ssim(pred_tensor, target_tensor).item()

        # Compute Spectral Angle Mapper (SAM)
        dot = torch.sum(pred_tensor * target_tensor, dim=1)
        p_norm = torch.norm(pred_tensor, dim=1)
        t_norm = torch.norm(target_tensor, dim=1)
        sam_rad = torch.acos(torch.clamp(dot / (p_norm * t_norm + 1e-7), -1.0, 1.0))
        sam_val = torch.mean(sam_rad).item()

        return {"PSNR_dB": round(psnr_val, 2), "SSIM": round(ssim_val, 4), "SAM_rad": round(sam_val, 4)}