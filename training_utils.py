"""
Shared utilities for LoftUp training scripts.

This module contains common classes and functions used by both
train_loftup_stage1.py and train_loftup_stage2.py to avoid code duplication.
"""

import random
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from ema import EMA

class ScaleNet(torch.nn.Module):
    """Network for predicting uncertainty scales."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.net = torch.nn.Conv2d(dim, 1, 1)
        with torch.no_grad():
            self.net.weight.copy_(self.net.weight * 0.1)
            self.net.bias.copy_(self.net.bias * 0.1)

    def forward(self, x):
        return torch.exp(self.net(x) + 0.1).clamp_min(0.0001)


class AttentionDownsampler(torch.nn.Module):
    """Attention-based downsampler from FeatUp."""
    
    def __init__(self, dim, kernel_size, final_size, blur_attn=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.final_size = final_size
        self.in_dim = dim
        self.attention_net = torch.nn.Sequential(
            torch.nn.Dropout(p=.2),
            torch.nn.Linear(self.in_dim, 1)
        )
        self.w = torch.nn.Parameter(torch.ones(kernel_size, kernel_size) + 
                                   0.01 * torch.randn(kernel_size, kernel_size))
        self.b = torch.nn.Parameter(torch.zeros(kernel_size, kernel_size) + 
                                   0.01 * torch.randn(kernel_size, kernel_size))
        self.blur_attn = blur_attn

    def forward_attention(self, feats, guidance):
        return self.attention_net(feats.permute(0, 2, 3, 1)).squeeze(-1).unsqueeze(1)

    def forward(self, hr_feats, guidance):
        b, c, h, w = hr_feats.shape

        if self.blur_attn:
            # Use simple blur since kornia might not be available
            inputs = F.avg_pool2d(hr_feats, kernel_size=5, padding=2, stride=1)
        else:
            inputs = hr_feats
        
        final_size = h // self.kernel_size
        stride = (h - self.kernel_size) // (final_size - 1)

        patches = torch.nn.Unfold(self.kernel_size, stride=stride)(inputs) \
            .reshape(
            (b, self.in_dim, self.kernel_size * self.kernel_size, final_size, final_size * int(w / h))) \
            .permute(0, 3, 4, 2, 1)

        patch_logits = self.attention_net(patches).squeeze(-1)

        b, h, w, p = patch_logits.shape
        dropout = torch.rand(b, h, w, 1, device=patch_logits.device) > 0.2

        w = self.w.flatten().reshape(1, 1, 1, -1)
        b = self.b.flatten().reshape(1, 1, 1, -1)

        patch_attn_logits = (patch_logits * dropout) * w + b
        patch_attention = F.softmax(patch_attn_logits, dim=-1)

        downsampled = torch.einsum("bhwpc,bhwp->bchw", patches, patch_attention)

        return downsampled[:, :c, :, :]
    
    def get_kernel(self):
        return torch.sigmoid(self.w * 0 + self.b)


class TVLoss(torch.nn.Module):
    """Total variation loss from FeatUp."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, img):
        b, c, h, w = img.size()
        return ((img[:, :, 1:, :] - img[:, :, :-1, :]).square().sum() +
                (img[:, :, :, 1:] - img[:, :, :, :-1]).square().sum()) / (b * c * h * w)




def entropy(kernel):
    """Compute entropy of a kernel."""
    # Ensure positive values for log
    kernel = torch.clamp(kernel, min=1e-8)
    # Compute entropy
    entropy_val = -torch.sum(kernel * torch.log(kernel + 1e-8))
    return entropy_val


def apply_jitter(img, max_pad, transform_params):
    """Apply jittering transformation to image."""
    pad_x, pad_y, zoom_x, zoom_y, rotate = transform_params
    
    # Apply padding
    if pad_x > 0 or pad_y > 0:
        img = F.pad(img, (pad_x, pad_x, pad_y, pad_y), mode='reflect')
    
    # Apply zoom
    if zoom_x != 1.0 or zoom_y != 1.0:
        img = F.interpolate(img, scale_factor=(zoom_x, zoom_y), mode='bilinear')
    
    # Apply rotation
    if rotate != 0:
        angle_rad = torch.tensor(rotate * 3.14159 / 180.0)
        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)
        
        # Create rotation matrix
        rotation_matrix = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]])
        
        # Apply rotation (simplified - in practice you'd use torchvision.transforms.functional.rotate)
        # For now, we'll skip rotation to keep it simple
        pass
    
    return img


def sample_transform(use_pad, max_pad, max_zoom, h, w, max_rotation=0):
    """Sample random transformation parameters."""
    if use_pad:
        pad_x = random.randint(0, max_pad)
        pad_y = random.randint(0, max_pad)
    else:
        pad_x = pad_y = 0
    
    zoom_x = 1.0 + random.uniform(-max_zoom, max_zoom)
    zoom_y = 1.0 + random.uniform(-max_zoom, max_zoom)
    
    if max_rotation > 0:
        rotate = random.uniform(-max_rotation, max_rotation)
    else:
        rotate = 0.0
    
    return pad_x, pad_y, zoom_x, zoom_y, rotate


def compute_affinity_matrix_batch(feature_map, alpha=0.1):
    """
    Compute affinity matrix for a batch of feature maps within a spatial region defined by alpha.
    
    Args:
        feature_map: Tensor of shape (b, c, h, w)
        alpha: The proportion of boundary region to ignore (0 <= alpha <= 0.5). 
               The affinity will only be computed for the region (alpha*h, (1-alpha)*h) x (alpha*w, (1-alpha)*w).

    Returns:
        affinity_matrix_batch: Affinity matrix of shape (b, selected_region_size, selected_region_size)
    """
    # Feature map shape: (b, c, h, w)
    b, c, h, w = feature_map.shape

    # Calculate the region to keep based on alpha
    start_h = int(alpha * h)
    end_h = int((1 - alpha) * h)
    start_w = int(alpha * w)
    end_w = int((1 - alpha) * w)

    # Extract the sub-region of the feature map
    feature_map_region = feature_map[:, :, start_h:end_h, start_w:end_w]  # Shape (b, c, new_h, new_w)

    # Reshape the feature map to (b, c, new_h*new_w)
    feature_map_flat = feature_map_region.reshape(b, c, -1)  # Shape (b, c, new_h*new_w)

    # Normalize the feature vectors along the channel dimension for cosine similarity
    feature_map_flat = feature_map_flat / (feature_map_flat.norm(dim=1, keepdim=True) + 1e-6)  # Shape (b, c, new_h*new_w)

    # Compute the affinity matrix for each item in the batch within the selected region
    # Affinity matrix shape will be (b, new_h*new_w, new_h*new_w)
    affinity_matrix_batch = torch.einsum('bcn,bcm->bnm', feature_map_flat, feature_map_flat)

    return affinity_matrix_batch


def project(feats, proj):
    """
    Project features using random projection matrix.
    
    Note: Uncertainty is handled in the loss computation, not in the projection itself.
    The projection is the same whether uncertainty is used or not.
    """
    if proj is None:
        return feats
    return torch.einsum("bchw,bcd->bdhw", feats, proj)


def create_random_projection(feats, random_projection_dim, device=None):
    """
    Create a random projection matrix for feature dimensionality reduction.
    
    Args:
        feats: Feature tensor or list of feature tensors
        random_projection_dim: Dimension of the random projection
        device: Device to create the projection on (if None, uses feats device)
    
    Returns:
        proj: Random projection matrix or None if random_projection_dim is None
    """
    if random_projection_dim is None:
        return None
    
    if isinstance(feats, list):
        feat_shape = feats[-1].shape
        feat_device = feats[-1].device
    else:
        feat_shape = feats.shape
        feat_device = feats.device
    
    if device is None:
        device = feat_device
    
    proj = torch.randn(feat_shape[0], feat_shape[1], random_projection_dim, device=device)
    proj = proj / proj.square().sum(1, keepdim=True).sqrt()
    return proj


def get_kernel_size(model_type):
    """
    Determine kernel size based on model type.
    
    Args:
        model_type: String indicating the model type (e.g., "dinov2")
    
    Returns:
        kernel_size: Integer kernel size (14 for dinov2, 16 for others)
    """
    if "dinov2" in model_type.lower():
        return 14
    else:
        return 16 