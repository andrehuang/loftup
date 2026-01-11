import torch
import torch.nn as nn
import sys

from torchvision import transforms
import torch.nn.functional as F
from einops import rearrange

from huggingface_hub import PyTorchModelHubMixin

import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import numpy as np
import torchvision.transforms as T
from .layers import ChannelNorm, LayerNorm, MinMaxScaler, ImplicitFeaturizer, CATransformer


class LoftUp(nn.Module, PyTorchModelHubMixin,
             repo_url="https://github.com/andrehuang/loftup",
             pipeline_tag="image-feature-extraction",
             license="mit",
             paper_url="https://huggingface.co/papers/2504.14032"):
    """
    We use Fourier features of images as inputs, and do cross attention with the LR features, the output is the HR features.
    """
    def __init__(self, dim, color_feats=True, n_freqs=20, num_heads=4, num_layers=2, num_conv_layers=1, lr_size=16, lr_pe_type="sine"):
        super(LoftUp, self).__init__()

        if color_feats:
            start_dim = 5 * n_freqs * 2 + 3
        else:
            start_dim = 2 * n_freqs * 2
        
        num_patches = lr_size * lr_size
        self.lr_pe_type = lr_pe_type
        if self.lr_pe_type == "sine":
            self.lr_pe = ImplicitFeaturizer(color_feats=False, n_freqs=5, learn_bias=True)
            self.lr_pe_dim = 2 * 5 * 2
        elif self.lr_pe_type == "learnable":
            self.lr_pe = nn.Parameter(torch.randn(1, num_patches, dim))
            self.lr_pe_dim = dim

        self.fourier_feat = torch.nn.Sequential(
                                MinMaxScaler(),
                                ImplicitFeaturizer(color_feats, n_freqs=n_freqs, learn_bias=True),
                            )
        if self.lr_pe_type == "sine": # LR PE is concatenated to LR
            self.first_conv = torch.nn.Sequential(
                                ChannelNorm(start_dim),
                                nn.Conv2d(start_dim, dim+self.lr_pe_dim, kernel_size=3, padding=1),
                                nn.BatchNorm2d(dim+self.lr_pe_dim),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(dim+self.lr_pe_dim, dim+self.lr_pe_dim, kernel_size=3, padding=1),
                                nn.BatchNorm2d(dim+self.lr_pe_dim),
                                nn.ReLU(inplace=True),
                                )


            self.final_conv = torch.nn.Sequential(
                nn.Conv2d(dim+self.lr_pe_dim, dim, kernel_size=1),
                LayerNorm(dim),
            )

            self.ca_transformer = CATransformer(dim+self.lr_pe_dim, depth=num_layers, heads=num_heads, dim_head=dim//num_heads, mlp_dim=dim, dropout=0.)
        elif self.lr_pe_type == "learnable": # LR PE is added to LR
            self.first_conv = torch.nn.Sequential(
                                ChannelNorm(start_dim),
                                nn.Conv2d(start_dim, dim, kernel_size=3, padding=1),
                                nn.BatchNorm2d(dim),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                                nn.BatchNorm2d(dim),
                                nn.ReLU(inplace=True),
                                )
            self.final_conv = LayerNorm(dim)
            self.ca_transformer = CATransformer(dim, depth=num_layers, heads=num_heads, dim_head=dim//num_heads, mlp_dim=dim, dropout=0.)

    def forward(self, lr_feats, img):
        # Step 1: Extract Fourier features from the input image
        x = self.fourier_feat(img) # Output shape: (B, dim, H, W)
        b, c, h, w = x.shape

        ## Resize and add LR feats to x? 
        x = self.first_conv(x)
    
        # Reshape for attention (B, C, H, W) -> (B, H*W, C)
        b, c, h, w = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # (B, H*W, C)

        # Step 2: Process LR features for keys and values
        b, c_lr, h_lr, w_lr = lr_feats.shape

        if self.lr_pe_type == "sine":
            lr_pe = self.lr_pe(lr_feats)
            lr_feats_with_pe = torch.cat([lr_feats, lr_pe], dim=1)
            lr_feats_with_pe = lr_feats_with_pe.flatten(2).permute(0, 2, 1)
        elif self.lr_pe_type == "learnable":
            lr_feats = lr_feats.flatten(2).permute(0, 2, 1) # (B, H*W, C)
            if lr_feats.shape[1] != self.lr_pe.shape[1]:
                len_pos_old = int(math.sqrt(self.lr_pe.shape[1]))
                pe = self.lr_pe.reshape(1, len_pos_old, len_pos_old, c_lr).permute(0, 3, 1, 2)
                pe = F.interpolate(pe, size=(h_lr, w_lr), mode='bicubic', align_corners=False)
                pe = pe.reshape(1, c_lr, h_lr*w_lr).permute(0, 2, 1)
                lr_feats_with_pe = lr_feats + pe
            else:
                lr_feats_with_pe = lr_feats + self.lr_pe
        x = self.ca_transformer(x, lr_feats_with_pe)     

        # Reshape back to (B, C, H, W)
        x = x.permute(0, 2, 1).reshape(b, c, h, w)

        return self.final_conv(x)

class UpsamplerwithChannelNorm(nn.Module):
    def __init__(self, upsampler, channelnorm):
        super(UpsamplerwithChannelNorm, self).__init__()
        self.upsampler = upsampler
        self.channelnorm = channelnorm

    def forward(self, lr_feats, img):
        lr_feats = self.channelnorm(lr_feats)
        return self.upsampler(lr_feats, img)

def load_loftup_checkpoint(upsampler_path, n_dim, lr_pe_type="sine", lr_size=16):
    channelnorm = ChannelNorm(n_dim)
    upsampler = LoftUp(n_dim, lr_pe_type=lr_pe_type, lr_size=16)
    ckpt_weight = torch.load(upsampler_path)['state_dict']
    channelnorm_checkpoint = {k: v for k, v in ckpt_weight.items() if 'model.1' in k} # dict_keys(['model.1.norm.weight', 'model.1.norm.bias'])
    # change the key names
    channelnorm_checkpoint = {k.replace('model.1.', ''): v for k, v in channelnorm_checkpoint.items()}
    # if the key starts with upsampler, remove the upsampler.
    upsampler_ckpt_weight = {k: v for k, v in ckpt_weight.items() if k.startswith('upsampler')}
    upsampler_ckpt_weight = {k.replace('upsampler.', ''): v for k, v in upsampler_ckpt_weight.items()}
    upsampler.load_state_dict(upsampler_ckpt_weight)
    channelnorm.load_state_dict(channelnorm_checkpoint)
    for param in upsampler.parameters():
        param.requires_grad = False
    for param in channelnorm.parameters():
        param.requires_grad = False
    # return channelnorm, upsampler
    return UpsamplerwithChannelNorm(upsampler, channelnorm)

class Bilinear(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, feats, img):
        _, _, h, w = img.shape
        return F.interpolate(feats, (h, w), mode="bilinear")

def get_upsampler(upsampler, dim, lr_size=16, n_freqs=20, cfg=None, lr_pe_type="sine"):
    if upsampler == "loftup":
        return LoftUp(dim, n_freqs=n_freqs, lr_size=lr_size, lr_pe_type=lr_pe_type)
    elif upsampler == "bilinear":
        return Bilinear()
    # elif upsampler == "lift":
        
    else:
        raise ValueError(f"Upsampler {upsampler} not implemented")

def load_upsampler_weights(upsampler, upsampler_path, dim, freeze=True):
    channelnorm = ChannelNorm(dim)
    ckpt_weight = torch.load(upsampler_path)['state_dict']
    channelnorm_checkpoint = {k: v for k, v in ckpt_weight.items() if 'model.1' in k} # dict_keys(['model.1.norm.weight', 'model.1.norm.bias'])
    channelnorm_checkpoint = {k.replace('model.1.', ''): v for k, v in channelnorm_checkpoint.items()}
    upsampler_ckpt_weight = {k: v for k, v in ckpt_weight.items() if k.startswith('upsampler')}
    upsampler_ckpt_weight = {k.replace('upsampler.', ''): v for k, v in upsampler_ckpt_weight.items()}

    upsampler.load_state_dict(upsampler_ckpt_weight)
    channelnorm.load_state_dict(channelnorm_checkpoint)
    if freeze:
        for param in upsampler.parameters():
            param.requires_grad = False
        for param in channelnorm.parameters():
            param.requires_grad = False
    return UpsamplerwithChannelNorm(upsampler, channelnorm)
