"""
LoftUp Stage 2 Training Script (High-Resolution Supervision)

Example training command:
python train_loftup_stage2.py ++dataset="sa1b" ++epochs=1 ++hr_res=896 ++batch_size=2 ++consistency_method="bilinear" ++model_type="dinov2" ++num_gpus=4 ++affinity_loss=True ++pytorch_data_dir='datasets' ++pretrained_upsampler="path/to/stage1_checkpoint.ckpt" ++upsampler_type="loftup" ++sam_mask_hr_alpha=0.5 ++sam_mask_reg=0.0 ++lr=1e-3 ++use_featup=False ++aug_size ++n_jitters=2

This script trains upsamplers with high-resolution supervision using a pretrained Stage 1 upsampler.
"""

import gc
import os
import random
from os.path import join

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

from upsamplers import get_upsampler, load_upsampler_weights, norm, unnorm
from datasets import get_dataset
from featurizers import get_featurizer
from utils import (
    pca, prep_image, adjust_features_with_masks, 
    mask_feature_similarity_loss
)
from training_utils import (
    ScaleNet, AttentionDownsampler, TVLoss, EMA, entropy, 
    apply_jitter, sample_transform, compute_affinity_matrix_batch,
    project, create_random_projection, get_kernel_size
)


class LoftUpStage2(pl.LightningModule):
    """LoftUp Stage 2 training module with high-resolution supervision."""
    
    def __init__(self,
                 model_type,
                 activation_type,
                 n_jitters,
                 max_pad,
                 max_zoom,
                 max_rotate,
                 kernel_size,
                 final_size,
                 lr,
                 random_projection,
                 predicted_uncertainty,
                 filter_ent_weight,
                 tv_weight,
                 upsampler,
                 downsampler,
                 chkpt_dir,
                 hr_res,
                 hr_weight,
                 consistency_method,
                 pretrained_upsampler,
                 affinity_loss,
                 rec_weight,
                 l1_affinity,
                 use_prototypes,
                 n_freqs,
                 sam_mask_reg,
                 sam_mask_hr_alpha,
                 sam_mask_hr_reg,
                 use_crop_upsampler,
                 cfg=None,
                 use_featup=False, # Default changed to False
                 zoom_only=False,
                 upsample_size=224,
                 multi_upsample_size=False,
                 clamp_featup=False,
                 aug_size=True, # Default changed to True
                 ):
        super().__init__()
        self.model_type = model_type
        self.activation_type = activation_type
        self.n_jitters = n_jitters
        self.max_pad = max_pad
        self.max_zoom = max_zoom
        self.max_rotate = max_rotate
        self.kernel_size = kernel_size
        self.final_size = final_size
        self.lr = lr
        self.random_projection = random_projection
        self.predicted_uncertainty = predicted_uncertainty
        self.filter_ent_weight = filter_ent_weight
        self.tv_weight = tv_weight
        self.chkpt_dir = chkpt_dir
        self.zoom_only = zoom_only
        self.upsample_size = upsample_size
        self.multi_upsample_size = multi_upsample_size
        self.clamp_featup = clamp_featup
        self.aug_size = aug_size
        
        # High-resolution supervision parameters
        self.hr_res = hr_res
        self.hr_weight = hr_weight
        self.consistency_method = consistency_method
        self.pretrained_upsampler = pretrained_upsampler
        self.affinity_loss = affinity_loss
        self.rec_weight = rec_weight
        self.l1_affinity = l1_affinity
        self.use_prototypes = use_prototypes
        self.n_freqs = n_freqs
        self.sam_mask_reg = sam_mask_reg
        self.sam_mask_hr_alpha = sam_mask_hr_alpha
        self.sam_mask_hr_reg = sam_mask_hr_reg
        self.use_crop_upsampler = use_crop_upsampler
        self.use_featup = use_featup
        self.upsampler_type = upsampler

        # Initialize feature extractor
        self.model, self.patch_size, self.dim = get_featurizer(model_type, activation_type, num_classes=1000)
        self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Freeze feature extractor
        for p in self.model.parameters():
            p.requires_grad = False

        # Initialize upsampler
        self.upsampler = get_upsampler(upsampler, self.dim, n_freqs=self.n_freqs, cfg=cfg)

        # Initialize downsampler
        if downsampler == 'attention':
            self.downsampler = AttentionDownsampler(self.dim, self.kernel_size, self.final_size, blur_attn=True)
        else:
            raise ValueError(f"Unknown downsampler {downsampler}")

        # Initialize uncertainty prediction network
        if self.predicted_uncertainty:
            self.scale_net = ScaleNet(self.dim)
            self.project = self._project_with_uncertainty
        else:
            self.project = self._project_simple

        # Initialize loss functions
        self.tv = TVLoss()

        # Initialize EMA for upsampler (hardcoded to always be active)
        if self.pretrained_upsampler is not None:
            # Load pretrained weights
            self.upsampler = load_upsampler_weights(self.upsampler, self.pretrained_upsampler)
            self.ema_upsampler = None
            print(f"Using pretrained upsampler weights. Upsampler type: {upsampler}. No EMA.")
        else:
            # Use EMA for upsampler training
            if self.use_crop_upsampler:
                self.ema_update_after = 0
                self.crop_upsampler = EMA(self.upsampler, beta=0.99, update_after_step=self.ema_update_after, update_every=10)
            else:
                # When there is no pretrained upsampler, we can still use EMA for the upsampler
                self.ema_update_after = 1000
                self.crop_upsampler = EMA(self.upsampler, beta=0.99, update_after_step=self.ema_update_after, update_every=10)

        self.automatic_optimization = False

    def project(self, feats, proj):
        """
        Project features using random projection matrix.
        
        Note: Uncertainty is handled in the loss computation, not in the projection.
        """
        return project(feats, proj)

    def _project_simple(self, feats, proj):
        """Default projection (same as project)."""
        return project(feats, proj)

    def _project_with_uncertainty(self, feats, proj):
        """Projection when uncertainty is enabled (currently identical)."""
        return project(feats, proj)


    def training_step(self, batch, batch_idx):
        """Training step with high-resolution supervision."""
        opt = self.optimizers()
        opt.zero_grad()

        # Process batch
        if isinstance(batch, dict):
            img = batch['img']
            binary_masks = batch['label']
        else:
            img, _ = batch
            binary_masks = None

        # Process high-resolution image
        # Determine input image size dynamically if aug_size is enabled
        if self.aug_size:
            if self.kernel_size == 14:  # DINOv2
                input_img_size = random.choice([224, 336, 448, 518])
            else:  # Other models
                input_img_size = random.choice([224, 336, 448, 512])
        else:
            input_img_size = 224
            
        hr_scale_factor = self.hr_res / input_img_size
        original_img = img
        original_binary_masks = binary_masks
        img = F.interpolate(original_img, input_img_size, mode='bilinear')  # 224x224 global image
        if binary_masks is not None:
            binary_masks = binary_masks.unsqueeze(1)
            binary_masks = F.interpolate(binary_masks, input_img_size, mode='nearest')

        # Extract features from global image
        lr_feats = self.model(img)

        # Initialize loss accumulators
        full_rec_loss = 0.0
        full_entropy_loss = 0.0
        full_tv_loss = 0.0
        full_total_loss = 0.0
        full_hr_loss = 0.0

        for i in range(self.n_jitters):
            # High-Resolution Loss
            wx = random.randint(0, original_img.shape[2] - input_img_size)
            hy = random.randint(0, original_img.shape[3] - input_img_size)
            # Ensure wx and hy are multiples of 16
            wx = wx - (wx % 16)
            hy = hy - (hy % 16)
            cropped_img = original_img[:, :, wx:wx+input_img_size, hy:hy+input_img_size]
            cropped_binary_masks = original_binary_masks[:, :, wx:wx+input_img_size, hy:hy+input_img_size] if original_binary_masks is not None else None

            # Upsample features
            hr_feats = self.upsampler(lr_feats, img)
            
            # Compute HR loss after EMA update
            if hasattr(self, 'ema_update_after') and self.global_step >= self.ema_update_after:
                with torch.no_grad():
                    cropped_feats = self.model(cropped_img)
                    
                    if self.use_crop_upsampler:
                        cropped_feats = self.crop_upsampler(cropped_feats, cropped_img)
                        
                        # Ensure cropped features match image size
                        if cropped_feats.shape[2] != cropped_img.shape[2]:
                            cropped_feats = F.interpolate(cropped_feats, cropped_img.shape[2:], mode='bilinear')
                            
                        # Apply SAM mask adjustment if enabled
                        if self.sam_mask_hr_alpha > 0.0 and cropped_binary_masks is not None:
                            cropped_feats = adjust_features_with_masks(cropped_feats, cropped_binary_masks, alpha=self.sam_mask_hr_alpha)

                # Determine feature region in global image
                if self.upsampler_type == "loftup":
                    feat_final_size = input_img_size  # Dynamic based on aug_size
                else:
                    feat_final_size = self.final_size * 16
                    
                cropped_hr_feat_loc_wx = int(wx / hr_scale_factor * feat_final_size / input_img_size)
                cropped_hr_feat_loc_hy = int(hy / hr_scale_factor * feat_final_size / input_img_size)
                cropped_hr_feat_size = int(input_img_size / hr_scale_factor * feat_final_size / input_img_size)

                # Crop HR features to corresponding region
                hr_feats_cropped = hr_feats[:, :, cropped_hr_feat_loc_wx:cropped_hr_feat_loc_wx+cropped_hr_feat_size, 
                                             cropped_hr_feat_loc_hy:cropped_hr_feat_loc_hy+cropped_hr_feat_size]
                
                # Resize cropped features to match HR features
                cropped_feats_resize = F.interpolate(cropped_feats, hr_feats_cropped.shape[2:], mode='bilinear')

                if self.use_crop_upsampler:
                    if self.affinity_loss:
                        aff_mat_hr = compute_affinity_matrix_batch(hr_feats_cropped)
                        aff_mat_cropped = compute_affinity_matrix_batch(cropped_feats_resize)
                        if self.l1_affinity:
                            hr_loss = F.l1_loss(aff_mat_hr, aff_mat_cropped)
                        else:
                            hr_loss = F.mse_loss(aff_mat_hr, aff_mat_cropped)
                    else:
                        hr_loss = F.mse_loss(hr_feats_cropped, cropped_feats_resize)
                else:
                    hr_loss = 0.0

                # Add SAM mask regularization
                if self.sam_mask_hr_reg > 0.0 and cropped_binary_masks is not None:
                    cropped_binary_masks_resize = F.interpolate(cropped_binary_masks, hr_feats_cropped.shape[2:], mode='nearest')
                    sam_mask_hr_reg = mask_feature_similarity_loss(hr_feats_cropped, cropped_binary_masks_resize)
                    hr_loss += sam_mask_hr_reg * self.sam_mask_hr_reg

                full_hr_loss += hr_loss / self.n_jitters

                # Ensure HR features match image size
                if hr_feats.shape[2] != img.shape[2]:
                    hr_feats = F.interpolate(hr_feats, img.shape[2:], mode="bilinear")
            else:
                hr_loss = 0
                full_hr_loss += hr_loss

            # Standard reconstruction loss with jittering
            with torch.no_grad():
                transform_params = sample_transform(
                    True, self.max_pad, self.max_zoom, img.shape[2], img.shape[3])
                jit_img = apply_jitter(img, self.max_pad, transform_params)
                lr_jit_feats = self.model(jit_img)

            # Random projection
            proj = create_random_projection(lr_feats, self.random_projection)

            hr_jit_feats = apply_jitter(hr_feats, self.max_pad, transform_params)
            proj_hr_feats = self.project(hr_jit_feats, proj)

            down_jit_feats_pre_proj = self.downsampler(hr_jit_feats, jit_img)
            down_jit_feats = self.project(down_jit_feats_pre_proj, proj)

            # Compute reconstruction loss
            if self.predicted_uncertainty:
                scales = self.scale_net(lr_jit_feats)
                scale_factor = (1 / (2 * scales ** 2))
                mse = (down_jit_feats - self.project(lr_jit_feats, proj)).square()
                rec_loss = (scale_factor * mse + scales.log()).mean() / self.n_jitters
            else:
                rec_loss = (self.project(lr_jit_feats, proj) - down_jit_feats).square().mean() / self.n_jitters

            if not self.use_featup:
                rec_loss = torch.clamp(rec_loss, min=0.0)

            full_rec_loss += rec_loss

            # Compute entropy loss
            if self.filter_ent_weight > 0.0:
                entropy_loss = entropy(self.downsampler.get_kernel())
                full_entropy_loss += entropy_loss.item()
            else:
                entropy_loss = 0

            # Compute TV loss
            if self.tv_weight > 0 and i == 0:
                tv_loss = self.tv(hr_feats)
                full_tv_loss += tv_loss.item()
            else:
                tv_loss = 0.0

            # Total loss for this jitter
            loss = rec_loss + self.tv_weight * tv_loss - self.filter_ent_weight * entropy_loss
            full_total_loss += loss

            torch.cuda.empty_cache()

        # Add HR loss to total loss
        full_total_loss += self.hr_weight * full_hr_loss

        # Manual backward pass
        self.manual_backward(full_total_loss)

        # Update EMA
        if hasattr(self, 'crop_upsampler'):
            self.crop_upsampler.update()

        # Logging
        full_total_loss = full_total_loss.item()
        self.log("loss/ent", full_entropy_loss)
        self.log("loss/tv", full_tv_loss)
        self.log("loss/rec", full_rec_loss)
        self.log("loss/hr", full_hr_loss)
        self.log("loss/total", full_total_loss)

        if self.global_step % 100 == 0:
            print(f"Step {self.global_step}: Total loss: {full_total_loss}, Rec loss: {full_rec_loss}, HR loss: {full_hr_loss}")

        if self.global_step % 5000 == 0:
            self.trainer.save_checkpoint(self.chkpt_dir[:-5] + f'_{self.global_step}.ckpt')

        # Gradient clipping for early steps
        if self.global_step < 10:
            self.clip_gradients(opt, gradient_clip_val=0.0001, gradient_clip_algorithm="norm")

        opt.step()
        return None

    def configure_optimizers(self):
        """Configure optimizers."""
        optimizer = torch.optim.AdamW([
            {'params': self.upsampler.parameters(), 'lr': self.lr},
            {'params': self.downsampler.parameters(), 'lr': self.lr},
        ])
        
        if self.predicted_uncertainty:
            optimizer.add_param_group({'params': self.scale_net.parameters(), 'lr': self.lr})
        
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        """Custom optimizer step to update EMA."""
        super().optimizer_step(*args, **kwargs)
        if hasattr(self, 'crop_upsampler'):
            self.crop_upsampler.update()  # Update the EMA upsampler

    def on_save_checkpoint(self, checkpoint):
        """Save checkpoint with additional information."""
        checkpoint['model_type'] = self.model_type
        checkpoint['upsampler_type'] = type(self.upsampler).__name__
        checkpoint['hr_res'] = self.hr_res
        checkpoint['sam_mask_hr_alpha'] = self.sam_mask_hr_alpha


@hydra.main(config_path="configs", config_name="train_loftup_stage2.yaml")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(cfg.output_root)
    seed_everything(seed=0, workers=True)

    # Determine kernel size based on model type
    kernel_size = get_kernel_size(cfg.model_type)
    final_size = cfg.load_size // kernel_size

    # Check if using pretrained upsampler
    if "bilinear" in cfg.pretrained_upsampler:
        ifpretrained = "bilinear"
    elif cfg.pretrained_upsampler != 'none' and cfg.pretrained_upsampler != 'None':
        ifpretrained = True
    else:
        ifpretrained = False

    # Generate experiment name
    name = (f"HR_{cfg.hr_res}_LR_{cfg.lr}_FeatUp{cfg.use_featup}_"
            f"{cfg.model_type}_{cfg.upsampler_type}_n_freqs_{cfg.n_freqs}_sam_reg_{cfg.sam_mask_reg}_sam_hr_alpha_{cfg.sam_mask_hr_alpha}_sam_hr_reg_{cfg.sam_mask_hr_reg}_"
            f"{cfg.dataset}_"
            f"crf_0.0"  # crf_weight is always 0
            f"_tv_{cfg.tv_weight}"
            f"_rec_{cfg.rec_weight}_n_jitters_{cfg.n_jitters}_ddp_stopgrad"
            )

    if "sam_mask_0.0" in cfg.pretrained_upsampler:
        name += "first_nosam"

    # Determine log and checkpoint directories
    if cfg.sam_mask_reg > 0.0 or cfg.sam_mask_hr_alpha > 0.0:
        log_dir = join(cfg.output_root, f"logs/{cfg.upsampler_type}_sam_hr/{name}")
        chkpt_dir = join(cfg.output_root, f"checkpoints/{cfg.upsampler_type}_sam_hr/{name}.ckpt")
    else:
        log_dir = join(cfg.output_root, f"logs/{cfg.upsampler_type}_hr/{name}")
        chkpt_dir = join(cfg.output_root, f"checkpoints/{cfg.upsampler_type}_hr/{name}.ckpt")
    
    os.makedirs(log_dir, exist_ok=True)

    # Create model
    model = LoftUpStage2(
        model_type=cfg.model_type,
        activation_type=cfg.activation_type,
        n_jitters=cfg.n_jitters,
        max_pad=cfg.max_pad,
        max_zoom=cfg.max_zoom,
        max_rotate=cfg.max_rotate,
        kernel_size=kernel_size,
        final_size=final_size,
        lr=cfg.lr,
        random_projection=cfg.random_projection,
        predicted_uncertainty=cfg.outlier_detection,
        filter_ent_weight=cfg.filter_ent_weight,
        tv_weight=cfg.tv_weight,
        upsampler=cfg.upsampler_type,
        downsampler=cfg.downsampler_type,
        chkpt_dir=chkpt_dir,
        hr_res=cfg.hr_res,
        hr_weight=cfg.hr_weight,
        consistency_method=cfg.consistency_method,
        pretrained_upsampler=cfg.pretrained_upsampler,
        affinity_loss=cfg.affinity_loss,
        rec_weight=cfg.rec_weight,
        l1_affinity=cfg.l1_affinity,
        use_prototypes=cfg.use_prototypes,
        n_freqs=cfg.n_freqs,
        sam_mask_reg=cfg.sam_mask_reg,
        sam_mask_hr_alpha=cfg.sam_mask_hr_alpha,
        sam_mask_hr_reg=cfg.sam_mask_hr_reg,
        use_crop_upsampler=cfg.use_crop_upsampler,
        cfg=cfg,
        use_featup=cfg.use_featup,
        zoom_only=cfg.zoom_only,
        upsample_size=cfg.upsample_size,
        multi_upsample_size=cfg.multi_upsample_size,
        clamp_featup=cfg.clamp_featup,
        aug_size=cfg.aug_size,
    )

    # Setup data transforms
    load_size = cfg.hr_res
    transform = T.Compose([
        T.Resize(load_size, InterpolationMode.BILINEAR),
        T.CenterCrop(load_size),
        T.ToTensor(),
        norm])
    
    target_transform = T.Compose([
        T.Lambda(lambda mask: TF.pil_to_tensor(mask).float()), 
        T.Resize(load_size, InterpolationMode.NEAREST),
        T.CenterCrop(load_size),
    ])

    # Setup dataset and dataloader
    dataset = get_dataset(
        cfg.pytorch_data_dir,
        cfg.dataset,
        "train",
        transform=transform,
        target_transform=target_transform,
        include_labels=False)

    loader = DataLoader(
        dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    
    # Simple validation dataset (single image)
    from datasets.util import SingleImageDataset
    val_dataset = get_dataset(
        cfg.pytorch_data_dir,
        cfg.dataset,
        "val",
        transform=transform,
        target_transform=target_transform,
        include_labels=False)
    val_loader = DataLoader(
        SingleImageDataset(0, val_dataset, 1), 1, shuffle=False, num_workers=cfg.num_workers)

    # Setup logging and callbacks
    tb_logger = TensorBoardLogger(log_dir, default_hp_metric=False)
    callbacks = [ModelCheckpoint(chkpt_dir[:-5], every_n_epochs=1)]

    # Create trainer
    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        strategy=DDPStrategy(find_unused_parameters=False) if cfg.num_gpus > 1 else None,
        devices=cfg.num_gpus if torch.cuda.is_available() else 1,
        max_epochs=cfg.epochs,
        logger=tb_logger,
        val_check_interval=100 if "debug" not in cfg.dataset else 10,
        log_every_n_steps=10,
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,
        precision=16 if torch.cuda.is_available() else 32,
    )

    # Clean up memory
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    # Train
    trainer.fit(model, loader, val_loader)
    trainer.save_checkpoint(chkpt_dir)
    print(f"Saved checkpoint to {chkpt_dir}")


if __name__ == "__main__":
    my_app() 