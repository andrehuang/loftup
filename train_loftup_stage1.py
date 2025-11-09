"""
LoftUp Stage 1 Training Script

Example training command:
python train_loftup_stage1.py ++dataset="sa1b" ++epochs=1 ++batch_size=2 ++num_gpus=4 ++model_type="dinov2" ++pytorch_data_dir='datasets' ++upsampler_type="loftup" ++sam_mask_alpha=0.8 ++load_size=224 ++upsample_size=224 ++tv_weight=0.001 ++clamp_featup=True

This script trains upsamplers to convert low-resolution features to high-resolution features.
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
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

from upsamplers import get_upsampler, load_upsampler_weights
from datasets import get_dataset
from featurizers import get_featurizer
from utils import (
    pca, norm, unnorm, prep_image, adjust_features_with_masks, 
    mask_feature_similarity_loss
)
from training_utils import (
    ScaleNet, AttentionDownsampler, TVLoss, entropy, 
    apply_jitter, sample_transform, project,
    create_random_projection, get_kernel_size
)

class LoftUpStage1(pl.LightningModule):
    """LoftUp Stage 1 training module for feature upsampling."""
    
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
                 zoom_only=False,
                 cfg=None,
                 upsample_size=224,
                 multi_upsample_size=False,
                 clamp_featup=False,
                 aug_size=False,
                 sam_mask_alpha=0.8,
                 sam_mask_reg=0.0):
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
        self.sam_mask_alpha = sam_mask_alpha
        self.sam_mask_reg = sam_mask_reg

        # Initialize feature extractor
        self.model, self.patch_size, self.dim = get_featurizer(model_type, activation_type, num_classes=1000)
        self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Freeze feature extractor
        for p in self.model.parameters():
            p.requires_grad = False

        # Initialize upsampler
        self.upsampler = get_upsampler(upsampler, self.dim, lr_size=self.final_size, cfg=cfg)

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

        self.automatic_optimization = False

    def forward(self, x):
        return self.upsampler(self.model(x))

    def project(self, feats, proj):
        """
        Project features using random projection matrix.
        
        Note: Uncertainty is handled in the loss computation, not in the projection.
        """
        return project(feats, proj)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        with torch.no_grad():
            if isinstance(batch, dict):
                original_img = batch['img']
                binary_masks = batch['label']
                if self.multi_upsample_size:
                    sample_size = random.choice([self.upsample_size//4, self.upsample_size//2, self.upsample_size])
                    guidance_img = F.interpolate(original_img, size=(sample_size, sample_size), mode='bilinear')
                    binary_masks = F.interpolate(binary_masks, size=(sample_size, sample_size), mode='nearest')
                else:
                    sample_size = self.upsample_size
                    guidance_img = F.interpolate(original_img, size=(self.upsample_size, self.upsample_size), mode='bilinear')
                    binary_masks = F.interpolate(binary_masks, size=(self.upsample_size, self.upsample_size), mode='nearest')
            else:
                img, _ = batch
                original_img = img
                guidance_img = img
                binary_masks = None

        # Determine input image size
        if self.aug_size:
            input_img_size = random.choice([224, 336])
        else:
            input_img_size = 224
            
        img = F.interpolate(original_img, size=(input_img_size, input_img_size), mode='bilinear')
        guidance_img = F.interpolate(guidance_img, size=(input_img_size, input_img_size), mode='bilinear')
        if binary_masks is not None:
            binary_masks = F.interpolate(binary_masks, size=(input_img_size, input_img_size), mode='nearest')

        # Extract features
        with torch.no_grad():
                lr_feats = self.model(img)
                final_lr_feats = lr_feats

        full_rec_loss = 0.0
        full_entropy_loss = 0.0
        full_tv_loss = 0.0
        full_total_loss = 0.0

        for i in range(self.n_jitters):
            # Upsample features
            hr_feats = self.upsampler(final_lr_feats, guidance_img)

            # Ensure HR features match image size
            if hr_feats.shape[2] != img.shape[2]:
                hr_feats = F.interpolate(hr_feats, img.shape[2:], mode="bilinear")

            # Apply jittering
            with torch.no_grad():
                if self.zoom_only:
                    transform_params = sample_transform(
                        False, 0, self.max_zoom, guidance_img.shape[2], guidance_img.shape[3])
                else:
                    transform_params = sample_transform(
                        True, self.max_pad, self.max_zoom, guidance_img.shape[2], guidance_img.shape[3], 
                        max_rotation=self.max_rotate)
                
                jit_img = apply_jitter(guidance_img, self.max_pad, transform_params)
                
                # Ensure jittered image has correct size
                if jit_img.shape[2] != guidance_img.shape[2]:
                    jit_img = F.interpolate(jit_img, guidance_img.shape[2:], mode="bilinear")
                
                lr_jit_feats = self.model(jit_img)

            # Random projection for efficiency
            proj = create_random_projection(final_lr_feats, self.random_projection)

            # Apply jittering to HR features
            hr_jit_feats = apply_jitter(hr_feats, self.max_pad, transform_params)
            if hr_jit_feats.shape[2] != guidance_img.shape[2]:
                hr_jit_feats = F.interpolate(hr_jit_feats, guidance_img.shape[2:], mode="bilinear")
            
            proj_hr_feats = self.project(hr_jit_feats, proj)
            down_jit_feats = self.project(self.downsampler(hr_jit_feats, jit_img), proj)

            # Compute reconstruction loss
            if self.predicted_uncertainty:
                scales = self.scale_net(lr_jit_feats)
                scale_factor = (1 / (2 * scales ** 2))
                mse = (down_jit_feats - self.project(lr_jit_feats, proj)).square()
                rec_loss = (scale_factor * mse + scales.log()).mean() / self.n_jitters
            else:
                rec_loss = (self.project(lr_jit_feats, proj) - down_jit_feats).square().mean() / self.n_jitters

            if self.clamp_featup:
                rec_loss = torch.clamp(rec_loss, min=0.0)
            
                full_rec_loss = full_rec_loss + rec_loss

            # Compute CRF loss (only for first jitter)


            # Compute entropy loss
            if self.filter_ent_weight > 0.0:
                entropy_loss = entropy(self.downsampler.get_kernel())
                full_entropy_loss += entropy_loss.item()
            else:
                entropy_loss = 0

            # Compute TV loss (only for first jitter)
            if self.tv_weight > 0 and i == 0:
                tv_loss = self.tv(proj_hr_feats.square().sum(1, keepdim=True))
                full_tv_loss += tv_loss.item()
            else:
                tv_loss = 0.0

            # Total loss
            loss = rec_loss + self.tv_weight * tv_loss - self.filter_ent_weight * entropy_loss
            full_total_loss += loss

            torch.cuda.empty_cache()

        # Apply SAM mask adjustment if enabled
        if self.sam_mask_alpha > 0.0:
            lr_feat = final_lr_feats
            
            # Create bilinear upsampled features for comparison
            up_bilinear_features = F.interpolate(lr_feat, size=(guidance_img.shape[2], guidance_img.shape[3]), mode='bicubic')
            
            # Adjust features with masks
            adjusted_bilinear_features = adjust_features_with_masks(up_bilinear_features, binary_masks, alpha=self.sam_mask_alpha)
            
            # Compute additional reconstruction loss with adjusted features
            if self.random_projection is not None:
                proj_hr_feats_no_jit = self.project(hr_feats, proj)
                sam_mask_bilinear_rec_loss = (self.project(adjusted_bilinear_features, proj) - proj_hr_feats_no_jit).square().mean()
            else:
                sam_mask_bilinear_rec_loss = (adjusted_bilinear_features - hr_feats).square().mean()
            
            # Add to total loss
            full_total_loss += sam_mask_bilinear_rec_loss
            self.log("loss/sam_mask_bilinear_rec", sam_mask_bilinear_rec_loss.item())

        # Apply SAM mask regularization if enabled
        if self.sam_mask_reg > 0.0:
            sam_mask_loss = mask_feature_similarity_loss(hr_feats, binary_masks)
            full_total_loss += sam_mask_loss * self.sam_mask_reg
            self.log("loss/sam_mask_reg", sam_mask_loss.item())

        # Manual backward pass
        self.manual_backward(full_total_loss)

        # Logging
        full_total_loss = full_total_loss.item()
        self.log("loss/ent", full_entropy_loss)
        self.log("loss/tv", full_tv_loss)
        self.log("loss/rec", full_rec_loss)
        self.log("loss/total", full_total_loss)

        if self.global_step % 100 == 0:
            print(f"Step {self.global_step}: Total loss: {full_total_loss}, Rec loss: {full_rec_loss}")

        if self.global_step % 5000 == 0:
            self.trainer.save_checkpoint(self.chkpt_dir[:-5] + f'_{self.global_step}.ckpt')

        # Gradient clipping for early steps
        if self.global_step < 10:
            self.clip_gradients(opt, gradient_clip_val=0.0001, gradient_clip_algorithm="norm")

        opt.step()
        return None

    def validation_step(self, batch, batch_idx):
        """Validation step with visualization."""
        with torch.no_grad():
            if self.trainer.is_global_zero and batch_idx == 0:
                if isinstance(batch, dict):
                    img = batch['img']
                    binary_masks = batch['label']
                    guidance_img = F.interpolate(img, size=(self.upsample_size, self.upsample_size), mode='bilinear')
                    binary_masks = F.interpolate(binary_masks, size=(self.upsample_size, self.upsample_size), mode='nearest')
                else:
                    img, _ = batch
                    guidance_img = img

                # Extract features
                lr_feats = self.model(img)
                final_lr_feats = lr_feats

                # Upsample features
                hr_feats = self.upsampler(final_lr_feats, guidance_img)

                # Ensure HR features match image size
                if hr_feats.shape[2] != img.shape[2]:
                    hr_feats = F.interpolate(hr_feats, img.shape[2:], mode="bilinear")

                # Apply jittering for validation
                if self.zoom_only:
                    transform_params = sample_transform(
                        False, 0, self.max_zoom, img.shape[2], img.shape[3])
                else:
                    transform_params = sample_transform(
                        True, self.max_pad, self.max_zoom, img.shape[2], img.shape[3], 
                        max_rotation=self.max_rotate)
                
                jit_img = apply_jitter(img, self.max_pad, transform_params)
                lr_jit_feats = self.model(jit_img)

                # Random projection
                proj = create_random_projection(final_lr_feats, self.random_projection)

                # Get uncertainty scales if enabled
                if self.predicted_uncertainty:
                    scales = self.scale_net(lr_jit_feats)
                else:
                    scales = torch.ones_like(lr_jit_feats[:, :1])

                # Visualization
                writer = self.logger.experiment

                hr_jit_feats = apply_jitter(hr_feats, self.max_pad, transform_params)
                if hr_jit_feats.shape[2] != guidance_img.shape[2] or hr_jit_feats.shape[3] != guidance_img.shape[3]:
                    hr_jit_feats = F.interpolate(hr_jit_feats, guidance_img.shape[2:], mode="bilinear")
                
                down_jit_feats = self.downsampler(hr_jit_feats, jit_img)

                # PCA visualization
                lr_feat = final_lr_feats
                
                [red_lr_feats], fit_pca = pca([lr_feat[0].unsqueeze(0)])
                [red_hr_feats], _ = pca([hr_feats[0].unsqueeze(0)], fit_pca=fit_pca)
                [red_lr_jit_feats], _ = pca([lr_jit_feats[0].unsqueeze(0)], fit_pca=fit_pca)
                [red_hr_jit_feats], _ = pca([hr_jit_feats[0].unsqueeze(0)], fit_pca=fit_pca)
                [red_down_jit_feats], _ = pca([down_jit_feats[0].unsqueeze(0)], fit_pca=fit_pca)

                # Log images to tensorboard
                writer.add_image("viz/image", unnorm(img[0].unsqueeze(0))[0], self.global_step)
                writer.add_image("viz/lr_feats", red_lr_feats[0], self.global_step)
                writer.add_image("viz/hr_feats", red_hr_feats[0], self.global_step)
                writer.add_image("jit_viz/jit_image", unnorm(jit_img[0].unsqueeze(0))[0], self.global_step)
                writer.add_image("jit_viz/lr_jit_feats", red_lr_jit_feats[0], self.global_step)
                writer.add_image("jit_viz/hr_jit_feats", red_hr_jit_feats[0], self.global_step)
                writer.add_image("jit_viz/down_jit_feats", red_down_jit_feats[0], self.global_step)

                # Log scales
                norm_scales = scales[0]
                norm_scales /= scales.max()
                writer.add_image("scales", norm_scales, self.global_step)
                writer.add_histogram("scales hist", scales, self.global_step)

                # Log downsampler information
                if isinstance(self.downsampler, AttentionDownsampler):
                    writer.add_image(
                        "down/att",
                        prep_image(self.downsampler.forward_attention(hr_feats, None)[0]),
                        self.global_step)
                    writer.add_image(
                        "down/w",
                        prep_image(self.downsampler.w.clone().squeeze()),
                        self.global_step)
                    writer.add_image(
                        "down/b",
                        prep_image(self.downsampler.b.clone().squeeze()),
                        self.global_step)

                writer.flush()

    def configure_optimizers(self):
        """Configure optimizers for trainable parameters."""
        all_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                all_params.append(param)
        return torch.optim.NAdam(all_params, lr=self.lr)


@hydra.main(config_path="configs", config_name="train_loftup_stage1.yaml")
def my_app(cfg: DictConfig) -> None:
    """Main training function."""
    print(OmegaConf.to_yaml(cfg))
    print(cfg.output_root)
    seed_everything(seed=0, workers=True)

    load_size = cfg.load_size
    upsample_size = cfg.upsample_size

    # Determine kernel size based on model type
    kernel_size = get_kernel_size(cfg.model_type)
    final_size = load_size // kernel_size
    
    # Create experiment name
    name = (f"{cfg.model_type}_{cfg.upsampler_type}_depth{cfg.upsampler_num_layers}_"
            f"loadsize_{cfg.load_size}_upsample_size_{cfg.upsample_size}_"
            f"{cfg.dataset}_{cfg.downsampler_type}_"
            f"tv_{cfg.tv_weight}_sam_alpha_{cfg.sam_mask_alpha}_sam_reg_{cfg.sam_mask_reg}"
            f"_RGB_{cfg.color_feats}_clamp_{cfg.clamp_featup}")

    # Setup logging and checkpoint directories
    log_dir = join(cfg.output_root, f"logs/loftup_stage1/{name}")
    chkpt_dir = join(cfg.output_root, f"checkpoints/loftup_stage1/{name}.ckpt")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to {log_dir}")

    # Initialize model
    model = LoftUpStage1(
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
        zoom_only=cfg.zoom_only,
        cfg=cfg,
        upsample_size=upsample_size,
        multi_upsample_size=cfg.multi_upsample_size,
        clamp_featup=cfg.clamp_featup,
        aug_size=cfg.aug_size,
        sam_mask_alpha=cfg.sam_mask_alpha,
        sam_mask_reg=cfg.sam_mask_reg,
    )

    # Setup data transforms
    transform = T.Compose([
        T.Resize(load_size, InterpolationMode.BILINEAR),
        T.CenterCrop(load_size),
        T.ToTensor(),
        norm])
    
    target_transform = T.Compose([
        T.Lambda(lambda mask: torch.from_numpy(mask).float().unsqueeze(0)), 
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
    val_dataset = get_dataset(
        cfg.pytorch_data_dir,
        cfg.dataset,
        "val",
        transform=transform,
        target_transform=target_transform,
        include_labels=False)
    val_loader = DataLoader(val_dataset, 1, shuffle=False, num_workers=cfg.num_workers)

    # Setup logging and callbacks
    tb_logger = TensorBoardLogger(log_dir, default_hp_metric=False)
    callbacks = [ModelCheckpoint(chkpt_dir[:-5], every_n_epochs=1)]

    # Setup trainer
    trainer = Trainer(
        accelerator='gpu',
        strategy=DDPStrategy(find_unused_parameters=True),
        devices=cfg.num_gpus,
        max_epochs=cfg.epochs,
        logger=tb_logger,
        val_check_interval=500 if "debug" not in cfg.dataset else 10,
        log_every_n_steps=10,
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,
    )

    # Clean up memory
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    # Start training
    trainer.fit(model, loader, val_loader)
    trainer.save_checkpoint(chkpt_dir)
    print(f"Saved model to {chkpt_dir}")


if __name__ == "__main__":
    my_app() 