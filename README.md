# [LoftUp: A Coordinate-Based Feature Upsampler for Vision Foundation Models](https://arxiv.org/abs/2504.14032)

ICCV2025 (oral)

[[Arxiv]](https://arxiv.org/abs/2504.14032) [[Project Page]](https://andrehuang.github.io/loftup-site/)

[Haiwen Huang](https://andrehuang.github.io/), Anpei Chen, Volodymyr Havrylov, Andreas Geiger, Dan Zhang

![Teaser](figures/loftup-teaser.png)

**TL;DR:** LoftUp achieves the strongest feature upsampling performance at a comparable speed to bilinear upsampling.

![bike-packing](examples/bike-packing.gif)
![camel](examples/clip-camel.gif)
![horsejump](examples/siglip2-horsejump.gif)


## Contents
- [Install](https://github.com/andrehuang/loftup/tree/main?tab=readme-ov-file#install)
- [Inference with pretrained upsamplers](https://github.com/andrehuang/loftup/tree/main?tab=readme-ov-file#inference-with-pretrained-upsamplers)
- [Evaluation on downstream tasks](https://github.com/andrehuang/loftup/tree/main?tab=readme-ov-file#inference-with-pretrained-upsamplers)
- [Training LoftUp upsamplers](https://github.com/andrehuang/loftup/tree/main?tab=readme-ov-file#inference-with-pretrained-upsamplers)
- [Citation](https://github.com/andrehuang/loftup/tree/main?tab=readme-ov-file#inference-with-pretrained-upsamplers)

## Install

In general, LoftUp can run with most recent pytorch environments. We encourage the users to try out LoftUp in their exisitng environment first.

We also provide two yaml file for installation. To use them, simply run:

```bash
conda env create -f environment_cuda11.yaml
```

or 

```bash
conda env create -f environment.yaml
```


## Inference with pretrained upsamplers

All pre-trained upsamplers are available on 🤗 here: https://huggingface.co/models?search=loftup.

We provide example code for using LoftUp in [example_usage.py](example_usage.py). Currently we provide:


|Backbone Name          | Featurizer Class              | HF hub                                  | Torch Hub Repo | Torch Hub Name |
|-------------------| ---|------------------------------------------------|------|-----|
| DINOv2 S/14     | [dinov2](featurizers/DINOv2.py)     | [haiwen/loftup-dinov2s](https://huggingface.co/haiwen/loftup-dinov2s)   | andrehuang/loftup | loftup_dinov2s|
| DINOv2 S/14 + Reg | [dinov2s_reg](featurizers/DINOv2.py)     | [haiwen/loftup-dinov2s_reg](https://huggingface.co/haiwen/loftup-dinov2s_reg)| andrehuang/loftup | loftup_dinov2s_reg|
| DINOv2 B/14 | [dinov2b](featurizers/DINOv2.py) | [haiwen/loftup-dinov2b](https://huggingface.co/haiwen/loftup-dinov2b) | andrehuang/loftup | loftup_dinov2b|
| DINOv2 B/14 + Reg | [dinov2b_reg](featurizers/DINOv2.py)     | [haiwen/loftup-dinov2b_reg](https://huggingface.co/haiwen/loftup-dinov2b_reg)|andrehuang/loftup | loftup_dinov2b_reg|
| CLIP ViT B/16 | [clip](featurizers/CLIP.py) |[haiwen/loftup-clip](https://huggingface.co/haiwen/loftup-clip) | andrehuang/loftup | loftup_clip|
|SigLIP ViT B/16 | [siglip](featurizers/SigLIP.py) | [haiwen/loftup-siglip](https://huggingface.co/haiwen/loftup-siglip)| andrehuang/loftup | loftup_siglip|
|SigLIP2 ViT B/16 | [siglip2](featurizers/SigLIP.py) | [haiwen/loftup-siglip2](https://huggingface.co/haiwen/loftup-siglip2)| andrehuang/loftup | loftup_siglip2|

To use torch hub checkpoints, simply run 
```python
upsampler = torch.hub.load('andrehuang/loftup', model_torch_hub_name, pretrained=True)
```
For example, ```upsampler = torch.hub.load('andrehuang/loftup', loftup_dinov2s, pretrained=True)```.

The upsampler class is defined at [UpsamplerwithChannelNorm](https://github.com/andrehuang/loftup/blob/7ce8a97e720465819a2a6b24a7c24c192da394b6/upsamplers/upsamplers.py#L109).

## Evaluation on Downstream Tasks

### Dataset Preparation

See [Preparing Datasets for Evaluation](datasets/README.md).

### Semantic Segmentation
For semantic segmentation, our implementation is adapted from [FeatUp](https://github.com/mhamilton723/FeatUp). You can use [eval_seg.py](eval_seg.py) by running:

```bash
python eval_seg.py  ++upsampler_path=/path/to/your/upsampler
```

You can also configure other hyper-parameters such as output_dir and dataset directory. The config file is [configs/eval_seg.yaml](configs/eval_seg.yaml). 

### Video Object Segmentation
For video object segmentation on DAVIS, our code is modified from the implementation in [LiFT](https://github.com/saksham-s/lift). Specifically, we first extract segmentaiton results by  running:

```bash
    python eval_davis.py --dataroot your_davis_data_dir --model_type "dinov2" --output_dir your_output_dir --imsize 224 --upsampler_path=your_upsampler_path
```

Then run the following to get evaluation results:

```bash
python davis2017-evaluation/evaluation_method.py --davis_path /your_davis_data_dir --task semi-supervised --results_path your_output_dir/davis_vidseg_224 --imsize 224
```

### Others
For interactive segmentation, please check out [iSegProbe](https://github.com/havrylovv/iSegProbe).

For open-vocabulary segmentation, please check out [ProxyCLIP](https://github.com/mc-lan/ProxyCLIP).

For depth and normal estimation, please check out [Probe3D](https://github.com/mbanani/probe3d).


## Training LoftUp Upsamplers

This repository contains training scripts for training LoftUp upsamplers. The training is done in two stages:

### Stage 1: Basic Feature Upsampling

Stage 1 training (`train_loftup_stage1.py`) trains upsamplers to convert low-resolution features to high-resolution features using reconstruction loss.

**Example training command:**
```bash
python train_loftup_stage1.py ++dataset="sa1b" ++epochs=1 ++batch_size=2 ++num_gpus=4 ++model_type="dinov2" ++pytorch_data_dir='datasets' ++upsampler_type="loftup" ++sam_mask_alpha=0.8 ++load_size=224 ++upsample_size=224 ++tv_weight=0.001 ++clamp_featup=True
```

### Stage 2: High-Resolution Supervision

Stage 2 training (`train_loftup_stage2.py`) fine-tunes the Stage 1 upsampler with high-resolution supervision for improved quality.

**Example training command:**
```bash
python train_loftup_stage2.py ++dataset="sa1b" ++epochs=1 ++hr_res=896 ++batch_size=2 ++consistency_method="bilinear" ++model_type="dinov2" ++num_gpus=4 ++affinity_loss=True ++pytorch_data_dir='datasets' ++pretrained_upsampler="path/to/stage1_checkpoint.ckpt" ++upsampler_type="loftup" ++sam_mask_hr_alpha=0.5 ++sam_mask_reg=0.0 ++lr=1e-3 ++use_featup=False ++aug_size ++n_jitters=2
```

### Configuration

Both training scripts use Hydra for configuration management. Configuration files are located in `configs/`:
- `configs/train_loftup_stage1.yaml` - Stage 1 configuration
- `configs/train_loftup_stage2.yaml` - Stage 2 configuration

**Key configuration parameters:**
- `model_type`: Feature extractor type (e.g., "dinov2", "clip")
- `upsampler_type`: Type of upsampler to train (e.g., "loftup")
- `batch_size`: Training batch size
- `epochs`: Number of training epochs
- `lr`: Learning rate
- `load_size`: Input image size for feature extraction
- `upsample_size`: Target size for upsampled features
- `n_jitters`: Number of jittering augmentations per training step
- `tv_weight`: Weight for total variation loss
- `sam_mask_alpha`: Weight for SAM mask adjustment (Stage 1)
- `sam_mask_hr_alpha`: Weight for SAM mask adjustment (Stage 2)


For more details, see the configuration files in `configs/` and the training scripts themselves.

## Citation
If you find our work helpful, please cite:

```
@misc{huang2025loftuplearningcoordinatebasedfeature,
      title={LoftUp: Learning a Coordinate-Based Feature Upsampler for Vision Foundation Models}, 
      author={Haiwen Huang and Anpei Chen and Volodymyr Havrylov and Andreas Geiger and Dan Zhang},
      year={2025},
      eprint={2504.14032},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.14032}, 
}
```
