import random
from os.path import join

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import glob
import torchvision.transforms.functional as F
from pycocotools import mask as mask_utils
import json

def rle_to_binary_mask(rle):
    """
    Converts an RLE annotation to a binary mask.
    
    Arguments:
        rle (dict): A dictionary with 'size' and 'counts' keys.
                    Example: {'size': [height, width], 'counts': 'RLE string...'}
    
    Returns:
        np.ndarray: A binary mask of shape (height, width).
    """
    binary_mask = mask_utils.decode(rle)
    return binary_mask

def bit_get(val, idx):
    """Gets the bit value.
    Args:
      val: Input value, int or numpy int array.
      idx: Which bit of the input val.
    Returns:
      The "idx"-th bit of input val.
    """
    return (val >> idx) & 1


class SA1B(Dataset):
    def __init__(self,
                 root,
                 split,
                 transform,
                 target_transform,
                 max_mask=150,
                 sample_size=None):
        super(SA1B, self).__init__()
        self.split = split
        self.root = join(root, "sa1b")
        self.transform = transform
        self.label_transform = target_transform
        self.max_mask = max_mask

        assert self.split in ["train"]


        self.image_files = []
        self.label_files = []
        for img in sorted(glob.glob(join(self.root, "*.jpg"))):
            self.image_files.append(img)
            label = img.replace('jpg', 'json')
            self.label_files.append(label)
        
        # Optionally keep only the first `sample_size` images (None = use all).
        if sample_size is not None:
            self.image_files = self.image_files[:sample_size]
            self.label_files = self.label_files[:sample_size]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        label_path = self.label_files[index]
        seed = np.random.randint(2147483647)
        batch = {}

        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transform(Image.open(image_path).convert("RGB"))
        annotation = json.load(open(label_path))['annotations']
        binary_masks = [rle_to_binary_mask(ann['segmentation']) for ann in annotation] # numpy array    

        if self.label_transform is not None:
            transformed_masks = torch.stack([
                self.label_transform(mask) for mask in binary_masks
                # if torch.any(self.label_transform(mask) > 0)  # Keep masks with non-zero values
            ])
            ## For collation, we need to pad the masks to the same size
            ## We will pad with -1, which is the ignore label
            transformed_masks = transformed_masks.squeeze(1) # (num_masks, h, w)
            if transformed_masks.size(0) < self.max_mask:
                pad_size = self.max_mask - transformed_masks.size(0)
                pad = torch.full((pad_size, transformed_masks.size(1), transformed_masks.size(2)), -1, dtype=torch.int32)
                transformed_masks = torch.cat([transformed_masks, pad], dim=0)
            else:
                transformed_masks = transformed_masks[:self.max_mask]
        else:
            transformed_masks = []

        batch["img"] = img # (3, h, w)
        batch["img_path"] = image_path
        batch["label_path"] = label_path    
        batch["label"] = transformed_masks # (num_masks, h, w)           

        return batch