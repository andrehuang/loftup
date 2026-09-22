from torch.utils.data import Dataset
from .COCO import Coco
from .cityscapes import Cityscapes
from .sa1b import SA1B
# from featup.datasets.nyu_probe import NYU

class SlicedDataset(Dataset):
    def __init__(self, ds, start, end):
        self.ds = ds
        self.start = max(0, start)
        self.end = min(len(ds), end)

    def __getitem__(self, index):
        if index >= self.__len__():
            raise StopIteration

        return self.ds[self.start + index]

    def __len__(self):
        return self.end - self.start



class SingleImageDataset(Dataset):
    def __init__(self, i, ds, l=None):
        self.ds = ds
        self.i = i
        self.l = len(self.ds) if l is None else l

    def __len__(self):
        return self.l

    def __getitem__(self, item):
        return self.ds[self.i]


def get_dataset(dataroot, name, split, transform, target_transform, include_labels, sample_size=None):
    if name == 'cocostuff':
        return Coco(dataroot, split, transform, target_transform, include_labels=include_labels)
    elif name == 'cityscapes':
        return Cityscapes(dataroot, split, transform, target_transform, include_labels=include_labels)
    elif name == "sa1b":
        return SA1B(
            root=dataroot,
            split=split,
            transform=transform,
            target_transform=target_transform,
            sample_size=sample_size,
        )
    else:
        raise ValueError(f"Unknown dataset {name}")
