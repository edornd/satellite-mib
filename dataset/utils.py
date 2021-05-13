import torch
import numpy as np
from tqdm import tqdm


def group_images(dataset, labels):
    # Group images based on the label in LABELS (using labels not reordered)
    idxs = {lab: [] for lab in labels}

    labels_cum = labels + [0, 255]
    for i in range(len(dataset)):
        cls = np.unique(np.array(dataset[i][1]))
        if all(x in labels_cum for x in cls):
            for x in cls:
                if x in labels:
                    idxs[x].append(i)
    return idxs


def filter_images(dataset, labels, labels_old=None, overlap=True):
    # Filter images without any label in LABELS (using labels not reordered)
    idxs = []

    if 0 in labels:
        labels = list(filter(lambda v: v != 0, labels))

    print(f"Filtering images...")
    if labels_old is None:
        labels_old = []
    all_labels = set(labels + labels_old + [0, 255])

    if overlap:
        fil = lambda c: any(x in labels for x in c)
    else:
        fil = lambda c: any(x in labels for x in c) and all(x in all_labels for x in c)

    for i, (_, mask) in tqdm(enumerate(dataset)):
        category = np.unique(np.array(mask))
        if fil(category):
            idxs.append(i)
    print(f"Subset length: {len(idxs)}/{len(dataset)}")
    return idxs


class Subset(torch.utils.data.Dataset):
    """
    Subset of a dataset at specified indices.
    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        transform (callable): way to transform the images and the targets
        target_transform(callable): way to transform the target labels
    """

    def __init__(self, dataset, indices, transform=None, target_transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        sample, target = self.dataset[self.indices[idx]]

        if self.transform is not None:
            sample, target = self.transform(sample, target)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.indices)


class ISPRSSubset(Subset):

    def __getitem__(self, idx):
        sample, target = self.dataset[self.indices[idx]]

        if self.transform is not None:
            out_dict = self.transform(image=sample, mask=target)
            sample = out_dict.get("image")
            target = out_dict.get("mask")

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class MaskLabels:
    """
    Use this class to mask labels that you don't want in your dataset.
    Arguments:
    labels_to_keep (list): The list of labels to keep in the target images
    mask_value (int): The value to replace ignored values (def: 0)
    """
    def __init__(self, labels_to_keep, mask_value=0):
        self.labels = labels_to_keep
        self.value = torch.tensor(mask_value, dtype=torch.uint8)

    def __call__(self, sample):
        # sample must be a tensor
        assert isinstance(sample, torch.Tensor), "Sample must be a tensor"

        sample.apply_(lambda t: t.apply_(lambda x: x if x in self.labels else self.value))

        return sample
