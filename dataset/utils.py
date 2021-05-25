import torch
import numpy as np
from tqdm import tqdm
from typing import List


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

def filter_with_overlap(current_labels: List[int], step_labels: List[int], *args, **kwargs) -> bool:
    """Returns whether the current image must be maintained or discarded for the given step,
    based on the labels on the current image and the labels required at the step.

    Args:
        current_labels (List[int]): indices of the labels present in the current image
        step_labels (List[int]): indices of the labels needed at the current step

    Returns:
        bool: true if any of the current labels are present, false otherwise
    """
    return any(x in step_labels for x in current_labels)

def filter_without_overlap(current_labels: List[int], step_labels: List[int], previous_labels: List[int]) -> bool:
    """Filters out any image that contains data with no labels belonging to the current step, including
    those images that contain future labels (potentially dangerous if an image contains more or less every label).add()

    Args:
        current_labels (List[int]): indices of unique labels for the current image
        step_labels (List[int]): indices of labels for the step T
        previous_labels (List[int]): indices of labels from steps 1 .. T - 1 + labels from step T + [0, 255]

    Returns:
        bool: true whether the image must be kept, false otherwise
    """
    return any(x in step_labels for x in current_labels) and all(x in previous_labels for x in current_labels)

def filter_images(dataset, labels, labels_old=None, overlap=True):
    # Filter images without any label in LABELS (using labels not reordered)
    idxs = []

    if 0 in labels:
        labels = list(filter(lambda v: v != 0, labels))

    print(f"Filtering images...")
    if labels_old is None:
        labels_old = []
    all_labels = set(labels + labels_old + [0, 255])

    filter_fn = filter_with_overlap if overlap else filter_without_overlap

    for i, (_, mask) in tqdm(enumerate(dataset)):
        unique_labels = np.unique(np.array(mask))
        if filter_fn(unique_labels, labels, all_labels):
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
