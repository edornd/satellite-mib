import os
import glob
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import tifffile as tif
from torch import Tensor
from torch import distributed
from torch.utils.data import Dataset as DatasetBase
import torchvision as tv

from .utils import ISPRSSubset, filter_images

classes = {
    0: "surface",
    1: "building",
    2: "low_vegetation",
    3: "tree",
    4: "car",
    5: "clutter",
}


class ISPRSDataset(DatasetBase):

    def __init__(self,
                 path: Path,
                 city: str = "potsdam",
                 subset: str = "train",
                 channels: str = "rgb",
                 channel_count: int = 3,
                 include_dsm: bool = False,
                 transform: Callable = None) -> None:
        super().__init__()
        self.channels = channels
        self.channel_count = channel_count
        self.include_dsm = include_dsm
        self.transform = transform
        path = os.path.join(path, city, subset)
        self.image_files = sorted(glob.glob(os.path.join(path, self.image_naming())))
        self.label_files = sorted(glob.glob(os.path.join(path, self.label_naming())))
        assert len(self.image_files) == len(self.label_files), \
            f"Length mismatch between tiles and masks: {len(self.image_files)} != {len(self.label_files)}"
        # check matching sub-tiles
        for image, mask in zip(self.image_files, self.label_files):
            image_tile = "_".join(os.path.basename(image).split("_")[:-1])
            mask_tile = "_".join(os.path.basename(mask).split("_")[:-1])
            assert image_tile == mask_tile, f"image: {image_tile} != mask: {mask_tile}"
        # add the optional digital surface map
        if include_dsm:
            self.dsm_files = sorted(glob.glob(os.path.join(path, self.dsm_naming())))
            assert len(self.image_files) == len(self.dsm_files), "Length mismatch between tiles and DSMs"
            for image, dsm in zip(self.image_files, self.dsm_files):
                image_tile = "_".join(os.path.basename(image).split("_")[:-1])
                dsm_tile = "_".join(os.path.basename(dsm).split("_")[:-1])
                assert image_tile == dsm_tile, f"image: {image_tile} != mask: {dsm_tile}"

    def image_naming(self) -> str:
        return f"*_{self.channels}.tif"

    def label_naming(self) -> str:
        return "*_mask.tif"

    def dsm_naming(self) -> str:
        return "*_dsm.tif"

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """Get the image/label pair, with optional augmentations and preprocessing steps.
        Augmentations should be provided for a training dataset, while preprocessing should contain
        the transforms required in both cases (normalizations, ToTensor, ...)

        :param index:   integer pointing to the tile
        :type index:    int
        :return:        image, mask tuple
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        image = tif.imread(self.image_files[index]).astype(np.float32)
        mask = tif.imread(self.label_files[index]).astype(np.uint8)
        mask[mask == 5] = 0
        image = image[:,:,:self.channel_count]
        # add Digital surface map as extra channel to the image
        if self.include_dsm:
            dsm = tif.imread(self.dsm_files[index]).astype(np.float32)
            image = np.dstack((image, dsm))
        # preprocess if required
        if self.transform is not None:
            pair = self.transform(image=image, mask=mask)
            image = pair.get("image")
            mask = pair.get("mask")
        return image, mask

    def __len__(self) -> int:
        return len(self.image_files)


class PotsdamDataset(ISPRSDataset):

    def __init__(self, path: Path, subset: str, include_dsm: bool = False, transform: Callable = None, channels: int = 3) -> None:
        super().__init__(path,
                         city="potsdam",
                         subset=subset,
                         channels="rgbir",
                         channel_count=channels,
                         include_dsm=include_dsm,
                         transform=transform)


class VaihingenDataset(ISPRSDataset):

    def __init__(self, path: Path, subset: str, include_dsm: bool = False, transform: Callable = None, channels: int = 3) -> None:
        super().__init__(path,
                         city="vaihingen",
                         subset=subset,
                         channels="rgb",
                         channel_count=channels,
                         include_dsm=include_dsm,
                         transform=transform)


class ISPRSDatasetIncremental(DatasetBase):
    """Incremental version of the ISPRS dataset
    """

    def __init__(self,
                 root: str,
                 city: str = "potsdam",
                 train: bool = True,
                 transform: Callable = None,
                 labels: List[int] = None,
                 labels_old: List[int] = None,
                 idxs_path: str = None,
                 masking: bool = True,
                 overlap: bool = True,
                 channels: int = 3):
        subset = "train" if train else "test"
        if city == "potsdam":
            full_set = PotsdamDataset(path=root, subset=subset, transform=None, channels=channels)
        else:
            full_set = VaihingenDataset(path=root, subset=subset, transform=None, channels=channels)
        self.labels = []
        self.labels_old = []
        # if we have labels, then we expect an ICL setup
        if labels is not None:
            labels_old = labels_old or []
            # remove void label from lists
            self.__strip_zero(labels)
            self.__strip_zero(labels_old)

            assert not any(l in labels_old for l in labels), "labels and labels_old must be disjoint sets"

            #? not clear: void always first?
            self.labels = [0] + labels
            self.labels_old = [0] + labels_old
            self.order = [0] + labels_old + labels

            # take index of images with at least one class in labels and all classes in labels+labels_old+[0,255]
            if idxs_path is not None and os.path.exists(idxs_path):
                idxs = np.load(idxs_path).tolist()
            else:
                idxs = filter_images(full_set, labels, labels_old, overlap=overlap)
                if idxs_path is not None and distributed.get_rank() == 0:
                    np.save(idxs_path, np.array(idxs, dtype=int))

            if train:
                masking_value = 0
            else:
                masking_value = 255

            #? Again not super clear
            self.inverted_order = {label: self.order.index(label) for label in self.order}
            self.inverted_order[255] = masking_value

            reorder_transform = tv.transforms.Lambda(lambda t: t.apply_(lambda x: self.inverted_order[x]
                                                                        if x in self.inverted_order else masking_value))

            if masking:
                tmp_labels = self.labels + [255]
                target_transform = tv.transforms.Lambda(lambda t: t.apply_(lambda x: self.inverted_order[x]
                                                                           if x in tmp_labels else masking_value))
            else:
                target_transform = reorder_transform

            # make the subset of the dataset
            self.dataset = ISPRSSubset(full_set, idxs, transform, target_transform)
        else:
            self.dataset = full_set

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def __strip_zero(labels):
        while 0 in labels:
            labels.remove(0)


class PotsdamIncremental(ISPRSDatasetIncremental):

    def __init__(self, root: str, train: bool, transform: Callable, labels: List[int], labels_old: List[int],
                 idxs_path: str, masking: bool, overlap: bool, channels: int = 3):
        super().__init__(root,
                         city="potsdam",
                         train=train,
                         transform=transform,
                         labels=labels,
                         labels_old=labels_old,
                         idxs_path=idxs_path,
                         masking=masking,
                         overlap=overlap,
                         channels=channels)


class VaihingenIncremental(ISPRSDatasetIncremental):

    def __init__(self, root: str, train: bool, transform: Callable, labels: List[int], labels_old: List[int],
                 idxs_path: str, masking: bool, overlap: bool, channels: int = 3):
        super().__init__(root,
                         city="vaihingen",
                         train=train,
                         transform=transform,
                         labels=labels,
                         labels_old=labels_old,
                         idxs_path=idxs_path,
                         masking=masking,
                         overlap=overlap,
                         channels=channels)
