from os.path import exists, join
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import tifffile as tif
import torch
import typer as tpr
from dataset.prepare import DatasetSplits
from dataset.prepare.isprs.config import (ISPRSChannels, ISPRSDatasetInfo, ISPRSDatasets, PotsdamInfo, VaihingenInfo)
from dataset.prepare.utils import makedirs, tile_overlapped
from PIL import Image
from tqdm import tqdm

# keep tiles until we go over 75% of missing data
VALID_PERCENT_THRESHOLD = 0.75


def convert_mask(image: np.ndarray, lut: Dict[tuple, int]) -> np.ndarray:
    """Converts a given RGB image containing labels in channels-last format (h, w, c)
    into a greyscale mask where each index indicates a given class.

    :param image: RGB input image with dimensions [height, width, channels]
    :type image: np.ndarray
    :param lut: look-up table containing the associations color -> index
    :type lut: Dict[tuple, int]
    :return: greyscale image with size [height, width] containing the mapped label indices
    :rtype: np.ndarray
    """
    result = np.zeros(image.shape[:2])
    for color, index in lut.items():
        result[np.all(image == color, axis=-1)] = index
    return result.astype(np.uint8)


def prefix_potsdam(area_id: Tuple[int, int]) -> str:
    """Generates the prefix to identify each main tile in the Potsdam dataset.

    :param area_id: tuple containing the coordinates of the tile
    :type area_id: Tuple[int, int]
    :return: string containing the coordinates separated by underscore
    :rtype: str
    """
    x, y = area_id
    return f"{x}_{y}"


def prefix_vaihingen(area_id: int) -> str:
    """Generates the prefix to identify each main tile in the Vaihingen dataset.

    :param area_id: integer defining the tile number
    :type area_id: int
    :return: string named area[area_id]
    :rtype: str
    """
    return f"area{area_id}"


def filenames_potsdam(area_id: Tuple[int, int],
                      channels: ISPRSChannels,
                      with_boundaries: bool = True) -> Tuple[str, str, str]:
    """Generates the filenames for the image, label and DSM files in the potsdam dataset.

    :param area_id: tuple containing x and y of the main tile
    :type area_id: Tuple[int, int]
    :param channels: enum with the possible channel combinations in the dataset
    :type channels: ISPRSChannels
    :param with_boundaries: use the boundary or noBoundary variants, defaults to True
    :type with_boundaries: bool, optional
    :return: image, label and DSM filenames
    :rtype: Tuple[str, str, str]
    """
    x, y = area_id
    label_suffix = "" if with_boundaries else "_noBoundary"
    image_filename = f"top_potsdam_{x}_{y}_{channels.name}.tif"
    label_filename = f"top_potsdam_{x}_{y}_label{label_suffix}.tif"
    dsurf_filename = f"dsm_potsdam_{x:02d}_{y:02d}_normalized_lastools.jpg"
    return image_filename, label_filename, dsurf_filename


def filenames_vaihingen(area_id: int, channels: ISPRSChannels, with_boundaries: bool = True) -> Tuple[str, str, str]:
    """Generates the filesnames for the image, label and DSM files in the Vaihingen dataset.

    :param area_id: ID of the main tile
    :type area_id: int
    :param channels: enum with the channel combination (only RGB in this case)
    :type channels: ISPRSChannels
    :param with_boundaries: use the boundary or the noBoundary variant, defaults to True
    :type with_boundaries: bool, optional
    :return: image, label and DSM filenames
    :rtype: Tuple[str, str, str]
    """
    label_suffix = "" if with_boundaries else "_noBoundary"
    image_filename = f"top_mosaic_09cm_area{area_id}.tif"
    label_filename = f"top_mosaic_09cm_area{area_id}{label_suffix}.tif"
    dsurf_filename = f"dsm_09cm_matching_area{area_id}.tif"
    return image_filename, label_filename, dsurf_filename


def prepare(config: ISPRSDatasetInfo, subset: DatasetSplits, naming_fn: Callable, prefix_fn: Callable,
            source_root: Path, destination_root: Path, channels: ISPRSChannels, target_size: int, overlap: int,
            exclude_index: int, with_boundaries: bool, normalize: bool) -> None:
    """Iterates over one of the two main datasets (Vaihingen or Potsdam) and over one of the three main splits
    (train, validation ,test), generating tiles with fixed size [target_size x target_size], ready to be provided
    as input to a segmentation model.

    :param config: instance of a DatasetInfo subclass, containing info about folders and labels structure
    :type config: ISPRSDatasetInfo
    :param subset: which subset, train, test or validation
    :type subset: DatasetSplits
    :param naming_fn: function required to generate filenames
    :type naming_fn: Callable
    :param prefix_fn: function required to generate the ad hoc prefix
    :type prefix_fn: Callable
    :param source_root: pointer to the directory containing one of the two datasets
    :type source_root: Path
    :param destination_root: pointer to the destination directory, that will contain train/test/valid folders
    :type destination_root: Path
    :param channels: enum containing the combination of channels
    :type channels: ISPRSChannels
    :param target_size: fixed size for the sub-tiles
    :type target_size: int
    :param overlap: how many pixels should be overlapped between neighboring tiles
    :type overlap: int
    :param exclude_index: index to use to delineate parts external to the original image (to be discarded)
    :type exclude_index: int
    :param with_boundaries: whether to use the boundary or noBoundary variant
    :type with_boundaries: bool
    :param normalize: whether to normalize image channels between 0 and 1
    """
    images_root = join(source_root, channels.value)  # suppose enum value is folder name
    labels_root = join(source_root, config.labels_dir)
    dsurfm_root = join(source_root, config.dsm_dir)
    destination = join(destination_root, subset.value)
    makedirs(destination)

    for block_id in tqdm(config.tiles_dict[subset]):
        prefix = prefix_fn(block_id)
        img_name, lab_name, dsm_name = naming_fn(block_id, channels, with_boundaries)
        image_path = join(images_root, img_name)
        label_path = join(labels_root, lab_name)
        dsurf_path = join(dsurfm_root, dsm_name)
        assert exists(image_path), f"Missing {channels.value} file for area {prefix})"
        assert exists(label_path), f"Missing mask file for area {prefix})"
        assert exists(dsurf_path), f"Missing DSM file for area {prefix})"

        # read images
        image = tif.imread(image_path)
        label = np.array(Image.open(label_path))
        dsm = np.array(Image.open(dsurf_path))
        # normalization, if requested
        if normalize:
            image = image.astype(np.float32) / 255.0
            dsm = (dsm.astype(np.float32) - config.dsm_min) / (config.dsm_max - config.dsm_min)
        # check height and width dimensions: no assert on 2nd pass since a DSM has a pixel row missing somehow
        assert image.shape[:2] == label.shape[:2], \
            f"Shape mismatch between image ({image.shape}) and labels ({label.shape})"
        if not image.shape[:2] == dsm.shape[:2]:
            height, width = image.shape[:2]
            tpr.echo(f"WARNING - shape mismatch between image ({image.shape}) and DSM  ({dsm.shape})")
            tmp = np.zeros((height, width))
            tmp[:dsm.shape[0], :dsm.shape[1]] = dsm
            dsm = tmp

        # divide in tiles with same dimension and dynamic overlap
        tiles_h = None
        tiles_w = None
        if overlap > 0:
            target_stride = target_size - overlap
            tiles_h = int(image.shape[0] / target_stride)
            tiles_w = int(image.shape[1] / target_stride)
        image_patches = tile_overlapped(image=image, tile_size=target_size, tile_cols=tiles_h, tile_rows=tiles_w)
        label_patches = tile_overlapped(image=label, tile_size=target_size, tile_cols=tiles_h, tile_rows=tiles_w)
        dsm_patches = tile_overlapped(image=dsm, tile_size=target_size, tile_cols=tiles_h, tile_rows=tiles_w)

        # iterate tiles and save images to the target folder
        rows, cols = image_patches.shape[:2]
        for x1 in range(rows):
            for y1 in range(cols):
                # generate file names programmatically
                tile_name = f"{prefix}_{x1}-{y1}"
                out_img_path = join(destination, f"{tile_name}_{channels.value}.tif")
                out_lbl_path = join(destination, f"{tile_name}_mask.tif")
                out_dsm_path = join(destination, f"{tile_name}_dsm.tif")
                # extract a single tile image, reordering channels
                image_tensor = image_patches[x1, y1]
                label_tensor = label_patches[x1, y1]
                dsm_tensor = dsm_patches[x1, y1].squeeze(-1)  # single channel, remove third dimension
                label_tensor = convert_mask(label_tensor, config.color2index)
                assert image_tensor.shape[:2] == label_tensor.shape[:2] == dsm_tensor.shape[:2]
                # save to file
                invalid_pixels = np.sum(label_tensor == config.color2index[(0, 0, 0)])
                total_pixels = label_tensor.size
                if (invalid_pixels / float(total_pixels)) < VALID_PERCENT_THRESHOLD:
                    tif.imwrite(out_img_path, image_tensor)
                    Image.fromarray(label_tensor).save(out_lbl_path, format="TIFF")
                    Image.fromarray(dsm_tensor).save(out_dsm_path, format="TIFF")
                else:
                    tpr.echo(f"WARNING: excluding tile {tile_name}: {invalid_pixels} "
                             f"invalid pixels out of {total_pixels}")


def init_tensors(channel_count: int):
    channel_maxs = np.ones(channel_count) * np.finfo(np.float).min
    channel_mins = np.ones(channel_count) * np.finfo(np.float).max
    channel_mean = np.zeros(channel_count)
    channel_stds = np.zeros(channel_count)
    return channel_maxs, channel_mins, channel_mean, channel_stds


def statistics(config: ISPRSDatasetInfo, naming_fn: Callable, prefix_fn: Callable, source: Path,
               channels: ISPRSChannels, with_boundaries: bool) -> None:
    """Computes the statistics on the current dataset.

    Args:
        config (ISPRSDatasetInfo): configuration of the specific dataset to use
        naming_fn (Callable): function required to retrieve the naming pattern
        prefix_fn (Callable): function returning the prefix pattern of each image
        source (Path): root path to the dataset
        channels (ISPRSChannels): enum defining which channels to compute the stats on
        with_boundaries (bool): whether to use the boundary version of no boundary
    """
    images_root = join(source, channels.value)  # suppose enum value is folder name
    dsurfm_root = join(source, config.dsm_dir)
    pixel_count = 0
    ch_max = None
    ch_min = None
    ch_avg = None
    ch_std = None
    # iterate on the large tiles
    tpr.echo("Computing  min, max and mean...")
    for block_id in tqdm(config.tiles):
        prefix = prefix_fn(block_id)
        img_name, _, dsm_name = naming_fn(block_id, channels, with_boundaries)
        image_path = join(images_root, img_name)
        dsurf_path = join(dsurfm_root, dsm_name)
        assert exists(image_path), f"Missing {channels.value} file for area {prefix})"
        assert exists(dsurf_path), f"Missing DSM file for area {prefix})"
        # read images
        image = tif.imread(image_path)
        dsm = np.array(Image.open(dsurf_path))
        dsm = dsm[..., np.newaxis]
        # check height and width dimensions: no assert since a DSM has a pixel row missing somehow
        if not image.shape[:2] == dsm.shape[:2]:
            tpr.echo(f"WARNING - shape mismatch: {image.shape}, {dsm.shape}")
        # initialize vectors if it's the first iteration
        if ch_max is None:
            ch_max, ch_min, ch_avg, ch_std = init_tensors(image.shape[-1] + dsm.shape[-1])
        image = image.reshape((-1, image.shape[-1]))
        dsm = dsm.reshape((-1, dsm.shape[-1]))
        pixel_count += image.shape[0]
        ch_max = np.maximum(ch_max, np.concatenate((image.max(axis=0), dsm.max(axis=0)), axis=-1))
        ch_min = np.minimum(ch_min, np.concatenate((image.min(axis=0), dsm.min(axis=0)), axis=-1))
        ch_avg += np.concatenate((image.sum(axis=0), dsm.sum(axis=0)), axis=-1)
    ch_avg /= float(pixel_count)
    # second pass to compute standard deviation
    tpr.echo("Computing standard deviation...")
    for block_id in tqdm(config.tiles):
        img_name, _, dsm_name = naming_fn(block_id, channels, with_boundaries)
        image = tif.imread(join(images_root, img_name))
        dsm = np.array(Image.open(join(dsurfm_root, dsm_name)))
        dsm = dsm[..., np.newaxis]
        # compute
        img_channels = image.shape[-1]
        dsm_channels = dsm.shape[-1]
        image = image.reshape((-1, img_channels))
        dsm = dsm.reshape((-1, dsm_channels))
        image_std = ((image - ch_avg[:img_channels])**2).sum(axis=0) / float(image.shape[0])
        dsm_std = ((dsm - ch_avg[img_channels:])**2).sum(axis=0) / float(image.shape[0])
        ch_std += np.concatenate((image_std, dsm_std), axis=-1)
    ch_std = np.sqrt(ch_std / len(config.tiles))
    # print stats
    print("channel-wise max: ", ch_max)
    print("channel-wise min: ", ch_min)
    print("channel-wise avg: ", ch_avg)
    print("channel-wise std: ", ch_std)
    print("normalized avg: ", (ch_avg - ch_min) / (ch_max - ch_min))
    print("normalized std: ", ch_std / (ch_max - ch_min))


def main(dataset: ISPRSDatasets = tpr.Argument(..., help="ISPRS dataset to be processed"),
         source: Path = tpr.Option(..., "--source", "-s", help="Path to the root folder of Potsdam or Vaihingen"),
         destination: Path = tpr.Option(..., "--dest", "-d", help="Destination folder where results will be stored"),
         sets: Optional[List[DatasetSplits]] = tpr.Option(None, "--sets", "-x", help="Which sets to process"),
         channels: ISPRSChannels = tpr.Option(ISPRSChannels.RGB,
                                              "--channels",
                                              "-c",
                                              help="Which channel combination to process"),
         target_size: int = tpr.Option(512, help="Output size for the smaller tiles"),
         overlap: int = tpr.Option(64, help="How many pixels should be overlapped between strides"),
         exclude_index: int = tpr.Option(0, help="Value to be placed on pixels outside boundaries (to be exluded)"),
         use_boundary: bool = tpr.Option(True, help="Whether to use labels with or without boundaries"),
         normalize: bool = tpr.Option(False, help="Whether to normalize image channels using min-max"),
         stats: bool = tpr.Option(False, help="Just compute some dataset statistics (mean, var)")):

    if dataset == ISPRSDatasets.potsdam:
        config = PotsdamInfo()
        naming_function = filenames_potsdam
        prefix_function = prefix_potsdam
    elif dataset == ISPRSDatasets.vaihingen:
        config = VaihingenInfo()
        naming_function = filenames_vaihingen
        prefix_function = prefix_vaihingen

    if stats:
        tpr.echo(f"Computing dataset statistics for {dataset}")
        statistics(config, naming_function, prefix_function, source, channels, use_boundary)
    else:
        if not sets:
            tpr.echo("WARNING: No dataset split selected, assuming all three (train, test, validation)")
            sets = [v for v in DatasetSplits]
        for subset in sets:
            tpr.echo(f"Processing {dataset.value}'s {subset.value} set...")
            prepare(config, subset, naming_function, prefix_function, source, destination, channels, target_size,
                    overlap, exclude_index, use_boundary, normalize)


if __name__ == "__main__":
    tpr.run(main)
