import os
import numpy as np
import typer as tpr
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from src.utils.template import makedirs
from src.data.agrivision.config import AgriVisionInfo


def main(source: Path = tpr.Option(..., "--source", "-s", help="Path to the root folder of Agriculture-Vision"),
         destination: Path = tpr.Option(..., "--dest", "-d", help="Destination folder where results will be stored"),
         target: Path = tpr.Option("train", help="Folder name of the set to be processed (train, val)"),
         check: bool = tpr.Option(False, help="Whether generating the ground truth or simply check its ")):
    # instantiate the class containing setup information about the dataset
    cfg = AgriVisionInfo()
    # only check for consistency
    if check:
        assert os.path.exists(destination), "Destination path to be validated does not exist"
        filenames = [f for f in os.listdir(destination) if os.path.isfile(os.path.join(destination, f))]
        for file in tqdm(filenames):
            image = np.array(Image.open(os.path.join(destination, file)))
            assert image.shape == (512, 512), f"Wrong shape for mask {file}: {image.shape}"
            assert ((image < 7) | (image == 255)).all(), f"Mask {file} contains out-of-range values"
        tpr.echo("Everything ok!")
    # otherwise, iterate masks and generate a single ground truth by merging together the binary components
    # this only needs to be run once
    else:
        makedirs(destination)
        base_path = os.path.join(source, target, cfg.rgb_dir)
        filenames = [os.path.basename(f) for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]

        for sample in tqdm(filenames):
            name = os.path.splitext(sample)[0]  # exclude the extension
            final_mask = np.zeros(cfg.image_dims, dtype=int)

            for index, label in cfg.index2label.items():
                if label == "background":
                    continue
                path = os.path.join(source, target, cfg.labels_dir, label, f"{name}.png")
                pixels = np.array(Image.open(path), dtype=int) // 255 * index
                remaining = final_mask == 0
                final_mask[remaining] = pixels[remaining]

            for folder in (cfg.boundaries_dir, cfg.masks_dir):
                path = os.path.join(source, target, folder, f"{name}.png")
                excluded = np.array(Image.open(path), dtype=int) // 255
                final_mask[excluded == 0] = 255

            image = Image.fromarray((final_mask).astype(np.uint8))
            image.save(os.path.join(destination, f"{name}.png"), format="PNG")


if __name__ == "__main__":
    tpr.run(main)
