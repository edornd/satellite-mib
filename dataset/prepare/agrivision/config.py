import os

from src.data import DatasetInfo


class AgriVisionInfo(DatasetInfo):

    def __init__(self) -> None:
        super().__init__()
        self.index2label: dict = {
            0: "background",
            1: "cloud_shadow",
            2: "double_plant",
            3: "planter_skip",
            4: "standing_water",
            5: "waterway",
            6: "weed_cluster",
        }
        self.label2index: dict = {v: k for k, v in self.index2label.items()}
        self.num_classes: int = 6  # not considering BG
        self.image_dims: tuple = (512, 512)
        self.image_dir: str = "images"
        self.rgb_dir: str = os.path.join(self.image_dir, "rgb")
        self.nir_dir: str = os.path.join(self.image_dir, "nir")
        self.masks_dir = "masks"
        self.boundaries_dir = "boundaries"
        self.labels_dir = "labels"
