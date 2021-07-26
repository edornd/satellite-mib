from enum import Enum

from dataset.prepare import DatasetInfo, DatasetSplits


class ISPRSDatasets(str, Enum):
    potsdam = "potsdam"
    vaihingen = "vaihingen"


class ISPRSChannels(str, Enum):
    RGB = "rgb"
    RGBIR = "rgbir"
    IRRG = "irrg"


ISPRSColorPalette = {
    0: (255, 255, 255),
    1: (0, 0, 255),
    2: (0, 255, 255),
    3: (0, 255, 0),
    4: (255, 255, 0),
    5: (255, 0, 0),
    255: (0, 0, 0)
}


class ISPRSDatasetInfo(DatasetInfo):

    def __init__(self) -> None:
        super().__init__()
        self.index2label: dict = {
            0: "impervious_surfaces",
            1: "building",
            2: "low_vegetation",
            3: "tree",
            4: "car",
            5: "clutter",
            255: "ignored"
        }
        self.index2color: dict = ISPRSColorPalette
        self.label2index: dict = {v: k for k, v in self.index2label.items()}
        self.color2index: dict = {v: k for k, v in self.index2color.items()}
        self.num_classes = len(self.index2label)


class PotsdamInfo(ISPRSDatasetInfo):

    def __init__(self) -> None:
        super().__init__()
        self.train_tiles: list = [(2, 10), (3, 10), (3, 11), (3, 12), (4, 11), (4, 12), (5, 10), (5, 12), (6, 10),
                                  (6, 11), (6, 12), (6, 8), (6, 9), (7, 11), (7, 12), (7, 7), (7, 9), (2, 11), (2, 12),
                                  (4, 10), (5, 11), (6, 7), (7, 10), (7, 8)]
        #! No valid tiles since the run script selects a validation set from the training set
        self.valid_tiles: list = []
        self.test_tiles: list = [(2, 13), (2, 14), (3, 13), (3, 14), (4, 13), (4, 14), (4, 15), (5, 13), (5, 14),
                                 (5, 15), (6, 13), (6, 14), (6, 15), (7, 13)]
        self.tiles: list = self.train_tiles + self.valid_tiles + self.test_tiles
        self.tiles_dict: dict = {
            DatasetSplits.train: self.train_tiles,
            DatasetSplits.valid: self.valid_tiles,
            DatasetSplits.test: self.test_tiles
        }
        self.dsm_max = 255.0
        self.dsm_min = 0.0
        self.rgb_dir: str = "rgb"
        self.dsm_dir: str = "dsm"
        self.rgbir_dir: str = "rgbir"
        self.irrg_dir: str = "irrg"
        self.labels_dir: str = "labels_all"


class VaihingenInfo(ISPRSDatasetInfo):

    def __init__(self) -> None:
        super().__init__()
        self.train_tiles = [1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32, 34, 37]
        #! No valid tiles since the run script selects a validation set from the training set
        self.valid_tiles = []
        self.test_tiles = [2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 27, 29, 31, 33, 35, 38]
        self.tiles: list = self.train_tiles + self.valid_tiles + self.test_tiles
        self.tiles_dict: dict = {
            DatasetSplits.train: self.train_tiles,
            DatasetSplits.valid: self.valid_tiles,
            DatasetSplits.test: self.test_tiles
        }
        self.rgb_dir: str = "top"
        self.dsm_dir: str = "dsm"
        self.labels_dir: str = "gt_complete"
        self.dsm_max = 361.0
        self.dsm_min = 240.0
