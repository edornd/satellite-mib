from src.data import DatasetInfo


class DeepGlobeInfo(DatasetInfo):

    def __init__(self) -> None:
        super().__init__()
        self.index2label: dict = {
            0: "urban",
            1: "agriculture",
            2: "rangeland",
            3: "forest",
            4: "water",
            5: "barren",
            255: "unknown"
        }
        self.index2color: dict = {
            0: (0, 255, 255),
            1: (255, 255, 0),
            2: (255, 0, 255),
            3: (0, 255, 0),
            4: (0, 0, 255),
            5: (255, 255, 255),
            255: (0, 0, 0)
        }
        self.label2index: dict = {v: k for k, v in self.index2label.items()}
        self.color2index: dict = {v: k for k, v in self.index2color.items()}
        self.num_classes = len(self.index2label) - 1
        self.train_dir = "train"
        self.valid_dir = "valid"
        self.test_dir = "test"
