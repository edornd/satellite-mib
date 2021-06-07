from pathlib import Path
import typer as tpr

from src.data.deepglobe.config import DeepGlobeInfo


def main(source: Path = tpr.Option(..., "--source", "-s", help="Source folder containing the unzipped archives"),
         destination: Path = tpr.Option(..., "--dest", "-d", help="Destination folder, where the data will be stored")):
    config = DeepGlobeInfo()
    print(config)


if __name__ == "__main__":
    main()
