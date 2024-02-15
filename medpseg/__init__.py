import os


__version__ = "4.1.0"


def check_weight(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"\n\nERROR: Weight {path} not found in installation.\n\nMake sure the .ckpt files downloaded from https://github.com/MICLab-Unicamp/medpseg/releases are in the folder before running 'pip install .'\n\nPlease follow the installation instructions listed in https://github.com/MICLab-Unicamp/medpseg\n")
