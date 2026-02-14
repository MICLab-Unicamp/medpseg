import os
import site


__version__ = "5.0.0"


def get_package_path():
    """
    Get the path to the medpseg package directory.
    Works for both regular and editable installs.
    """
    # Get the directory where this __init__.py file is located
    package_dir = os.path.dirname(os.path.abspath(__file__))
    
    # If we're in an editable install, the .ckpt files are in the source directory
    # If we're in a regular install, they should be in the same directory
    return package_dir


def check_weight(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"\n\nERROR: Weight {path} not found in installation.\n\nMake sure the .ckpt files downloaded from https://github.com/MICLab-Unicamp/medpseg/releases are in the folder before running 'pip install .'\n\nPlease follow the installation instructions listed in https://github.com/MICLab-Unicamp/medpseg\n")
