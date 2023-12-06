'''
Copyright (c) Diedre Carmo, Medical Imaging Computing Lab (MICLab)
https://miclab.fee.unicamp.br/
https://github.com/MICLab-Unicamp/medpseg
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
'''

import argparse
import multiprocessing as mp
from torch import cuda
from medpseg.gui import MainWindow   


def arg_parse():
    '''
    Parses CLI arguments. If nothing is given still opens GUI!
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', type=str, default=None, help="Use to indicate input folder to run without using the GUI.")
    parser.add_argument('-o', '--output_folder', type=str, default=None, help="Use to indicate output folder without using the GUI. Will try to create if it doesn't exist.")
    parser.add_argument('--debug', action="store_true", help="Debug.")
    parser.add_argument('-bs', '--batch_size', type=int, default=1)
    parser.add_argument('-win_itk_path', type=str, default="C:\\Program Files\\ITK-SNAP 3.8\\bin\\ITK-SNAP.exe", help="Path for ITKSnap exe location in Windows. Default: 'C:\\Program Files\\ITK-SNAP 3.8\\bin\\ITK-SNAP.exe'")
    parser.add_argument('-linux_itk_path', type=str, default="itksnap", help="Command for itksnap execution in Linux. Default: 'itksnap'")
    parser.add_argument('--post', action="store_true", help="Run largest component seleciton on circulatory outputs.")
    parser.add_argument('-mih', '--min_hu', type=int, default=-1024, help="Min HU for reverse engineering 0-255 2D images as input files. We assume Clip to min_hu and max_hu and min-max normalization (0-1 range) was performed. ONLY APPLIES TO .JPG/.PNG INPUTS. We recommend using raw NifT/DICOM files and ignoring this argument.")
    parser.add_argument('-mah', '--max_hu', type=int, default=600, help="Max HU for reverse engineering 0-255 2D images as input files. We assume Clip to min_hu and max_hu and min-max normalization (0-1 range) was performed. ONLY APPLIES TO .JPG/.PNG INPUTS. We recommend using raw NifT/DICOM files and ignoring this argument.")
    parser.add_argument('--slicify', action="store_true", help="Saves ALL slices as image files in output")
    parser.add_argument('--disable_lobe', action="store_true", help="Disable lobe segmentation for faster prediction")
    args = parser.parse_args()
    
    return args

def main():
    '''
    Main entrypoint for attempting to use the GPU
    Referenced in setup.py
    '''
    args = arg_parse()
    print(f"Running MEDPSeg with GPU support. CUDA available? {cuda.is_available()}.")

    args.cpu = False

    MainWindow(args, mp.Queue()).join()

def main_cpu():
    '''
    Main entrypoint which forces CPU usage.
    Referenced in setup.py
    '''
    args = arg_parse()
    print("Running MEDPSeg in CPU-only mode.")

    args.cpu = True

    MainWindow(args, mp.Queue()).join()


if __name__ == "__main__":
    '''
    If called as a script just call main
    '''
    main()
    