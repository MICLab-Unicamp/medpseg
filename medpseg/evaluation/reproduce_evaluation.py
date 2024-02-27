'''
This file should allow for reproduction of CoronaCases and SemiSeg results automatically

1) Download and unpack data in dataset folder
2) Predict using MEDPSeg tool functionality
3) Run evaluation metrics using output and target
4) Save result artifacts in results folder
'''
import os
import wget
import glob
import copy
import json
import torch
import zipfile
import imageio
import argparse
import subprocess
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from typing import Union, Tuple, Dict
from medpseg.seg_metrics import initialize_metrics_dict, seg_metrics


# As of Feb. 2024
SEMISEG_FILE_NAME = os.path.join("dataset", "semiseg", "13521488.zip")
CORONACASES_FILE_NAME = os.path.join("dataset", "coronacases", "3757476.zip")


# Adapted from genius idea on stackoverflow
def all_idx(idx, axis):
    grid = np.ogrid[tuple(map(slice, idx.shape))]
    grid.insert(axis, idx)
    return tuple(grid)


def int_to_onehot(matrix, overhide_max=None):
    '''
    Converts a matrix of int values (will try to convert) to one hot vectors
    '''
    if overhide_max is None:
        vec_len = int(matrix.max() + 1)
    else:
        vec_len = overhide_max

    onehot = np.zeros((vec_len,) + matrix.shape, dtype=int)

    int_matrix = matrix.astype(int)
    onehot[all_idx(int_matrix, axis=0)] = 1

    return onehot


def download_data():
    '''
    Uses wget to download original source zip files if not already downloaded
    '''
    if not os.path.isfile(SEMISEG_FILE_NAME):
        print(f"Semiseg file {SEMISEG_FILE_NAME} not found, downloading SemiSeg...")
        wget.download("https://figshare.com/ndownloader/articles/13521488/versions/2", out="dataset/semiseg")
    else:
        print(f"SemiSeg file {SEMISEG_FILE_NAME} already downloaded.")

    if not os.path.isfile(CORONACASES_FILE_NAME):
        print(f"\nCoronaCases file {CORONACASES_FILE_NAME} not found, downloading CoronaCases...")
        wget.download("https://zenodo.org/api/records/3757476/files-archive", out="dataset/coronacases")
    else:
        print(f"CoronaCases file {CORONACASES_FILE_NAME} already downloaded.")


def unzip():
    '''
    Check zip files for consistency
    '''
    tqdm_it = tqdm(range(7), desc="Checking and extracting data...")
    tqdm_iter = iter(tqdm_it)
    sseg_zip = zipfile.ZipFile(SEMISEG_FILE_NAME)
    ccases_zip = zipfile.ZipFile(CORONACASES_FILE_NAME)
    
    # Test and extract SemiSeg data
    tqdm_it.write("Testing SemiSeg zip file...")
    sseg_test_result = sseg_zip.testzip()
    if sseg_test_result is not None:
        raise ValueError(f"{SEMISEG_FILE_NAME} corrupt at {sseg_test_result}. Please delete the file and run this again.")
    tqdm_it.write("PASSED")
    next(tqdm_iter)
    
    tqdm_it.write(f"Extracting SemiSeg zip file {SEMISEG_FILE_NAME}...")
    sseg_zip.extractall(os.path.dirname(SEMISEG_FILE_NAME))
    tqdm_it.write("Done.")
    next(tqdm_iter)
    
    # Test and extract CoronaCases data
    tqdm_it.write("Testing CoronaCases zip file...")
    ccases_test_result = ccases_zip.testzip()
    if ccases_test_result is not None:
        raise ValueError(f"{CORONACASES_FILE_NAME} corrupt at {ccases_test_result}. Please delete the file and run this again.")
    tqdm_it.write("PASSED")
    next(tqdm_iter)
    
    # Extract CoronaCases zip
    tqdm_it.write(f"Extracting CoronaCases zip file {CORONACASES_FILE_NAME}...")
    ccases_dst = os.path.join(os.path.dirname(CORONACASES_FILE_NAME), "COVID-19-CT-Seg_20cases")
    ccases_zip.extractall(ccases_dst)
    next(tqdm_iter)

    # Extract CoronaCases scan zip (was inside original zip)
    scan_zip_file_name = os.path.join(ccases_dst, "COVID-19-CT-Seg_20cases.zip")
    tqdm_it.write(f"Extracting scans from {scan_zip_file_name}...")
    scan_zip = zipfile.ZipFile(scan_zip_file_name)
    scan_zip.extractall(ccases_dst)
    tqdm_it.write("Done.")
    next(tqdm_iter)
    
    # Extract CoronaCases mask zip (was inside original zip)
    target_zip_file_name = os.path.join(ccases_dst, "Lung_and_Infection_Mask.zip")
    tqdm_it.write(f"Extracting targets from {target_zip_file_name}...")
    target_zip = zipfile.ZipFile(target_zip_file_name)
    mask_dst = os.path.join(os.path.dirname(CORONACASES_FILE_NAME), "Lung_and_Infection_Mask")
    target_zip.extractall(mask_dst)
    tqdm_it.write("Done.")
    next(tqdm_iter)

    # Remove unused files
    for radiopaedia_file in tqdm(glob.glob(os.path.join(ccases_dst, "*radiopaedia*")) + glob.glob(os.path.join(mask_dst, "*radiopaedia*")), 
                                 desc="Removing radiopaedia scans...", leave=False):
        os.remove(radiopaedia_file)
    next(tqdm_iter)


def read_image_mask(data_path: str, mask_path: str, C: int) -> Union[torch.Tensor, torch.Tensor, Tuple[float]]:
    data = sitk.ReadImage(data_path)
    spacing = data.GetSpacing()
    data = torch.from_numpy(sitk.GetArrayFromImage(data)).unsqueeze(0)
    masks = torch.from_numpy(int_to_onehot(sitk.GetArrayFromImage(sitk.ReadImage(mask_path)), overhide_max=C))

    return data, masks, spacing


def save_semiseg_pngs():
    '''
    Saves semiseg input data as PNGs
    '''
    data = sitk.ReadImage(os.path.join("dataset", "semiseg", "tr_im.nii.gz"))
    data = sitk.GetArrayFromImage(data)
    HU_MIN, HU_MAX = -1024, 600
    data = np.clip(data, HU_MIN, HU_MAX)
    data = (data - HU_MIN)/(HU_MAX - HU_MIN)
    data = (data*255).astype(np.uint8)

    for i, slice_data in enumerate(data):
        imageio.imwrite(os.path.join("dataset", "semiseg", "tr_im_slices", f"{i}.png"), slice_data)

    # Build .txt file for predicting over test images
    with open(os.path.join("dataset", "semiseg", "covid_semi_seg_split.json"), 'r') as split_file:
        test_split = json.load(split_file)["test"]
    with open(os.path.join("dataset", "semiseg", "tr_im_slices", "test_slices.txt"), 'w') as test_slices_file:
        for test_idx in test_split:
            test_slices_file.write(os.path.join("dataset", "semiseg", "tr_im_slices", f"{test_idx}.png") + '\n')


def check_data():
    if not os.path.isfile(os.path.join("dataset", "semiseg", "tr_im_slices", "test_slices.txt")):
        raise FileNotFoundError("test_slices.txt file not found, rerun reproduce_evaluation.py")
    
    if not os.path.isfile(os.path.join("dataset", "semiseg", "tr_mask.nii.gz")):
        raise FileNotFoundError("SemiSeg target file not found, rerun reproduce_evaluation.py")

    if len(glob.glob(os.path.join("dataset", "semiseg", "tr_im_slices", "*.png"))) != 100:
        raise FileNotFoundError("Number of slices in tr_im_slices incorrect, rerun reproduce_evaluation.py")
    
    if len(glob.glob(os.path.join("dataset", "coronacases", "COVID-19-CT-Seg_20cases", "*.nii.gz"))) != 10:
        raise FileNotFoundError("Number of scans in coronacases/COVID-19-CT-Seg_20cases incorrect, rerun reproduce_evaluation.py")
    
    if len(glob.glob(os.path.join("dataset", "coronacases", "Lung_and_Infection_Mask", "*.nii.gz"))) != 10:
        raise FileNotFoundError("Number of targets in coronacases/Lung_and_Infection_Mask incorrect, rerun reproduce_evaluation.py")


def print_stats(stats):
    for struct_name, struct_metrics in stats.items():
        print(f"\n{struct_name}\n")
        for metric_name, metric_mean_std in struct_metrics.items():
            print(f"{metric_name}: {metric_mean_std['mean']}+-{metric_mean_std['std']}")


def semiseg_eval(metrics):
    with open(os.path.join("dataset", "semiseg", "covid_semi_seg_split.json"), 'r') as split_file:
        test_IDs = json.load(split_file)["test"]

    target_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join("dataset", "semiseg", "tr_mask.nii.gz")))
    target_array[target_array == 3] = 0
    ggo_targets = (target_array == 1).astype(np.uint8)
    con_targets = (target_array == 2).astype(np.uint8)

    for test_ID in tqdm(test_IDs, desc="Computing 2D metrics for SemiSeg..."):
        ggo_pred = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join("dataset", "semiseg", "medpseg_output", f"{test_ID}_ggo.nii.gz"))).astype(np.uint8)
        con_pred = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join("dataset", "semiseg", "medpseg_output", f"{test_ID}_consolidation.nii.gz"))).astype(np.uint8)
        pred = np.stack([ggo_pred, con_pred], axis=0)
        
        ggo_target = ggo_targets[test_ID:test_ID+1]
        con_target = con_targets[test_ID:test_ID+1]
        target = np.stack([ggo_target, con_target], axis=0)
        
        seg_metrics(gts=target, preds=pred, metrics=metrics, struct_names=["semiseg_2d_ggo", "semiseg_2d_consolidation"])


def coronacases_eval(metrics):
    targets = glob.glob(os.path.join("dataset", "coronacases", "Lung_and_Infection_Mask", "*.nii.gz"))

    for target_name in tqdm(targets, desc="Computing 3D metrics for CoronaCases..."):
        ID = os.path.basename(target_name).replace(".nii.gz", '')
        target = sitk.GetArrayFromImage(sitk.ReadImage(target_name))
        inf_target = np.expand_dims((target == 3).astype(np.uint8), 0)
        inf_pred = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join("dataset", "coronacases", "medpseg_output", f"{ID}_findings.nii.gz"))).astype(np.uint8), 0)

        seg_metrics(gts=inf_target, preds=inf_pred, metrics=metrics, struct_names=["coronacases_3d_inf"])


def generate_table(stats_file):
    from json2table import convert
    INCLUDED_METRICS = ["dice", "false_negative_error", "false_positive_error", "sensitivity", "specificity"]
    
    stats_dict: Dict[str, Dict[str, Dict[str, float]]] = json.load(stats_file)
    
    reference_dict = copy.deepcopy(stats_dict)
    for struct, metrics in reference_dict.items():
        for metric in metrics.keys():
            if metric not in INCLUDED_METRICS:
                stats_dict[struct].pop(metric)

    html = convert(stats_dict, build_direction="TOP_TO_BOTTOM", table_attributes={"style": "width:90%", "border" : 1})

    with open(os.path.join("results", "table.html"), 'w') as table_file:
        table_file.write(html)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_to_prediction", action="store_true", help="Use if data is already downloaded and organized correctly.")
    parser.add_argument("--skip_to_evaluation", action="store_true", help="Use if outputs are already computed.")
    parser.add_argument("--skip_to_table", action="store_true", help="Use if metrics are already computed.")
    args = parser.parse_args()

    if not args.skip_to_table:
        if not args.skip_to_evaluation:
            if not args.skip_to_prediction:
                download_data()
                unzip()
                save_semiseg_pngs()

            check_data()

            subprocess.run(["medpseg", 
                            "-i", os.path.join("dataset", "semiseg", "tr_im_slices", "test_slices.txt"), 
                            "-o", os.path.join("dataset", "semiseg", "medpseg_output")])
            
            subprocess.run(["medpseg", 
                            "-i", os.path.join("dataset", "coronacases", "COVID-19-CT-Seg_20cases"), 
                            "-o", os.path.join("dataset", "coronacases", "medpseg_output"),
                            "--disable_lobe"])
        
        tqdm.write("Computing metrics...")
        metrics = initialize_metrics_dict()
        semiseg_eval(metrics)
        coronacases_eval(metrics)
        json_dict_path = os.path.join("results", "metrics.json")
        metrics.save_dictionary(json_dict_path)
        print_stats(metrics.stats())

    with open(os.path.join("results", "metrics_stats.json"), 'r') as json_dict_file:
        generate_table(json_dict_file)

    tqdm.write("\nDone! Metrics are in results folder.")
