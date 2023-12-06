'''
Copyright (c) Livia Rodrigues, Diedre Carmo, Medical Imaging Computing Lab (MICLab)
https://miclab.fee.unicamp.br/
https://github.com/MICLab-Unicamp/medpseg
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Modified from Livia code, this was used for all reportings of FPE, FNE and Dice
'''
import numpy as np
import SimpleITK as sitk
from math import nan
from typing import List, Dict
from collections import defaultdict


def initialize_metrics_dict():
    '''
    Initializes an empty metrics dict to be given to seg_metrics
    '''
    return defaultdict(lambda: defaultdict(list))


def seg_metrics(gts: np.ndarray, preds: np.ndarray, metrics: Dict[str, Dict[str, List[float]]], struct_names=["bg", "healthy", "unhealthy"]):
    '''
    Last change: add FPE, FNE and Dice edge cases
    finds overlap and distance measures of two given segmentations.
    "Overlap measures: Dice, FNError, FPError, jaccard, Volume Similarity (SimpleITK) and Volume Similarity(Taha et al)
    "Distance measures: Hausdorff distance and average hausdorff distance
    '''
    assert (len(gts.shape) == len(preds.shape) and 
            isinstance(gts, np.ndarray) and isinstance(preds, np.ndarray) and gts.dtype == np.uint8 and preds.dtype == np.uint8 and
            (gts >= 0).all() and (gts <= 1).all() and (preds <= 1).all() and (preds >= 0).all()), "Malformed input for seg_metrics"
    
    for gt, pred, str_label in zip(gts, preds, struct_names):
        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
        overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
        
        empty_gt = gt.sum() == 0
        empty_pred = pred.sum() == 0
        img_gt_sitk = sitk.GetImageFromArray(gt)
        img_pred_sitk = sitk.GetImageFromArray(pred)
        
        overlap_measures_filter.Execute(img_pred_sitk, img_gt_sitk)  # this was reversed in an old version, fixed 
        
        if empty_gt and empty_pred:
            print(f"WARNING: No {str_label} target or prediction, considering Dice 1.0.")
            metrics[str_label]["dice"].append(1.0)
        else:
            metrics[str_label]["dice"].append(overlap_measures_filter.GetDiceCoefficient())

        if empty_gt:
            print(f"WARNING: No {str_label} target, considering no false negatives present.")
            metrics[str_label]["false_negative_error"].append(0.0)
        else:
            metrics[str_label]["false_negative_error"].append(overlap_measures_filter.GetFalseNegativeError())

        if empty_pred:
            print(f"WARNING: No {str_label} prediction, considering no false positives present.")
            metrics[str_label]["false_positive_error"].append(0.0)
        else:
            metrics[str_label]["false_positive_error"].append(overlap_measures_filter.GetFalsePositiveError())

        metrics[str_label]["jaccard"].append(overlap_measures_filter.GetJaccardCoefficient())
        metrics[str_label]["volume_similarity"].append(overlap_measures_filter.GetVolumeSimilarity())
        metrics[str_label]["abs_volume_similarity"].append(1-abs(overlap_measures_filter.GetVolumeSimilarity())/2)
        
        try:
            hausdorff_distance_filter.Execute(img_pred_sitk, img_gt_sitk)  # this was reversed in an old version, fixed 
            metrics[str_label]["avg_hd"].append(hausdorff_distance_filter.GetAverageHausdorffDistance())
            metrics[str_label]["hd"].append(hausdorff_distance_filter.GetHausdorffDistance())
        except:
            metrics[str_label]["avg_hd"].append(nan)
            metrics[str_label]["hd"].append(nan)
    