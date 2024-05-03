'''
Copyright (c) Livia Rodrigues, Diedre Carmo, Medical Imaging Computing Lab (MICLab)
https://miclab.fee.unicamp.br/
https://github.com/MICLab-Unicamp/medpseg
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Modified from Livia code, this was used for all reportings of FPE, FNE and Dice
'''
import os
import torch
import json
import numpy as np
import SimpleITK as sitk
from math import nan
from typing import List, Dict, Union
from collections import defaultdict
from medpseg.atm_evaluation import dice_coefficient_score_calculation, precision_calculation, false_negative_rate_calculation, false_positive_rate_calculation, sensitivity_calculation, specificity_calculation, branch_detected_calculation, tree_length_calculation


## Extracted from DLPT fixed version 21 August 23
def sensitivity(pred: Union[torch.Tensor, np.ndarray], 
                tgt: Union[torch.Tensor, np.ndarray]):
    '''
    True positive rate, how many positives are actually positive
    Supports torch or numpy
    '''
    tgt_sum = tgt.sum()
    if tgt_sum == 0:
        return nan
    else:
        return (pred*tgt).sum() / tgt.sum()


def specificity(pred: Union[torch.Tensor, np.ndarray], 
                tgt: Union[torch.Tensor, np.ndarray]):
    '''
    True negative rate, how many negatives are actually negative
    Doesnt work well with too many true negatives
    '''
    assert pred.shape == tgt.shape

    ones_minus_tgt = 1 - tgt
    ones_minus_pred = 1 - pred

    one_minus_tgt_sum = ones_minus_tgt.sum()

    if one_minus_tgt_sum == 0:
        return nan

    return ((ones_minus_pred)*(ones_minus_tgt)).sum() / one_minus_tgt_sum


def precision(pred, tgt):
    '''
    True positives / (true positives + false positives)
    '''
    assert pred.shape == tgt.shape

    one_minus_tgt = 1 - tgt

    TP = (pred*tgt).sum()  # Postiv
    FP = (pred*one_minus_tgt).sum()  # Negatives that are in prediction

    TPplusFP = TP + FP
    if TPplusFP == 0:
        return nan
    else:
        return TP/TPplusFP
##



class MetricDict(defaultdict):
    '''
    Extend defaultdict with a string representation method to make printins metrics easier
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self) -> str:
        _str: str = ""
        for struct, metrics in self.items():
            _str += f"\n{struct}\n"
            if isinstance(metrics, dict):
                for metric_name, metric_values in metrics.items():
                    _str += f"{metric_name}: {metric_values}\n"
            else:
                _str += str(metrics)

        return _str
    
    def stats(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        '''
        Returns a dict that indexes structure, metric and mean and std values
        '''
        stats: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

        for struct, metrics in self.items():
            if isinstance(metrics, dict):
                for metric_name, metric_values in metrics.items():
                    value_ndarray = np.array(metric_values)
                    stats[struct][metric_name]["mean"] = np.nanmean(value_ndarray)
                    stats[struct][metric_name]["std"] = np.nanstd(value_ndarray)

        return stats

    def save_dictionary(self, out_path: str):
        '''
        Saves dictionary contents and statistics
        '''
        if not out_path.endswith(".json") or not os.path.exists(os.path.dirname(out_path)):
            raise ValueError(f"{out_path} does not exist or does not end with .json")

        with open(out_path, 'w') as out_path_file:
            json.dump(self, out_path_file)

        with open(out_path.replace(".json", "_stats.json"), 'w') as out_path_file:
            json.dump(self.stats(), out_path_file)

def initialize_metrics_dict():
    '''
    Initializes an empty metrics dict to be given to seg_metrics
    '''
    return MetricDict(lambda: defaultdict(list))


def seg_metrics(gts: np.ndarray, preds: np.ndarray, metrics: Dict[str, Dict[str, List[float]]], struct_names=["bg", "healthy", "unhealthy"]):
    '''
    Last change: add FPE, FNE and Dice edge cases
    finds overlap and distance measures of two given segmentations.
    "Overlap measures: Dice, FNError, FPError, jaccard, Volume Similarity (SimpleITK) and Volume Similarity(Taha et al)
    "Distance measures: Hausdorff distance and average hausdorff distance
    '''
    assert len(gts.shape) == 4, f"{gts.shape}"
    assert len(gts.shape) == len(preds.shape), f"{gts.shape}/{preds.shape}"
    assert isinstance(gts, np.ndarray) and isinstance(preds, np.ndarray), f"{type(gts)}/{type(preds)}"
    assert gts.dtype == np.uint8 and preds.dtype == np.uint8, f"{gts.dtype}/{preds.dtype}"
    assert (gts >= 0).all() and (gts <= 1).all() and (preds <= 1).all() and (preds >= 0).all(), f"gts: {gts.min()}/{gts.max()}, preds: {preds.min()}/{preds.max()}"
    
    for gt, pred, str_label in zip(gts, preds, struct_names):
        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
        overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
        
        empty_gt = gt.sum() == 0
        empty_pred = pred.sum() == 0
        sen = sensitivity(pred=pred, tgt=gt)
        spec = specificity(pred=pred, tgt=gt)
        prec = precision(pred=pred, tgt=gt)
        img_gt_sitk = sitk.GetImageFromArray(gt)
        img_pred_sitk = sitk.GetImageFromArray(pred)
        
        overlap_measures_filter.Execute(img_pred_sitk, img_gt_sitk)  # this was reversed in an old version, fixed 
        
        if empty_gt and empty_pred:
            metrics[str_label]["dice"].append(1.0)
        else:
            metrics[str_label]["dice"].append(overlap_measures_filter.GetDiceCoefficient())

        if empty_gt:
            metrics[str_label]["false_negative_error"].append(0.0)
        else:
            metrics[str_label]["false_negative_error"].append(overlap_measures_filter.GetFalseNegativeError())

        if empty_pred:
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

        metrics[str_label]["sensitivity"].append(sen)
        metrics[str_label]["specificity"].append(spec)
        metrics[str_label]["precision"].append(prec)

        # Smooth ATM metrics
        metrics[str_label]["smooth_dice"].append(dice_coefficient_score_calculation(pred=pred, label=gt))
        metrics[str_label]["false_positive_rate"].append(false_positive_rate_calculation(pred=pred, label=gt))
        metrics[str_label]["false_negative_rate"].append(false_negative_rate_calculation(pred=pred, label=gt))
        metrics[str_label]["smooth_sensitivity"].append(sensitivity_calculation(pred=pred, label=gt))
        metrics[str_label]["smooth_specificity"].append(specificity_calculation(pred=pred, label=gt))
        metrics[str_label]["smooth_precision"].append(precision_calculation(pred=pred, label=gt))
    