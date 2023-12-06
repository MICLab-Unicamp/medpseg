'''
Copyright (c) Diedre Carmo, Medical Imaging Computing Lab (MICLab)
https://miclab.fee.unicamp.br/
https://github.com/MICLab-Unicamp/medpseg
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

A collection of general utils extracted from DLPT v0.2.2.0
'''
import os
import sys
import time
import torch
import psutil
import numpy as np
import SimpleITK as sitk
import multiprocessing as mp
from tqdm import tqdm
from torch import optim, nn
from scipy.ndimage import zoom


current_point = None


class DummyTkIntVar():
    def __init__(self, value):
        self.value = value

    def __bool__(self):
        return bool(self.value)

    def get(self):
        return self.value


def zoom_worker(x):
    channel, zoom_factor, order = x
    return zoom(channel, zoom_factor, order=order)


def check_params(hparams):
    '''
    Avoids errors while loading old models while needing new hparams.
    '''
    '''
    hparams.noschedule = getattr(hparams, "noschedule", False)
    hparams.batchfy = getattr(hparams, "batchfy", False)
    hparams.f1_average = getattr(hparams, "f1_average", 'macro')
    hparams.datasets = getattr(hparams, "datasets", 'all')
    hparams.eval_batch_size = getattr(hparams, "eval_batch_size", hparams.batch_size)
    hparams.return_path = getattr(hparams, "return_path", False)
    hparams.return_path = getattr(hparams, "patched_eval", False)
    hparams.bn = getattr(hparams, "bn", "group")
    hparams.wd = getattr(hparams, "wd", 0)
    if getattr(hparams, "noschedule", False):
        hparams.scheduling_factor = None
    '''

    return hparams


def monitor_itksnap(check_delay: int = 1):
    '''
    Blocks process checking with psutil if an itksnap instance is opened.
    Delay between checks is specified by check_delay
    '''
    itksnap_found = True
    while itksnap_found:
        process_list = [proc.name() for proc in psutil.process_iter()]
        itksnap_instances = ['itk-snap' in name.lower() for name in process_list]
        itksnap_found = any(itksnap_instances)
        if itksnap_found:
            print(' '*100, end='\r')
            time.sleep(check_delay/2)
            itk = process_list[itksnap_instances.index(True)]
            print(f"Waiting for {itk} to be closed.", end='\r')
            time.sleep(check_delay/2)


def multi_channel_zoom(full_volume, zoom_factors, order, C=None, tqdm_on=True, threaded=False):
    '''
    full_volume: Full 4D volume (numpy)
    zoom_factors: intented shape / current shape
    order: 0 - 5, higher is slower but better results, 0 is fast and bad results
    C: how many cores to spawn, defaults to number of channels in volume
    tqdm_on: verbose computation
    '''
    assert len(full_volume.shape) == 4 and isinstance(full_volume, np.ndarray)

    if C is None:
        C = full_volume.shape[0]

    if threaded:
        pool = mp.pool.ThreadPool(C)
    else:
        pool = mp.Pool(C)

    channels = [(channel, zoom_factors, order) for channel in full_volume]

    zoomed_volumes = []

    pool_iter = pool.map(zoom_worker, channels)

    if tqdm_on:
        iterator = tqdm(pool_iter, total=len(channels), desc="Computing zooms...")
    else:
        iterator = pool_iter

    for output in iterator:
        zoomed_volumes.append(output)

    return np.stack(zoomed_volumes)


def get_optimizer(name, params, lr, wd=0):
    if name == "RAdam":
        return optim.RAdam(params, lr=lr, weight_decay=wd)
    elif name == "Adam":
        return optim.Adam(params, lr=lr, weight_decay=wd)
    elif name == "AdamW":
        return optim.AdamW(params, lr=lr, weight_decay=wd)
    elif name == "SGD":
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    else:
        raise ValueError("Invalid optimizer name")


class DICELoss(torch.nn.Module):
    '''
    Calculates DICE Loss
    Use per channel for multiple targets.
    '''
    def __init__(self, volumetric=False, negative_loss=False, per_channel=False, check_bounds=True):
        self.name = "DICE Loss"
        super(DICELoss, self).__init__()
        self.volumetric = volumetric
        self.negative_loss = negative_loss
        self.per_channel = per_channel
        self.check_bounds = check_bounds

    def __call__(self, probs, targets):
        '''
        probs: output of last convolution, sigmoided or not (use apply_sigmoid=True if not)
        targets: binary target mask
        '''
        p_min = probs.min()
        p_max = probs.max()
        bounded = p_max <= 1.0 and p_min >= 0.0
        if self.check_bounds:
            if not bounded:
                raise ValueError(f"FATAL ERROR: DICE loss input not bounded between 1 and 0! {p_min} {p_max}")
        else:
            if not bounded:
                print(f"WARNING: DICE loss input not bounded between 1 and 0! {p_min} {p_max}")


        score = 0

        if self.per_channel:
            assert len(targets.shape) >= 4, ("less than 4 dimensions makes no sense with multi channel in a batch of 2D or 3D"
                                             "volumes")
            nchannels = targets.shape[1]
            if self.volumetric:
                score = torch.stack([vol_dice(probs[:, c], targets[:, c]) for c in range(nchannels)]).mean()
            else:
                score = torch.stack([batch_dice(probs[:, c], targets[:, c]) for c in range(nchannels)]).mean()
        else:
            if self.volumetric:
                score = vol_dice(probs, targets)
            else:
                score = batch_dice(probs, targets)

        if self.negative_loss:
            loss = -score
        else:
            loss = 1 - score

        return loss


def vol_dice(inpt, target, smooth=1.0):
    '''
    Calculate DICE of volume
    '''
    # q = inpt.size(0)
    assert len(inpt) != 0, " trying to compute DICE of nothing"

    iflat = inpt.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    eps = 0
    if smooth == 0.0:
        eps = sys.float_info.epsilon

    iflat_sum = iflat.sum()
    tflat_sum = tflat.sum()

    if iflat_sum.item() == 0.0 and tflat_sum.item() == 0.0:
        # print("DICE Metric got black mask and prediction!")
        dice = torch.tensor(1.0, requires_grad=True, device=inpt.device)
    else:
        dice = (2. * intersection + smooth) / (iflat_sum + tflat_sum + smooth + eps)

    value = dice.item()
    assert value >= 0.0 or value <= 1.0, " DICE not between 0 and 1! something is wrong"

    return dice


def batch_dice(inpt, target, smooth=1.0):
    '''
    Calculate DICE of a batch of two binary masks
    Returns mean dice of all slices
    '''
    q = inpt.size(0)
    assert len(inpt) != 0, " trying to compute DICE of nothing"

    iflat = inpt.contiguous().view(q, -1)
    tflat = target.contiguous().view(q, -1)
    intersection = (iflat * tflat).sum(dim=1)

    eps = 0
    if smooth == 0.0:
        eps = sys.float_info.epsilon

    iflat_sum = iflat.sum(dim=1)
    tflat_sum = tflat.sum(dim=1)

    dice = (2. * intersection + smooth) / (iflat_sum + tflat_sum + smooth + eps)

    dice = dice.mean()
    value = dice.item()
    assert value >= 0.0 or value <= 1.0, " DICE not between 0 and 1! something is wrong"

    return dice


class DICEMetric(nn.Module):
    '''
    Calculates DICE Metric
    '''
    def __init__(self, apply_sigmoid=False, mask_ths=0.5, skip_ths=False, per_channel_metric=False, check_bounds=True):
        self.name = "DICE"
        self.lower_is_best = False
        super(DICEMetric, self).__init__()
        self.apply_sigmoid = apply_sigmoid
        self.mask_ths = mask_ths
        self.skip_ths = skip_ths
        self.per_channel_metric = per_channel_metric
        self.check_bounds = check_bounds

    def __call__(self, probs, target):
        '''
        Returns only DICE metric, as volumetric dice
        probs: output of last convolution, sigmoided or not (use apply_sigmoid=True if not)
        targets: float binary target mask
        '''
        probs = probs.type(torch.float32)
        target = target.type(torch.float32)

        if self.apply_sigmoid:
            probs = probs.sigmoid()

        p_min = probs.min()
        if self.check_bounds:
            assert p_min >= 0.0, f"FATAL ERROR: DICE metric input not positive! {p_min}"
        else:
            if p_min < 0:
                print(f"WARNING: Negative probabilities entering into DICE! {p_min}")

        if self.skip_ths:
            mask = probs
        else:
            mask = (probs > self.mask_ths).float()

        if self.per_channel_metric:
            assert len(target.shape) >= 4, ("less than 4 dimensions makes no sense with multi channel in a batch of 2D or 3D"
                                            "volumes")
            nchannels = target.shape[1]
            return [vol_dice(mask[:, c], target[:, c], smooth=0.0).item() for c in range(nchannels)]
        else:
            return vol_dice(mask, target, smooth=0.0).item()


def itk_snap_spawner(nparray: np.ndarray, title: str = "ITKSnap", itksnap_path: str = "/usr/bin/itksnap",
                     block: bool = True):
    '''
    Displays a three dimensional numpy array using SimpleITK and itksnap.
    Assumes itksnap is installed on /usr/bin/itksnap.
    Blocks process until all itksnap instances openend on the computer are closed. 
    '''
    assert os.path.isfile(itksnap_path), f"Couldn't find itksnap on {itksnap_path}"
    assert len(nparray.shape) in [3, 4], "Array not three dimensional"

    if len(nparray.shape) == 4 and np.array(nparray.shape).argmin() == 0:
        adjusted_nparray = nparray.transpose(1, 2, 3, 0)
    else:
        adjusted_nparray = nparray

    image_viewer = sitk.ImageViewer()
    image_viewer.SetTitle(title)
    image_viewer.SetApplication(itksnap_path)
    image_viewer.Execute(sitk.GetImageFromArray(adjusted_nparray))
    if block:
        monitor_itksnap()


class CoUNet3D_metrics():
    def __init__(self, classes=["P", "L"]):
        self.dice = DICEMetric(per_channel_metric=True)
        self.classes = classes

    def __call__(self, preds, tgt):
        dices = self.dice(preds, tgt)
        report = {}

        for i, c in enumerate(self.classes):
            report[f"{c}_dice"] = dices[i]

        return report