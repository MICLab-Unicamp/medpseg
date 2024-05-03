'''
Copyright (c) Diedre Carmo, Medical Imaging Computing Lab (MICLab)
https://miclab.fee.unicamp.br/
https://github.com/MICLab-Unicamp/medpseg
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Lite version of old pipeline only doing lung segmentation for initial lung detection and L/R separation.

Code thrown away in commit from 31 Oct 2023
'''
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from medpseg.poly_seg_3d_module import PolySeg3DModule


class PrintInterface():
    def __init__(self, tqdm_iter):
        self.tqdm_iter = tqdm_iter
        self.rot90 = False

    def write(self, x):
        self.tqdm_iter.put(("write", x))

    def progress(self, x):
        self.tqdm_iter.put(("iterbar", x))

    def image_to_front_end(self, x):
        if self.rot90:
            x = np.rot90(x, k=2, axes=(0, 1))

        self.tqdm_iter.put(("slice", x))

    def icon(self):
        self.tqdm_iter.put(("icon", ''))


class SegmentationPipeline():
    def __init__(self,
                 best_3d="/home/diedre/diedre_phd/phd/models/wd_step_poly_lung_softm-epoch=85-val_loss=0.03.ckpt",  # POLY LUNG SOFTM
                 # These are from v1, still here but not used.
                 best_25d="/home/diedre/diedre_phd/phd/models/wd_step_sme2d_coedet_fiso-epoch=72-val_loss=0.06-bg_dice=1.00-healthy_dice=0.92-unhealthy_dice=0.74.ckpt",  # SME2D COEDET FISO Full train wd step  better
                 best_25d_raw="/home/diedre/diedre_phd/phd/models/sing_a100_up_awd_step_raw_medseg_pos-epoch=99-val_loss=0.15-bg_dice=1.00-healthy_dice=0.84-unhealthy_dice=0.71.ckpt",  # Best raw axial 2.5D model, trained on positive 256 slices only
                 airway="/home/diedre/diedre_phd/phd/models/atm_baseline-epoch=75-val_loss=0.10-bg_dice=0.00-healthy_dice=0.00-unhealthy_dice=0.90.ckpt",  # Air way model used in ATM22 challenge
                 parse="/home/diedre/diedre_phd/phd/models/parse_baseline-epoch=103-val_loss=0.26-bg_dice=0.00-healthy_dice=0.00-unhealthy_dice=0.74.ckpt",  # First attempt at training in parse data
                 # 
                 batch_size=4,
                 n=10,
                 cpu=False):  
        self.version = 'wd_step_poly_lung_softm_only'
        self.batch_size = batch_size
        self.n = n
        self.device = torch.device("cpu") if cpu else torch.device("cuda:0")
        self.model_3d = PolySeg3DModule.load_from_checkpoint(best_3d, map_location="cpu").eval()
        if best_25d is None:
            self.model_25d = None
        else:
            raise ValueError("Seg2DModule is not used anymore in the new version.")
            # self.model_25d = Seg2DModule.load_from_checkpoint(best_25d).eval()
        if best_25d_raw is None:
            self.model_25d_raw = None
        else:
            raise ValueError("Seg2DModule is not used anymore in the new version.")
            # self.model_25d_raw = Seg2DModule.load_from_checkpoint(best_25d_raw).eval()
        if airway is None:
            self.airway_model = None
        else:
            raise ValueError("Seg2DModule is not used anymore in the new version.")
            # self.airway_model = Seg2DModule.load_from_checkpoint(airway).eval()
        if parse is None:
            self.parse_model = None
        else:
            raise ValueError("Seg2DModule is not used anymore in the new version.")
            # self.parse_model = Seg2DModule.load_from_checkpoint(parse).eval()

    def __call__(self, input_volume, spacing, tqdm_iter, minimum_return=False, atm_mode=False, act=False, lung_only=False):
        assert input_volume.max() <= 1.0 and input_volume.min() >= 0.0
        pre_shape = input_volume.shape[2:]
        
        if tqdm_iter is not None and not isinstance(tqdm_iter, tqdm):
            tqdm_iter = PrintInterface(tqdm_iter)

        assert lung_only, "This code is meant to be used only for Polymorphic Lung segmentation"

        with torch.no_grad():
            # 3D Lung detection. Everything else removed, and in old medseg repository.
            tqdm_iter.write("3D Prediction...")
            tqdm_iter.progress(20)
            self.model_3d = self.model_3d.to(self.device)
            input_volume = input_volume.to(self.device)
            left_right_label = self.model_3d(F.interpolate(input_volume, (128, 256, 256), mode="trilinear"), get_bg=True)[0]
            self.model_3d.cpu()
            input_volume = input_volume.cpu()
            left_right_label = F.interpolate(left_right_label, pre_shape, mode="nearest").squeeze().cpu().numpy()
            
            left_lung, right_lung = left_right_label[1], left_right_label[2]
            voxvol = spacing[0]*spacing[1]*spacing[2]
            left_lung_volume = round((left_lung.sum()*voxvol)/1e+6, 3)
            right_lung_volume = round((right_lung.sum()*voxvol)/1e+6, 3)
            lung_volume = left_lung_volume + right_lung_volume

            tqdm_iter.write(f"Lung volume: {lung_volume}L, left: {left_lung_volume}L, right: {right_lung_volume}L")
            
            if lung_volume < .5:
                # Lung not detected!
                tqdm_iter.write("\nCritical WARNING: Lung doesn't seen to be present in image! Continuing, but this might not be a chest CT image and results might be weird.\n")
                
            # If lung only mode return here, this is the only use case currently
            if lung_only:
                return left_right_label, left_lung_volume, right_lung_volume

            # Removed all deprecated code from old medseg in (31 Oct 2023)
    

def get_atts_work(encoder, pre_shape):
    '''
    Gets attention from unt encoder specifically
    '''
    atts = torch.stack([torch.from_numpy(x) for x in encoder.return_atts()]).unsqueeze(0)
    atts = F.interpolate(atts, pre_shape, mode="trilinear", align_corners=False).squeeze().numpy()
    return atts
