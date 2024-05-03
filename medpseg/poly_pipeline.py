'''
Copyright (c) Diedre Carmo, Medical Imaging Computing Lab (MICLab)
https://miclab.fee.unicamp.br/
https://github.com/MICLab-Unicamp/medpseg
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Independent script
Updated pipeline using a single weight
'''
import os
import torch
import numpy as np
import cc3d
from medpseg.poly_seg_2d_module import PolySeg2DModule
from medpseg.eval_2d_utils import E2DStackDataset, argon_cpu_count
import SimpleITK as sitk
from torch.nn import functional as F
from tqdm import tqdm
from collections import defaultdict
from operator import itemgetter
from typing import Dict, Optional
from queue import Queue
from threading import Thread
from medpseg.utils.turbo_colormap import turbo_colormap_data


def get_connected_components(volume, return_largest=2, verbose=False):
    '''
    volume: input volume
    return_largest: how many of the largest labels to return. If 0, nothing is changed in input volume
    verbose: prints label_count

    returns:
        filtered_volume, label_count, labeled_volume
    '''
    labels_out = cc3d.connected_components(volume.astype(np.int32))
    label_count = np.unique(labels_out, return_counts=True)[1]

    # Indicate which was the original label and sort by count
    label_count = [(label, count) for label, count in enumerate(label_count)]
    label_count.sort(key=itemgetter(1), reverse=True)
    label_count.pop(0)  # remove largest which should be background

    if verbose:
        print(f"Label count: {label_count}")

    filtered = None
    if return_largest > 0:
        for i in range(return_largest):
            try:
                id_max = label_count[i][0]
                if filtered is None:
                    filtered = (labels_out == id_max)
                else:
                    filtered += (labels_out == id_max)
            except IndexError:
                # We want more components that what is in the image, stop
                break

        volume = filtered * volume
        labels_out = filtered * labels_out

    return volume, label_count, labels_out


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


def attention_worker(input_q: Queue, model:  PolySeg2DModule, info_q: PrintInterface):
    '''
    Computes attention and input image for front end in a thread
    '''
    print("Attention worker started.")
    while True:
        input_slice = input_q.get()
        
        if input_slice is None:
            print("Attention worker done.")
            return

        package = build_front_end_package(input_slice, model)
        info_q.image_to_front_end(package)

def build_front_end_package(input_slice: torch.Tensor, model: PolySeg2DModule, outs: Dict[str, torch.Tensor]):
    '''
    Builds data to be sent to front end for display
    '''
    atts, circulatory_atts = model.model.return_atts(order=1, hr_only=True)

    # Weight the same as loss: 0.75 + exponential decay on weight starting from [1:]
    atts = np.stack([atts[0]*0.75] + [att*(2**(-1*(r+3))) for r, att in enumerate(atts[1:])], axis=0).sum(axis=0).squeeze()
    circulatory_atts = np.stack([circulatory_atts[0]*0.75] + [catt*(1/(2**(-1*(r+3)))) for r, catt in enumerate(circulatory_atts[1:])], axis=0).sum(axis=0).squeeze()
    
    # Colormapping, setting RGB values, 2x2 mosaic: [[input, att], [seg, att]]
    rgb_input = input_slice[0].numpy().transpose(1, 2, 0).copy()
    integer_scalar_array = (np.vstack([atts, circulatory_atts])*255).astype(np.uint8)
    att_stack = turbo_colormap_data[integer_scalar_array]
    rgb_out = outs["main"].detach().cpu().squeeze()[1:]
    atm = outs["atm"].detach().cpu().squeeze()[1]
    vessel = outs["vessel"].detach().cpu().squeeze()[1]
    rgb_out[1] = rgb_out[1]/2 + atm/2
    rgb_out[2] = rgb_out[2]/2 + atm/2
    rgb_out[0] = rgb_out[0]/2 + vessel/2
    rgb_out[1] = rgb_out[1]/2 + vessel/2
    rgb_out = rgb_out.permute(1, 2, 0).numpy()
    input_output = np.vstack([rgb_input, rgb_out])
    return np.hstack([input_output, att_stack])


def poly_stack_predict(model: PolySeg2DModule, volume: torch.Tensor, batch_size: int, device=torch.device("cuda:0"), info_q: Optional[Queue] = None, uncertainty: Optional[int] = None, cli: bool = True):
    '''
    DEVING uncertainty: epistemic uncerainty, predict n times and return the mean and std prediction
    '''
    e2d_stack_dataloader = E2DStackDataset(volume, extended_2d=1).get_dataloader(batch_size=batch_size, pin_memory=False, num_workers=argon_cpu_count())
    
    outs = defaultdict(list)
    np_outs = {}
    np_means = {}
    np_stds = {}
    
    uncertainty_means = defaultdict(list)
    uncertainty_stds = defaultdict(list)

    for input_slice in tqdm(e2d_stack_dataloader, desc=f"Slicing with batch size {batch_size}."):
        if uncertainty is None:
            out = model(input_slice.to(device), stacking=True)
            for key, y_hat in out.items():
                outs[key].append(y_hat.cpu())
        else:
            raise DeprecationWarning("Uncertainty deprecated for now, needs update")
            # Save outputs for each branch in buffer
            uncertainty_buffer = defaultdict(list)
            model.train()
            for _ in tqdm(range(uncertainty)):  # 8 equivalent to (0, 1, 2) flips 
                out = model(input_slice.to(device), stacking=True) 
                for key, y_hat in out.items():
                    uncertainty_buffer[key].append(y_hat.cpu())
            model.eval()

            # Collect buffer items into means and STDs for each branch
            for key, buffer in uncertainty_buffer.items():
                # use stack to keep batch dimension separate from acumulation dimension
                # statistics will take that dimension out
                buffer = torch.stack(buffer, dim=0)  
                uncertainty_means[key].append(buffer.mean(dim=0))
                uncertainty_stds[key].append(buffer.std(dim=0))

        # Front end update
        if info_q is not None and isinstance(info_q, PrintInterface) and not cli:
            package = build_front_end_package(input_slice, model, out)
            info_q.image_to_front_end(package)
            # input_q.put(input_slice)  # sync problems
            # atts, circulatory_atts = model.model.return_atts()
            # atts = np.stack(atts).mean(axis=0).squeeze()
            # circulatory_atts = np.stack(circulatory_atts).mean(axis=0).squeeze()
            # package = np.hstack([input_slice[0].numpy().transpose(1, 2, 0).copy(), np.stack([atts, np.zeros_like(atts), circulatory_atts], axis=-1)])
            # info_q.image_to_front_end(package)

    # Certain prediction volumes. Will no run if in uncertain mode.
    for key, y_hat in outs.items():
        np_outs[key] = torch.cat(y_hat).unsqueeze(0).permute(0, 2, 1, 3, 4)
    
    # Compute final volume for uncertainty mean and uncertainty itself (STD)
    if uncertainty is not None:
        raise DeprecationWarning("Uncertainty deprecated for now, needs update")
        for key, y_hat in uncertainty_means.items():
            np_means[key] = torch.cat(y_hat).unsqueeze(0).permute(0, 2, 1, 3, 4)
        for key, y_hat in uncertainty_stds.items():
            np_stds[f"{key}_uncertainty"] = torch.cat(y_hat).unsqueeze(0).permute(0, 2, 1, 3, 4)

    if uncertainty is None:
        return np_outs
    else:
        return np_means, np_stds


class PolySegmentationPipeline():
    '''
    This pipeline does all targets in a single weight
    '''
    def __init__(self,
                 weight="/home/diedre/diedre_phd/phd/models/medseg_25d_a100_long_silver_gold_gdl-epoch=22-step=76176-val_loss_3d=0.25-healthy_dice_3d=0.93-unhealthy_dice_3d=0.71-ggo_dice_3d=0.71-con_dice_3d=0.62-airway_dice_3d=0.90-vessel_dice_3d=0.87.ckpt",  
                 batch_size=1,  # increase with high memory gpus
                 cpu=False,
                 output_dir=None,
                 post=False,
                 cli=True):  
        self.version = 'silver_gold_gdl'
        self.batch_size = batch_size
        self.device = torch.device("cpu") if cpu else torch.device("cuda:0")
        self.model = PolySeg2DModule.load_from_checkpoint(weight, map_location="cpu").eval()
        print(self.model.hparams.experiment_name)
        EXPECTED_WEIGHT = "medseg_25d_a100_long_silver_gold_gdl"
        assert EXPECTED_WEIGHT in self.model.hparams.experiment_name, f"Incorrect experiment name {EXPECTED_WEIGHT} in given weight: {weight}, please update weights."
        self.output_dir = output_dir
        self.hparams = self.model.hparams
        self.post = post
        self.cli = cli

    def save_activations(self, poly_out: np.ndarray, airway: np.ndarray, vessel: np.ndarray, uncertainty_std: Optional[Dict[str, np.ndarray]], original_image: sitk.Image, ID: str, dir_array: np.ndarray):
        all_arrays: Dict[str, np.ndarray] = {"polymorphic_lung": poly_out, "airway": airway, "vessel": vessel}
        if uncertainty_std is not None:
            all_arrays.update(uncertainty_std)

        for name, array in all_arrays.items():
            array = array.squeeze()
            if array.ndim == 4:
                C = array.shape[0]
                new_array = np.zeros_like(array, shape=array.shape[1:])
                for i, channel_array in enumerate(array):
                    # Representing multichannels as heat map
                    # channel 0 of 4 adds 0
                    # channel 1 of 4 adds 0.333...
                    # channel 2 of 4 adds 0.666...
                    # channel 3 of 4 adds 0.999...
                    new_array += (i/(C-1))*channel_array
                array = new_array

            array = np.flip(array, np.where(dir_array[[0,4,8]][::-1]<0)[0]).copy()  # old n reliable lungmask trick
            out_image = sitk.GetImageFromArray(array)
            out_image.CopyInformation(original_image)
            sitk.WriteImage(out_image, os.path.join(self.output_dir, ID + f"_{name}_activation.nii.gz"))
            del array  # just to be sure but should be automatically cleaned anyway

    def __call__(self, input_volume: torch.Tensor, tqdm_iter: Optional[tqdm], uncertainty: Optional[int], original_image: sitk.Image, ID: str, dir_array: np.ndarray, act: bool):
        print(f"Input volume max: {input_volume.max()} min {input_volume.min()}")
        print("Deprecated uncertainty computation for now, forcing it to None.") 
        uncertainty = None

        if tqdm_iter is not None and not isinstance(tqdm_iter, tqdm):
            tqdm_iter = PrintInterface(tqdm_iter)

        with torch.no_grad():
            tqdm_iter.write("Starting segmentation...")
            tqdm_iter.progress(10)
            pre_shape = input_volume.shape[2:]
            
            if pre_shape[1] != 512 or pre_shape[2] != 512:
                adjust_shape = (pre_shape[0], 512, 512)
                tqdm_iter.write(f"Unusual shape {pre_shape} being transformed to {adjust_shape}")
                print(f"Unusual shape {pre_shape} being transformed to {adjust_shape}")
                adjusted_volume = F.interpolate(input_volume.to(self.device), adjust_shape, mode="trilinear")
                adjusted_volume = adjusted_volume.cpu()
            else:
                adjust_shape = None
                adjusted_volume = input_volume

            tqdm_iter.write("Polymorphic prediction running...")
            tqdm_iter.progress(40)

            if uncertainty is not None:
                tqdm_iter.write(f"Using epistemic uncertainty with {uncertainty} as ensembling strategy.")

            poly_stack_output = poly_stack_predict(self.model.to(self.device), adjusted_volume, batch_size=self.batch_size, device=self.device, info_q=tqdm_iter, uncertainty=uncertainty, cli=self.cli)

            if uncertainty is not None:
                poly_stack_output, uncertainty_std = poly_stack_output
            else:
                uncertainty_std = None
                
            self.model.cpu()
            poly_out = poly_stack_output["main"]
            airway = poly_stack_output["atm"]
            vessel = poly_stack_output["vessel"]

            if adjust_shape is not None:
                a_shape = airway.shape
                tqdm_iter.write(f"Airway output {a_shape} being transformed back to {pre_shape}")
                print(airway.shape)
                airway = F.interpolate(airway.cpu(), pre_shape, mode="nearest").numpy()
                
                v_shape = vessel.shape
                tqdm_iter.write(f"Vessel output {v_shape} being transformed back to {pre_shape}")
                vessel = F.interpolate(vessel.cpu(), pre_shape, mode="nearest").numpy()
                
                poly_shape = poly_out.shape
                tqdm_iter.write(f"Main output {poly_shape} being transformed back to {pre_shape}")
                poly_out = F.interpolate(poly_out.cpu(), pre_shape, mode="nearest").numpy()

                if uncertainty_std is not None:
                    for key in list(uncertainty_std.keys()):
                        array_shape = uncertainty_std[key].shape
                        tqdm_iter.write(f"{key} output {array_shape} being transformed back to {pre_shape}")
                        uncertainty_std[key] = F.interpolate(uncertainty_std[key].cpu(), pre_shape, mode="nearest").numpy()
            else:
                airway = airway.cpu().numpy()
                vessel = vessel.cpu().numpy()
                poly_out = poly_out.cpu().numpy()
                if uncertainty_std is not None:
                    for key in list(uncertainty_std.keys()):
                        uncertainty_std[key] = uncertainty_std[key].cpu().numpy()
        
        tqdm_iter.progress(60)

        # Save activations before post processing to save memory
        # Uncertainty stuff is deprecated for now 
        if act:
            tqdm_iter.write("Saving output activations. This might use more RAM.")
            self.save_activations(poly_out, airway, vessel, None, original_image, ID, dir_array)

        airway = (airway[:, 1:] > 0.5).astype(np.int32)
        vessel = (vessel[:, 1:] > 0.5).astype(np.int32)
        if self.post:
            tqdm_iter.write("Circulatory Post-processing...\n(WARNING: Largest component selection might make everything worse if the scan is of low resolution!)")
            print(f"Airway max activation {airway.max()}")
            airway = get_connected_components(airway.squeeze(0).squeeze(0), return_largest=1)[0].astype(np.uint8)  # save ram with uint8
            print(f"Vessel max activation {vessel.max()}")
            vessel = get_connected_components(vessel.squeeze(0).squeeze(0), return_largest=1)[0].astype(np.uint8)
        else:
            tqdm_iter.write("Connected component post processing is turned off.")
            airway = airway.squeeze(0).squeeze(0).astype(np.uint8)
            vessel = vessel.squeeze(0).squeeze(0).astype(np.uint8)
        
        print(f"Final airway max activation {airway.max()}")
        print(f"Final vessel max activation {vessel.max()}")

        # Activations
        poly_out = poly_out[0]

        tqdm_iter.progress(80)

        return poly_out, airway, vessel
        