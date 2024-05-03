'''
Copyright (c) Diedre Carmo, Medical Imaging Computing Lab (MICLab)
https://miclab.fee.unicamp.br/
https://github.com/MICLab-Unicamp/medpseg
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
'''
import os
import torch
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from tqdm import tqdm
from medpseg.utils.uncertainty import get_epistemic_uncertainty
from collections import defaultdict


class E2DStackDataset():
    '''
    Speed up evaluation time slice stacking with dataloader compatible dataset
    '''
    def __init__(self, volume, extended_2d):
        self.volume = volume
        self.limits = [0, volume.shape[2] - 1 ]
        self.extended_2d = extended_2d
    
    def __len__(self):
        return self.volume.shape[2]

    def __getitem__(self, i):
        if self.extended_2d is None:
            input_slice = self.volume[:, :, i]
        else:
            central_slice = self.volume[:, :, i]
            input_slice = []
            for extend_i in range(-self.extended_2d, self.extended_2d + 1):
                if extend_i == 0:
                    input_slice.append(central_slice)
                    continue

                new_i = i + extend_i
                if new_i > self.limits[1]:
                    new_i = self.limits[1]
                if new_i < self.limits[0]:
                    new_i = self.limits[0]
                
                input_slice.append(self.volume[:, :, new_i])
            input_slice = torch.cat(input_slice, dim=1)
        '''
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(input_slice[0, 0].detach().cpu().numpy(), cmap="gray")
        plt.subplot(1, 3, 2)
        plt.imshow(input_slice[0, 1].detach().cpu().numpy(), cmap="gray")
        plt.subplot(1, 3, 3)
        plt.imshow(input_slice[0, 2].detach().cpu().numpy(), cmap="gray")
        plt.show()
        '''
        return input_slice[0]

    def get_dataloader(self, batch_size, pin_memory, num_workers):
        return DataLoader(self, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers)



def stack_predict(model, volume, batch_size, extended_2d=None, device=torch.device("cuda:0"), get_uncertainty=None, info_q=None):
    '''
    Stack predict function for evaluation prediction of 3D volumes with E2D (2.5D) inputs
    '''
    e2d_stack_dataloader = E2DStackDataset(volume, extended_2d=extended_2d).get_dataloader(batch_size=batch_size, pin_memory=True, num_workers=cpu_count()//2)
    
    outs = []
    if get_uncertainty is not None:
        mean_predictions = []
        epistemic_uncertainties = []

    for input_slice in tqdm(e2d_stack_dataloader, desc=f"Slicing with batch size {batch_size}. Uncertainty: {get_uncertainty}"):
        package = input_slice[0].numpy().transpose(1, 2, 0).copy()
        info_q.image_to_front_end(package)
        outs.append(model(input_slice.to(device)).cpu())
        if get_uncertainty is not None:
            epistemic_uncertainty, mean_prediction = get_epistemic_uncertainty(model, input_slice.to(device), n=get_uncertainty)
            mean_predictions.append(mean_prediction)
            epistemic_uncertainties.append(epistemic_uncertainty)

    outs = torch.cat(outs).unsqueeze(0).permute(0, 2, 1, 3, 4).numpy()
    if get_uncertainty is not None:
        epistemic_uncertainties = torch.cat(epistemic_uncertainties).unsqueeze(0).permute(0, 2, 1, 3, 4).numpy()
        mean_predictions = torch.cat(mean_predictions).unsqueeze(0).permute(0, 2, 1, 3, 4).numpy()
        
    if get_uncertainty:
        return outs[0], epistemic_uncertainties[0], mean_predictions[0]
    else:  # old multitask validator behaviour
        return outs
    

def multi_view_consensus(models, orientations, tqdm_iter, batch_size, extended_2d, device, info_q=None): 
    '''
    Currently not in use, aggregates models predicting in multiple directions
    '''       
    y_hats = []
    for i, x_axis in enumerate(orientations):
        if tqdm_iter is not None:
            tqdm_iter.write(f"Multi View Consensus Slicing from axis {i}")
        if info_q is not None and i != 0:
            info_q.rot90 = True
        y_hat_axis = stack_predict(models[i].to(device), x_axis, batch_size=batch_size, extended_2d=extended_2d, device=device, info_q=info_q)
        
        if i == 1:
            corrected_output = y_hat_axis.transpose(0, 1, 3, 2, 4)
        elif i == 2:
            corrected_output = y_hat_axis.transpose(0, 1, 3, 4, 2)
        else:
            corrected_output = y_hat_axis

        y_hats.append(corrected_output)
        models[i].cpu()
    y_hat = y_hats[0] + y_hats[1] + y_hats[2]
    y_hat = y_hat/3

    if info_q is not None:
        info_q.rot90 = False

    return y_hat


def argon_cpu_count() -> int:
    if os.getenv("NSLOTS") is not None:
        return int(os.getenv("NSLOTS"))
    else:
        return cpu_count()
    

def real_time_stack_predict(model, volume, batch_size, extended_2d=None, num_workers=argon_cpu_count(), device=torch.device("cuda:0")):
    '''
    For use during training validation with a PL module. If you want to stack predictions on a testing script, use the "stack_predict" function
    Need to be transient on vRAM usage, we don't want to hog GPU memory with 3D tensors
    '''
    with torch.no_grad():
        volume = volume.cpu()  # There should be a better way to do this
        l_outputs = []
        d_outputs = defaultdict(list)
        for input_slice in E2DStackDataset(volume, extended_2d=extended_2d).get_dataloader(batch_size=batch_size, pin_memory=False, num_workers=num_workers):
            output = model(input_slice.to(device), stacking=True)
            if isinstance(output, dict):
                for k, v in output.items():
                    if k != "latent_space":  # ATTENTION: Latent space is excluded from val_3d, makes no sense in this context
                        d_outputs[k].append(v.cpu())  # unfortunately we cant stack gpu memory here 
            else:
                l_outputs.append(output.cpu())
        
        # Stack on the GPU but send out in the CPU for metrics and dice computation. 
        # This shouldnt impact performance much because the move to CPU will be necessary for seg_metrics anyway
        if len(l_outputs) == 0:
            for k in d_outputs.keys():
                d_outputs[k] = torch.cat(d_outputs[k]).unsqueeze(0).permute(0, 2, 1, 3, 4).cpu()
            # return torch.cat([v for v in d_outputs.values()], dim=1)  # no reason to concatenate in a single return just return the dict
            return d_outputs
        else:
            return torch.cat(l_outputs).unsqueeze(0).permute(0, 2, 1, 3, 4).cpu()
    