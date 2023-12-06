'''
Copyright (c) Diedre Carmo, Medical Imaging Computing Lab (MICLab)
https://miclab.fee.unicamp.br/
https://github.com/MICLab-Unicamp/medpseg
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

This is a way more complicated version of seg_2d_module, to deal with ANY target (polymorphic and multitasking targets)
coming from the dataloader.

In this public verison of the file, I have removed all deprecated and old experiment code that makes sense being removed (31 Oct 2023)
-Removed all deprecated code
-Removed Con Detect and POI loss experiments

It should be close to minimal to being able to reproduce our best experiment
'''
import os
import torch
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from collections import defaultdict
from typing import Dict, Callable, Union, List
from torchvision.transforms import Resize, InterpolationMode

from medpseg.architecture import MEDSeg
from medpseg.utils import get_optimizer
from medpseg.utils import DICEMetric
from medpseg.eval_2d_utils import real_time_stack_predict
from medpseg.combined_loss import CombinedLoss
from medpseg.seg_metrics import seg_metrics


class PolySeg2DModule(pl.LightningModule):
    '''
    Regarding of the name, also works with 3D networks
    '''
    def __init__(self, hparams):
        '''
        Check starter.py for description of all hparams
        '''
        super().__init__()
        self.save_hyperparameters(hparams)

        ####### Hyperparameters used during development, ignore this its confusing #######
        self.pretraining = self.hparams.pretraining
        self.findings_only = getattr(self.hparams, "findings_only", False)
        self.weight_decay = getattr(self.hparams, "weight_decay", None)
        self.scheduling_factor = getattr(self.hparams, "scheduling_factor", None)
        self.scheduling = getattr(self.hparams, "scheduling", "step")
        self.scratch = getattr(self.hparams, "scratch", False)
        self.expand_bifpn = getattr(self.hparams, "expand_bifpn", "conv")
        self.backbone = getattr(self.hparams, "backbone", "effnet")
        self.val_3d = getattr(self.hparams, "val_3d", False)
        self.gdl = getattr(self.hparams, "gdl", False)
        self.bdl = getattr(self.hparams, "bdl", False)
        self.focal = getattr(self.hparams, "focal", False)
        self.atmbranch = getattr(self.hparams, "atmbranch", None)
        self.vesselbranch = getattr(self.hparams, "vesselbranch", None)
        self.recbranch = getattr(self.hparams, "recbranch", None)
        self.include_bg = getattr(self.hparams, "include_background", False)
        self.unet = getattr(self.hparams, "unet", False)
        self.unettr = getattr(self.hparams, "unettr", False)
        self.poly_level = getattr(self.hparams, "poly_level", None)
        self.flag_3d_metric = '_3d' if self.val_3d or self.unettr else ''
        self.excluded_average_metric_keys = ["volume_similarity", "avg_hd", "hd"]
        self.downstream_method = getattr(self.hparams, "downstream_method", None)
        self.perceptual_loss = getattr(self.hparams, "perceptual_loss", False)
        self.stem_replacement = getattr(self.hparams, "stem_replacement", False)
        self.new_latent_space = getattr(self.hparams, "new_latent_space", False)
        self.compound_coef = getattr(self.hparams, "compound_coef", 4)
        self.consistency = getattr(self.hparams, "consistency", False)
        self.imnet_norm = getattr(self.hparams, "imnet_norm", False)
        self.learnable_norm = getattr(self.hparams, "learnable_norm", False)
        self.circulatory_branch = getattr(self.hparams, "circulatory_branch", None)
        self.bifpn_channels = getattr(self.hparams, "bifpn_channels", 128)
        self.combined_loss = getattr(self.hparams, "combined_loss", False)
        self.sam = getattr(self.hparams, "sam", False)
        self.freeze_encoder = getattr(self.hparams, "freeze_encoder", False)
        self.batchfy_e2d = getattr(self.hparams, "batchfy_e2d", False)
        self.circulatory_regularization = getattr(self.hparams, "circulatory_regularization", False)
        self.medseg3d = getattr(self.hparams, "medseg3d", False)
        self.fpn_c = getattr(self.hparams, "fpn_c", None)
        # Post ATS ideas
        self.soft_circulatory = getattr(self.hparams, "soft_circulatory", False)
        self.poi_loss = getattr(self.hparams, "poi_loss", False)
        self.nrdice_loss = getattr(self.hparams, "nrdice_loss", False)
        self.polyunet25d = getattr(self.hparams, "polyunet25d", False)
        self.polyunet3d = getattr(self.hparams, "polyunet3d", False)
        self.mccl = getattr(self.hparams, "mccl", False)
        self.tversky = getattr(self.hparams, "tversky", False)
        self.airway_ths = getattr(self.hparams, "airway_ths", 0.5)
        self.vessel_ths = getattr(self.hparams, "vessel_ths", 0.5)
        self.self_attention = getattr(self.hparams, "self_attention", False)
        self.deep_supervision = getattr(self.hparams, "deep_supervision", False)
        self.con_detect = getattr(self.hparams, "con_detect", False)
        self.celoss = getattr(self.hparams, "celoss", False)
        self.large = getattr(self.hparams, "large", False)
        self.combined_gdl = getattr(self.hparams, "combined_gdl", False)
        self.full_silver = getattr(self.hparams, "preprocess", '') == "full_silver_poly_3levels_circulatory"
        if self.full_silver:
            print("Full silver mode detected, every item on batch must be fullsilver preprocess")
        ####### Hyperparameters used during development, ignore this its confusing #######

        # Determine offset for polymorphic labels depending on poly level
        # Poly level:
        # None: supervised training only
        # 0: self supervised only
        # 2: lung -> unhealthy/healthy
        # 3: unhealthy -> GGO/CON
        self.nlossterms = 0
        if self.poly_level == 3:  # Previous logic for this was wrong, changing to count from beginning
            self.simple_offset = 2  #  BG + Lung
            self.detailed_offset = 3  # BG + Healthy + Unhealthy
        else:
            self.simple_offset = 2 # BG + Lung
            self.detailed_offset = None  # Not present if not poly_level 3

        # Redundant argument necessary to not tie module to data preprocessing
        if "poly_3levels" in self.hparams.preprocess:
            assert self.poly_level == 3 or self.poly_level == 2

        self.two5d = True
        self.model = MEDSeg(self.hparams.nin, self.hparams.seg_nout, apply_sigmoid=False, backbone=self.backbone, expand_bifpn=self.expand_bifpn, pretrained=not self.scratch,
                            num_classes_atm=self.atmbranch, num_classes_vessel=self.vesselbranch, num_classes_rec=self.recbranch, stem_replacement=self.stem_replacement, new_latent_space=self.new_latent_space,
                            compound_coef=self.compound_coef, imnet_norm=self.imnet_norm, learnable_norm=self.learnable_norm, circulatory_branch=self.circulatory_branch,
                            bifpn_channels=self.bifpn_channels, sam_embedding=self.sam, self_attention=self.self_attention, deep_supervision=self.deep_supervision,
                            con_detecting=self.con_detect, large=self.large, soft_circulatory=self.soft_circulatory)
    
        self.pretrained_weights = self.hparams.pretrained_weights
        if self.pretrained_weights is not None:
            print(f"Loading pretrained weights from {self.pretrained_weights}")
            self.model = PolySeg2DModule.load_from_checkpoint(self.pretrained_weights).model

        # Supervised loss
        assert (not(self.combined_loss) or not(self.nrdice_loss)) and (not(self.combined_loss) or not(self.mccl)) and (not(self.nrdice_loss) or not(self.mccl)), "Cant do combined loss and nrdice loss or combined loss and mccl at the same time"
        
        if self.combined_loss:
            print("Combined Loss")
            self.lossfn = CombinedLoss(include_background=self.include_bg, cross_entropy=self.celoss, gdl=self.combined_gdl, soft_circulatory=self.soft_circulatory)
        self.dicer = DICEMetric(per_channel_metric=True, check_bounds=False)

        print('-'*100 + 
              f"\nPoly2D Module in the following configuration:"
              f"\npoly_level: {self.poly_level} soft_circulatory: {self.soft_circulatory}"
              f"\nnin: {self.hparams.nin} main_nout: {self.hparams.seg_nout}, DS: {self.deep_supervision}, SA: {self.self_attention}"
              f"\nMEDSeg 3D? {self.medseg3d}\n" +
              '-'*100)

    def save_pt_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_pt_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def visual_debug(self, x, y, label):
        pass

    def forward(self, x, stacking=False):
        if self.val_3d and not self.training and not stacking:  # either training, or bein in val_3d or stacking flag avoids this branch and...
            return real_time_stack_predict(self, x, self.hparams.eval_batch_size, extended_2d=self.hparams.extended_2d, num_workers=self.hparams.nworkers, device=torch.device("cpu") if self.hparams.cpu else x.device)
        else:  #  ...we return direct slice activations
            y_hat = self.model(x) 
            if isinstance(y_hat, dict):
                for k in y_hat.keys():
                    if 'atm' in k or 'vessel' in k:
                        if self.soft_circulatory:
                            y_hat[k] = y_hat[k].softmax(dim=1)    
                        else:
                            y_hat[k] = y_hat[k].sigmoid()
                    elif 'main' in k:
                        y_hat[k] = y_hat[k].softmax(dim=1)
                    else:
                        raise ValueError(f"Unexpected key in MEDSeg return: {k}")
                if self.hparams.debug and not stacking:
                    print("y_hat state:")
                    for k, v in y_hat.items():
                        print(f"{k}: {v.shape}")
            else:
                y_hat = y_hat.softmax(dim=1)
                if self.hparams.debug and not stacking:
                    print(f"y_hat state: {y_hat.shape}")
            
            return y_hat

    # Main branch forms ##################################
    def simple_level(self, y_hat, y, simple, ds, do_loss):
        '''
        Where we train on lung masks only. 
        '''
        if self.full_silver and self.training:
            raise RuntimeError("Shouldn't be running simple_level on full_silver")
        
        if isinstance(y_hat, dict):
            lung = y_hat["main"][simple, 1:].sum(dim=1, keepdim=True)  # lung is everything after bg summed
            y_hat_simple = torch.cat([y_hat["main"][simple, :1], lung], dim=1)  # 2 channel bg + lung on simple cases
        else:
            lung = y_hat[simple, 1:].sum(dim=1, keepdim=True) # lung is everything after bg summed
            y_hat_simple = torch.cat([y_hat[simple, :1], lung], dim=1)  # bg + lung on simple cases
        
        # WANING: Boundary Loss deprecated, no significant difference shown 
        if self.simple_offset is None:  # poly simplification removes unhealthy label
            y_simple = y[simple]  
        else:
            y_simple = y[simple, :self.simple_offset]  
        NS = y_simple.shape[0]
        
        # Loss can be disabled to accelerate validation
        if do_loss:
            simple_loss = self.lossfn(y_hat_simple, y_simple)
        else:
            simple_loss = 0
        
        # Complex metrics on real time
        if not self.training:
            if self.val_3d:
                y_hat_simple_argmax = y_hat_simple.argmax(dim=1, keepdim=True)
                y_hat_lung = y_hat_simple_argmax == 1
                for ns in range(NS):
                    struct_names = ["lung"]
                    seg_metrics(gts=y_simple[ns, 1:2].cpu().numpy().astype(np.uint8), preds=y_hat_lung.detach().cpu().numpy().astype(np.uint8),
                                metrics=self.metrics, struct_names=struct_names)
                    for key, value in self.metrics.items():
                        for metric, metric_value in value.items():
                            if key in struct_names:
                                self.log(f"{key}_{metric}_3d", metric_value[-1], on_epoch=True, on_step=False, prog_bar=False)
            else:
                raise NotImplementedError("2D validation for simplified level not implemented")
        
        return simple_loss

    def detailed_level(self, y_hat, y, detailed, ds, do_loss):
        '''
        Where we train on Healthy/Unhealthy masks
        Still supports old 2.5D validation metrics do pretraining project
        '''
        if self.full_silver and self.training:
            raise RuntimeError("Shouldn't be running detailed_level on full_silver")
        
        if isinstance(y_hat, dict): 
            if self.poly_level == 3:  # if we have ggo and con outputs, reduce then
                unhealthy = y_hat["main"][detailed, 2:].sum(dim=1, keepdim=True)  # GGO + CON = unhealthy
                y_hat_detailed = torch.cat([y_hat["main"][detailed, :2], unhealthy], dim=1)  # Concating BG, Healthy with unhealthy
            else:
                y_hat_detailed = y_hat["main"][detailed]
        else:
            if self.poly_level == 3:  # if we have ggo and con outputs, reduce then
                unhealthy = y_hat[detailed, 2:].sum(dim=1, keepdim=True)  # GGO + CON = unhealthy
                y_hat_detailed = torch.cat([y_hat[detailed, :2], unhealthy], dim=1)  # Concating BG, Healthy with unhealthy
            else:
                y_hat_detailed = y_hat[detailed]
        
        # Logic to separate concatenations on x and y. Kind of complicated
        # Although boundary loss is implemented, early experiments showed it not being signifcantly better so, deprecated.
        if self.detailed_offset is None:
            y_detailed = y[detailed]
        else:
            y_detailed = y[detailed, :self.detailed_offset]  
        ND = y_detailed.shape[0]

        # Loss can be disabled to accelerate validation
        if do_loss:
            detailed_loss = self.lossfn(y_hat_detailed, y_detailed)
        else:
            detailed_loss = 0
        
        # Complex metrics on real time
        if not self.training:
            if self.val_3d:
                y_hat_detailed_argmax = y_hat_detailed.argmax(dim=1, keepdim=True)
                y_hat_detailed = torch.cat((y_hat_detailed_argmax == 1,  y_hat_detailed_argmax == 2), dim=1)
                for nd in range(ND):
                    struct_names = ["healthy", "unhealthy"]
                    seg_metrics(gts=y_detailed[nd, 1:3].cpu().numpy().astype(np.uint8), preds=y_hat_detailed[nd, :2].detach().cpu().numpy().astype(np.uint8),
                                metrics=self.metrics, struct_names=struct_names)
                    for key, value in self.metrics.items():
                        for metric, metric_value in value.items():
                            if key in struct_names:
                                self.log(f"{key}_{metric}_3d", metric_value[-1], on_epoch=True, on_step=False, prog_bar=False)
            else:
                healthy_metric, unhealthy_metric = self.dicer(y_hat_detailed[:, 1:3], y_detailed[:, 1:3])
                self.log("healthy_dice", healthy_metric, on_epoch=True, on_step=False, prog_bar=False)
                self.log("unhealthy_dice", unhealthy_metric, on_epoch=True, on_step=False, prog_bar=False)

        return detailed_loss

    def separation_level(self, y_hat, y, separation, ds, do_loss):
        '''
        Where we train on separating GGO and Consolidations 
        (semi-supervised through threshold + unhealthy label)

        One day might be manual labels too
        '''
        if isinstance(y_hat, dict):
            y_hat_separation = y_hat["main"][separation][:, :4]
        else:
            y_hat_separation = y_hat[separation][:, :4]

        y_separation = y[separation][:, :4]
        ND = y_separation.shape[0]

        # Loss can be disabled to accelerate validation
        if do_loss:
            separation_loss = self.lossfn(y_hat_separation, y_separation)
        else:
            separation_loss = 0
        
        # Complex metrics on real time
        if not self.training:
            if self.val_3d:
                y_hat_separation_argmax = y_hat_separation.argmax(dim=1, keepdim=True)
                y_hat_separation = torch.cat((y_hat_separation_argmax == 2,  y_hat_separation_argmax == 3), dim=1)
                for nd in range(ND):
                    struct_names = ["ggo", "con"]
                    seg_metrics(gts=y_separation[nd, 2:4].cpu().numpy().astype(np.uint8), preds=y_hat_separation[nd, :2].detach().cpu().numpy().astype(np.uint8),
                                metrics=self.metrics, struct_names=struct_names)
                    for key, value in self.metrics.items():
                        for metric, metric_value in value.items():
                            if key in struct_names:
                                self.log(f"{key}_{metric}_3d", metric_value[-1], on_epoch=True, on_step=False, prog_bar=False)

        return separation_loss
    ####################################################

    # ATM branch computations
    def atm_branch(self, y_hat, y, atm, ds, do_loss):
        '''
        where we optimize atm parts of the batch, binary label
        '''
        if self.full_silver and self.training:
            if self.soft_circulatory:
                bg = torch.ones_like(y[atm, 5:6]) - y[atm, 5:6]
                y_airway = torch.cat([bg, y[atm, 5:6]], dim=1)
                y_hat_airway = y_hat["atm"][atm, :2]  
            else:
                raise RuntimeError("Why are you running full_silver without SoftCirculatory")
        else:
            if self.soft_circulatory:
                y_airway = y[atm, :2]  # Taking one hot map
                y_hat_airway = y_hat["atm"][atm, :2]  # output has 2 channels
            else:
                y_airway = y[atm, 1:2]  # 0 is BG, taking binary airway map
                y_hat_airway = y_hat["atm"][atm, :1]  # output has only 1 channel
        NS = y_airway.shape[0]  # nsamples
        
        # Loss can be disabled to accelerate validation
        if do_loss:
            atm_loss = self.lossfn(y_hat_airway, y_airway)
        else:
            atm_loss = 0
        
        # Complex metrics on real time
        if not self.training:
            if self.val_3d:
                # Making sure to get the correct activation when softmax (soft_circulatory) is turned on.
                if self.soft_circulatory:
                    # Note that this is already 0 and 1 after argmax
                    binary_y_hat_airway = y_hat_airway.detach().argmax(dim=1, keepdim=True).cpu().numpy().astype(np.uint8)
                    binary_y_airway = y_airway[:, 1:2].cpu().numpy().astype(np.uint8)
                else:
                    # Split sigmoid on THS
                    binary_y_hat_airway = (y_hat_airway.detach() > self.airway_ths).cpu().numpy().astype(np.uint8)
                    binary_y_airway = y_airway[:, 0:1].cpu().numpy().astype(np.uint8)
                assert binary_y_hat_airway.shape[1] == 1 and binary_y_hat_airway.max() <= 1

                for ns in range(NS):
                    struct_names = ["airway"]
                    seg_metrics(gts=binary_y_airway[ns], 
                                preds=binary_y_hat_airway[ns],
                                metrics=self.metrics, 
                                struct_names=struct_names)
                    for key, value in self.metrics.items():
                        for metric, metric_value in value.items():
                            if key in struct_names:
                                self.log(f"{key}_{metric}_3d", metric_value[-1], on_epoch=True, on_step=False, prog_bar=False)
            else:
                raise NotImplementedError("2D validation for atm not implemented")
        
        return atm_loss

    # Vessel branch computations
    def vessel_branch(self, y_hat, y, vessel, ds, do_loss):
        '''
        where we optimize atm parts of the batch
        '''
        '''
        where we optimize atm parts of the batch, binary label
        '''
        if self.full_silver and self.training:
            if self.soft_circulatory:
                bg = torch.ones_like(y[vessel, 4:5]) - y[vessel, 4:5]
                y_vessel = torch.cat([bg, y[vessel, 4:5]], dim=1)
                y_hat_vessel = y_hat["vessel"][vessel, :2]  
            else:
                raise RuntimeError("Why are you running full_silver without SoftCirculatory")
        else:
            if self.soft_circulatory:
                y_vessel = y[vessel, :2]  # Taking one hot map
                y_hat_vessel = y_hat["vessel"][vessel, :2]  # output has 2 channels
            else:
                y_vessel = y[vessel, 1:2]  # 0 is BG, taking binary airway map
                y_hat_vessel = y_hat["vessel"][vessel, :1]  # output has only 1 channel
        
        NS = y_vessel.shape[0]  # nsamples
        
        # Loss can be disabled to accelerate validation
        if do_loss:
            vessel_loss = self.lossfn(y_hat_vessel, y_vessel)
        else:
            vessel_loss = 0
        
        # Complex metrics on real time
        if not self.training:
            if self.val_3d:
                # Making sure to get the correct activation when softmax (soft_circulatory) is turned on.
                if self.soft_circulatory:
                    # Note that this is already 0 and 1 after argmax
                    binary_y_hat_vessel = y_hat_vessel.detach().argmax(dim=1, keepdim=True).cpu().numpy().astype(np.uint8)
                    binary_y_vessel = y_vessel[:, 1:2].cpu().numpy().astype(np.uint8)
                else:
                    # Split sigmoid on THS
                    binary_y_hat_vessel = (y_hat_vessel.detach() > self.vessel_ths).cpu().numpy().astype(np.uint8)
                    binary_y_vessel = y_vessel[:, 0:1].cpu().numpy().astype(np.uint8)
                assert binary_y_hat_vessel.shape[1] == 1 and binary_y_hat_vessel.max() <= 1

                for ns in range(NS):
                    struct_names = ["vessel"]
                    seg_metrics(gts=binary_y_vessel[ns], 
                                preds=binary_y_hat_vessel[ns],
                                metrics=self.metrics, 
                                struct_names=struct_names)
                    for key, value in self.metrics.items():
                        for metric, metric_value in value.items():
                            if key in struct_names:
                                self.log(f"{key}_{metric}_3d", metric_value[-1], on_epoch=True, on_step=False, prog_bar=False)
            else:
                raise NotImplementedError("2D validation for vessel not implemented")
        
        return vessel_loss

    def debug_batch(self, simple, detailed, separation, atm, vessel, y, meta):
        if self.hparams.debug:
            print(f"Training? {self.training}")
            print("Simple")
            print(simple)
            print("Detailed")
            print(detailed)
            print("Separation")
            print(separation)
            print("ATM")
            print(atm)
            print("Vessel (parse)")
            print(vessel)
        
            # Assuming B, C, ... format
            preprocess = meta["preprocess"]
            import matplotlib.pyplot as plt
            for i, y_item in enumerate(y):
                item_preprocess = preprocess[i]
                print(y_item.max())
                display_buffer = y_item.cpu().argmax(dim=0).numpy()
                print(display_buffer.max())
                print(f"Display buffer: {display_buffer.shape}")
                if os.getenv("NSLOTS") is None:
                    if len(display_buffer.shape) == 3:
                        pass
                    else:
                        plt.title(f"Batch target {i} preprocess {item_preprocess}")
                        plt.imshow(display_buffer)
                        plt.show()

    def deep_supervision_fn(self, 
                            loss_fn: Callable, 
                            key: str, 
                            y_hat: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                            y: torch.Tensor, 
                            index: np.ndarray, 
                            do_loss: bool):
        loss_acum = []
        
        for i in range(1, 5):
            current_size = (y_hat[key].shape[-2], y_hat[key].shape[-1])
            current_size = (current_size[0]//(2**(i)), current_size[1]//(2**(i)))
            
            transform = Resize(current_size, interpolation=InterpolationMode.NEAREST)
            
            # Craft prediction and target for deep supervision outputs
            new_y_hat = {}

            if key == "main":
                new_y_hat[key] = y_hat[f"{key}{i}"]
            elif key == "vessel" or key == "atm":
                new_y_hat[key] = y_hat[f"{key}{i}"]
            else:
                raise ValueError(f"Key {key} not valid")

            new_y = transform(y)
            loss = loss_fn(new_y_hat, new_y, index, True, do_loss)

            loss_acum.append(loss)

        return loss_acum

    def compute_loss(self, 
                     loss_fn: Callable, 
                     key: str, 
                     y_hat: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                     y: torch.Tensor, 
                     index: np.ndarray, 
                     do_loss: bool, 
                     deep_supervision: bool):
        if index.sum() > 0:
            loss = loss_fn(y_hat, y, index, False, do_loss)
            if deep_supervision and self.training:
                loss_acum = self.deep_supervision_fn(loss_fn, key, y_hat, y, index, do_loss)
                # Due to observing good results with only high resolution loss in poly, bumping high resolution weight in optimization
                # To 0.75, with rest of DS contributing to 0.25 of optimization
                loss = ((2**-1)+(2**-2))*loss + (2**-3)*loss_acum[0] + (2**-4)*loss_acum[1] + (2**-5)*loss_acum[2] + (2**-6)*loss_acum[3]
                for i in range(5):
                    self.log(f"{loss_fn.__name__}_deep_supervision_{i}", loss if i == 0 else loss_acum[i-1], prog_bar=False, on_step=True, on_epoch=True)
        else:
            loss = 0

        return loss

    def loss_wrapper(self, 
                     y_hat: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                     y: torch.Tensor, 
                     indexes: Dict[str, np.ndarray], 
                     do_loss: bool, 
                     deep_supervision: bool):
        simple, detailed, separation, atm, vessel = indexes["simple"], indexes["detailed"], indexes["separation"], indexes["atm"], indexes["vessel"]

        simple_loss = self.compute_loss(self.simple_level, "main", y_hat, y, simple, do_loss, deep_supervision)
        detailed_loss = self.compute_loss(self.detailed_level, "main", y_hat, y, detailed, do_loss, deep_supervision)
        separation_loss = self.compute_loss(self.separation_level, "main", y_hat, y, separation, do_loss, deep_supervision)
        atm_loss = self.compute_loss(self.atm_branch, "atm", y_hat, y, atm, do_loss, deep_supervision)
        vessel_loss = self.compute_loss(self.vessel_branch, "vessel", y_hat, y, vessel, do_loss, deep_supervision)

        if do_loss and simple_loss == 0 and detailed_loss == 0 and atm_loss == 0 and separation_loss == 0 and vessel_loss == 0:
            print(">>>>>>>>>>>>>WARNING: Malformed batch, didn't find any level of polymorphism!<<<<<<<<<<<<<")

        return simple_loss, detailed_loss, separation_loss, atm_loss, vessel_loss

    def polymorphic_loss_metrics(self, 
                                 y: torch.Tensor, 
                                 y_hat: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                                 meta: Dict[str, List[str]], 
                                 do_loss: bool = True):
        '''
        ####### Polymorphic training #############
        # Indexes whole batch and perform loss computations separately
        '''
        detailed = np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.array(meta["preprocess"]) == "seg_raw_new", np.array(meta["preprocess"]) == "seg_raw"), np.array(meta["preprocess"]) == "msd_seg"),  np.array(meta["preprocess"]) == "seg_raw_new_hu"), np.array(meta["preprocess"]) == "msd_seg_hu")   # Level 2 polymorphism, healthy/unhealthy annotation, cancer
        simple = np.logical_or(np.logical_or(np.array(meta["preprocess"]) == "pretrain_preprocessing", np.array(meta["preprocess"]) == "classification_pretrain_preprocessing"), np.array(meta["preprocess"]) == "pretrain_preprocessing_hu")  # Level 1 polymorphism, lung annotation
        separation = np.logical_or(np.array(meta["preprocess"]) == "separation", np.array(meta["preprocess"]) == "manual_split_msc_hu")  # Level 3 polymorphism detect artificial con/ggo separation and correction with transform
        atm = np.logical_or(np.array(meta["preprocess"]) == "new_atm", np.array(meta["preprocess"]) == "new_atm_hu")  # Auxiliary task, airway segmentation
        vessel = np.logical_or(np.array(meta["preprocess"]) == "parse", np.array(meta["preprocess"]) == "parse_hu")  # Auxiliary task, vessel segmentation

        if self.full_silver and self.training:
            # The case where every batch item has everything, from teacher network labeling
            separation = np.array([True]*y.shape[0])
            atm = np.array([True]*y.shape[0])
            vessel = np.array([True]*y.shape[0])

        self.debug_batch(simple, detailed, separation, atm, vessel, y, meta)

        indexes = {"simple": simple, "detailed": detailed, "separation": separation, "atm": atm, "vessel": vessel}

        return self.loss_wrapper(y_hat, y, indexes, do_loss, deep_supervision=self.deep_supervision)

    def supervised_loss(self, y, y_hat, meta, prestr):
        '''
        Does all the dozens of losses involved in this training
        This function also computes and logs metrics internally. Only losses are returned to compute the final loss
        '''
        simple_loss, detailed_loss, separation_loss, atm_loss, vessel_loss = self.polymorphic_loss_metrics(y=y, y_hat=y_hat, meta=meta, do_loss=True)
        
        loss = simple_loss + detailed_loss + separation_loss + atm_loss + vessel_loss
        if loss is not None:
            if self.training:
                if simple_loss > 0:
                    self.nlossterms += 1
                    self.log(f"{prestr}simple_loss", simple_loss, on_step=True, on_epoch=True)
                if detailed_loss > 0: 
                    self.nlossterms += 1
                    self.log(f"{prestr}detailed_loss", detailed_loss, on_step=True, on_epoch=True)
                if separation_loss > 0:
                    self.nlossterms += 1
                    self.log(f"{prestr}separation_loss", separation_loss, on_step=True, on_epoch=True)
                if atm_loss > 0:
                    self.nlossterms += 1
                    self.log(f"{prestr}atm_loss", atm_loss, on_step=True, on_epoch=True)
                if vessel_loss > 0:
                    self.nlossterms += 1
                    self.log(f"{prestr}vessel_loss", vessel_loss, on_step=True, on_epoch=True)
                
                self.log(f"{prestr}loss", loss, on_step=True, on_epoch=True)
            else:
                if simple_loss > 0:
                    self.nlossterms += 1
                    self.log(f"{prestr}simple_loss{self.flag_3d_metric}", simple_loss, on_step=True, on_epoch=True)
                if detailed_loss > 0:
                    self.nlossterms += 1
                    self.log(f"{prestr}detailed_loss{self.flag_3d_metric}", detailed_loss, on_step=True, on_epoch=True)
                if separation_loss > 0:
                    self.nlossterms += 1
                    self.log(f"{prestr}separation_loss{self.flag_3d_metric}", separation_loss, on_step=True, on_epoch=True)
                if atm_loss > 0:
                    self.nlossterms += 1
                    self.log(f"{prestr}atm_loss{self.flag_3d_metric}", atm_loss, on_step=True, on_epoch=True)
                if vessel_loss > 0:
                    self.nlossterms += 1
                    self.log(f"{prestr}vessel_loss", vessel_loss, on_step=True, on_epoch=True)
                
                self.log(f"{prestr}loss{self.flag_3d_metric}", loss, on_step=True, on_epoch=True)

        return loss

    def training_step(self, train_batch, batch_idx):
        '''
        Training step does different things if on exclusive pretraining mode or 
        doing traditional supervision.

        We only need to return loss for optimizer, metrics are not computed
        '''
        self.nlossterms = 0
        x, y, meta = train_batch
        self.visual_debug(x, y, "Training")
        
        y_hat = None

        if self.poly_level != 0:  # zero polymorphic means pretraining only
            # Traditional supervision
            if y_hat is None:
                y_hat = self.forward(x)

            supervised_loss = self.supervised_loss(y=y, y_hat=y_hat, meta=meta, prestr='')
            self.log("supervised_loss", supervised_loss, on_step=True, on_epoch=True)
        else:
            supervised_loss = 0
        
        final_loss = supervised_loss/self.nlossterms
        self.log("nlossterms", self.nlossterms, on_step=True, on_epoch=True)
        self.log("loss", final_loss, on_step=True, on_epoch=True)

        if final_loss == 0:
            raise ValueError("Loss is equal to 0. Something is misconfigured.")

        return final_loss  # for outside optimization

    def validation_step(self, val_batch, batch_idx):
        '''
        Validation step does different things if on exclusive pretraining mode or 
        doing traditional supervision

        There is no return but metrics are computed in 3D (takes a while)
        for pretraining loss is used as a validation metric. 

        When using boundary loss, we are not computing it in 3D validation.
        '''
        self.nlossterms = 0
        x, y, meta = val_batch
        self.visual_debug(x, y, "Validation")
        
        y_hat = None
        preproc = meta["preprocess"][0]
        if preproc == "pretrain_preprocessing" and self.val_3d:
            print(f"Skipping no label 3D validation {preproc}")
            return
        
        
        if self.poly_level != 0:
            # Traditional supervision
            if y_hat is None:
                y_hat = self.forward(x)
            
            # Compute loss and metrics on CPU due to val_3d memory usage
            if self.val_3d:
                if isinstance(y_hat, dict):
                    for _, value in y_hat.items():
                        if value.device == torch.device("cpu"):
                            y = y.to(value.device)
                            break
                elif y_hat.device == torch.device("cpu"):
                    y = y.to(y_hat.device)
            
            supervised_loss = self.supervised_loss(y=y, y_hat=y_hat, meta=meta, prestr="val_")
        else:
            supervised_loss = 0
        
        # We only compute validation loss when not using val_3d, since 3D validation loss is very heavy on gpu[
        if self.nlossterms != 0:
            final_loss = supervised_loss/self.nlossterms
            self.log("val_nlossterms", self.nlossterms, on_step=True, on_epoch=True)
            self.log("val_supervised_loss", supervised_loss, on_step=True, on_epoch=True)
            self.log("val_loss", final_loss, on_step=True, on_epoch=True)
    
    def on_validation_epoch_start(self):
        '''
        Start of validation epoch tasks:
        Initialize metric dictionary and list of IDs
        '''
        # Reset metric dict
        if self.val_3d:
            self.metrics: Dict = defaultdict(lambda: defaultdict(list))
        
    def on_validation_epoch_end(self):
        '''
        End of epoch tasks:
        - Increment BDL weights
        - Print results so far in terminal (stdout) for backup logging
        '''
        if self.bdl:
            self.lossfn.increment_weights()

        if self.trainer.fast_dev_run or self.trainer.sanity_checking:
            print("Fast dev run or sanity checking detected, not logging")
        elif not self.pretraining and self.val_3d:
            for key, value in self.metrics.items():
                print(f"\n{key}")
                selected_metrics = {"names": [], "values": []}
                for metric, metric_value in value.items():
                    np_metric_value = np.array(metric_value)
                    mean = np_metric_value.mean() 
                    std = np_metric_value.std() 
                    print(f"{key} {metric}: {mean}+-{std}")
                    
                    # Stopped logging std for every metric, too much not very useful data on neptune
                    # self.logger.experiment[f"training/{key}_{metric}_3d_std"].log(std)
                    
                    if metric not in self.excluded_average_metric_keys:
                        if "error" in metric:
                            selected_metrics["names"].append(f"1 - {metric}")
                            selected_metrics["values"].append(1 - mean)
                        else:
                            selected_metrics["names"].append(metric)
                            selected_metrics["values"].append(mean)
                    
                np_selected_metrics = np.array(selected_metrics["values"])
                np_selected_metrics_mean = np_selected_metrics.mean()
                np_selected_metrics_std = np_selected_metrics.std()
                print(f"Building end-of-epoch composite metric:")
                for metric, value in zip(selected_metrics["names"], selected_metrics["values"]):
                    print(f"{metric}: {value}")
                print(f"{key}_composite_metric: {np_selected_metrics_mean} +- {np_selected_metrics_std}")
                
                self.logger.experiment[f"training/{key}_composite_metric"].log(np_selected_metrics_mean)
                self.logger.experiment[f"training/{key}_composite_metric_std"].log(np_selected_metrics_std)
                    

    def configure_optimizers(self):
        '''
        Select optimizer and scheduling strategy according to hparams.
        '''
        opt = getattr(self.hparams, "opt", "Adam")
        optimizer = get_optimizer(opt, self.model.parameters(), self.hparams.lr, wd=self.weight_decay)
        print(f"Opt: {opt}, Weight decay: {self.weight_decay}")

        if self.scheduling == "poly":
            print("Polynomial LR")
            # scheduler = PolynomialLR(optimizer, total_iters=self.hparams.max_epochs, power=0.9, verbose=True)
        elif self.scheduling == "step" and self.scheduling_factor is None:
            print("Not using any scheduler")
            return optimizer
        elif self.scheduling_factor is not None and self.scheduling == "step":
            print(f"Using step LR {self.scheduling_factor}!")
            scheduler = StepLR(optimizer, 1, self.scheduling_factor, verbose=True)
            return [optimizer], [scheduler]
        elif self.scheduling == "cosine":
            print(f"Using CosineAnnealingLR with tmax {self.scheduling_factor}!")
            scheduler = CosineAnnealingLR(optimizer, T_max=self.scheduling_factor, verbose=True)
            return [optimizer], [scheduler]
