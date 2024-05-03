'''
Copyright (c) Diedre Carmo, Medical Imaging Computing Lab (MICLab)
https://miclab.fee.unicamp.br/
https://github.com/MICLab-Unicamp/medpseg
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Renaming coedet to medseg

Ablation results:
-ImageNet B6 weights
-upsample_sum feature pyramid feature fusion
-Raw HU and clipped HU dont have any significant changes, raw hu is numerically unstable
-Dice + L1Loss: no significant change
'''
import os
import torch
import numpy as np
from typing import List, Optional, Tuple
from scipy.ndimage import zoom
from torch import nn
from efficientnet_pytorch.utils import round_filters
from multiprocessing.pool import ThreadPool
from medpseg.edet.modeling_efficientdet import EfficientDetForSemanticSegmentation
from medpseg.hu_clip_norm_module import CTHUClipNormModule, CTHUClipZNormModule


sam_model_registry = {}  # placeholder, avoid SAM dependency since it is not being used in public release
SAM_WEIGHT = "sam/sam_vit_b_01ec64.pth"


class MEDSeg(nn.Module):
    def __init__(self, nin=3, nout=3, backbone="effnet", pretrained=True, expand_bifpn="upsample_sum", imnet_norm=False,
                 stem_replacement=False,
                 compound_coef=4,  # compound always has been 4 by default before, 6 is a large version
                 learnable_norm=False,  # Learning hu clip and norm parameters in the input 
                 circulatory_branch=None,
                 bifpn_channels=128,  # option to increase girth of bifpn channels, increases memory consumption
                 sam_embedding=False,  # compute frozen sam embedding, requires 2x3x1024x1024 -1, 1 normalized input
                 deep_supervision=False,  # use BiFPN outputs to compute multiple low resolution segmentations
                 self_attention=False,  # Self-attention gate at bifpn outputs
                 con_detecting=False,  # Train in consolidation presence
                 large=False,  # Turn on gigantic model
                 soft_circulatory=False, 
                 **kwargs):  # dump for unused arguments
        super().__init__()
        squeeze = False if expand_bifpn is None else "cat" in expand_bifpn  # Mostly false, lets deprecate
        if squeeze:
            raise DeprecationWarning("Not squeezing bifpn output anymore")
        self.att_pool = None
        self.circulatory_branch = circulatory_branch
        self.self_attention = self_attention
        self.con_detecting = con_detecting
        if self.con_detecting:
            self.attention_pooler = nn.AdaptiveAvgPool1d(2048)
            self.con_detector = nn.Sequential(nn.Linear(2048, 1024), 
                                              nn.Dropout1d(p=0.5), 
                                              nn.LeakyReLU(), 
                                              nn.BatchNorm1d(1024),
                                              nn.Linear(1024, 1))  # To be used with BCE, return logit
        print("WARNING: default expand_bifpn changed to upsample_sum after many ablations and literature support.")
        self.model = EfficientDetForSemanticSegmentation(num_classes=nout, 
                                                         load_weights=pretrained,
                                                         expand_bifpn=expand_bifpn, 
                                                         backbone=backbone,
                                                         compound_coef=compound_coef,
                                                         circulatory_branch=circulatory_branch,
                                                         bifpn_channels=bifpn_channels*(1 + int(large)),  # scale width
                                                         squeeze=squeeze,
                                                         deep_supervision=deep_supervision,
                                                         self_attention=self_attention,
                                                         repeat=3*(1 + int(large)),
                                                         soft_circulatory=soft_circulatory)  # scale depth

        # SAM frozen embedding computer
        if sam_embedding:
            if isinstance(sam_embedding, str) and os.path.exists(sam_embedding):
                path = sam_embedding
            else:
                path = SAM_WEIGHT
            self.sam = sam_model_registry["vit_b"](checkpoint=path).image_encoder
            print("Initialized SAM embedding computer")
        else:
            self.sam = None
            print("SAM embedding disabled")

        # Backward compatiblity with True False learnable_norm and string parametrization
        if learnable_norm == True or learnable_norm == "clip_norm":
            print("Performing learnable clip with 0-1 normalization")
            self.learnable_norm_module = CTHUClipNormModule(vmin=-1024, vmax=600, norm=True)
        elif learnable_norm == "clipz":
            print("Performing learnable clip, with -1 1 normalization")
            self.learnable_norm_module = CTHUClipZNormModule(vmin=-1024, vmax=600, norm=True)
        elif learnable_norm == "clipz_1000":
            self.learnable_norm_module = CTHUClipZNormModule(vmin=-1024, vmax=1024, norm=True)
        elif learnable_norm == "clip":
            print("Performing learnable clip, with NO normalization")
            self.learnable_norm_module = CTHUClipNormModule(vmin=-1024, vmax=600, norm=False)
        elif learnable_norm is None or learnable_norm == False:
            print("Not using learnable clip, whatever comes from dataloader is entering the network!")
            self.learnable_norm_module = DisabledLearnableNorm()

        self.feature_adapters = self.model.feature_adapters

        if imnet_norm:
            raise DeprecationWarning("No need to do imagenet normalization")
        else:
            print("No imagenet specific normalization being performed")
            self.imnet_norm = DisabledImNetNorm()

        self.nin = nin
        if self.nin not in [1, 3]:
            self.in_conv = nn.Conv2d(in_channels=self.nin, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False)

        if stem_replacement:
            assert backbone == "effnet", "Stem replacement only valid for efficientnet"
            print("Performing stem replacement on EfficientNet backbone (this runs after initialization)")
            self.model.backbone_net.model._conv_stem = EffNet3DStemReplacement(self.model.backbone_net.model)

        print(f"MEDSeg initialized. nin: {nin}, nout: {nout} bifpn channels {bifpn_channels} * 2 if large {large}" 
              f"backbone: {backbone}, pretrained: {pretrained}, expand_bifpn: {expand_bifpn}, pad align DISABLED, stem_replacement {stem_replacement} compound_coef {compound_coef} "
              f"learnable norm {self.learnable_norm_module} circulatory_branch {circulatory_branch} deep_supervision {deep_supervision} self attention {self_attention}")

    def __del__(self):
        if self.att_pool is not None:
            self.att_pool.close()

    def extract_backbone_features(self, inputs):
        return self.model.extract_backbone_features(inputs)

    def extract_bifpn_features(self, features):
        return self.model.extract_bifpn_features(features)
    
    def triplify_x(self, x):
        x_in = torch.zeros(size=(x.shape[0], 3) + x.shape[2:], device=x.device, dtype=x.dtype)
        x_in[:, 0] = x[:, 0]
        x_in[:, 1] = x[:, 0]
        x_in[:, 2] = x[:, 0]
        return x_in

    def forward(self, x):
        self.input_shape = [x.shape[0], x.shape[2], x.shape[3]]

        # Sam embedding, didnt work
        if self.sam is not None:
            with torch.no_grad():
                B, _, _, _ = x.shape
                x = self.sam(x).reshape(B, 1, 1024, 1024)
                x = self.triplify_x(x)

        # Learnable HU Clip and Norm!
        x = self.learnable_norm_module(x)  # this is identity if disabled

        if self.nin == 1:
            x = self.triplify_x(x)
        elif self.nin == 3:
            pass
        else:
            x = self.in_conv(x)

        x = self.imnet_norm(x)  # this is identity if disabled

        x = self.model(x)

        if isinstance(x, dict):
            con_detection_logit = self.con_detect()
            if con_detection_logit is not None:
                x["con_detection_logit"] = con_detection_logit
        else:
            raise RuntimeError("Can't apply con_detection logic with non dictionary return from underlying model in MEDSeg")
        
        return x

    def return_atts(self, order=3, hr_only=False) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        '''
        Returns attentions interpolated to input size (ndarray)

        Atts in return list are higher to low resolution
        '''
        # High resolution only for faster computation
        if hr_only:
            if self.att_pool is None or (self.att_pool is not None and self.att_pool._processes != 2):
                self.att_pool = ThreadPool(2)
            atts: List[np.ndarray] = [self.model.attention_modules[0].att.detach().cpu().squeeze(1).numpy()]  # [B, X, Y]
            circulatory_atts: List[np.ndarray] = [self.model.circulatory_classifier.attention_modules[0].att.detach().cpu().squeeze(1).numpy()]
        else:
            if self.att_pool is None or (self.att_pool is not None and self.att_pool._processes != 10):
                self.att_pool = ThreadPool(10)
            atts: List[np.ndarray] = [module.att.detach().cpu().squeeze(1).numpy() for module in self.model.attention_modules]  # [B, X, Y]
            circulatory_atts: List[np.ndarray] = [module.att.detach().cpu().squeeze(1).numpy() for module in self.model.circulatory_classifier.attention_modules]  # [B, X, Y]

        att_packages: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = [(np.array(self.input_shape), 
                                                                               np.array(att.shape), 
                                                                               att, 
                                                                               order) for att in atts]        
        zoomed_atts_result =  self.att_pool.map_async(return_att_worker, iterable=att_packages)
    
        if self.circulatory_branch:
            catt_packages: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = [(np.array(self.input_shape), 
                                                                                    np.array(catt.shape), 
                                                                                    catt, 
                                                                                    order) for catt in circulatory_atts]        
            zoomed_circulatory_atts =  self.att_pool.map(return_att_worker, iterable=catt_packages)
    
        zoomed_atts = zoomed_atts_result.get()
        # Sinlge thread version
        # atts: List[np.ndarray] = [module.att.squeeze(1) for module in self.model.attention_modules]  # [B, X, Y]
        # zoomed_atts = []
        # for att in atts:
        #     ishape = np.array(self.input_shape)
        #     ashape = np.array(att.shape)
        #     assert len(ishape) == len(ashape), f"{ishape} != {ashape}, attention only works in 3D"
        #     zoom_factors = (ishape/ashape).tolist()
        #     zoomed_atts.append(zoom(att.detach().cpu().numpy(), zoom_factors, order=order))
        # zoomed_circulatory_atts = []
        # if self.circulatory_branch:
        #     circulatory_atts: List[torch.Tensor] = [module.att.squeeze(1) for module in self.model.circulatory_classifier.attention_modules]
        #     for att in circulatory_atts:
        #         ishape = np.array(self.input_shape)
        #         ashape = np.array(att.shape)
        #         assert len(ishape) == len(ashape), f"{ishape} != {ashape}, attention only works in 3D"
        #         zoom_factors = (ishape/ashape).tolist()
        #         zoomed_circulatory_atts.append(zoom(att.detach().cpu().numpy(), zoom_factors, order=order))

        return zoomed_atts, zoomed_circulatory_atts

    def con_detect(self) -> Optional[torch.Tensor]:
        '''
        Reasoning: It is very common to make false positive judgement on the presence or not presence of consolidations. It is something somewhat rare.
                   We will use attention maps from deep supervision BEFORE UPSAMPLING (more weight to high resolution maps) to classify the presence of Con.
                   Optimization of this module should only be done for "separation" class images.

        DEPRECATED
        '''
        ret = None
        if self.con_detecting:
            # Main branch attention
            atts: List[torch.Tensor] = [module.att.squeeze(1) for module in self.model.attention_modules]  # List[B, Y (varies), X(varies)]
            B = atts[0].shape[0]
            x: torch.Tensor = torch.cat([att.reshape(B, -1) for att in atts], dim=-1)  # [B, attention_features]
                                                                                       # e. g. [B, 512x512 + 256*256 + 128*128 + 64*64 + 32*32]
            x = self.attention_pooler(x)
            ret = self.con_detector(x)

        return ret

def return_att_worker(x):
    '''
    Attempt to real time compute attentions with multiprocessing
    '''
    ishape, ashape, att, order = x
    assert len(ishape) == len(ashape), f"{ishape} != {ashape}, attention only works in 3D"
    zoom_factors = (ishape/ashape).tolist()
    return zoom(att, zoom_factors, order=order)


class EffNet3DStemReplacement(nn.Module):
    '''
    Replaces input stem convolution with 3D convolution
    '''
    def __init__(self, effnet_pytorch_instance):
        super().__init__()
        out_channels = round_filters(32, effnet_pytorch_instance._global_params)
        self.conv = nn.Conv3d(1, out_channels, kernel_size=3, stride=1, padding="valid", bias=False)
        self.pad = nn.ZeroPad2d(1)
        self.conv_pool = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=False)
        
    def forward(self, x):
        '''
        x is 4D batch but will be treated as 5D
        '''
        x = self.conv(x.unsqueeze(1)).squeeze(2)  # [B, 3, X, Y] -> [B, 1, 3, X, Y] 
                                                  # -> [B, OUT_CH, 1, X, Y] -> [B, OUT_CH, X, Y]
        x = self.pad(x)
        x = self.conv_pool(x)
        return x
        

class ImNetNorm():
    '''
    Assumes input between 1 and 0
    '''
    def __init__(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, xim):
        with torch.no_grad():
            for i in range(3):
                xim[:, i] = (xim[:, i] - self.mean[i])/self.std[i]
        
        return xim
    

class DisabledImNetNorm(nn.Identity):
    def __init__(self):
        super().__init__()
    
    def __str__(self):
        return "Disabled ImNetNorm"
    

class DisabledLearnableNorm(nn.Identity):
    def __init__(self):
        super().__init__()
    
    def __str__(self):
        return "Disabled Learnable Norm"