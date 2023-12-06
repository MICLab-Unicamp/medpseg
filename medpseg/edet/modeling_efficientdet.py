'''
Copyright (c) Diedre Carmo, Medical Imaging Computing Lab (MICLab https://miclab.fee.unicamp.br/.
https://github.com/MICLab-Unicamp/medpseg
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Original Author: Zylo117
Modified by Israel, adopted by Diedre as an initial EfficientDet implementation and extended to MEDPSeg related implementations
April 2023: removed code not directly related to MEDSeg and extra deprecations
'''
from typing import List, Optional, Tuple, Union
from collections import OrderedDict
import torch
from torch import nn
from medpseg.self_attention import SelfAttention
from medpseg.edet.efficientnet.utils import MemoryEfficientSwish

from medpseg.edet.efficientdet.model import BiFPN, EfficientNet, SegmentationClasssificationHead, CirculatoryBranch


class FeatureFusion(nn.Module):
    '''
    Feature fusion module that makes use of all BiFPN features for segmentation instead of only
    upsampling the highest spatial resolution.

    upsample_sum: upsamples and sums all features 
    (ESC) exponential_stride_compression: increases kernel size and dilation and exponentially increases the stride to compress features, from B, C, x, y into a B, C, x/256, y/256 array that can be linearized easily with reshape. Minimum input size 256x256.
    seg_exponential_stride_compression: use values derived from ESC to weight high resolution features
    '''
    SUPPORTED_STRATS = ["cat", "upsample_sum", "upsample_cat", "exponential_stride_compression", "seg_exponential_stride_compression", "nonlinear_esc"]
    def __init__(self, in_c: int, out_c: int, key: Union[bool, str]):
        super().__init__()
        print(f"SELECTING FEATURE ADAPTER: {key}")
        self.key = key
        if key == "cat":
            # Concatenate features without over upsampling (results in features /2 the spatial resolution of the input)
            self.feature_adapters =  nn.ModuleList([nn.Identity(),
                                                    nn.UpsamplingBilinear2d(scale_factor=2),
                                                    nn.UpsamplingBilinear2d(scale_factor=4),
                                                    nn.UpsamplingBilinear2d(scale_factor=8),
                                                    nn.UpsamplingBilinear2d(scale_factor=16)])
        elif key == "upsample_sum" or key == "upsample_cat":
            self.feature_adapters =  nn.ModuleList([nn.UpsamplingBilinear2d(scale_factor=2),
                                                    nn.UpsamplingBilinear2d(scale_factor=4),
                                                    nn.UpsamplingBilinear2d(scale_factor=8),
                                                    nn.UpsamplingBilinear2d(scale_factor=16),
                                                    nn.UpsamplingBilinear2d(scale_factor=32)])
        elif key == "exponential_stride_compression":
            self.feature_adapters =  nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=11, padding=5, stride=128, dilation=6, bias=False, groups=in_c),
                                                                  nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding=0, stride=1, bias=False)),
                                                    nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=9, padding=4, stride=64, dilation=5, bias=False, groups=in_c),
                                                                  nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding=0, stride=1, bias=False)),
                                                    nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=7, padding=3, stride=32, dilation=4, bias=False, groups=in_c),
                                                                  nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding=0, stride=1, bias=False)),
                                                    nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=5, padding=2, stride=16, dilation=3, bias=False, groups=in_c),
                                                                  nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding=0, stride=1, bias=False)),
                                                    nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=3, padding=1, stride=8, dilation=2, bias=False, groups=in_c),
                                                                  nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding=0, stride=1, bias=False))])
        elif key == "seg_exponential_stride_compression":
            self.feature_adapters =  nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=11, padding=5, stride=128, dilation=6, bias=False, groups=in_c),
                                                                  nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding=0, stride=1, bias=False)),
                                                    nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=9, padding=4, stride=64, dilation=5, bias=False, groups=in_c),
                                                                  nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding=0, stride=1, bias=False)),
                                                    nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=7, padding=3, stride=32, dilation=4, bias=False, groups=in_c),
                                                                  nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding=0, stride=1, bias=False)),
                                                    nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=5, padding=2, stride=16, dilation=3, bias=False, groups=in_c),
                                                                  nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding=0, stride=1, bias=False)),
                                                    nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=3, padding=1, stride=8, dilation=2, bias=False, groups=in_c),
                                                                  nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding=0, stride=1, bias=False))])
            self.upsampler = nn.UpsamplingBilinear2d(scale_factor=2)
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif key == "nonlinear_esc":  # Save this for future embbedding building for transformers 
            # Reduced stride progression, trusting average pooling, makes network work with 128x128 inputs minimum
            self.feature_adapters =  nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=11, padding=5, stride=64, dilation=4, bias=False, groups=in_c),
                                                                  nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding=0, stride=1, bias=False),
                                                                  nn.LeakyReLU()),
                                                    nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=9, padding=4, stride=32, dilation=3, bias=False, groups=in_c),
                                                                  nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding=0, stride=1, bias=False),
                                                                  nn.LeakyReLU()),
                                                    nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=7, padding=3, stride=16, dilation=3, bias=False, groups=in_c),
                                                                  nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding=0, stride=1, bias=False),
                                                                  nn.LeakyReLU()),
                                                    nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=5, padding=2, stride=8, dilation=2, bias=False, groups=in_c),
                                                                  nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding=0, stride=1, bias=False),
                                                                  nn.LeakyReLU()),
                                                    nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=3, padding=1, stride=4, dilation=2, bias=False, groups=in_c),
                                                                  nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding=0, stride=1, bias=False),
                                                                  nn.LeakyReLU())])
            self.upsampler = nn.UpsamplingBilinear2d(scale_factor=2)
            self.pooling = nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError(f"Unsupported feature adapter {key}. Use one of {FeatureFusion.SUPPORTED_STRATS}")
        self.latent_space = None
    
    def get_latent_space(self):
        # Save this for future transformer involvement
        B, C, _, _ = self.latent_space.shape
        return self.latent_space.reshape(B, C)

    def forward(self, in_features: List[torch.Tensor]) -> Optional[torch.Tensor]:
        out_features = None
        for feature_adapter, in_feature in zip(self.feature_adapters, in_features):
            if out_features is None:  # first thing
                out_features = feature_adapter(in_feature)
            elif self.key == "upsample_cat":
                out_features = torch.cat([out_features, feature_adapter(in_feature)], dim=1)  # upsample cat concatenates in channel dimension
            else:
                out_features += feature_adapter(in_feature)
        
        if self.key in ["nonlinear_esc", "seg_exponential_stride_compression"]:
            self.latent_space = self.pooling(out_features)
            return self.upsampler(in_features[0]) * self.latent_space  # latent space weights channel contributions
        else:
            return out_features


class EfficientDetForSemanticSegmentation(nn.Module):

    def __init__(self, 
                 load_weights:bool = True, 
                 num_classes: int = 2, 
                 compound_coef: int = 4, 
                 repeat: int = 3, 
                 expand_bifpn: Union[bool, str] = False, 
                 backbone: str = "effnet", 
                 circulatory_branch: bool = False,
                 bifpn_channels: int = 128,
                 squeeze:bool = False,
                 deep_supervision: bool = False,
                 self_attention: bool = False,
                 soft_circulatory: bool = False,
                 **kwargs):  # dump for old variables
        '''
        load_weights: wether to load pre trained as backbone
        num_classes: number of classes for primary downstream segmentation task 
        compound_coef: which efficientnet variation to base the architecture of, only supports 4.
        repeat: how many conv blocks on the segmentation head
        expand_bifpn: how to expand the bifpn features. Upsample is best
        backbone: efficientnet or convnext as backbone
        num_classes_aux: number of classes for secondary segmentation task. If None will not initialize second output.
        '''
        super().__init__()
        
        for k, v in kwargs.items():
            print(f"WARNING: MEDSeg Argument {k}={v} being ignored")

        self.compound_coef = compound_coef
        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 7]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.num_classes = num_classes
        self.expand_bifpn = expand_bifpn
        self.backbone = backbone
        self.self_attention = self_attention
        self.deep_supervision = deep_supervision
        if self.self_attention:
            self.attention_modules: nn.ModuleList = nn.ModuleList([SelfAttention(bifpn_channels, dim='2d') for _ in range(5)])
        if self.expand_bifpn == "upsample_cat":
            self.upsample_cat_scaling = 5
        else:
            self.upsample_cat_scaling = 1  # scale expected input of segmentation heads

        # Check if expand_bifpn requires 
        feature_fusion = self.set_expand_conv()
        
        conv_channel_coef = {
            # the channels of P2/P3/P4.
            0: [16, 24, 40],
            4: [24, 32, 56],
            6: [32, 40, 72],
            7: [32, 48, 80],
            -1: [96, 192, 384]
        }

        if self.backbone == "convnext":
            print("Changing compound coeff of BiFPN due to convnext backbone")
            compound_coef = -1
            print(f"Convnext upsample scale {self.convnext_upsample_scale}")

        self.bifpn = nn.Sequential(*[BiFPN(bifpn_channels,
                                           conv_channel_coef[compound_coef],
                                           True if i == 0 else False,
                                           attention=True if self.compound_coef < 6 else False)
                                     for i in range(repeat)])

        # Main classifier
        self.classifier = SegmentationClasssificationHead(in_channels=self.upsample_cat_scaling*bifpn_channels,
                                                          num_classes=self.num_classes,
                                                          num_layers=repeat,
                                                          squeeze=squeeze,
                                                          deep_supervision=deep_supervision
                                                          )

        # Where bifpn upsampling happens
        if feature_fusion:
            self.feature_adapters: Optional[FeatureFusion] = FeatureFusion(bifpn_channels, bifpn_channels, key=self.expand_bifpn)   
        else:
            self.feature_adapters = None

        # Experimenting with a mixed vessel and ATM branch
        if circulatory_branch:
            self.circulatory_classifier: Optional[CirculatoryBranch] = CirculatoryBranch(bifpn=nn.Sequential(*[BiFPN(bifpn_channels,
                                                                                                                     conv_channel_coef[compound_coef],
                                                                                                                     True if i == 0 else False,
                                                                                                                     attention=True if self.compound_coef < 6 else False)
                                                                                                             for i in range(repeat)]),
                                                                                         bifpn_channels = bifpn_channels,
                                                                                         in_channels=self.upsample_cat_scaling*bifpn_channels,
                                                                                         num_classes=1 + soft_circulatory*1, 
                                                                                         num_layers=repeat,
                                                                                         squeeze=squeeze,
                                                                                         feature_adapters=self.feature_adapters,
                                                                                         expand_bifpn=self.expand_bifpn,
                                                                                         expand_conv=self.expand_conv,
                                                                                         deep_supervision=deep_supervision,
                                                                                         self_attention=self_attention
                                                                                        )               
        else:
            self.circulatory_classifier = None
            
        # Choose backbone
        if self.backbone == "effnet":
            self.backbone_net = EfficientNet(self.backbone_compound_coef[self.compound_coef], load_weights)
        elif self.backbone == "convnext":
            raise NotImplementedError("ConvNext implementation removed for public code release. Check src code for commented initialization.")
            # self.backbone_net = convnext_tiny(pretrained=load_weights)
        elif self.backbone == "sam":
            raise NotImplementedError("SAM hasnt been implemented yet")

    def set_expand_conv(self):
        '''
        Sets feature expansion convolution or hands down the responsability to feature fusion module
        '''
        feature_fusion = False
        if self.expand_bifpn == True or self.expand_bifpn == "conv":
            print("Using transposed convolution for bifpn result expansion and upsample 2 for convnext features")
            self.expand_conv = nn.Sequential(nn.ConvTranspose2d(128, 128, 2, 2),
                                             nn.BatchNorm2d(128),
                                             MemoryEfficientSwish())
            self.convnext_upsample_scale = 2
        elif self.expand_bifpn == "upsample":
            print("Using upsample for bifpn result expansion and upsample 2 expansion of convnext features")
            self.expand_conv = nn.UpsamplingBilinear2d(scale_factor=2)
            self.convnext_upsample_scale = 2
        elif self.expand_bifpn == "upsample_4":
            print("Using upsample 4 for bifpn result expansion and no expansion of convnext features")
            self.expand_conv = nn.UpsamplingBilinear2d(scale_factor=4)
            self.convnext_upsample_scale = 0
        elif self.expand_bifpn == False or self.expand_bifpn == "no":
            print("Bifpn expansion disabled")
            self.convnext_upsample_scale = 4
        elif self.expand_bifpn in FeatureFusion.SUPPORTED_STRATS:
            print(f"Enabling feature fusion through {self.expand_bifpn}")
            feature_fusion = True
            self.expand_conv = None
        else:
            raise ValueError(f"Expand bifpn {self.expand_bifpn} not supported!")
        
        return feature_fusion

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        
    def extract_backbone_features(self, inputs):
        if self.backbone == "effnet":
            p2, p3, p4 = self.backbone_net(inputs)
        elif self.backbone == "convnext":
            p2, p3, p4 = self.backbone_net.forward_seg_features(inputs, self.convnext_upsample_scale)

        features = (p2, p3, p4)
        return features

    def extract_bifpn_features(self, features):
        features = self.bifpn(features)
        return features

    def build_return_dict(self, backbone_features, bifpn_features):
        classification = self.classifier(bifpn_features)

        rdict = OrderedDict()
        return_dict = False

        if isinstance(classification, dict):
            # If main branch result is a dict, we need return dict
            rdict.update(classification)
            return_dict = True
        else:
            rdict[f"main"] = classification

        for branch in ["circulatory"]:
            branch_module = getattr(self, f"{branch}_classifier", None)
            if branch_module is not None:
                return_dict = True
                branch_return = branch_module(backbone_features)
                rdict.update(branch_return)
        
        return rdict, return_dict

    def forward(self, inputs):
        features: Tuple[torch.Tensor] = self.extract_backbone_features(inputs)
        feat_map: Tuple[torch.Tensor] = self.extract_bifpn_features(features)

        # Apply attention gates in BiFPN outputs
        if self.self_attention:
            assert len(feat_map) == len(self.attention_modules)
            feat_map = [attention_module(x) for x, attention_module in zip(feat_map, self.attention_modules)]
        
        if self.feature_adapters is not None:
            if self.deep_supervision:
                raise RuntimeError("Can't do deep supervision with feature adapters")
            feat_map = self.feature_adapters(feat_map)
        else:
            if self.deep_supervision:
                # Apply expand bifpn on all feat_maps and still keep the list for deep supervision
                new_feat_map = []
                for feat in feat_map:
                    if self.expand_bifpn is not None and self.expand_bifpn != True and self.expand_bifpn != "no":
                        # Simple upsampling happends here
                        new_feat_map.append(self.expand_conv(feat))
                feat_map = new_feat_map
            else:
                # Selection of only higher resolution feature
                feat_map = feat_map[0]

                if self.expand_bifpn is not None and self.expand_bifpn != True and self.expand_bifpn != "no":
                    feat_map = self.expand_conv(feat_map)

        rdict, return_dict = self.build_return_dict(features, feat_map)
        
        # Dict return form in case auxiliary branches are involved
        if return_dict:
            return rdict
        else:
            return rdict["main"]

    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')
