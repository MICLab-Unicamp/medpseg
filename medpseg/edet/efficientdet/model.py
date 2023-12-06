import torch.nn as nn
import torch
from torchvision.ops.boxes import nms as nms_torch
from medpseg.self_attention import SelfAttention
from efficientnet_pytorch import EfficientNet as EffNet
from medpseg.edet.efficientnet.utils import MemoryEfficientSwish, Swish
from medpseg.edet.efficientnet.utils_extra import Conv2dStaticSamePadding, MaxPool2dStaticSamePadding


def nms(dets, thresh):
    return nms_torch(dets[:, :4], dets[:, 4], thresh)


class SeparableConvBlock(nn.Module):
    """
    created by Zylo117
    """

    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(
            in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(
                num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x


def align_sum(to_be_summed, upsample, align=True):
    '''
    Aligns x1 with x2 and sums
    Based on UNet upsample concatenation alignment code
    '''
    raise DeprecationWarning("align_sum deprecated due to alignment problems with some shapes. Use external padding?")
    if align:
        diffX = to_be_summed.size()[2] - upsample.size()[2]
        diffY = to_be_summed.size()[3] - upsample.size()[3]
        upsample = torch.nn.functional.pad(upsample, (diffY // 2, diffY - diffY // 2, diffX // 2, diffX - diffX // 2))
    
    return to_be_summed + upsample


class BiFPN(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False, attention=True):
        """

        Args:
            num_channels:
            conv_channels:
            first_time: whether the input comes directly from the efficientnet,
                        if True, downchannel it first, and downsample P5 to generate P6 then P7
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
            onnx_export: if True, use Swish instead of MemoryEfficientSwish
        """
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        # Conv layers
        self.conv6_up = SeparableConvBlock(
            num_channels, onnx_export=onnx_export)
        self.conv5_up = SeparableConvBlock(
            num_channels, onnx_export=onnx_export)
        self.conv4_up = SeparableConvBlock(
            num_channels, onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock(
            num_channels, onnx_export=onnx_export)
        self.conv4_down = SeparableConvBlock(
            num_channels, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock(
            num_channels, onnx_export=onnx_export)
        self.conv6_down = SeparableConvBlock(
            num_channels, onnx_export=onnx_export)
        self.conv7_down = SeparableConvBlock(
            num_channels, onnx_export=onnx_export)

        # Feature scaling layers
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p6_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p7_downsample = MaxPool2dStaticSamePadding(3, 2)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        self.first_time = first_time
        if self.first_time:
            self.p5_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p4_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p3_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

            self.p5_to_p6 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
                MaxPool2dStaticSamePadding(3, 2)
            )
            self.p6_to_p7 = nn.Sequential(
                MaxPool2dStaticSamePadding(3, 2)
            )

            self.p4_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

        # Weight
        self.p6_w1 = nn.Parameter(torch.ones(
            2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(
            2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(
            2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(
            2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(
            3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(
            3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(
            3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(
            2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()

        self.attention = attention

    def forward(self, inputs):
        """
        illustration of a minimal bifpn unit
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """

        # downsample channels using same-padding conv2d to target phase's if not the same
        # judge: same phase as target,
        # if same, pass;
        # elif earlier phase, downsample to target phase's by pooling
        # elif later phase, upsample to target phase's by nearest interpolation
        if self.attention:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward_fast_attention(
                inputs)
        else:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward(inputs)

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward_fast_attention(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # P7_0 to P7_2

        # Weights for P6_0 and P7_0 to P6_1
        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

        # Weights for P5_0 and P6_1 to P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_0 to P5_1 respectively
        p5_up = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_up)))

        # Weights for P4_0 and P5_1 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_0 to P4_1 respectively
        p4_up = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_up)))

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out)))

        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish(weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out)))

        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.swish(weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p6_downsample(p5_out)))

        # Weights for P7_0 and P6_2 to P7_2
        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(self.swish(
            weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # P7_0 to P7_2

        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))

        # Connections for P5_0 and P6_1 to P5_1 respectively
        p5_up = self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_up)))

        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_up = self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_up)))

        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(p4_in + p4_up + self.p4_downsample(p3_out)))

        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish(p5_in + p5_up + self.p5_downsample(p4_out)))

        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.swish(p6_in + p6_up + self.p6_downsample(p5_out)))

        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(self.swish(
            p7_in + self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out


class Regressor(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, in_channels, num_anchors, num_layers, onnx_export=False):
        super(Regressor, self).__init__()
        self.num_layers = num_layers
        self.num_layers = num_layers

        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(5)])
        self.header = SeparableConvBlock(
            in_channels, num_anchors * 4, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], -1, 4)

            feats.append(feat)

        feats = torch.cat(feats, dim=1)

        return feats


class Classifier(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, in_channels, num_anchors, num_classes, num_layers, onnx_export=False):
        super(Classifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(5)])
        self.header = SeparableConvBlock(
            in_channels, num_anchors * num_classes, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], feat.shape[1], feat.shape[2], self.num_anchors,
                                          self.num_classes)
            feat = feat.contiguous().view(feat.shape[0], -1, self.num_classes)

            feats.append(feat)

        feats = torch.cat(feats, dim=1)
        feats = feats.sigmoid()

        return feats


class SegmentationClasssificationHead(nn.Module):
    '''
    DLPT v3.5 changes removed some arguments
    '''
    def __init__(self, in_channels, num_classes, num_layers, onnx_export=False, squeeze=False, deep_supervision=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.squeeze = squeeze  # squeeze channels before header, use when in_channels is too large
        
        # Squeezing channels changes
        if self.squeeze:
            self.internal_channels = [2**i for i in range(num_layers+1)]
            self.conv_list = nn.ModuleList([SeparableConvBlock(in_channels//self.internal_channels[i], in_channels//self.internal_channels[i+1], norm=False, activation=False) for i in range(num_layers)])
            self.bn_list = nn.ModuleList([nn.BatchNorm2d(in_channels//self.internal_channels[i+1], momentum=0.01, eps=1e-3) for i in range(num_layers)])
            self.header = SeparableConvBlock(in_channels//self.internal_channels[num_layers], num_classes, norm=False, activation=False)
        else:
            self.internal_channels = None
            self.conv_list = nn.ModuleList([SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
            self.bn_list = nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)])
            self.header = SeparableConvBlock(in_channels, num_classes, norm=False, activation=False)

        # Swishes are swishes
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        self.deep_supervision = deep_supervision
        if self.deep_supervision:
            self.ds_seg_heads: nn.ModuleList = nn.ModuleList([SegmentationClasssificationHead(in_channels=in_channels, 
                                                                                              num_classes=num_classes, 
                                                                                              num_layers=num_layers, 
                                                                                              onnx_export=onnx_export, 
                                                                                              squeeze=squeeze, 
                                                                                              deep_supervision=False) for _ in range(4)])

    def forward(self, feat, return_tensor=False):
        # Main head
        if torch.is_tensor(feat):
            main_feat = feat
        else:
            main_feat = feat[0]

        for i, bn, conv in zip(range(self.num_layers), self.bn_list, self.conv_list):
            main_feat = conv(main_feat)
            main_feat = bn(main_feat)
            main_feat = self.swish(main_feat)
        main_feat = self.header(main_feat)

        return_dict = {"main": main_feat} 
        if self.deep_supervision:
            assert isinstance(feat, list)
            for i in range(1, 5):
                return_dict[f"main{i}"] = self.ds_seg_heads[i-1](feat[i])["main"]

        if return_tensor:
            assert not self.deep_supervision, "Deep supervision should not attempt to return a single tensor"
            return return_dict["main"]
        else:
            return return_dict
        

class CirculatoryBranch(nn.Module):
    '''
    This is intented to be used to optimize vessels and airways at the same time.
    '''
    def __init__(self, bifpn, bifpn_channels, in_channels, num_classes, num_layers, squeeze, feature_adapters, expand_bifpn, expand_conv, deep_supervision, self_attention):
        super().__init__()
        self.bifpn = bifpn
        self.feature_adapters = feature_adapters
        self.expand_bifpn = expand_bifpn
        self.expand_conv = expand_conv
        self.deep_supervision = deep_supervision
        self.attention = self_attention
        self.airway_head = SegmentationClasssificationHead(in_channels=in_channels,
                                                           num_classes=num_classes, 
                                                           num_layers=num_layers,
                                                           squeeze=squeeze
                                                           )
        self.vessel_head = SegmentationClasssificationHead(in_channels=in_channels,
                                                           num_classes=num_classes, 
                                                           num_layers=num_layers,
                                                           squeeze=squeeze
                                                           )
        if self.attention:
            self.attention_modules: nn.ModuleList = nn.ModuleList([SelfAttention(bifpn_channels, dim='2d') for _ in range(5)])
        if self.deep_supervision:
            self.ds_airway_heads: nn.ModuleList = nn.ModuleList([SegmentationClasssificationHead(in_channels=in_channels,
                                                                                                 num_classes=num_classes, 
                                                                                                 num_layers=num_layers,
                                                                                                 squeeze=squeeze
                                                                                                 ) for _ in range(4)])
            self.ds_vessel_heads: nn.ModuleList = nn.ModuleList([SegmentationClasssificationHead(in_channels=in_channels,
                                                                                                 num_classes=num_classes, 
                                                                                                 num_layers=num_layers,
                                                                                                 squeeze=squeeze
                                                                                                 ) for _ in range(4)])
    
    def forward(self, backbone_features):
        '''
        Replicates the original medseg forward actually
        '''
        feat_map = self.bifpn(backbone_features)
        
        # Apply attention gates in BiFPN outputs
        if self.attention:
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
                        new_feat_map.append(self.expand_conv(feat))
                feat_map = new_feat_map
            else:
                feat_map = feat_map[0]  # higher resolution feature is first

                if self.expand_bifpn is not None and self.expand_bifpn != True and self.expand_bifpn != "no":
                    feat_map = self.expand_conv(feat_map)
                
        
        if self.deep_supervision:
            assert isinstance(feat_map, list)
            return_dict = {"atm": self.airway_head(feat_map[0], return_tensor=True), "vessel": self.vessel_head(feat_map[0], return_tensor=True)}
            for i in range(1, 5):
                return_dict[f"atm{i}"] = self.ds_airway_heads[i-1](feat_map[i], return_tensor=True)
                return_dict[f"vessel{i}"] = self.ds_vessel_heads[i-1](feat_map[i], return_tensor=True)
        else:
            return_dict = {"atm": self.airway_head(feat_map, return_tensor=True), "vessel": self.vessel_head(feat_map, return_tensor=True)}

        return return_dict

class EfficientNet(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, compound_coef, load_weights=True):
        super(EfficientNet, self).__init__()
        if load_weights:
            model = EffNet.from_pretrained(f'efficientnet-b{compound_coef}')
        else:
            model = EffNet.from_name(f'efficientnet-b{compound_coef}')
        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc
        self.model = model

    def forward(self, x):
        x = self.model._conv_stem(x)
        x = self.model._bn0(x)
        x = self.model._swish(x)
        feature_maps = []

        # TODO: temporarily storing extra tensor last_x and del it later might not be a good idea,
        #  try recording stride changing when creating efficientnet,
        #  and then apply it here.
        last_x = None
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            
            if len(feature_maps) < 3:
                # avoid unnecessary forwards
                x = block(x, drop_connect_rate=drop_connect_rate)

                if block._depthwise_conv.stride == [2, 2]:
                    feature_maps.append(last_x)
                elif idx == len(self.model._blocks) - 1:
                    feature_maps.append(x)
                last_x = x

        return feature_maps
