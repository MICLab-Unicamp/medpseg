'''
Copyright (c) Diedre Carmo, Medical Imaging Computing Lab (MICLab)
https://miclab.fee.unicamp.br/
https://github.com/MICLab-Unicamp/medpseg
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

An early experiment in polymorphic 3D segmentatiom of the lung 
This should handle transforming targets, currently only implemented for lung variation
'''
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
from medpseg.utils import get_optimizer, DICELoss, DICEMetric, itk_snap_spawner
from medpseg.unet_v2 import UNet


class PolySeg3DModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        dropout = getattr(self.hparams, "dropout", None)
        self.weight_decay = getattr(self.hparams, "weight_decay", None)
        self.scheduling_factor = getattr(self.hparams, "scheduling_factor", None)
        if dropout == True:
            print("WARNING: Replacing old hparams true dropout by full dropout")
            dropout = "full"
        
        self.model = UNet(self.hparams.nin, self.hparams.seg_nout, "instance", "3d", self.hparams.init_channel)

        self.lossfn = DICELoss(volumetric=False, per_channel=True, check_bounds=False)
        self.dicer = DICEMetric(per_channel_metric=True, check_bounds=False)

    def forward(self, x, get_bg=False):
        '''
        Testing with 2 channel output for both lungs
        '''
        logits = self.model(x)  # 3 canais, bg, ll, rl
        if get_bg:
            y_hat = logits.softmax(dim=1)
        else:
            y_hat = logits.softmax(dim=1)[:, 1:]
        y_hat_lung = y_hat.sum(dim=1, keepdim=True)

        return y_hat, y_hat_lung
        
    def polymorphic_loss_metrics(self, x, y, y_hat, y_hat_lung, meta, mode=None):
        target_format = meta["target_format"][0]
        assert target_format in ["simple", "has_left_right", "has_ggo_con", "full_poly"]
        loss, metrics = None, None

        if target_format in ["simple", "has_ggo_con"]:
            # Format returns only lung binary mask, extract lung target
            y = y[:, 0:1]
            if mode == "val":
                metrics = self.dicer(y_hat_lung, y)[0]
            elif mode == "train":
                loss = self.lossfn(y_hat_lung, y)
        elif target_format in ["has_left_right", "full_poly"]:
            # Extract left right lung
            y = y[:, 0:2]
            if mode == "val":
                metrics = self.dicer(y_hat, y)
            elif mode == "train":
                loss = self.lossfn(y_hat, y)

        return y, loss, metrics

    def compute_loss(self, x, y, y_hat, y_hat_lung, meta, prestr):
        _, loss, _ = self.polymorphic_loss_metrics(x, y, y_hat, y_hat_lung, meta, mode="train")
        
        self.log(f"{prestr}loss", loss, on_step=True, on_epoch=True)

        return loss

    def compute_metrics(self, x, y, y_hat, y_hat_lung, meta):
        _, _, metrics = self.polymorphic_loss_metrics(x, y, y_hat, y_hat_lung, meta, mode="val")

        if isinstance(metrics, list):
            left_lung_dice, right_lung_dice = metrics
            self.log("left_lung_dice", left_lung_dice, on_epoch=True, on_step=True, prog_bar=True)
            self.log("right_lung_dice", right_lung_dice, on_epoch=True, on_step=True, prog_bar=True)
        else:
            self.log("lung_dice", metrics, on_epoch=True, on_step=True, prog_bar=True)

        
    def training_step(self, train_batch, batch_idx):
        x, y, meta = train_batch

        y_hat, y_hat_lung = self.forward(x)

        loss = self.compute_loss(x, y, y_hat, y_hat_lung, meta, prestr='')

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y, meta = val_batch
        
        y_hat, y_hat_lung = self.forward(x)
        
        self.compute_loss(x, y, y_hat, y_hat_lung, meta, prestr="val_")
        self.compute_metrics(x, y, y_hat, y_hat_lung, meta)

    def configure_optimizers(self):
        '''
        Select optimizer and scheduling strategy according to hparams.
        '''
        opt = getattr(self.hparams, "opt", "Adam")
        optimizer = get_optimizer(opt, self.model.parameters(), self.hparams.lr, wd=self.weight_decay)
        print(f"Weight decay: {self.weight_decay}")

        if self.scheduling_factor is not None:
            print(f"Using step LR {self.scheduling_factor}!")
            scheduler = StepLR(optimizer, 1, self.scheduling_factor)
            return [optimizer], [scheduler]
        else:
            return optimizer
