'''
Copyright (c) Jean Ribeiro, Diedre Carmo, Medical Imaging Computing Lab (MICLab)
https://miclab.fee.unicamp.br/
https://github.com/MICLab-Unicamp/medpseg
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Reproduce Jean Ribeiro lobe segmentation network segmentation capabilities
'''
import argparse
import torch
import cc3d
import scipy
import monai
import numpy as np
import pytorch_lightning as pl

from typing import Optional

from monai.inferers import sliding_window_inference
from monai.transforms import LoadImaged, EnsureChannelFirstd, Orientationd, LoadImaged, Spacingd, SaveImaged


class LoberModule(pl.LightningModule):
	'''
	Simplified module to only reproduce prediction capabilities
	'''
	def __init__(self, hparams):
		super().__init__()

		self.save_hyperparameters(hparams)
		self.loader = LoadImaged(keys=["image"], image_only=False)
		self.transform = monai.transforms.Compose([EnsureChannelFirstd(keys=["image"]),
												   Orientationd(keys=["image"], axcodes="PLI"),
												   Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear"))])
		
		self.model = monai.networks.nets.VNet(in_channels=hparams["nin"], out_channels=hparams["snout"])

	def forward(self, x):
		y_hat = self.model(x)

		return y_hat

	def inverse_transform(self, y_hat: torch.Tensor, transform: monai.transforms.Compose) -> dict:
		'''
		Apply inverse transform in output, also correcting to depth last order
		WARNING: Use this only for integer tensors!
		'''
		y_hat_inv = y_hat.permute(0, 2, 3, 1)

		y_hat_inv = transform.inverse({"image": y_hat_inv})

		# Inverse of bilinear spacing may cause non binary values to appear
		y_hat_inv["image"] = np.round(y_hat_inv["image"])

		return y_hat_inv

	def post_processing(self, y_hat: np.ndarray):
		'''
		component isolation and fill holes
		'''
		y_hat = y_hat[0]
		new_y_hat = np.zeros_like(y_hat)
		for channel in range(1, 6):
			binary_array = cc3d.largest_k((y_hat == channel).astype(np.int32), k=1) > 0
			binary_array = scipy.ndimage.binary_fill_holes(binary_array)
			new_y_hat[binary_array] = channel

		return np.expand_dims(new_y_hat, 0)

	def predict(self, image_path: str, cpu: bool = False) -> dict:
		'''
		Disables all model training features and performs an eval prediction 
		on the current LightningModule device

		Returns image dictionary ready for Saved from monai

		image_path: image path

		cpu: forces computation to happen in CPU, will take a tremendous amount of time
		
		returns:
			numpy array
		'''
		self.eval()
		
		if cpu:
			self.cpu()
		else:
			self.cuda()

		reference = self.loader({"image": image_path})
		x = self.transform(reference)
		x = x['image'].permute(0, 3, 1, 2)
		x = torch.clip(x, -1024, 600)
		x = (x - x.min())/(x.max() - x.min())
		x = x.unsqueeze(0)
		x = x.float()

		with torch.no_grad():
			y_hat = sliding_window_inference(x.to(self.device),
											 roi_size=(80, 160, 160),  # For lower GPU usage, results could be better with 128, 256, 256
											 sw_batch_size=1, 
											 predictor=self,
											 overlap=0.2,
											 mode="gaussian",
											 progress=True,
											 device=self.device).softmax(dim=1).detach().float().cpu()  # 5 dims
		
		# Put yourself on the cpu after predicting
		self.cpu()

		# Per channel inverse transform and rounding
		y_hat = np.concatenate([np.round(self.inverse_transform(y_hat[:, c], self.transform)["image"]) for c in range(y_hat.shape[1])], axis=0).argmax(axis=0, keepdims=True)

		# Post processing
		y_hat = self.post_processing(y_hat)

		return {"image": y_hat, "image_meta_dict": reference["image_meta_dict"]}


def debug(image_path):
	test_model = LoberModule.load_from_checkpoint("lober.ckpt")
	output = test_model.predict(image_path)

	monai_saver = SaveImaged(keys=["image"],
							 meta_keys=["image_meta_dict"],
							 output_ext=".nii.gz",
							 output_dir=".",
							 output_dtype=np.uint8,
							 output_postfix="lobes",
							 separate_folder=False)
	monai_saver(output)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("input")
	args = parser.parse_args()

	debug(args.input)
