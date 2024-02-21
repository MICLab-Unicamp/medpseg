'''
To read covidsemiseg slices following Inf-Net's[1] train/val/test split

[1] Fan, Deng-Ping, et al. "Inf-net: Automatic covid-19 lung infection segmentation from ct images." IEEE transactions on medical imaging 39.8 (2020): 2626-2637.
'''
import os
import json
import torch
import numpy as np
from sanitize_filename import sanitize
from torch.utils.data import Dataset
from datasets.base import PATHS, DataModule
from DLPT.utils.unified_img_reading import unified_img_reading


class CovidSemiSeg(Dataset):
    def __init__(self, mode, transform=None, include_removed=False, extended_2d=1):
        base_folder = os.path.join(PATHS["ideiagov"], "slice_datasets", "COVID-19-CT100")
        with open(os.path.join(base_folder, "covid_semi_seg_split.json"), 'r') as splits_file:
            self.splits = json.load(splits_file)
        
        if mode == "test" and include_removed:
            print("Including images that InfNet author removed.")
            self.splits["test"] = self.splits["test"] + self.splits["removed"]
        
        self.extended_2d = extended_2d
        self.keys = self.splits[mode]
        self.transform = transform
        self.mode = mode
        self.data_path = os.path.join(base_folder, "tr_im.nii.gz")
        self.mask_path = os.path.join(base_folder, "merged.nii.gz")
        self.data, self.masks, self.spacing = unified_img_reading(self.data_path, self.mask_path, torch_convert=True, convert_to_onehot=4)
        print(f"Initialized {mode} CovidSemiSeg with and {len(self.keys)} items. Slices are axial and positive.")
        
    def __len__(self):
        return len(self.keys)
    
    def turn_on_poly_processing(self):
        # Nothing to do this is already poly
        pass
    
    def __getitem__(self, i):
        idx = self.keys[i]
        assert idx >= 0 and idx < 100
        data, mask = self.data[:, idx], self.masks[:, idx]  # InfSeg used 0 indexing
        
        # Extended 2d 1
        if self.extended_2d == 1:
            data_25d = torch.zeros(size=(3,) + data.shape[1:])
            data_25d[0] = data[0]
            data_25d[1] = data[0]
            data_25d[2] = data[0]
            data = data_25d

        if self.transform is not None:
            data, mask = self.transform(data, mask)

        ID = f"semiseg_{idx}"
        metadata = {"ID": ID, 
                    "sID": sanitize(ID),
                    "desc": f"semiseg {self.mode} {i} {ID}",
                    "path": self.data_path,
                    "spacing": list(self.spacing),
                    "label": 1,
                    "preprocess": "separation",
                    "bbox": [-1, -1, -1, -1, -1, -1]}

        return data, mask, metadata


class CovidSemiSegDataModule(DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def dataset_setup(self):
        self.train = CovidSemiSeg("train", transform=self.train_transform)
        self.val = CovidSemiSeg("val", transform=self.eval_transform)
        self.test = CovidSemiSeg("test", transform=self.eval_transform)
