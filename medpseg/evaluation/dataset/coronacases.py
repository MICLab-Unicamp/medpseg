'''
Abstract CoronaCases volumes, this was only used for testing in MEDPSeg, and not included in training.
'''
from .h5covidseg import H52DCovid19Seg
import os
import glob
import numpy as np

from datasets.base import DataModule, MANUAL_SEED, PATHS
from datasets.segmentation.seg_dataset import SegDataset
from sklearn.model_selection import train_test_split, KFold


#######################################################################################################
################################ CoronaCases ###############################################################
class CoronaCases(SegDataset):
    '''
    Reading data from CoronaCases (20 cases)
    '''
    def init(self):
        self.nclasses = 3
        
        self.keys = sorted(glob.glob(os.path.join(PATHS["public"], "COVID-19-CT-Seg_20cases", "*.nii.gz")))
        
        # Radiopaedia volumes not playing well with pre processing
        new_keys = []
        for x in self.keys:
            if "radiopaedia" not in x:
                new_keys.append(x)
        self.keys = new_keys

        mode = self.mode
        test_fold = self.test_fold

        if test_fold is None:
            print("Using Holdout approach.")
            self.train, test = train_test_split(self.keys, train_size=0.8, test_size=0.2, shuffle=True, random_state=MANUAL_SEED)
            self.val, self.test = train_test_split(test, train_size=0.5, shuffle=False)
            if self.mode is not None:
                self.keys = getattr(self, mode)
        else:
            print(f"Using 5-Fold approach, fold {test_fold}.")
            kfold = KFold(5, shuffle=True, random_state=MANUAL_SEED)
            splits = [x for x in kfold.split(self.keys)]
            subjects_idx = {}
            subjects_idx["train"], subjects_idx["test"] = splits[test_fold]
            subjects_idx["train"], subjects_idx["val"] = train_test_split(subjects_idx["train"], test_size=0.125, shuffle=False)
            idxs = subjects_idx[mode]
            self.keys = np.array(self.keys)[idxs]

        self.len = len(self.keys)
        self.idxs = [os.path.basename(path).replace(".nii.gz", "") for path in self.keys]

    def __init__(self, mode, transform=None, test_fold=None, isometric=False):
        self.test_fold = test_fold
        super().__init__(name="CoronaCases", mode=mode, transform=transform, isometric=isometric)

    def get_paths(self, path, ID):
        mask_path = os.path.join(os.path.dirname(path), "Lung_and_Infection_Mask", f"{ID}.nii.gz")

        return path, mask_path

    def correction_callback(self, img, msk):
        msk[msk == 2] = 1
        msk[msk == 3] = 2

        return img, msk

    def __getitem__(self, i):
        return super().__getitem__(i, pre_transform_callback=self.correction_callback)

class CoronaCasesDataModule(DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_data = kwargs.get("all_data", False)

    def dataset_setup(self):
        test_fold = getattr(self.hparams, "test_fold", None)
        self.train = CoronaCases("train", transform=self.train_transform, test_fold=test_fold, isometric=self.hparams.isometric)
        self.val = CoronaCases("val", transform=self.eval_transform, test_fold=test_fold, isometric=self.hparams.isometric)
        if self.all_data:
            print("WARNING: Using all data from CoronaCases!")
            self.test = CoronaCases(None, transform=self.eval_transform, test_fold=test_fold, isometric=self.hparams.isometric)
        else:
            self.test = CoronaCases("test", transform=self.eval_transform, test_fold=test_fold, isometric=self.hparams.isometric)

#######################################################################################################


class CoronaCases2DDataModule(DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.volumetric = False

    def dataset_setup(self):
        self.train = H52DCovid19Seg(CoronaCases("train", test_fold=self.hparams.test_fold).idxs,
                                    os.path.join("/home", "diedre", "processed_data", "big_seg_dataset_v3.hdf5"),
                                    transform=self.train_transform)
        self.val = H52DCovid19Seg(CoronaCases("val", test_fold=self.hparams.test_fold).idxs,
                                    os.path.join("/home", "diedre", "processed_data", "big_seg_dataset_v3.hdf5"),
                                    transform=self.eval_transform)
        self.test = H52DCovid19Seg(CoronaCases("test", test_fold=self.hparams.test_fold).idxs,
                                    os.path.join("/home", "diedre", "processed_data", "big_seg_dataset_v3.hdf5"),
                                    transform=self.eval_transform)
