# Reproducing Test Evaluation in CoronaCases and SemiSeg

Follow the main README to install MEDPSeg.

After having installed MEDPSeg, that is, the command "medpseg" works in your environment, this will guide you through reproducing our published results on the public CoronaCases and SemiSeg datasets. These datasets were chosen for this due to their ease of access without requiring registration to medical imaging segmentation challenges. CoronaCases was used as an external test dataset, not included in training, being a fair comparison site for other methods. SemiSeg was used in training validation and testing, following Inf-Net's splits. 

All metrics are calculated using the code in medpseg/seg_metrics.py and medpseg/atm_evaluation.py

The remainder of this README will guide you through reproducing our evaluation metrics on CoronaCases and SemiSeg test split.

## Download raw data

CoronaCases: https://zenodo.org/records/3757476
SemiSeg: https://figshare.com/articles/dataset/MedSeg_Covid_Dataset_1/13521488

TODO

