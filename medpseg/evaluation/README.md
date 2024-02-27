# Reproducing Test Evaluation in CoronaCases and SemiSeg

Follow the main README to install MEDPSeg.

After having installed MEDPSeg, that is, the command "medpseg" works in your environment, this will guide you through reproducing our published results on the public CoronaCases and SemiSeg datasets. These datasets were chosen for this due to their ease of access without requiring registration to medical imaging segmentation challenges. CoronaCases was used as an external test dataset, not included in training, being a fair comparison site for other methods. SemiSeg was used in training validation and testing, following Inf-Net's splits. 

All metrics are calculated using the code in medpseg/seg_metrics.py or medpseg/atm_evaluation.py for BD and TD.

The remainder of this README will guide you through reproducing our evaluation metrics on CoronaCases and SemiSeg test split.

## Evaluation Reproduction

The file code in reproduce_evaluation.py will reproduce our evaluation results in the CoronaCases and SemiSeg datasets. Just running:

    python reproduce_evaluation.py

Following is a description of each step:

1. Donwload data for [CoronaCases](https://zenodo.org/records/3757476) and [SemiSeg](https://figshare.com/articles/dataset/MedSeg_Covid_Dataset_1/13521488)
2. Unpack and preprocess.
3. Run predictions in both datasets using the MEDPSeg CLI.
4. Compute metrics using implementations in [seg_metrics.py](medpseg/seg_metrics.py).
5. Generate a results [table](medpseg/evaluation/results/table.html).

The results generated in the table should be equal to what an user achieves through running the script on their environment. Evaluation in different datasets can be easily achieved through minor editing to [reproduce_evaluation.py](medpseg/evaluation/results/reproduce_evaluation.py), mainly downloading and data organization code. Note that some results are slightly different (0.00x variations) when using pre-processed .png instead of direct HU values, due to conversions from int16 to uint8 and back.
