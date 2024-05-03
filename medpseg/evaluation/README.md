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
4. Compute metrics using implementations in [seg_metrics.py](../seg_metrics.py).
5. Generate a results [table](results/table.html).

The following table is what we have achieved through running the script in February 2024:

<table style="width:90%" border="1"><tr><th>semiseg_2d_ggo</th><th>semiseg_2d_consolidation</th><th>coronacases_3d_inf</th></tr><tr><td><table style="width:90%" border="1"><tr><th>dice</th><th>false_negative_error</th><th>false_positive_error</th><th>sensitivity</th><th>specificity</th></tr><tr><td><table style="width:90%" border="1"><tr><th>mean</th><th>std</th></tr><tr><td>0.6493887637341156</td><td>0.20984685178535936</td></tr></table></td><td><table style="width:90%" border="1"><tr><th>mean</th><th>std</th></tr><tr><td>0.3486784022810951</td><td>0.20904111222696362</td></tr></table></td><td><table style="width:90%" border="1"><tr><th>mean</th><th>std</th></tr><tr><td>0.29967016007280345</td><td>0.194346475130671</td></tr></table></td><td><table style="width:90%" border="1"><tr><th>mean</th><th>std</th></tr><tr><td>0.6367933309571926</td><td>0.20060453882862272</td></tr></table></td><td><table style="width:90%" border="1"><tr><th>mean</th><th>std</th></tr><tr><td>0.9893210889893039</td><td>0.011094264979578237</td></tr></table></td></tr></table></td><td><table style="width:90%" border="1"><tr><th>dice</th><th>false_negative_error</th><th>false_positive_error</th><th>sensitivity</th><th>specificity</th></tr><tr><td><table style="width:90%" border="1"><tr><th>mean</th><th>std</th></tr><tr><td>0.5769309854556685</td><td>0.29307313426454484</td></tr></table></td><td><table style="width:90%" border="1"><tr><th>mean</th><th>std</th></tr><tr><td>0.3634770364650775</td><td>0.2696565459942699</td></tr></table></td><td><table style="width:90%" border="1"><tr><th>mean</th><th>std</th></tr><tr><td>0.3294236091277356</td><td>0.2810979814012969</td></tr></table></td><td><table style="width:90%" border="1"><tr><th>mean</th><th>std</th></tr><tr><td>0.5340037994037468</td><td>0.21319093055192054</td></tr></table></td><td><table style="width:90%" border="1"><tr><th>mean</th><th>std</th></tr><tr><td>0.994367999682688</td><td>0.01025442270113634</td></tr></table></td></tr></table></td><td><table style="width:90%" border="1"><tr><th>dice</th><th>false_negative_error</th><th>false_positive_error</th><th>sensitivity</th><th>specificity</th></tr><tr><td><table style="width:90%" border="1"><tr><th>mean</th><th>std</th></tr><tr><td>0.8135291858795479</td><td>0.06652757121479057</td></tr></table></td><td><table style="width:90%" border="1"><tr><th>mean</th><th>std</th></tr><tr><td>0.1652826958455593</td><td>0.1218182096414322</td></tr></table></td><td><table style="width:90%" border="1"><tr><th>mean</th><th>std</th></tr><tr><td>0.18548197424264676</td><td>0.08417444667991547</td></tr></table></td><td><table style="width:90%" border="1"><tr><th>mean</th><th>std</th></tr><tr><td>0.8347173041544407</td><td>0.1218182096414322</td></tr></table></td><td><table style="width:90%" border="1"><tr><th>mean</th><th>std</th></tr><tr><td>0.9988212383395796</td><td>0.0010165322636769925</td></tr></table></td></tr></table></td></tr></table>

The results generated in the table should be equal to what an user achieves through running the script on their environment. Evaluation in different datasets can be easily achieved through minor editing to [reproduce_evaluation.py](reproduce_evaluation.py), mainly downloading and data organization code. Note that some results are slightly different (0.00x variations) when using pre-processed .png instead of direct HU values, due to conversions from int16 to uint8 and back.
