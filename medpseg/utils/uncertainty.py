'''
Copyright (c) Diedre Carmo, Medical Imaging Computing Lab (MICLab)
https://miclab.fee.unicamp.br/
https://github.com/MICLab-Unicamp/medpseg
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Attempts at exploring epistemic uncertainty, did not work
'''
import torch
from torch.nn import Module
from torch import Tensor
from typing import Tuple
from tqdm import tqdm


def get_epistemic_uncertainty(model: Module, x: Tensor, n: int = 10) -> Tuple[Tensor, Tensor]:
    '''
    Estimates epistemic uncertainty with n monte carlo predictions of model on x.

    Returns:
        standard deviation uncertainty, mean prediction
    '''
    model = model.train()
    with torch.no_grad():
        uncertain_preds = [model(x).detach().cpu() for _ in tqdm(range(n), leave=False)]
    model = model.eval()

    uncertain_preds_tensor = torch.stack(uncertain_preds)
    epistemic_uncertainty = uncertain_preds_tensor.std(dim=0)
    mean_prediction = uncertain_preds_tensor.mean(dim=0)
    
    return epistemic_uncertainty, mean_prediction
