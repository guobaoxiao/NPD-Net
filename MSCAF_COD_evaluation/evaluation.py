# -*- coding: utf-8 -*-
# @Time    : 2020/11/21
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import os
import cv2
from tqdm import tqdm
import torch
import sod_metrics as M
import torch.nn.functional as F

FM = M.Fmeasure()
WFM = M.WeightedFmeasure()
SM = M.Smeasure()
EM = M.Emeasure()
MAE = M.MAE()

mask_root = '/data/Jenny/NPD-Net/data/TestDataset/ETIS-LaribPolypDB/masks/'
pred_root = '/data/Jenny/NPD-Net/weight/after-test/ETIS-LaribPolypDB/'

def _upsample_like(src, tar):
    src = torch.tensor(src, dtype=torch.float32)
    tar = torch.tensor(tar)
    src = F.interpolate(src.unsqueeze(0).unsqueeze(0), size=tar.shape, mode='bilinear')
    src = src.squeeze(0).squeeze(0).numpy()
    return src
mask_name_list = sorted(os.listdir(mask_root))
for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
    mask_path = os.path.join(mask_root, mask_name)
    pred_path = os.path.join(pred_root, mask_name)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    if len(pred.shape) != 2:
        pred = pred[:, :, 0]  # 返回(height, width)
    if len(mask.shape) != 2:
        mask = mask[:, :, 0]
    pred = _upsample_like(pred, mask)
    assert pred.shape == mask.shape

    FM.step(pred=pred, gt=mask)
    WFM.step(pred=pred, gt=mask)
    SM.step(pred=pred, gt=mask)
    EM.step(pred=pred, gt=mask)
    MAE.step(pred=pred, gt=mask)

fm = FM.get_results()['fm']
wfm = WFM.get_results()['wfm']
sm = SM.get_results()['sm']
em = EM.get_results()['em']
mae = MAE.get_results()['mae']

print(
    'Smeasure:', sm.round(3), '; ',
    'wFmeasure:', wfm.round(3), '; ',
    'MAE:', mae.round(3), '; ',
    'adpEm:', em['adp'].round(3), '; ',
    'meanEm:', '-' if em['curve'] is None else em['curve'].mean().round(3), '; ',
    'maxEm:', '-' if em['curve'] is None else em['curve'].max().round(3), '; ',
    'adpFm:', fm['adp'].round(3), '; ',
    'meanFm:', fm['curve'].mean().round(3), '; ',
    'maxFm:', fm['curve'].max().round(3),
    sep=''
)

with open("../result.txt", "a+") as f:
    print('Smeasure:', sm.round(3), '; ',
          'meanEm:', '-' if em['curve'] is None else em['curve'].mean().round(3), '; ',
          'wFmeasure:', wfm.round(3), '; ',
          'MAE:', mae.round(3), '; ',
          file=f
          )
