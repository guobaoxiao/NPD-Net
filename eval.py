import os
import tqdm
import sys
import yaml
import numpy as np
from PIL import Image

filepath = os.path.split(os.path.abspath(__file__))[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from utilss.eval_functions import *
from utilss.utils import *


def evaluate(opt):
    if os.path.isdir(opt['Eval']['result_path']) is False:
        os.makedirs(opt['Eval']['result_path'])

    method = os.path.split(opt['Eval']['pred_root'])[-1]
    Thresholds = np.linspace(1, 0, 256)
    headers = opt['Eval']['metrics']  # ['meanDic', 'meanIoU', 'wFm', 'Sm', 'meanEm', 'mae', 'maxEm', 'maxDic', 'maxIoU', 'meanSen', 'maxSen', 'meanSpe', 'maxSpe']
    results = []

    if True is True:
        print('#' * 20, 'Start Evaluation', '#' * 20)
        datasets = tqdm.tqdm(opt['Eval']['datasets'], desc='Expr - ' + method, total=len(
            opt['Eval']['datasets']), position=0, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
    else:
        datasets = opt['Eval']['datasets']

    for dataset in datasets:
        pred_root = os.path.join(opt['Eval']['pred_root'], dataset)
        gt_root = os.path.join(opt['Eval']['gt_root'], dataset, 'masks')

        preds = os.listdir(pred_root)
        gts = os.listdir(gt_root)

        preds.sort()
        gts.sort()

        threshold_IoU = np.zeros((len(preds), len(Thresholds)))
        threshold_Dice = np.zeros((len(preds), len(Thresholds)))

        Smeasure = np.zeros(len(preds))
        wFmeasure = np.zeros(len(preds))
        MAE = np.zeros(len(preds))

        if True is True:
            samples = tqdm.tqdm(enumerate(zip(preds, gts)), desc=dataset + ' - Evaluation', total=len(
                preds), position=1, leave=False, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
        else:
            samples = enumerate(zip(preds, gts))

        for i, sample in samples:
            pred, gt = sample
            assert os.path.splitext(pred)[0] == os.path.splitext(gt)[0]

            pred_mask = np.array(Image.open(os.path.join(pred_root, pred)))
            gt_mask = np.array(Image.open(os.path.join(gt_root, gt)))

            if len(pred_mask.shape) != 2:
                pred_mask = pred_mask[:, :, 0]
            if len(gt_mask.shape) != 2:
                gt_mask = gt_mask[:, :, 0]

            assert pred_mask.shape == gt_mask.shape

            gt_mask = gt_mask.astype(np.float64) / 255
            gt_mask = (gt_mask > 0.5).astype(np.float64)

            pred_mask = pred_mask.astype(np.float64) / 255

            Smeasure[i] = StructureMeasure(pred_mask, gt_mask)
            wFmeasure[i] = original_WFb(pred_mask, gt_mask)
            MAE[i] = np.mean(np.abs(gt_mask - pred_mask))

            threshold_Iou = np.zeros(len(Thresholds))
            threshold_Dic = np.zeros(len(Thresholds))

            for j, threshold in enumerate(Thresholds):
               threshold_Dic[j], threshold_Iou[j] = Fmeasure_calu(pred_mask, gt_mask, threshold)

            threshold_Dice[i, :] = threshold_Dic
            threshold_IoU[i, :] = threshold_Iou

        result = []


        column_Dic = np.mean(threshold_Dice, axis=0)
        column_IoU = np.mean(threshold_IoU, axis=0)
        meanDic = np.mean(column_Dic)
        meanIoU = np.mean(column_IoU)
        out = []
        for metric in opt['Eval']['metrics']:
            out.append(eval(metric))
            print(out)
        result.extend(out)

        csv = os.path.join(opt['Eval']['result_path'], 'result_' + dataset + '.csv')
        if os.path.isfile(csv) is True:
            csv = open(csv, 'a')
        else:
            csv = open(csv, 'w')
            csv.write(', '.join(['method', *headers]) + '\n')

        out_str = method + ','
        # print(result)
        for metric in result:
            out_str += '{:.4f}'.format(metric) + ','
        out_str += '\n'
        # print(out_str)
        csv.write(out_str)
        csv.close()

    if True is True:
        print("#" * 20, "End Evaluation", "#" * 20)

    return 0


if __name__ == "__main__":
    with open('./config.yaml', 'r', encoding='utf8') as file:  # utf8可识别中文
        opt = yaml.safe_load(file)
    evaluate(opt)





