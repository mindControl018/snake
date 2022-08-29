import os
import cv2
import json
import numpy as np
from lib.utils.snake import snake_config, snake_cityscapes_utils, snake_eval_utils, snake_poly_utils

from lib.config import cfg
from lib.datasets.dataset_catalog import DatasetCatalog
from lib.utils import data_utils


class Evaluator:
    def __init__(self, result_dir):
        self.dice_results = []

        self.result_dir = result_dir
        os.system('mkdir -p {}'.format(self.result_dir))

        args = DatasetCatalog.get(cfg.test.dataset)


    def evaluate(self, output, batch):

        gt_mask = batch['query_label']
        pred_poly = output['py_pred'][-1]
        bg = np.zeros(gt_mask.shape, np.uint8)

        for i in range(gt_mask.shape[0]):
            pred_mask = cv2.drawContours(bg[i], pred_poly[i], -1, 1, -1)
            dice = self.dice_coeff(pred_mask, gt_mask[i])
            self.dice_results.append(dice)

    def dice_coeff(self, pred, target):
        smooth = 1.
        intersection = (pred * target).sum()
        return (2 * intersection +smooth) / (pred.sum() + target.sum() + smooth)

    def summarize(self):
        json.dump(self.dice_results, open(os.path.join(self.result_dir, 'results.json'), 'w'))
        return {'dice': np.array(self.result_dir).mean()}
