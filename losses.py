# Author: Alexander Decurnou

import math
import numpy as np


def multitask_loss(num_dists, pred_sal_map_set, gt_sal_map_set):

    loss = 0

    for pred, gt in zip(pred_sal_map_set, gt_sal_map_set):
        loss += multitask_loss_helper(pred, gt)

    loss = loss * (1 / 2 * num_dists)

    return loss


def multitask_loss_helper(pred_sal_map, ground_truth_sal_map, alpha=5):

    if ground_truth_sal_map > 0.5:
        distance = alpha * math.pow((pred_sal_map - ground_truth_sal_map), 2)
    else:
        distance = math.pow((pred_sal_map - ground_truth_sal_map), 2)

    return distance


def final_multitask_loss(multitask_loss, multinomial_loss, weight_factor=0.5):

    final_loss = multitask_loss + (weight_factor * multinomial_loss)

    return final_loss
