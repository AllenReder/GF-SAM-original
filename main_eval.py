# main_eval.py
r""" Matcher testing code for one-shot segmentation """
import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append('./')

from matcher.common.logger import Logger, AverageMeter
from matcher.common.vis import Visualizer
from matcher.common.evaluation import Evaluator
from matcher.common import utils
from matcher.data.dataset import FSSDataset
from matcher.GFSAM import build_model
from matcher.data.task import SUPP_CLASS_IDS

from tool import *

import random
random.seed(0)


def test(GFSAM, dataloader, args=None):
    r""" Test GFSAM """

    # Freeze randomness during testing for reproducibility
    # Follow HSNet
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        batch = utils.to_cuda(batch)
        # (1, 3, H, W) (1, H, W) (1, supp_class_num*shot, 3, H, W) (1, supp_class_num*shot, H, W)
        query_img, query_mask, support_imgs, support_masks = \
            batch['query_img'], batch['query_mask'], \
            batch['support_imgs'], batch['support_masks']
        
        # show_image([query_img[0] * 255, query_mask[0].long()], save='debug/query.png')
        # for i in range(support_imgs.shape[1]):
        #     show_image([support_imgs[0, i] * 255, support_masks[0, i].long()], save=f'debug/support_{i+1}.png')
        #     Logger.info(f"support_masks_{i+1} save: {f'debug/support_{i+1}.png'}")

        final_pred_mask = torch.zeros_like(query_mask)

        for supp_cls_idx in range(support_masks.shape[1]):

            support_cls_img = support_imgs[:, supp_cls_idx, :, :, :].unsqueeze(1)
            support_cls_mask = support_masks[:, supp_cls_idx, :, :].unsqueeze(1)

            # 1. GFSAM准备参考图像和目标图像
            GFSAM.set_reference(support_cls_img, support_cls_mask)
            GFSAM.set_target(query_img)

            # 2. 预测目标的掩码
            pred_mask, _, confidence = GFSAM.predict()
            GFSAM.clear()

            if confidence > 0.5:
                final_pred_mask[pred_mask > 0] = dataloader.dataset.supp_class_ids[supp_cls_idx]
        
        ### DEBUG
        # show_image([query_img[0] * 255, query_mask[0].long(), final_pred_mask[0].long()], save='output.png')
        # Logger.info(f"pred_mask save: {f'output.png'}")
        # input()

        # 3. 评估预测结果
        area_inter, area_union = Evaluator.classify_prediction(final_pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union)
        average_meter.write_process(idx, len(dataloader), write_batch_idx=1)

    # Write evaluation results
    average_meter.write_result(0)
    miou, fb_iou, _ = average_meter.compute_iou()

    return miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='GFSAM Pytorch Implementation for One-shot Segmentation')

    # 数据集参数
    parser.add_argument('--datapath', type=str, default='datasets')
    parser.add_argument('--benchmark', type=str, default='pascal',
                        choices=['fss', 'coco', 'pascal', 'lvis', 'paco_part', 'pascal_part', 'deepglobe', 'isic', 'isaid'])
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--fold', type=int, default=3)
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--img-size', type=int, default=1024)
    parser.add_argument('--use_original_imgsize', action='store_true')
    parser.add_argument('--log-root', type=str, default='output/debug')
    parser.add_argument('--visualize', type=int, default=0)

    # DINOv2和SAM参数
    parser.add_argument('--dinov2-size', type=str, default="vit_large")
    parser.add_argument('--sam-size', type=str, default="vit_h")
    parser.add_argument('--dinov2-weights', type=str, default="models/dinov2_vitl14_pretrain.pth")
    parser.add_argument('--sam-weights', type=str, default="models/sam_vit_h_4b8939.pth")



    args = parser.parse_args()

    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    Logger.initialize(args, root=args.log_root)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())

    # Model initialization
    GFSAM = build_model(args)

    # Helper classes (for testing) initialization
    Evaluator.initialize(SUPP_CLASS_IDS[args.benchmark][args.fold])
    Visualizer.initialize(args.visualize)

    # Dataset initialization
    FSSDataset.initialize(img_size=args.img_size, datapath=args.datapath, use_original_imgsize=args.use_original_imgsize)
    dataloader_test = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot)

    # Test GFSAM
    with torch.no_grad():
        test_miou, test_fb_iou = test(GFSAM, dataloader_test, args=args)
    Logger.info('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')