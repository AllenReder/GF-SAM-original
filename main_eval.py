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

import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# sim map
def show_sim_map(sim_map, image):
    """
    在原图上半透明叠加热力图并显示 colorbar 与最小/最大值标注。
    sim_map: 2D numpy array, 值域任意（会归一化到 [0,1]）
    image: HxWx3 numpy array (RGB, 0-1 或 0-255)
    """
    # 确保 image 为 0-1 范围的 float
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    
    # 如果 sim_map 为 tensor，转换为 numpy array
    if isinstance(sim_map, torch.Tensor):
        sim_map = sim_map.numpy()

    # 缩放到相同大小
    # img = cv2.resize(img, (sim_map.shape[1], sim_map.shape[0]), interpolation=cv2.INTER_LINEAR)
    sim_map = cv2.resize(sim_map, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 归一化到 [0,1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # 先显示原图
    plt.imshow(img)
    # 再叠加热力图（alpha 控制透明度）
    # 自定义 jet cmap, 修改当值为0时显示透明色, 其他按原版jet
    cmap = plt.cm.get_cmap('jet')
    cmap.set_under('none')
    sim_map_masked = np.where(sim_map == 0, -0.001, sim_map)
    im = plt.imshow(sim_map_masked, cmap=cmap, vmin=0, vmax=1, alpha=0.5)
    cbar = plt.colorbar(im)

    # 显示并标注最小/最大值
    vmin, vmax = im.get_clim()
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f'{vmin:.3f}', f'{vmax:.3f}'])

    # 在 colorbar 上画点并带文本（使用 axes fraction，y∈[0,1]）
    for val, color in [(sim_map.min(), 'black'), (sim_map.max(), 'white')]:
        y = (val - vmin) / (vmax - vmin + 1e-12)
        cbar.ax.scatter(0.5, y, transform=cbar.ax.transAxes, color=color, s=30)
        cbar.ax.text(0.62, y, f'{val:.3f}', transform=cbar.ax.transAxes, va='center', fontsize=8)

def draw_prompt(coord_xy):
    for i in range(coord_xy.shape[0]):
        x, y = coord_xy[i]
        # 注意 imshow 的坐标是 (y, x)
        plt.scatter(x, y, c='red', s=20, marker='o', edgecolors='white', linewidths=1.5, label='prompt' if i == 0 else "")
    if coord_xy.shape[0] > 0:
        plt.legend(loc='upper right', fontsize=8)

def test(GFSAM, dataloader, args=None):
    r""" Test GFSAM """

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
            support_cls_mask = (support_masks[:, supp_cls_idx, :, :].unsqueeze(1) == dataloader.dataset.supp_class_ids[supp_cls_idx]).float()
        
            # 1. GFSAM准备参考图像和目标图像
            GFSAM.set_reference(support_cls_img, support_cls_mask)
            GFSAM.set_target(query_img)

            # 2. 预测目标的掩码
            pred_mask, _, confidence, dbg = GFSAM.predict()
            GFSAM.clear()

            max_sim_map = dbg["max_sim_map"]
            mean_sim_map = dbg["mean_sim_map"]
            cross_sim_map = dbg["cross_sim_map"]
            neg_mean_sim_map = dbg["neg_mean_sim_map"]
            coord_xy = dbg["coord_xy"] # 属于 (1024, 1024) 坐标

            cmap = plt.cm.get_cmap('nipy_spectral', 21)
            bounds = np.arange(22) - 0.5
            norm = mcolors.BoundaryNorm(bounds, cmap.N)
            # img & mask
            plt.figure(figsize=(20, 10))
            plt.subplot(2, 4, 1)
            query_img_np = query_img[0].cpu().permute(1, 2, 0).numpy() # (1, 3, H, W) -> (H, W, 3)
            draw_prompt(coord_xy)
            plt.imshow(query_img_np)
            plt.title(f'query_img')
            plt.subplot(2, 4, 2)
            query_mask_np = query_mask[0].cpu().squeeze(0).numpy().astype(np.uint8)
            plt.imshow(query_mask_np, cmap=cmap, norm=norm)
            draw_prompt(coord_xy)
            query_cls = list(np.unique(query_mask_np))
            plt.title(f'query_mask: {query_cls}')
            plt.subplot(2, 4, 3)
            pred_mask_np = pred_mask[0].cpu().squeeze(0).numpy().astype(np.uint8)
            pred_mask_np[pred_mask_np > 0] = dataloader.dataset.supp_class_ids[supp_cls_idx]
            plt.imshow(pred_mask_np, cmap=cmap, norm=norm)
            draw_prompt(coord_xy)
            plt.title(f'pred_mask_{dataloader.dataset.supp_class_ids[supp_cls_idx]}')

            plt.subplot(2, 4, 4)
            neg_mean_sim_map_std = (neg_mean_sim_map - neg_mean_sim_map.min()) / (neg_mean_sim_map.max() - neg_mean_sim_map.min() + 1e-6)
            neg_region = neg_mean_sim_map_std > cross_sim_map
            pos_region_sim = cross_sim_map * ~neg_region
            show_sim_map(pos_region_sim, query_img_np)
            draw_prompt(coord_xy)
            plt.title('pos_region_sim')
            plt.subplot(2, 4, 5)
            show_sim_map(max_sim_map, query_img_np)
            plt.title('max_sim_map')
            plt.subplot(2, 4, 6)
            show_sim_map(mean_sim_map, query_img_np)
            plt.title('mean_sim_map')
            plt.subplot(2, 4, 7)
            show_sim_map(cross_sim_map, query_img_np)
            plt.title('cross_sim_map')
            plt.subplot(2, 4, 8)
            show_sim_map(neg_mean_sim_map, query_img_np)
            plt.title('neg_mean_sim_map')

            plt.tight_layout()
            plt.savefig('debug.png')

            print(f"cls {supp_cls_idx} debug.png saved")
            input()

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
    parser.add_argument('--benchmark', type=str, default='coco',
                        choices=['fss', 'coco', 'pascal', 'lvis', 'paco_part', 'pascal_part', 'deepglobe', 'isic', 'isaid'])
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
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

    utils.fix_randseed(0)

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