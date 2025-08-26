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
from matcher.data.task import SUPP_CLASS_IDS, CLASS_NAME

from tool import *

import random
random.seed(0)

import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import matplotlib.patches as patches

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

def add_confusion_matrix(original_matrix, gt, pred):
    """
    根据 gt 和 pred 计算出混淆矩阵, 加到原始混淆矩阵并返回
    gt, pred: numpy array，shape一致，值为类别索引（0~num_classes-1）
    original_matrix: shape=(num_classes, num_classes)
    返回：更新后的混淆矩阵
    """
    num_classes = original_matrix.shape[0]
    # 展平成一维
    # 只统计合法类别
    mask = (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
    inds = gt[mask] * num_classes + pred[mask]
    batch_cm = np.bincount(inds, minlength=num_classes*num_classes).reshape(num_classes, num_classes)
    return original_matrix + batch_cm


def test(GFSAM, dataloader, args=None):
    r""" Test GFSAM """

    ### DEBUG 混淆矩阵

    class_labels = [0] + list(SUPP_CLASS_IDS[args.benchmark][args.fold])
    num_classes = len(class_labels)
    class_names = CLASS_NAME[args.benchmark]
    class_names = [class_names[i] for i in class_labels] # 类别名
    # 映射
    label_to_index = dict()
    for i, lbl in enumerate(class_labels):
        label_to_index[int(lbl)] = i
    vec_map = np.vectorize(lambda v: label_to_index.get(int(v), 0))

    prompt_confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.float32) # 提示点真实类别与预测类别的混淆矩阵
    pixel_confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.float32) # 真实类别与预测类别的混淆矩阵

    ###
    total_pixel_num = 0
    overlap_pixel_num = np.zeros((num_classes, ), dtype=np.int32) # 被 k 个类别预测伪掩码重叠的像素数量
    overlap_pixel_wrong_pred_num = np.zeros((num_classes, ), dtype=np.int32) # 被 k 个类别预测伪掩码重叠且预测错误的像素数量

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

        ### DEBUG
        overlap_count = torch.zeros(query_mask.shape[-2:], dtype=torch.int32) # 每个像素被不同类别掩码重叠的次数
        overlap_might_right = torch.zeros(query_mask.shape[-2:], dtype=bool) # 像素被正确类别覆盖的掩码

        # DEBUG
        total_pixel_num += 1024 * 1024

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
            prompt_coord_xy = dbg["coord_xy"] # 属于 (1024, 1024) 坐标
            

            ### DEBUG ---------- 提示点混淆矩阵记录 ---------- 
            query_mask_np = query_mask[0].cpu().squeeze(0).numpy().astype(np.int32)
            coords = prompt_coord_xy.astype(np.int64)
            xs = coords[:, 0]
            ys = coords[:, 1]
            # 过滤越界坐标
            h, w = query_mask_np.shape
            valid_mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            if valid_mask.any():
                xs = xs[valid_mask]
                ys = ys[valid_mask]
                # 获取这些坐标上的真实标签
                prompt_gt_labels = query_mask_np[ys, xs]
                # 把真实标签映射为混淆矩阵索引（label_to_index），未知标签映射为 0（背景）
                gt = vec_map(prompt_gt_labels)
                pred_idx = label_to_index.get(int(dataloader.dataset.supp_class_ids[supp_cls_idx]), 0)
                pred = np.ones_like(gt) * pred_idx
                prompt_confusion_matrix = add_confusion_matrix(prompt_confusion_matrix, gt, pred)

            ### DEBUG ---------- 重叠记录 ---------- 
            overlap_count[pred_mask[0].cpu() > 0] += 1
            overlap_might_right[(pred_mask[0] > 0).cpu() & (query_mask[0].cpu() == dataloader.dataset.supp_class_ids[supp_cls_idx])] = 1


            ### ---------- 相似度图可视化 ---------- 
            # cmap = plt.cm.get_cmap('nipy_spectral', 21)
            # bounds = np.arange(22) - 0.5
            # norm = mcolors.BoundaryNorm(bounds, cmap.N)
            # # img & mask
            # plt.figure(figsize=(20, 10))
            # plt.subplot(2, 4, 1)
            # query_img_np = query_img[0].cpu().permute(1, 2, 0).numpy() # (1, 3, H, W) -> (H, W, 3)
            # draw_prompt(prompt_coord_xy)
            # plt.imshow(query_img_np)
            # plt.title(f'query_img')
            # plt.subplot(2, 4, 2)
            # query_mask_np = query_mask[0].cpu().squeeze(0).numpy().astype(np.uint8)
            # plt.imshow(query_mask_np, cmap=cmap, norm=norm)
            # draw_prompt(prompt_coord_xy)
            # query_cls = list(np.unique(query_mask_np))
            # plt.title(f'query_mask: {query_cls}')
            # plt.subplot(2, 4, 3)
            # pred_mask_np = pred_mask[0].cpu().squeeze(0).numpy().astype(np.uint8)
            # pred_mask_np[pred_mask_np > 0] = dataloader.dataset.supp_class_ids[supp_cls_idx]
            # plt.imshow(pred_mask_np, cmap=cmap, norm=norm)
            # draw_prompt(prompt_coord_xy)
            # plt.subplot(2, 4, 4)
            # neg_mean_sim_map_std = (neg_mean_sim_map - neg_mean_sim_map.min()) / (neg_mean_sim_map.max() - neg_mean_sim_map.min() + 1e-6)
            # neg_region = neg_mean_sim_map_std > cross_sim_map
            # pos_region_sim = cross_sim_map * ~neg_region
            # show_sim_map(pos_region_sim, query_img_np)
            # draw_prompt(prompt_coord_xy)
            # plt.title('pos_region_sim')
            # plt.subplot(2, 4, 5)
            # show_sim_map(max_sim_map, query_img_np)
            # plt.title('max_sim_map')
            # plt.subplot(2, 4, 6)
            # show_sim_map(mean_sim_map, query_img_np)
            # plt.title('mean_sim_map')
            # plt.subplot(2, 4, 7)
            # show_sim_map(cross_sim_map, query_img_np)
            # plt.title('cross_sim_map')
            # plt.subplot(2, 4, 8)
            # show_sim_map(neg_mean_sim_map, query_img_np)
            # plt.title('neg_mean_sim_map')

            # plt.tight_layout()
            # plt.savefig('debug.png')

            # print(f"cls {supp_cls_idx} debug.png saved")
            # input()

            if confidence > 0.5:
                final_pred_mask[pred_mask > 0] = dataloader.dataset.supp_class_ids[supp_cls_idx]
        
        # 3. 评估预测结果
        area_inter, area_union = Evaluator.classify_prediction(final_pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union)
        average_meter.write_process(idx, len(dataloader), write_batch_idx=1)

        ### DEBUG ---------- 更新像素混淆矩阵 ----------

        # 使用第一张 query 的掩码
        query_mask = query_mask[0].cpu().squeeze(0).numpy().astype(np.uint8)  # (H, W)
        final_pred_mask = final_pred_mask[0].cpu().squeeze(0).numpy().astype(np.uint8)  # (H, W)

        gt = vec_map(query_mask).flatten()
        pred = vec_map(final_pred_mask).flatten()

        pixel_confusion_matrix = add_confusion_matrix(pixel_confusion_matrix, gt, pred)
        
        # 按每行归一化
        norm_pixel_confusion_matrix = pixel_confusion_matrix.copy()
        row_sum = norm_pixel_confusion_matrix.sum(axis=1, keepdims=True)
        norm_pixel_confusion_matrix = pixel_confusion_matrix / (row_sum + 1e-6)

        ### DEBUG ---------- 可视化混淆矩阵 ----------
        plt.figure(figsize=(len(class_labels) * 2, len(class_labels) * 2))
        plt.imshow(norm_pixel_confusion_matrix, cmap='jet', vmin=0, vmax=1)
        plt.colorbar()
        # 在每个方格内显示数值，并为每行的 top1/top2 绘制边框
        for i in range(norm_pixel_confusion_matrix.shape[0]):
            # 找出本行的 top2 索引
            row = norm_pixel_confusion_matrix[i]
            sorted_idx = np.argsort(row)[::-1]
            top1 = sorted_idx[0]
            top2 = sorted_idx[1] if sorted_idx.size > 1 else None
            for j in range(norm_pixel_confusion_matrix.shape[1]):
                txt = plt.text(j, i, f"{norm_pixel_confusion_matrix[i, j]: .2f}", ha='center', va='center', color='white', fontsize=12)
                txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])
            # 绘制 top1/top2 的矩形（坐标系以像素为单位，矩形左上角为 (col-0.5, row-0.5)）
            rect1 = patches.Rectangle((top1 - 0.5, i - 0.5), 1, 1, linewidth=2.0, edgecolor='red', facecolor='none')
            plt.gca().add_patch(rect1)
            if top2 is not None:
                rect2 = patches.Rectangle((top2 - 0.5, i - 0.5), 1, 1, linewidth=1.5, edgecolor='red', facecolor='none', linestyle='--')
                plt.gca().add_patch(rect2)
        plt.xticks(range(len(class_names)), class_names, rotation=45)
        plt.yticks(range(len(class_names)), class_names)
        plt.title('pixel confusion matrix')
        plt.savefig('debug/norm_pixel_confusion_matrix.png')
        plt.close()

        # 绘制提示点混淆矩阵热力图
        norm_prompt_confusion_matrix = prompt_confusion_matrix.copy()
        row_sum = norm_prompt_confusion_matrix.sum(axis=1, keepdims=True)
        norm_prompt_confusion_matrix = prompt_confusion_matrix / (row_sum + 1e-6)
        plt.figure(figsize=(len(class_labels) * 2, len(class_labels) * 2))
        plt.imshow(norm_prompt_confusion_matrix, cmap='jet', vmin=0, vmax=1)
        plt.colorbar()
        for i in range(norm_prompt_confusion_matrix.shape[0]):
            row = norm_prompt_confusion_matrix[i]
            sorted_idx = np.argsort(row)[::-1]
            top1 = sorted_idx[0]
            top2 = sorted_idx[1] if sorted_idx.size > 1 else None
            for j in range(norm_prompt_confusion_matrix.shape[1]):
                txt = plt.text(j, i, f"{norm_prompt_confusion_matrix[i, j]: .2f}", ha='center', va='center', color='white', fontsize=12)
                txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])
            rect1 = patches.Rectangle((top1 - 0.5, i - 0.5), 1, 1, linewidth=2.0, edgecolor='red', facecolor='none')
            plt.gca().add_patch(rect1)
            if top2 is not None:
                rect2 = patches.Rectangle((top2 - 0.5, i - 0.5), 1, 1, linewidth=1.5, edgecolor='red', facecolor='none', linestyle='--')
                plt.gca().add_patch(rect2)
        plt.xticks(range(len(class_names)), class_names, rotation=45)
        plt.yticks(range(len(class_names)), class_names)
        plt.savefig('debug/norm_prompt_confusion_matrix.png')
        plt.close()

        ### DEBUG ---------- 重叠情况 ----------
        # 任何被重叠区域
        mask = ((overlap_count > 0) & (overlap_might_right > 0)).cpu().numpy()
        overlap_pixel_num[0] += mask.sum()
        wrong_pred = (query_mask != final_pred_mask) & mask
        overlap_pixel_wrong_pred_num[0] += wrong_pred.sum()
        for cnt in np.unique(overlap_count): # 被指定类别数量重叠区域
            if cnt == 0:
                continue
            mask = ((overlap_count == cnt) & (overlap_might_right > 0)).cpu().numpy()
            overlap_pixel_num[cnt] += mask.sum()
            wrong_pred = (query_mask != final_pred_mask) & mask
            overlap_pixel_wrong_pred_num[cnt] += wrong_pred.sum()

        # 可视化overlap的两个数组的数据
        # 获取重叠次数的范围
        cnts = list(range(num_classes))
        cnts = sorted(cnts)

        overlap_nums = [overlap_pixel_num[cnt] / total_pixel_num for cnt in cnts]
        overlap_wrong_nums = [overlap_pixel_wrong_pred_num[cnt] / total_pixel_num for cnt in cnts]

        x = np.arange(len(cnts))  # 横坐标位置
        width = 0.35  # 柱宽

        plt.figure(figsize=(num_classes*2, 6))
        bar1 = plt.bar(x - width/2, overlap_nums, width, label='overlap_pixel_num')
        bar2 = plt.bar(x + width/2, overlap_wrong_nums, width, label='overlap_pixel_wrong_pred_num')

        # 在bar上绘制数字
        for rect in bar1:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2, height, f'{height*100:.1f}%', ha='center', va='bottom', fontsize=10)
        for rect in bar2:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2, height, f'{height*100:.1f}', ha='center', va='bottom', fontsize=10)

        # 在两个bar中间上方标注百分比（错误像素/重叠像素）
        for i, (rect1, rect2) in enumerate(zip(bar1, bar2)):
            total = overlap_nums[i]
            wrong = overlap_wrong_nums[i]
            if total > 0:
                ratio = wrong / total
                # 中点的 x：两根柱子中心的平均
                mid_x = (rect1.get_x() + rect1.get_width()/2 + rect2.get_x() + rect2.get_width()/2) / 2
                # y 放在两柱较高者之上，加一点偏移
                top_y = max(rect1.get_height(), rect2.get_height())
                plt.text(mid_x, top_y + 0.2, f'{ratio*100:.1f}%', ha='center', va='bottom', fontsize=10)

        # 绘制total_pixel_num的横向虚线
        plt.axhline(y=1, color='r', linestyle='--', label='total_pixel_num')

        plt.xlabel('overlap count')
        plt.ylabel('num')
        plt.xticks(x, cnts)
        plt.legend()
        plt.tight_layout()
        plt.savefig('debug/overlap_bar.png')
        plt.close()



    # Write evaluation results
    average_meter.write_result(0)
    miou, fb_iou, _ = average_meter.compute_iou()

    return miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='GFSAM Pytorch Implementation for One-shot Segmentation')

    # 数据集参数
    parser.add_argument('--datapath', type=str, default='datasets')
    parser.add_argument('--benchmark', type=str, default='coco', choices=['coco', 'pascal'])
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