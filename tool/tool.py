import torch
import numpy as np
from PIL import Image
import os
import cv2
import random
import logging
from PIL import Image

import torch.nn.functional as F
import torch.distributed as dist


def debug(msg:str, only_main=True):
    """
    调试输出
    
    Args:
        - msg: 调试信息
        - only_main: (分布式的情况下) 是否只在主进程输出 
    """
    if dist.is_initialized():
        local_rank = dist.get_rank()
        if local_rank == 0 or not only_main:
            print(f"[DEBUG] [GPU {local_rank}] {msg}")
    else:
        print(f"[DEBUG] {msg}")


def setup_seed(seed, deterministic=False):
    """
    设置随机种子

    Args:
        seed (int): 随机种子值
        deterministic (bool): 是否启用确定性模式（可能影响性能）
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(log_file, level=logging.INFO, clear_existing_handlers=True):
    """
    创建日志记录器，支持清除已有处理器避免重复输出

    Args:
        log_file (str): 日志文件路径
        level (int): 日志级别（默认INFO）
        clear_existing_handlers (bool): 是否清除已有的处理器

    Returns:
        logger (logging.Logger): 日志记录器
    """
    # 创建日志目录
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # 配置日志格式
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)

    # 创建日志记录器
    logger = logging.getLogger()
    
    # 清除已有的处理器以避免重复输出
    if clear_existing_handlers:
        while logger.handlers:
            logger.handlers.pop()
    
    logger.setLevel(level)

    # 文件处理器
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def get_box(mask):
    """
    获取 mask 的包围框
    """
    x, y = np.where(mask)
    return [x.min(), y.min(), x.max(), y.max()]


def box_include(box1, box2):
    """
    判断 box1 是否包含 box2
    """
    return box1[0] < box2[0] and box1[1] < box2[1] and box1[2] > box2[2] and box1[3] > box2[3]


def box_area(box):
    """
    计算 box 的面积
    """
    return (box[2] - box[0]) * (box[3] - box[1])


def denormalize(image):
    """
    将归一化后的 image (归一化参数与 transforms.Normalize 中保持一致)
    反归一化回原始图像，并转换为 numpy 格式 (uint8，范围 [0, 255])

    Args:
        - image: shape: [H, W, 3] 或 [B, H, W, 3]

    Returns:
        - 如果输入为 3 维，则返回 shape [H, W, 3] 的 numpy 数组;
        - 如果输入为 4 维，则返回 shape [B, H, W, 3] 的 numpy 数组。
    """

    # 先统一转为 numpy 数组
    is_tensor = False
    if isinstance(image, torch.Tensor):
        is_tensor = True
        image = image.cpu().numpy()  # 转为 numpy 数组

    if image.ndim == 3:
        image = image.copy()
        image = (image - image.min()) / (image.max() - image.min()) * 255
        image = image.astype(np.uint8)
        if is_tensor:
            image = torch.from_numpy(image)
        return image
    elif image.ndim == 4:
        # 对每个 batch 分别处理
        images = []
        for i in range(image.shape[0]):
            img = image[i]
            img = denormalize(img)
            images.append(img)
        if is_tensor:
            images = [torch.from_numpy(img) for img in images]
            return torch.stack(images, axis=0)
        return np.stack(images, axis=0)
    else:
        raise ValueError("image_tensor 的维度应为 3 或 4")


def iouGPU(preds, labels, K, ignore_index=255):
    """
    计算预测结果与真实标签的交集和并集（支持GPU加速）

    Args:
        preds (torch.Tensor): 预测结果 [B, H, W]
        labels (torch.Tensor): 真实标签 [B, H, W]
        K (int): 类别数量（包含背景）
        ignore_index (int): 忽略的标签值（默认255）

    Returns:
        intersection (torch.Tensor): 交集 [K]
        union (torch.Tensor): 并集 [K]
        target (torch.Tensor): 真实标签中每个类别的像素数 [K]
    """
    assert preds.shape == labels.shape, "预测结果和真实标签的形状必须一致"

    # 忽略指定标签
    mask = (labels != ignore_index)
    preds = preds[mask]
    labels = labels[mask]

    # 计算交集和并集
    intersection = preds[preds == labels]  # 预测正确的像素
    area_intersection = torch.histc(
        intersection.float(), bins=K, min=0, max=K-1)  # 统计每个类别的交集

    area_pred = torch.histc(preds.float(), bins=K,
                            min=0, max=K-1)  # 预测结果中每个类别的像素数
    area_label = torch.histc(labels.float(), bins=K,
                             min=0, max=K-1)  # 真实标签中每个类别的像素数
    area_union = area_pred + area_label - area_intersection  # 并集 = 预测 + 真实 - 交集

    return area_intersection, area_union, area_label


def get_mean_features(features, labels, cls=[]):
    """
    计算指定类别的平均特征

    Args:
        - features: (torch.Tensor) (N, feat_dim) 特征
        - labels: (torch.Tensor) (N,) 标签
        - cls: (list) 类别列表

    Returns:
        - cls_feature_mean: (torch.Tensor) (len(cls), feat_dim) 平均特征
    """
    assert len(cls) > 0, "类别列表不能为空"
    feat_dim = features.shape[1]

    cls_feature_mean = torch.zeros(
        len(cls), feat_dim).to(features.device)
    for i, cls in enumerate(cls):
        cls_feature = features[labels == cls]
        if cls_feature.size(0) > 0:
            cls_feature_mean[i] = torch.mean(cls_feature, dim=0)

    return cls_feature_mean


def get_example_image(name='1'):
    image = Image.open(f'data/example/{name}.jpg')
    return np.array(image.convert('RGB'))


VOC_DATASET_DIR = 'data/pascal'


def get_image_name_list():
    return sorted([f.split('.')[0] for f in os.listdir(f'{VOC_DATASET_DIR}/SegmentationClassAug') if f.endswith('.png')])


def get_image_pair(name):
    image = Image.open(f'{VOC_DATASET_DIR}/JPEGImages/{name}.jpg')
    gt = Image.open(
        f'{VOC_DATASET_DIR}/SegmentationClassAug/{name}.png').convert('L')
    return np.array(image.convert('RGB')), np.array(gt)


if __name__ == "__main__":
    pass
