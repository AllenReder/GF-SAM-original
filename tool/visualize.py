import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import ListedColormap
import cv2
import matplotlib.colors as mcolors
import random
import io
from PIL import Image

from .tool import *


def show_image(image, save=None, rows=1, scale=1):
    """
    显示一张或多张图片。

    Args: 
        - image: 图片 (numpy 数组、torch Tensor 或 PIL.Image) 或图片列表
        - save: 保存图片的路径，默认为 None 即直接显示, 如果有字符串值则改为保存而不显示
        - rows: 每行显示的图片数量，默认为1
    """
    def to_numpy(img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
            if img.ndim == 3 and img.shape[0] in [1, 3, 4]:  # CWH to HWC
                img = img.permute(1, 2, 0)
            if img.ndim == 3 and img.min() < 0:
                img = denormalize(img)
            else:
                img = img.long()
            img = img.numpy()

        if isinstance(img, np.ndarray):
            if img.ndim == 3 and img.shape[0] in [1, 3, 4]:  # CWH to HWC
                img = np.transpose(img, (1, 2, 0))
            if img.ndim == 3 and img.min() < 0:
                img = denormalize(img)

        return img

    def custom_colormap():
        # 保存当前的随机状态
        state = random.getstate()
        try:
            # 设置固定的种子
            random.seed(42)
            colors = {0: (0, 0, 0), 255: (1, 1, 1)}  # 0 -> 黑色, 255 -> 白色
            for i in range(1, 255):
                colors[i] = (random.random(), random.random(),
                             random.random())  # 随机颜色
            cmap_list = [colors[i] for i in range(256)]
            cmap = mcolors.ListedColormap(cmap_list)
        finally:
            # 恢复随机状态
            random.setstate(state)
        return cmap

    if not isinstance(image, (list, tuple)):
        image = [image]

    images = [to_numpy(img) for img in image]
    num_images = len(images)
    cols = (num_images + rows - 1) // rows
    # 记录最大的图像尺寸
    max_height, max_width = 0, 0
    for img in images:
        img_height, img_width = img.shape[:2]
        max_height = max(max_height, img_height)
        max_width = max(max_width, img_width)

    # 设置画布大小
    fig_width = cols * (max_width / 100) * scale
    fig_height = rows * (max_height / 100) * scale
    plt.figure(figsize=(fig_width, fig_height))

    cmap = custom_colormap()

    for idx, img in enumerate(images):
        plt.subplot(rows, cols, idx + 1)
        plt.axis('off')
        if img.ndim == 2 and np.issubdtype(img.dtype, np.integer):
            plt.imshow(img, cmap=cmap, vmin=0, vmax=255,
                       interpolation='nearest')  # 禁用插值
        elif img.ndim == 2 and np.issubdtype(img.dtype, np.floating):
            plt.imshow(img, cmap='jet', interpolation='bilinear')
        else:
            plt.imshow(img)
    plt.tight_layout()

    if save:
        save_dir = os.path.dirname(save)
        if save_dir != '' and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save, bbox_inches='tight', pad_inches=0.1)
        plt.close()  # 关闭图像以释放内存
    else:
        plt.show()


def get_color_sample():
    """
    生成一张色块图, 灰度图, 用来显示灰度值对应颜色
    """
    size = 5  # 每行色块数 (列数)
    rect_size = 50  # 色块边长
    map = np.zeros((rect_size * size, rect_size * size), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            map[i*rect_size:(i+1)*rect_size, j*rect_size:(j+1)
                * rect_size] = i * size + j + 1
    return map


def draw_points(image, points, size=3, color=(255, 0, 0)):
    """
    在图像上绘制点

    Args:
        - image: 图像
        - points: 点 (N, 2)

    Returns:
        - result_image: 结果图像
    """
    result_image = image.copy()
    for point in points:
        result_image = cv2.circle(result_image, point, size, color, -1)
    return result_image


def draw_masks(image, masks, opacity=0.5, borders=True, draw_index=False) -> np.array:
    """
    在 image 上绘制半透明随机颜色的 masks

    Args:
        - image: 输入的 image
        - masks: 输入的 masks, 每个 mask 是一个二维数组 (H, W)
        - opacity: 透明度
        - borders: 是否绘制边界
        - draw_index: 是否绘制 mask 的 id
    Returns:
        - img: 绘制后的 image
    """
    assert len(masks) > 0, "masks 不能为空"
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()  # (3, H, W)
        if image.ndim == 3 and image.shape[0] == 3:
            image = image.permute(1, 2, 0)  # (H, W, 3)
            image = denormalize(image)  # (H, W, 3)
        image = image.numpy()  # (H, W, 3)

    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()  # (num_masks, H, W)
        masks = [mask.astype(bool) for mask in masks]

    image = image.copy()
    image = (image - image.min()) / (image.max() - image.min()) * 255
    image = image.astype(np.uint8)

    # 将输入图像转换为 RGBA 格式
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    else:
        image = image.copy()

    for idx, mask in enumerate(masks):
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        # 将 mask 缩小到 image 的大小
        mask = cv2.resize(mask.astype(
            np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(bool)
        # 生成随机颜色
        color_mask = np.concatenate(
            [np.random.randint(0, 256, 3), [int(opacity * 255)]])

        # 将 mask 应用到图像上
        image[mask] = image[mask] * (1 - opacity) + color_mask * opacity

        if borders:
            contours, _ = cv2.findContours(mask.astype(
                np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(image, contours, -1, (0, 0, 0, 1), thickness=1)

        if draw_index:
            # 计算 mask 的中心点
            ys, xs = np.where(mask)
            if len(xs) > 0 and len(ys) > 0:
                center_x = int(np.mean(xs))
                center_y = int(np.mean(ys))
                # 在中心点绘制 id
                cv2.putText(image, str(idx), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image


def visualize_sim(features, labels=None):
    """
    绘制相似度热力图

    Args:
        - features: (torch.Tensor) 特征 (len(labels), feat_dim)
        - labels: 每行特征所属标签
    
    Returns:
        - sim_img: (torch.Tensor) 相似度热力图
    """
    if labels is None:
        labels = np.arange(features.shape[0])
    elif isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    cls_num = len(labels)
    features = F.normalize(features, p=2, dim=1)
    matrix = torch.matmul(features, features.T)  # (cls_num, cls_num)
    plt.figure(figsize=(10, 10))
    plt.imshow(matrix.cpu().numpy(), cmap='jet',
               interpolation='nearest', vmin=-1, vmax=1)
    # 坐标轴标记 labels
    plt.xticks(range(cls_num), labels)
    plt.yticks(range(cls_num), labels)
    plt.colorbar()
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    sim_img = torch.tensor(np.array(Image.open(buffer))).permute(2, 0, 1)
    plt.close()

    return sim_img
