r""" PASCAL-5i 少样本语义分割数据集 """
import os

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np

from matcher.common.logger import Logger


class DatasetPASCAL(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize, supp_class_ids, random_sample=1000):
        """
        初始化PASCAL数据集
        Args:
            datapath: 数据集根目录路径
            fold: 当前使用的折叠编号(0-3)
            transform: 图像变换函数
            split: 数据集划分('train'/'val'/'test')
            shot: 支持集样本数量
            use_original_imgsize: 是否使用原始图像尺寸
            supp_class_ids: 支持集类别ID列表
            random_sample: 查询集随机采样数量
        """
        self.split = 'val' if split in ['val', 'test'] else 'trn'  # 'trn'表示训练集
        self.fold = fold  # 当前折叠编号
        self.nfolds = 4  # 总共4个折叠
        self.nclass = 20  # PASCAL VOC共20个类别
        self.benchmark = 'pascal'  # 数据集名称
        self.base_path = os.path.join(datapath, 'VOC2012')  # VOC2012数据集路径
        self.shot = shot  # 支持集样本数量
        self.use_original_imgsize = use_original_imgsize  # 是否使用原始图像尺寸

        self.img_path = os.path.join(datapath, 'VOC2012/JPEGImages/')  # 图像路径
        self.ann_path = os.path.join(datapath, 'VOC2012/SegmentationClassAug/')  # 标注路径
        self.class_map_path = os.path.join(datapath, 'VOC2012/class_map/')  # 类别映射路径
        self.transform = transform  # 图像变换函数

        self.class_ids = self.build_class_ids()  # 构建类别ID列表 - shape: [类别数量]
        # self.cats = [
        #     ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        #      "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"][i] for i in
        #     self.class_ids]  # 类别名称列表 - shape: [类别数量]
        self.img_metadata = self.build_img_metadata()  # 构建图像元数据 - shape: [图像数量, 2]，每项为[图像名称, 类别ID]
        self.img_metadata_classwise = self.build_img_metadata_classwise()  # 按类别构建图像元数据 - dict，键为类别ID，值为图像名称列表
        for i in range(len(self.img_metadata_classwise)):
            print(f"DEBUG 类别 {i} 的图像数量: {len(self.img_metadata_classwise[i])}")
        
        self.supp_class_ids = supp_class_ids

        # 加载查询集可用图像
        self.query_name_list = self.build_query_name_list()
        if random_sample is not None and len(self.query_name_list) > random_sample:
            self.query_name_list = np.random.choice(self.query_name_list, random_sample, replace=False)
        
        Logger.info(f"查询集数量: {len(self.query_name_list)}")


    def __len__(self):
        """返回数据集长度"""
        return len(self.query_name_list)

    def __getitem__(self, idx):
        """
        获取一个训练/测试样本
        Args:
            idx: 样本索引
        Returns:
            batch: 包含查询图像和支持集的字典
        """
        query_name, support_names= self.sample_episode(idx)  # 采样一个episode
        query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize = self.load_frame(query_name, support_names)  # 加载图像和掩码

        query_img = self.transform(query_img)  # 变换查询图像 - shape: [3, H, W]
        if not self.use_original_imgsize:
            query_cmask = F.interpolate(query_cmask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()  # 调整查询掩码尺寸 - shape: [H, W]
        query_ignore_idx = self.extract_ignore_idx(query_cmask.float())  # 提取查询掩码和忽略索引 - shape: [H, W]
        # 去掉 query_cmask 中不存在于 supp_class_ids 的像素
        unique_query_mask = np.unique(query_cmask.numpy())
        for cls_id in unique_query_mask:
            if cls_id not in self.supp_class_ids:
                query_cmask[query_cmask == cls_id] = 0

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])  # 变换支持集图像 - shape: [shot, 3, H, W]

        support_masks = []  # 支持集掩码列表
        for scmask in support_cmasks:
            scmask = F.interpolate(scmask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()  # 调整支持集掩码尺寸 - shape: [H, W]
            support_masks.append(scmask)
        support_masks = torch.stack(support_masks)  # shape: [shot, H, W]

        batch = {'query_img': query_img,  # 查询图像 - shape: [3, H, W]
                 'query_mask': query_cmask,  # 查询掩码 - shape: [H, W]
                 'query_name': query_name,  # 查询图像名称
                 'query_ignore_idx': query_ignore_idx,  # 查询忽略索引 - shape: [H, W]

                 'org_query_imsize': org_qry_imsize,  # 原始查询图像尺寸 - (W, H)

                 'support_imgs': support_imgs,  # 支持集图像 - shape: [shot, 3, H, W]
                 'support_masks': support_masks,  # 支持集掩码 - shape: [shot, H, W]
                 'support_names': support_names,  # 支持集图像名称列表
                }

        return batch

    def extract_ignore_idx(self, mask):
        """
        提取掩码和忽略索引
        Args:
            mask: 原始掩码
        Returns:
            boundary: 边界掩码
        """
        boundary = (mask / 255).floor()  # 创建边界掩码 - shape: 与mask相同

        return boundary

    def load_frame(self, query_name, support_names):
        """
        加载查询图像和支持集图像及其掩码
        Args:
            query_name: 查询图像名称
            support_names: 支持集图像名称列表
        Returns:
            query_img: 查询图像
            query_mask: 查询掩码
            support_imgs: 支持集图像列表
            support_masks: 支持集掩码列表
            org_qry_imsize: 原始查询图像尺寸
        """
        query_img = self.read_img(query_name)  # 读取查询图像
        query_mask = self.read_mask(query_name)  # 读取查询掩码 - shape: [H, W]
        support_imgs = [self.read_img(name) for name in support_names]  # 读取支持集图像列表
        support_masks = [self.read_mask(name) for name in support_names]  # 读取支持集掩码列表 - shape: [shot, H, W]

        org_qry_imsize = query_img.size  # 原始查询图像尺寸 (W, H)

        return query_img, query_mask, support_imgs, support_masks, org_qry_imsize

    def read_mask(self, img_name):
        """
        读取分割掩码
        Args:
            img_name: 图像名称
        Returns:
            mask: 分割掩码张量 - shape: [H, W]
        """
        mask = torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name) + '.png')))
        return mask

    def read_img(self, img_name):
        """
        读取RGB图像
        Args:
            img_name: 图像名称
        Returns:
            img: PIL图像对象
        """
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg')

    def sample_episode(self, idx):
        """
        采样一个episode（查询图像和支持集）
        Args:
            idx: 样本索引
        Returns:
            query_name: 查询图像名称
            support_names: 支持集图像名称列表
            class_sample: 类别ID
        """
        query_name = self.query_name_list[idx]  # 获取查询图像名称和类别

        support_names = []
        for supp_cls in self.supp_class_ids:
            support_cls_names = []
            while True:  # 持续采样支持集，直到满足条件
                support_name = np.random.choice(self.img_metadata_classwise[supp_cls], 1, replace=False)[0]
                if query_name == support_name: 
                    continue
                if support_name in support_cls_names:
                    continue
                label_path = os.path.join(self.ann_path, support_name + '.png')
                label = np.array(Image.open(label_path))
                # label 中只能包含 supp_cls 和 0/255
                unique_labels = set(np.unique(label)) - {0, 255}
                if len(unique_labels) > 1 or unique_labels.pop() != supp_cls:
                    continue
                support_cls_names.append(support_name)  # 确保支持集不包含查询图像
                if len(support_cls_names) == self.shot: break  # 达到所需支持集数量时停止
            support_names.extend(support_cls_names)

        return query_name, support_names

    def build_class_ids(self):
        """
        构建类别ID列表
        Returns:
            class_ids_trn/class_ids_val: 训练/验证集的类别ID列表
        """
        nclass_trn = self.nclass // self.nfolds  # 每个折叠的类别数量
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]  # 验证集类别ID - shape: [nclass_trn]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]  # 训练集类别ID - shape: [nclass - nclass_trn]

        if self.split == 'trn':
            return class_ids_trn
        else:
            return class_ids_val

    def build_img_metadata(self):
        """
        构建图像元数据
        Returns:
            img_metadata: 图像元数据列表，每项为[图像名称, 类别ID]
        """
        def read_metadata(split, fold_id):
            # fold_n_metadata = os.path.join(self.base_path, 'splits/%s/fold%d.txt' % (split, fold_id))
            fold_n_metadata = os.path.join('matcher/data/splits/pascal/%s/fold%d.txt' % (split, fold_id))
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1])] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.split == 'trn':  # 对于训练集，读取"其他"折叠的图像元数据
            for fold_id in range(self.nfolds):
                if fold_id == self.fold:  # 跳过验证折叠
                    continue
                img_metadata += read_metadata(self.split, fold_id)
        elif self.split == 'val':  # 对于验证集，读取"当前"折叠的图像元数据
            img_metadata = read_metadata(self.split, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.split)

        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))

        return img_metadata

    def build_img_metadata_classwise(self):
        """
        按类别构建图像元数据
        Returns:
            img_metadata_classwise: 按类别组织的图像元数据字典，键为类别ID，值为图像名称列表
        """
        img_metadata_classwise = {}
        for class_id in range(self.nclass + 1):
            img_metadata_classwise[class_id] = []  # 初始化每个类别的图像列表

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]  # 将图像添加到对应类别
        return img_metadata_classwise
    
    def build_query_name_list(self):
        """
        构建查询集可用图像列表, 根据 supp_class_ids 构建
        Returns:
            query_name_list: 查询集可用图像列表
        """
        query_name_list = []
        for cls in self.supp_class_ids:
            class_map_path = os.path.join(self.class_map_path, f"{cls}.txt")
            cls_name_list = []
            with open(class_map_path, 'r') as f:
                for line in f:
                    cls_name_list.append(line.strip())
            query_name_list.extend(cls_name_list)
        query_name_list = list(set(query_name_list))
        return query_name_list


def build_class_map(datapath='datasets', class_num=20):
    from tqdm import tqdm
    ann_dir = os.path.join(datapath, 'VOC2012/SegmentationClassAug/')
    name_list = os.listdir(ann_dir)
    name_list = [name.split('.')[0] for name in name_list]
    cls_name_list = [[] for _ in range(class_num)]
    for name in tqdm(name_list, desc='build class map'):
        label_path = os.path.join(ann_dir, f"{name}.png")
        label = np.array(Image.open(label_path))
        unique_labels = set(np.unique(label)) - {0, 255}
        for cls in unique_labels:
            cls_name_list[cls - 1].append(name)
    # 保存到 class_map_dir 中的 {cls}.txt 中
    class_map_dir = os.path.join(datapath, 'VOC2012/class_map/')
    os.makedirs(class_map_dir, exist_ok=True)
    for cls in tqdm(range(1, class_num + 1), desc='save txt'):
        with open(os.path.join(class_map_dir, f"{cls}.txt"), 'w') as f:
            for name in cls_name_list[cls - 1]:
                f.write(f"{name}\n")

if __name__ == '__main__':
    import cv2
    from torchvision import transforms
    import matplotlib.pyplot as plt
    import numpy as np
    from tool import *

    # build_class_map()

    # class_map_path = '/data3/flyu/ywy/GF-SAM/datasets/VOC2012/class_map/15.txt'
    # name_list = []
    # with open(class_map_path, 'r') as f:
    #     for line in f:
    #         name_list.append(line.strip())
    # print(f"name_list: {name_list}")
    # exit()

    img_size = 1024
    transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor()
    ])
    dataset = DatasetPASCAL(
        datapath='datasets',
        fold=0,
        transform=transform,
        split='test',
        shot=1,
        use_original_imgsize=False,
        supp_class_ids=[1,2,3,4,5]
    )

    exit()

    # print(f"len(dataset): {len(dataset)}")
    # for i in range(20):
    #     query_name = dataset.img_metadata[i][0]
    #     print(f"query_name: {query_name}")
    #     image_path = os.path.join(dataset.img_path, query_name + '.jpg')
    #     label_path = os.path.join(dataset.ann_path, query_name + '.png')
    #     image = cv2.imread(image_path)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    #     print(np.unique(label))
    #     show_image([image, label], save=f"debug/{i}.png")

    
    ### DEBUG 支持集选取
    query_name, support_names, class_sample = dataset.sample_episode(0)
    print(f"query_name: {query_name}")
    print(f"support_names: {support_names}")
    print(f"class_sample: {class_sample}")

    for supp_name in support_names:
        print(f"supp_name: {supp_name}")
        label_path = os.path.join(dataset.ann_path, supp_name + '.png')
        label = np.array(Image.open(label_path))
        print(np.unique(label))

    ### DEBUG getitem
    batch = dataset[0]
    print(batch.keys())
    print(batch['query_name'])
    print(batch['support_names'])
    print(batch['class_id'])
    print(batch['query_img'].shape)
    print(batch['query_mask'].shape)
    print(batch['support_imgs'].shape)

    # for metadata in dataset.img_metadata:
    #     img_name, class_id = metadata
    #     print(f"img_name: {img_name}, class_id: {class_id}")
    #     image_path = os.path.join(dataset.img_path, img_name + '.jpg')
    #     mask_path = os.path.join(dataset.ann_path, img_name + '.png')
    #     image = cv2.imread(image_path)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    #     print(np.unique(mask))
    #     # 将颜色数字对应关系和图像标签绘制在同一张图中
    #     from matplotlib import gridspec

    #     fig = plt.figure(figsize=(12, 7))
    #     gs = gridspec.GridSpec(2, 2, height_ratios=[5, 1])

    #     # 显示原图
    #     ax_img = fig.add_subplot(gs[0, 0])
    #     ax_img.imshow(image)
    #     ax_img.set_title("Image")
    #     ax_img.axis('off')

    #     # 显示mask
    #     cmap = plt.get_cmap('tab20', 20)
    #     ax_mask = fig.add_subplot(gs[0, 1])
    #     im = ax_mask.imshow(mask, cmap=cmap, vmin=0, vmax=19)
    #     ax_mask.set_title("Mask (tab20)")
    #     ax_mask.axis('off')

    #     # 在下方合并显示tab20颜色与数字对应关系
    #     ax_legend = fig.add_subplot(gs[1, :])
    #     for i in range(20):
    #         ax_legend.add_patch(
    #             plt.Rectangle((i, 0), 1, 1, color=cmap(i))
    #         )
    #         ax_legend.text(
    #             i + 0.5, 0.5, str(i),
    #             va='center', ha='center', fontsize=12,
    #             color='white' if i not in [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] else 'black'
    #         )
    #     ax_legend.set_xlim(0, 20)
    #     ax_legend.set_ylim(0, 1)
    #     ax_legend.axis('off')
    #     ax_legend.set_title("tab20")

    #     plt.tight_layout()
    #     plt.show()
    #     plt.savefig(f"test.png")
    #     input()