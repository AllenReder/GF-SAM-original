r""" COCO-20i few-shot semantic segmentation dataset """
import os
import pickle

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np


class DatasetCOCO(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize, supp_class_ids, random_sample=1000):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 80
        self.benchmark = 'coco'
        self.shot = shot
        self.split_coco = split if split == 'val2014' else 'train2014'
        self.base_path = os.path.join(datapath, 'COCO2014')
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize
        self.supp_class_ids = supp_class_ids
        self.class_ids = self.build_class_ids()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        for k, v in self.img_metadata_classwise.items():
            if len(v) > 0:
                print(f"DEBUG 类别 {k} 的图像数量: {len(v)}")
        self.img_metadata = self.build_img_metadata()

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        # ignores idx during training & testing and perform uniform sampling over object classes to form an episode
        # (due to the large size of the COCO dataset)
        query_name, support_names = self.sample_episode()  # 采样一个 episode
        query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize = self.load_frame(query_name, support_names)  # 加载图像和掩码

        # query image
        query_img = self.transform(query_img)

        # query mask
        if not self.use_original_imgsize:
            query_cmask = F.interpolate(query_cmask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()  # 调整查询掩码尺寸 - shape: [H, W]

        # 去掉 query_cmask 中不存在于 supp_class_ids 的像素
        unique_query_mask = np.unique(query_cmask.numpy())
        for cls_id in unique_query_mask:
            if cls_id not in self.supp_class_ids:
                query_cmask[query_cmask == cls_id] = 0

        # support images
        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        # support masks
        support_masks = []  # 支持集掩码列表
        for scmask in support_cmasks:
            scmask = F.interpolate(scmask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()  # 调整支持集掩码尺寸 - shape: [H, W]
            support_masks.append(scmask)
        support_masks = torch.stack(support_masks)  # shape: [shot, H, W]

        batch = {'query_img': query_img,
                 'query_mask': query_cmask,
                 'query_name': query_name,

                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 }

        return batch

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold + self.nfolds * v + 1 for v in range(nclass_trn)]
        class_ids_trn = [x for x in range(1, self.nclass + 1) if x not in class_ids_val]
        class_ids = class_ids_trn if self.split == 'trn' else class_ids_val

        return class_ids

    def build_img_metadata_classwise(self):
        with open(f'{self.base_path}/splits/{self.split}/fold{self.fold}.pkl', 'rb') as f:
            img_metadata_classwise = pickle.load(f)
        img_metadata_classwise = {k+1: v for k, v in img_metadata_classwise.items()} # 索引类别从 1 开始
        return img_metadata_classwise

    def build_img_metadata(self):
        img_metadata = []
        for k in self.img_metadata_classwise.keys():
            img_metadata += self.img_metadata_classwise[k]
        return sorted(list(set(img_metadata)))
    
    def read_img(self, name):
        img = Image.open(os.path.join(self.base_path, name)).convert('RGB')
        return img

    def read_mask(self, name):
        mask_path = os.path.join(self.base_path, 'annotations', name)
        mask = torch.tensor(np.array(Image.open(mask_path[:mask_path.index('.jpg')] + '.png')))
        return mask

    def load_frame(self, query_name, support_names):
        query_img = self.read_img(query_name)  # 读取查询图像
        query_mask = self.read_mask(query_name)  # 读取查询掩码 - shape: [H, W]
        support_imgs = [self.read_img(name) for name in support_names]  # 读取支持集图像列表
        support_masks = [self.read_mask(name) for name in support_names]  # 读取支持集掩码列表 - shape: [shot, H, W]

        org_qry_imsize = query_img.size  # 原始查询图像尺寸 (W, H)

        return query_img, query_mask, support_imgs, support_masks, org_qry_imsize
    
    def sample_episode(self):
        class_sample = np.random.choice(self.class_ids, 1, replace=False)[0] # 随机类别
        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0].replace('COCO_val2014_', '')

        support_names = []
        for supp_cls in self.supp_class_ids:
            support_cls_names = []
            while True:  # 持续采样支持集，直到满足条件
                support_name = np.random.choice(self.img_metadata_classwise[supp_cls], 1, replace=False)[0].replace('COCO_val2014_', '')
                if query_name == support_name: 
                    continue
                if support_name in support_cls_names:
                    continue
                label_path = os.path.join(self.base_path, 'annotations', support_name.replace('.jpg', '.png'))
                label = np.array(Image.open(label_path))
                # label 中只能包含 supp_cls 和 0/255
                unique_labels = set(np.unique(label)) - {0, 255}
                if len(unique_labels) > 1 or unique_labels.pop() != supp_cls:
                    continue
                support_cls_names.append(support_name)  # 确保支持集不包含查询图像
                if len(support_cls_names) == self.shot: break  # 达到所需支持集数量时停止
            support_names.extend(support_cls_names)
        
        return query_name, support_names


# def build_class_map(datapath='datasets', class_num=80):
#     from tqdm import tqdm
#     ann_dir = os.path.join(datapath, 'COCO2014/annotations/val2014/')
#     name_list = os.listdir(ann_dir)
#     name_list = [name.split('.')[0] for name in name_list]
#     cls_name_list = [[] for _ in range(class_num)]
#     for name in tqdm(name_list, desc='build class map'):
#         label_path = os.path.join(ann_dir, f"{name}.png")
#         label = np.array(Image.open(label_path))
#         unique_labels = set(np.unique(label)) - {0, 255}
#         for cls in unique_labels:
#             cls_name_list[cls - 1].append(name)
#     # 保存到 class_map_dir 中的 {cls}.txt 中
#     class_map_dir = os.path.join(datapath, 'COCO2014/class_map/')
#     os.makedirs(class_map_dir, exist_ok=True)
#     for cls in tqdm(range(1, class_num + 1), desc='save txt'):
#         with open(os.path.join(class_map_dir, f"{cls}.txt"), 'w') as f:
#             for name in cls_name_list[cls - 1]:
#                 f.write(f"{name}\n")

if __name__ == '__main__':
    import cv2
    from torchvision import transforms
    import matplotlib.pyplot as plt
    import numpy as np
    from tool import *
    from matcher.data.task import SUPP_CLASS_IDS

    img_size = 1024
    transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor()
    ])
    dataset = DatasetCOCO(datapath='datasets', 
                          fold=0, 
                          transform=transform, 
                          split='val', 
                          shot=1, 
                          use_original_imgsize=True, 
                          supp_class_ids=SUPP_CLASS_IDS['coco'][0]
                          )
    item = dataset[0]
    for k, v in item.items():
        print(f"{k}: {v.shape if isinstance(v, torch.Tensor) else v}")
    exit()