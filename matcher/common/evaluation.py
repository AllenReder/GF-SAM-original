r""" Evaluate mask prediction """
import torch


class Evaluator:
    r""" Computes intersection and union between prediction and ground-truth """
    @classmethod
    def initialize(cls, supp_class_ids):
        cls.ignore_index = 255
        cls.supp_class_ids = supp_class_ids # list[cls_id, ...]

    @classmethod
    def classify_prediction(cls, pred_mask, batch):
        """
        Args:
            pred_mask: (1, H, W)
            batch: 
                query_mask: (1, H, W)
                query_ignore_idx: (1, H, W)
        """
        gt_mask = batch.get('query_mask')

        # Apply ignore_index in PASCAL-5i masks (following evaluation scheme in PFE-Net (TPAMI 2020))
        query_ignore_idx = batch.get('query_ignore_idx')
        if query_ignore_idx is not None:
            assert torch.logical_and(query_ignore_idx, gt_mask).sum() == 0
            query_ignore_idx *= cls.ignore_index
            gt_mask = gt_mask + query_ignore_idx
            pred_mask[gt_mask == cls.ignore_index] = cls.ignore_index

        # compute intersection and union for each class
        area_inter, area_union = [], []
        classes = [0] + cls.supp_class_ids  # 假设背景是0，加上支持的类别


        _inter = []
        _union = []
        for c in classes:
            pred_c = (pred_mask == c)
            gt_c = (gt_mask == c)
            inter = (pred_c & gt_c).sum().item()
            pred_area = pred_c.sum().item()
            gt_area = gt_c.sum().item()
            union = pred_area + gt_area - inter if pred_area + gt_area > 0 else 0
            area_inter.append(inter)
            area_union.append(union)
        
        area_inter = torch.tensor(area_inter).float().cuda()
        area_union = torch.tensor(area_union).float().cuda()
        
        return area_inter, area_union
