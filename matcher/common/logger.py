r""" Logging during training/testing """
import datetime
import logging
import os

from tensorboardX import SummaryWriter
import torch


class AverageMeter:
    r""" Stores evaluation results """
    def __init__(self, dataset):
        self.benchmark = dataset.benchmark
        self.class_ids_interest = dataset.class_ids
        self.class_ids_interest = torch.tensor(self.class_ids_interest).cuda()

        if self.benchmark == 'pascal':
            self.nclass = 20
        elif self.benchmark == 'coco':
            self.nclass = 80
        elif self.benchmark == 'fss':
            self.nclass = 1000
        elif self.benchmark == 'paco_part':
            self.nclass = 448
        elif self.benchmark == 'pascal_part':
            self.nclass = 100
        elif self.benchmark == 'lvis':
            self.nclass = 1203
        else:
            self.nclass = dataset.nclass

        self.supp_class_ids = dataset.supp_class_ids

        self.intersection_buf = torch.zeros(1 + len(self.supp_class_ids)).float().cuda()
        self.union_buf = torch.zeros(1 + len(self.supp_class_ids)).float().cuda()
        self.ones = torch.ones_like(self.union_buf)


    def update(self, inter, union):
        """
        Args:
            inter: (num_classes)
            union: (num_classes)
        """
        self.intersection_buf += inter.float()
        self.union_buf += union.float()


    def compute_iou(self):
        class_iou = self.intersection_buf.float() / self.union_buf.float()
        miou = class_iou.mean()
        miou_without_bg = class_iou[1:].mean()
        return class_iou, miou, miou_without_bg

    def write_result(self, split):
        class_iou, miou, miou_without_bg = self.compute_iou()

        msg = '\n*** %s ' % split
        msg += 'mIoU: %5.2f   ' % miou
        msg += 'mIoU(No Bg): %5.2f   ' % miou_without_bg
        for cat, cat_iou in enumerate(class_iou):
            cat_iou = cat_iou * 100
            msg += f' |  {cat}:'+' %5.2f   ' % cat_iou

        msg += '***\n'
        Logger.info(msg)

    def write_process(self, batch_idx, datalen, write_batch_idx=20):
        if batch_idx % write_batch_idx == 0:
            msg = ''
            msg += '[Batch: %04d/%04d] ' % (batch_idx+1, datalen)
            class_iou, miou, miou_without_bg = self.compute_iou()
            msg += 'mIoU: %5.2f  |  ' % miou
            msg += 'mIoU(No Bg): %5.2f' % miou_without_bg
            for cat, cat_iou in enumerate(class_iou):
                cat_iou = cat_iou * 100
                msg += f' |  {cat}:' + ' %5.2f   ' % cat_iou

            Logger.info(msg)


class Logger:
    r""" Writes evaluation results of training/testing """
    @classmethod
    def initialize(cls, args, root='logs'):
        logtime = datetime.datetime.now().__format__('%m%d_%H%M%S')
        logpath = '_TEST_' + logtime

        cls.logpath = os.path.join(root, logpath + '.log')
        cls.benchmark = args.benchmark
        os.makedirs(cls.logpath)

        logging.basicConfig(filemode='w',
                            filename=os.path.join(cls.logpath, 'log.txt'),
                            level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S')

        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # Tensorboard writer
        cls.tbd_writer = SummaryWriter(os.path.join(cls.logpath, 'tbd/runs'))

        # Log arguments
        logging.info('\n:=========== Few-shot Seg. with Matcher ===========')
        for arg_key in args.__dict__:
            logging.info('| %20s: %-24s' % (arg_key, str(args.__dict__[arg_key])))
        logging.info(':================================================\n')

    @classmethod
    def info(cls, msg):
        r""" Writes log message to log.txt """
        logging.info(msg)

    @classmethod
    def save_model_miou(cls, model, epoch, val_miou):
        torch.save(model.state_dict(), os.path.join(cls.logpath, 'best_model.pt'))
        cls.info('Model saved @%d w/ val. mIoU: %5.2f.\n' % (epoch, val_miou))

    @classmethod
    def log_params(cls, model):
        backbone_param = 0
        learner_param = 0
        for k in model.state_dict().keys():
            n_param = model.state_dict()[k].view(-1).size(0)
            if k.split('.')[0] in 'backbone':
                if k.split('.')[1] in ['classifier', 'fc']:
                    continue
                backbone_param += n_param
            else:
                learner_param += n_param
        Logger.info('Backbone # param.: %d' % backbone_param)
        Logger.info('Learnable # param.: %d' % learner_param)
        Logger.info('Total # param.: %d' % (backbone_param + learner_param))

