import os
import copy
import numpy as np
from tqdm import tqdm

import torch
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather
import torch.nn.functional as F
import torch.nn as nn

import pandas as pd

import encoding.utils as utils
from encoding.nn import SyncBatchNorm
from encoding.parallel import DataParallelModel, DataParallelCriterion

import cv2
import gc
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue, Resize,
    RandomBrightness, RandomContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RandomSizedCrop
)

import sys
sys.path.append('..')
from mask_functions import mask2rle
from options_parse import OptionsParser
from dataset import SegmentationDataset, apply_revert_shscro, apply_hflip
from models.encnet import EncNet
from models.loss import SegmentationLosses, dice_loss

torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable

class Tester():
    def __init__(self, args):
        self.args = args
        self.args.start_epoch = 0
        self.args.cuda = True
        # data transforms
        input_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize([.490, .490, .490], [.247, .247, .247])]) # TODO: change mean and std
        
        # dataset
        testset = SegmentationDataset(
                    os.path.join(args.imagelist_path, 'test_stage2.csv'),
                    args.image_path,
                    args.masks_path,
                    input_transform=input_transform, 
                    transform_chain=Compose([Resize(self.args.size, self.args.size)], p=1),
                    base_size=480, is_flip=True, is_clahe=True, is_sh_sc_ro=True
        )
        # dataloader
        kwargs = {'num_workers': args.workers }#, 'pin_memory': True} 
        self.testloader = data.DataLoader(testset, batch_size=args.batch_size,
                                           drop_last=False, shuffle=False, **kwargs)
        self.nclass = 1
        model = EncNet(
            nclass=self.nclass, backbone=args.backbone,
            aux=args.aux, se_loss=args.se_loss, norm_layer=SyncBatchNorm
        )
        print(model)

        self.model = model

        # resuming checkpoint
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            state_dict = {k[7:] : v for k,v in checkpoint['state_dict'].items()}
            self.model.load_state_dict(state_dict)
            self.best_pred = checkpoint['best_pred']
            if 'best_loss' in checkpoint.keys(): 
                self.best_loss = checkpoint['best_loss']
            else:
                self.best_loss = 0
            print("=> loaded checkpoint '{}' (epoch {}, best pred: {}, best loss, {})"
                  .format(args.resume, checkpoint['epoch'], self.best_pred, self.best_loss))
        
        self.model = DataParallelModel(self.model).cuda()

        self.mode2func = {
            0 : lambda x, y: (x, y),
            1 : apply_hflip,
            2 : lambda x, y: (x, y),
            3 : lambda x, y: apply_revert_shscro(x, y, angle=5, scale=0.9, dx=0, dy=0),
            4 : lambda x, y: apply_revert_shscro(x, y, angle=10, scale=0.9, dx=0, dy=0),
            5 : lambda x, y: apply_revert_shscro(x, y, angle=15, scale=0.9, dx=0, dy=0),
            6 : lambda x, y: apply_revert_shscro(x, y, angle=20, scale=0.9, dx=0, dy=0),
            7 : lambda x, y: apply_revert_shscro(x, y, angle=-5, scale=0.9, dx=0, dy=0),
            8 : lambda x, y: apply_revert_shscro(x, y, angle=-10, scale=0.9, dx=0, dy=0),
            9 : lambda x, y: apply_revert_shscro(x, y, angle=-15, scale=0.9, dx=0, dy=0),
            10 : lambda x, y: apply_revert_shscro(x, y, angle=-20, scale=0.9, dx=0, dy=0),
        }

    def predict(self):
        train_loss = 0.0
        self.model.eval()
        tbar = tqdm(self.testloader)
        img_ids = []
        encode_pixels = []
        for i, (img_id, image, _, mode) in enumerate(tbar):
            image = image.cuda()
            with torch.no_grad():
                outputs = self.model(image)
            
            preds_ten = [v[0].data.cpu() for v in outputs]
            cls_preds_ten = [v[1].data.cpu() for v in outputs]
            preds_ten = torch.cat(preds_ten)
            cls_preds_ten = torch.cat(cls_preds_ten)
            preds = torch.sigmoid(preds_ten).data.cpu().numpy()[:, 0, :, :]
            mask_pred = torch.sigmoid(cls_preds_ten).data.cpu().numpy().reshape(-1)

            l_img_id = list(img_id)
            img_ids += l_img_id
            for k, imid in enumerate(l_img_id):
                npy_file = os.path.join(self.args.pred_path,
                        str(imid) + f'_{mode[k].item()}.npy')
                if mode[k].item() < 2:
                    np.save(npy_file, cv2.resize(self.mode2func[mode[k].item()](preds[k], None)[0], (1024, 1024)))
                encode_pixels.append(mask_pred[k])
        pd.DataFrame({'ImageId' : img_ids, 'EncodedPixels' : encode_pixels}).to_csv(
            os.path.join(self.args.pred_path, 'stage2_new_model_submit_16.csv'), index=None)


    def __del__(self):
        del self.model
        gc.collect()


if __name__ == "__main__":
    args = OptionsParser().parse()
    tester = Tester(args)
    tester.predict()
