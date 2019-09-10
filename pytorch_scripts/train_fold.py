
# sudo CUDA_VISIBLE_DEVICES=1,3 python3 train.py --backbone resnet50 --imagelist_path /home/sggpls/Documents/siim-pneumothorax/data_for_train/ --image_path /home/sggpls/Documents/siim-pneumothorax/data_for_train/origin_img/ --masks_path /home/sggpls/Documents/siim-pneumothorax/data_for_train/masks_img/ --lr=0.001 --wd=0.0 --epochs 100 --batch-size 4 --se-loss --aux --ckpt_name ../checkponts/encnet_resnet50_masked_dice_sgd_vallr_001_wd_0_1024.pth --use-dice --resume ../../models_pth/encnet_resnet50_masked_sgd_vallr_001_wd_00001_best.pth --logger-dir ../tb_logs/encnet_resnet50_masked_dice_sgd_vallr_001_wd_0_1024 | tee ../logs/encnet_resnet50_masked_dice_sgd_vallr_001_wd_0_1024.log

import os
import sys
import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import torch
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime as dtm

import encoding.utils as utils
from encoding.nn import SyncBatchNorm
from encoding.parallel import DataParallelModel, DataParallelCriterion

sys.path.append('..')
from logger import Logger
from dataset import SegmentationDataset
from options_parse import OptionsParser
from models.encnet import EncNet
from models.unet_model import UNet
from models.loss import SegmentationLosses, dice_loss, SegmentationLossesBCE
import cv2
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue, Resize,
    RandomBrightness, RandomContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RandomSizedCrop
)

import gc

torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable

def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight

def drop_bn_parameters(model):
    param_list = []
    for k, v in model.named_parameters():
        if k.find('bn') == -1:
            param_list.append(v)
            print(f'{k:50s} - added')
        else:
            print(f'{k:50s} - droped')
    return param_list

def to_np(x, mode=None):
    if mode == 'cpu':
        return x.data.numpy()
    return x.data.cpu().numpy()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class Trainer():
    def __init__(self, args):
        self.args = args
        self.args.start_epoch = 0
        self.args.cuda = True
        # data transforms
        input_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize([.490, .490, .490], [.247, .247, .247])]) # TODO: change mean and std
        
        # dataset
        train_chain = Compose([
            HorizontalFlip(p=0.5),
            OneOf([
                ElasticTransform(alpha=300, sigma=300 * 0.05, alpha_affine=300 * 0.03),
                GridDistortion(),
                OpticalDistortion(distort_limit=2, shift_limit=0.5),
                ], p=0.3),
            RandomSizedCrop(min_max_height=(900, 1024), height=1024, width=1024, p=0.5),
            ShiftScaleRotate(rotate_limit=20, p=0.5),
            Resize(self.args.size, self.args.size)
        ], p=1)

        val_chain = Compose([Resize(self.args.size, self.args.size)], p=1)
        num_fold = self.args.num_fold
        df_train = pd.read_csv(os.path.join(args.imagelist_path, 'train.csv'))
        df_val = pd.read_csv(os.path.join(args.imagelist_path, 'val.csv'))
        df_full = pd.concat((df_train, df_val), ignore_index=True, axis=0)
        df_full['lbl'] = (df_full['mask_name'].astype(str) == '-1').astype(int)
        skf = StratifiedKFold(8, shuffle=True, random_state=777)
        train_ids, val_ids = list(skf.split(df_full['mask_name'], df_full['lbl']))[num_fold]

        df_test = pd.read_csv(os.path.join(args.imagelist_path, 'test_true.csv'))

        df_new_train = pd.concat((df_full.iloc[train_ids], df_test),
                ignore_index=True, axis=0, sort=False
        )
        df_new_val = df_full.iloc[val_ids]

        df_new_train.to_csv(f'/tmp/train_new_pneumo_{num_fold}.csv')
        df_new_val.to_csv(f'/tmp/val_new_pneumo_{num_fold}.csv')

        trainset = SegmentationDataset(
                    f'/tmp/train_new_pneumo_{num_fold}.csv',
                    args.image_path,
                    args.masks_path,
                    input_transform=input_transform, 
                    transform_chain=train_chain,
                    base_size=1024
        )
        testset = SegmentationDataset(
                    f'/tmp/val_new_pneumo_{num_fold}.csv',
                    args.image_path,
                    args.masks_path,
                    input_transform=input_transform, 
                    transform_chain=val_chain,
                    base_size=1024
        )
        
        imgs = trainset.mask_img_map[:, [0, 3]]
        weights = make_weights_for_balanced_classes(imgs, 2)
        weights = torch.DoubleTensor(weights)                                       
        train_sampler = (torch.utils
            .data.sampler.WeightedRandomSampler(weights, len(weights))
        )

        # dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True} 
        self.trainloader = data.DataLoader(trainset, batch_size=args.batch_size,
                                           drop_last=True, sampler=train_sampler, #shuffle=True, 
                                           **kwargs)
        self.valloader = data.DataLoader(testset, batch_size=args.batch_size,
                                         drop_last=False, shuffle=False, **kwargs)
                                         
        self.nclass = 1
        if self.args.model == 'unet':
            model = UNet(n_classes=self.nclass, norm_layer=SyncBatchNorm)
            params_list = [{'params': model.parameters(), 'lr': args.lr},]
        elif self.args.model == 'encnet':
            model = EncNet(
                nclass=self.nclass, backbone=args.backbone,
                aux=args.aux, se_loss=args.se_loss, norm_layer=SyncBatchNorm#nn.BatchNorm2d
            )
        
            # optimizer using different LR
            params_list = [{'params': model.pretrained.parameters(), 'lr': args.lr},]
            if hasattr(model, 'head'):
                params_list.append({'params': model.head.parameters(), 'lr': args.lr*10})
            if hasattr(model, 'auxlayer'):
                params_list.append({'params': model.auxlayer.parameters(), 'lr': args.lr*10})

        print(model)
        optimizer = torch.optim.SGD(params_list, lr=args.lr,
            momentum=0.9, weight_decay=args.wd)

        # criterions
        if self.nclass == 1:
            self.criterion = SegmentationLossesBCE(se_loss=args.se_loss,
                                            aux=args.aux,
                                            nclass=self.nclass, 
                                            se_weight=args.se_weight,
                                            aux_weight=args.aux_weight,
                                            use_dice=args.use_dice)
        else:
            self.criterion = SegmentationLosses(se_loss=args.se_loss,
                                            aux=args.aux,
                                            nclass=self.nclass, 
                                            se_weight=args.se_weight,
                                            aux_weight=args.aux_weight,
                                            )
        self.model, self.optimizer = model, optimizer
        
        self.best_pred = 0.0
        self.model = DataParallelModel(self.model).cuda()
        self.criterion = DataParallelCriterion(self.criterion).cuda()
        
        # resuming checkpoint
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)#, map_location='cpu')
            self.args.start_epoch = checkpoint['epoch']
            state_dict = {k : v for k, v in checkpoint['state_dict'].items()}
            self.model.load_state_dict(state_dict)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            for g in self.optimizer.param_groups:
                g['lr'] = args.lr
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            print(f'Best dice: {checkpoint["best_pred"]}')
            print(f'LR: {get_lr(self.optimizer):.5f}')
            
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=4,
        threshold=0.001, threshold_mode='abs', min_lr=0.00001)
        self.logger = Logger(args.logger_dir)
        self.step_train = 0
        self.best_loss = 20
        self.step_val = 0

    def logging(self, loss, running_acc, total, is_train, step, 
            is_per_epoch, inputs=None, pred_masks=None, true_masks=None):
        #============ TensorBoard logging ============#
        # Log the scalar values
        accuracy = 100.0 * running_acc / total
        loss_str = 'Loss per epoch' if is_per_epoch else 'Loss per step' 
        accuracy_str = 'Accuracy per epoch' if is_per_epoch else 'Accuracy per step'
        if is_per_epoch:
            loss = loss / len(self.trainloader) if is_train else loss / len(self.valloader)
        info = {
            loss_str: loss,
            accuracy_str: accuracy
        }

        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, step, is_train)

        # Log values and gradients of the parameters (histogram)
        for tag, value in filter(lambda p: p[1].requires_grad, self.model.named_parameters()):
            tag = tag.replace('.', '/')
            self.logger.histo_summary(tag, to_np(value), step, 1000, is_train)
            if value.grad is not None:
                self.logger.histo_summary(tag + '/grad', to_np(value.grad), step, 1000, is_train)

        if inputs is not None:
            # Log the images
            inputs = to_np(inputs)[:10].transpose(0, 2, 3, 1)
            for i in range(inputs.shape[0]):
                inputs[i] *= np.array([.247, .247, .247])
                inputs[i] += np.array([.490, .490, .490])
            inputs = (255 * inputs).astype(np.uint8)#.transpose(0, 3, 1, 2)
            pred_masks = (255 * pred_masks)[:10].astype(np.uint8)
            true_masks = (255 * true_masks)[:10].astype(np.uint8)
            inputs[..., 0] = (0.5 * inputs[..., 0] + 0.5 * pred_masks)
            inputs[..., 1] = (0.5 * inputs[..., 1] + 0.5 * true_masks).astype(np.uint8)
            info = {
                'images': inputs,
            }

            for tag, inputs in info.items():
                self.logger.image_summary(tag, inputs, step, is_train)

    def training(self, epoch):
        self.model.train()
        # tbar = tqdm(self.trainloader)
        total_score = 0
        total_score_simple = 0
        total_count = 0
        total_loss = 0
        for i, (_, image, target) in enumerate(self.trainloader):
            # if i >= 1000:
            #     break
            start_t = dtm.now()
            torch.cuda.empty_cache()
            if torch_ver == "0.3":
                image = Variable(image)
                target = Variable(target)

            outputs = self.model(image)
            loss = self.criterion(outputs, target)
            # loss = loss / 4
            loss.backward()
            # if (i+1) % 4 == 0:
            self.optimizer.step()
                # self.model.zero_grad()
            self.optimizer.zero_grad()
                
            total_loss += loss.item()

            preds_ten = [v[0].data.cpu() for v in outputs]
            preds_ten = torch.cat(preds_ten)
            if self.args.model == 'encnet':
                cls_preds_ten = [v[1].data.cpu() for v in outputs]
                cls_preds_ten = torch.cat(cls_preds_ten)
                cls_mask = torch.sigmoid(cls_preds_ten).numpy().reshape(-1) < 0.5
                preds = torch.sigmoid(preds_ten).numpy()[:, 0, :, :]
            elif self.args.model == 'unet':
                cls_mask = np.zeros(preds_ten.size(0))
                preds = preds_ten.numpy()[:, 0, :, :]
            trues = target.numpy()

            local_score = dice_loss(trues, preds, cls_mask=cls_mask)
            local_score_simple = dice_loss(trues, preds)
            batch_size = preds.shape[0]
            
            total_score += local_score
            total_score_simple += local_score_simple
            total_count += batch_size

            print(( f'Epoch: {epoch}, Batch: {i + 1} / {len(self.trainloader)}, '#{len(self.trainloader)}, '
                f'loss: {total_loss / (i + 1):.3f}, batch loss: {loss.item():.3f}'
                f', batch simple DICE: {local_score_simple / batch_size:.3f}'
                f', total simple DICE: {total_score_simple / total_count:.3f}'
                f', batch DICE: {local_score / batch_size:.3f}'
                f', total DICE: {total_score / total_count:.5f}'
                f', lr: {get_lr(self.optimizer):.5f}'
                f', time: {(dtm.now() - start_t).total_seconds():.2f}'))
                
            # if i > 5:
            #     break
            self.step_train += 1
            if i % 10 == 0:
                sys.stdout.flush()
                pred_masks = np.array([preds[i] * cls_mask[i] for i in range(len(cls_mask))])
                pred_masks = (pred_masks > 0.5).astype(int)
                self.logging(loss.item(), total_score, total_count, 
                is_train=True, step=self.step_train, is_per_epoch=False, inputs=image, 
                pred_masks=pred_masks, true_masks=trues)


    def validation(self, epoch):
        # Fast test during the training
        def eval_batch(model, image, target):
            outputs = model(image)
            loss = self.criterion(outputs, target)

            preds_ten = [v[0].data.cpu() for v in outputs]
            preds_ten = torch.cat(preds_ten)
            if self.args.model == 'encnet':
                cls_preds_ten = [v[1].data.cpu() for v in outputs]
                cls_preds_ten = torch.cat(cls_preds_ten)
                cls_mask = torch.sigmoid(cls_preds_ten).numpy().reshape(-1) < 0.5
                preds = torch.sigmoid(preds_ten).numpy()[:, 0, :, :]
            elif self.args.model == 'unet':
                cls_mask = np.zeros(preds_ten.size(0))
                preds = preds_ten.numpy()[:, 0, :, :]
                
            trues = target.numpy()

            local_score = dice_loss(trues, preds, cls_mask=cls_mask)
            batch_size = preds.shape[0]

            return preds, trues, cls_mask, local_score, batch_size, loss

        is_best = False
        self.model.eval()
        total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        total_score = 0
        total_loss = 0
        total_count = 0

        for i, (_, image, target) in enumerate(self.valloader):
            if torch_ver == "0.3":
                image = Variable(image, volatile=True)
                correct, labeled, inter, union = eval_batch(self.model, image, target)
            else:
                with torch.no_grad():
                    preds, trues, cls_mask, local_score, batch_size, loss = (
                        eval_batch(self.model, image, target))

            total_score += local_score
            total_loss += loss.item()
            total_count += batch_size

            dice = total_score / total_count
            print(f'val epoch: {epoch}, batch: {i + 1} / {len(self.valloader)}, DICE: {dice:.5f}')
            self.step_val += 1
            # if i > 15:
            #     break
            if i % 10 == 0:
                sys.stdout.flush()
                pred_masks = np.array([preds[i] * cls_mask[i] for i in range(len(cls_mask))])
                pred_masks = (pred_masks > 0.5).astype(int)
                self.logging(loss.item(), total_score, total_count, 
                is_train=False, step=self.step_val, is_per_epoch=False, inputs=image, 
                pred_masks=pred_masks, true_masks=trues)

        new_pred = dice
        if new_pred > self.best_pred:
            self.best_pred = new_pred
            torch.save({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
                'best_loss': self.best_loss
            }, self.args.ckpt_name[:-4] + '_best.pth')

        new_loss = total_loss / total_count
        if new_loss < self.best_loss:
            self.best_loss = new_loss
            torch.save({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
                'best_loss': self.best_loss
            }, self.args.ckpt_name[:-4] + '_best_loss.pth')
        

        print(f'Validation DICE: {dice:.5f}, loss: {new_loss:.5f}')
        print(f'Validation best DICE: {self.best_pred:.5f}, best loss: {self.best_loss:.5f}')
        torch.save({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': dice,
        }, self.args.ckpt_name[:-4] + '_last.pth')
        return new_loss
        

    def __del__(self):
        del self.model
        gc.collect()


if __name__ == "__main__":
    args = OptionsParser().parse()
    trainer = Trainer(args)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        val_loss = trainer.validation(epoch)
        trainer.scheduler.step(val_loss)
