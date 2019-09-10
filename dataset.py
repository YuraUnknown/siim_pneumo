import random
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter
import torch
import torch.utils.data as data
import torchvision.transforms as transform
import torchvision.transforms.functional as F
import albumentations.augmentations.functional as F_alb
import random
import cv2
import os

def apply_hflip(img, mask, **params):
    img = cv2.flip(img, 1)
    if mask is not None:
        mask = cv2.flip(mask, 1)
    return img, mask

def apply_clahe(img, mask, **params):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img[..., 0])
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img, mask

def apply_shscro(img, mask, **params):
    angle = params['angle']
    scale = params['scale']
    dx = params['dx']
    dy = params['dy']
    img = F_alb.shift_scale_rotate(img, angle, scale, dx, dy, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
    if mask is not None:
        mask = F_alb.shift_scale_rotate(mask, angle, scale, dx, dy, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, 0)
    return img, mask

def apply_revert_shscro(img, mask, **params):
    angle = params['angle']
    scale = params['scale']
    dx = params['dx']
    dy = params['dy']
    img = F_alb.shift_scale_rotate(img, -angle, 1 / scale, -dx, -dy, 
            cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
    if mask is not None:
        mask = F_alb.shift_scale_rotate(mask, -angle, 1 / scale, -dx, -dy, 
                cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, 0)
    return img, mask

class SegmentationDataset(data.Dataset):
    def __init__(self, imagelist, image_path, masks_path,
                input_transform, transform_chain, base_size,
                is_flip=False, is_clahe=False, 
                is_sh_sc_ro=False
                ):
        print(f'Load from {imagelist}')
        imagelist = pd.read_csv(imagelist)
        if 'mask_name' not in imagelist.columns:
            imagelist['mask_name'] = '-1'
        imagelist['mask_name'] = imagelist['mask_name'].astype(str)
        imagelist['has_mask'] = (imagelist['mask_name'] != '-1').astype(int)
        self.num_modes = is_flip + is_clahe + is_sh_sc_ro
        if self.num_modes == 0:
            self.mask_img_map = imagelist[['image_id', 'image_name', 'mask_name', 'has_mask']].values
        else:
            len_imlist = len(imagelist)
            imagelist['mode'] = 0
            imagelist_c = imagelist.copy()
            for i in range(1, self.num_modes + 8):
                imagelist_copy = imagelist_c.copy()
                imagelist_copy['mode'] = i
                imagelist = pd.concat((imagelist, imagelist_copy), axis=0, ignore_index=True)
            self.mask_img_map = imagelist[['image_id', 'image_name', 
            'mask_name', 'has_mask', 'mode']].values
        
        self.image_path = image_path
        self.masks_path = masks_path
        self.base_size = base_size
        self.transform_chain = transform_chain
        self.input_transform = input_transform
        self.mode2func = {
            0 : lambda x, y: (x, y),
            1 : apply_hflip,
            2 : apply_clahe,
            3 : lambda x, y: apply_shscro(x, y, angle=5, scale=0.9, dx=0, dy=0),
            4 : lambda x, y: apply_shscro(x, y, angle=10, scale=0.9, dx=0, dy=0),
            5 : lambda x, y: apply_shscro(x, y, angle=15, scale=0.9, dx=0, dy=0),
            6 : lambda x, y: apply_shscro(x, y, angle=20, scale=0.9, dx=0, dy=0),
            7 : lambda x, y: apply_shscro(x, y, angle=-5, scale=0.9, dx=0, dy=0),
            8 : lambda x, y: apply_shscro(x, y, angle=-10, scale=0.9, dx=0, dy=0),
            9 : lambda x, y: apply_shscro(x, y, angle=-15, scale=0.9, dx=0, dy=0),
            10 : lambda x, y: apply_shscro(x, y, angle=-20, scale=0.9, dx=0, dy=0),
        }

    def __getitem__(self, index):
        if self.num_modes:
            img_id, imname, mask_name, _, mode = self.mask_img_map[index]
        else:
            img_id, imname, mask_name, _ = self.mask_img_map[index]
            is_flip = 0
        img = cv2.imread(os.path.join(self.image_path, imname))
        if img is None:
            img = cv2.imread(imname)

        if img.shape[:2] != (1024, 1024):
            h, w = img.shape[:2]
            factor = 1024 / w if w < h else 1024 / h
            img = cv2.resize(img, None, fx=factor, fy=factor)
            # h, w = img.shape[:2]
            # h_offset = (h - 1024) / 2
            # w_offset = (w - 1024) / 2

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        if mask_name != '-1':
            mask = np.load(os.path.join(self.masks_path, mask_name)).astype(bool).astype(np.uint8)

        if self.num_modes:
            img, mask = self.mode2func[mode](img, mask)
#         img = cv2.resize(img, (self.base_size, self.base_size), interpolation=cv2.INTER_NEAREST)
#         mask = cv2.resize(mask, (self.base_size, self.base_size), interpolation=cv2.INTER_NEAREST)
        
        augmented = self.transform_chain(image=img, mask=mask)
        
        img = Image.fromarray(augmented['image'])
        mask = Image.fromarray(augmented['mask'])
#         for name, args in self.transform_chain.items():
#             img, mask = self.mapping_transform[name](img, mask, args)
        
        img_t = self.input_transform(img)
        mask_t = torch.tensor(np.array(mask, dtype=int))
        if not isinstance(img_id, str):
            img_id = str(img_id) 
        if self.num_modes:
            return img_id, img_t, mask_t, mode
        return img_id, img_t, mask_t.float()

    def __len__(self):
        return len(self.mask_img_map)

    # def random_crop(self, img_t, mask_t, args):
    #     target_size = args['target_size']
    #     h, w = img_t.size
    #     start_h = np.random.randint(h - target_size)
    #     start_w = np.random.randint(w - target_size)
    #     img_t = img_t.crop((start_w, start_h, start_w + target_size, start_h + target_size))
    #     mask_t = mask_t.crop((start_w, start_h, start_w + target_size, start_h + target_size))
    #     return img_t, mask_t

    # def random_horizontal_flip(self, img_t, mask_t, args):
    #     p = args['p']
    #     if random.random() < p:
    #         img_t = F.hflip(img_t)
    #         mask_t = F.hflip(mask_t)
    #     return img_t, mask_t


class ClassificationDataset(data.Dataset):
    def __init__(self, imagelist, image_path, 
                input_transform, transform_chain, base_size
                ):
        imagelist = pd.read_csv(imagelist)
        if 'mask_name' not in imagelist.columns:
            imagelist['mask_name'] = '-1'
        imagelist['has_mask'] = (imagelist['mask_name'] != '-1').astype(int)
        self.img_list = imagelist[['image_id', 'image_name', 'has_mask']].values
        self.image_path = image_path
        self.base_size = base_size
        self.transform_chain = transform_chain
        self.input_transform = input_transform

    def __getitem__(self, index):
        img_id, imname, has_mask = self.img_list[index]
        img = cv2.imread(os.path.join(self.image_path, imname))

        augmented = self.transform_chain(image=img)
        
        img = Image.fromarray(augmented['image'])
        
        img_t = self.input_transform(img)
        return img_id, img_t, has_mask

    def __len__(self):
        return len(self.img_list)