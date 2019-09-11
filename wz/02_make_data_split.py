import glob
import shutil
import os

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from config import *
from utils import *

## Train/Val split

all_mask_fn = glob.glob(f'{DATA_DIR}/masks/*')
mask_df = pd.DataFrame()
mask_df['file_names'] = all_mask_fn
mask_df['mask_percentage'] = 0
mask_df.set_index('file_names',inplace=True)
for fn in all_mask_fn:
    mask_df.loc[fn,'mask_percentage'] = np.array(Image.open(fn)).sum()/(1024*1024*255) #255 is bcz img range is 255
    
mask_df.reset_index(inplace=True)
mask_df['labels'] = 0
mask_df.loc[mask_df.mask_percentage>0,'labels'] = 1

all_train_fn = glob.glob(f'{DATA_DIR}/train/train/*')
total_samples = len(all_train_fn)
idx = np.arange(total_samples)
train_fn, val_fn = train_test_split(all_train_fn,stratify=mask_df.labels,test_size=0.1,random_state=SEED)
train_fn = all_train_fn

masks_train_fn = [fn.replace(f'{DATA_DIR}/train/train',f'{DATA_DIR}/masks') for fn in train_fn]    
masks_val_fn = [fn.replace(f'{DATA_DIR}/train/train',f'{DATA_DIR}/masks') for fn in val_fn]

train_dir = f'{DATA_DIR}/keras_im_train'
for full_fn in train_fn:
    fn = full_fn.split('/')[-1]
    shutil.copy(full_fn, os.path.join(train_dir,fn))
    
train_dir = f'{DATA_DIR}/keras_mask_train'
for full_fn in masks_train_fn:
    fn = full_fn.split('/')[-1]
    shutil.copy(full_fn,os.path.join(train_dir,fn))
    
train_dir = f'{DATA_DIR}/keras_im_val'
for full_fn in val_fn:
    fn = full_fn.split('/')[-1]
    shutil.copy(full_fn,os.path.join(train_dir,fn))
    
train_dir = f'{DATA_DIR}/keras_mask_val'
for full_fn in masks_val_fn:
    fn = full_fn.split('/')[-1]
    shutil.copy(full_fn,os.path.join(train_dir,fn))

train_im_path, train_mask_path = f'{DATA_DIR}/keras_im_train',f'{DATA_DIR}/keras_mask_train'
train_im_paths = glob.glob(train_im_path+'/*')

masks_bool = []
for i, im_path in enumerate(train_im_paths):
    mask_path = im_path.replace(train_im_path, train_mask_path)
    mask = np.array(Image.open(mask_path))
    masks_bool.append(mask.sum() > 0)
    
write_pickle_obj(masks_bool, 'masks_bool.pickle')