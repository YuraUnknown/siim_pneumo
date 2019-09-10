# siim_pneumo

## Pytorch part
Пример запуска EncNet:
```
CUDA_VISIBLE_DEVICES=1,2,3 python3 train_fold.py --model encnet --backbone resnet50 --imagelist_path /home/sggpls/Documents/siim_pneumo/data_for_train/ --image_path /home/sggpls/Documents/siim_pneumo/data_for_train/origin_img/ --masks_path /home/sggpls/Documents/siim_pneumo/data_for_train/masks_img/ --lr=0.001 --wd=0.0 --epochs 50 --batch-size 24 --size=512 --se-loss --aux --ckpt_name /tmp/tmp.pth --use-dice --logger-dir /tmp | tee /tmp/log
```

Пример запуска Unet:
```
CUDA_VISIBLE_DEVICES=1,2,3 python3 train_fold.py --model unet --backbone seresnext50 --imagelist_path /home/sggpls/Documents/siim_pneumo/data_for_train/ --image_path /home/sggpls/Documents/siim_pneumo/data_for_train/origin_img/ --masks_path /home/sggpls/Documents/siim_pneumo/data_for_train/masks_img/ --lr=0.1 --wd=0.0 --epochs 50 --batch-size 24 --size=512 --ckpt_name /tmp/tmp.pth --use-dice --logger-dir /tmp | tee /tmp/log
```

Дополнительно установить: pytorch-encoding
