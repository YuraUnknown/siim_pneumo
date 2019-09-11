import cv2
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise, CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RandomSizedCrop, CLAHE
)


AUGMENTATIONS_TRAIN = Compose([
#     HorizontalFlip(p=0.5),
#     OneOf([
# #         RandomContrast(),
#         RandomGamma(),
#         CLAHE(clip_limit=2.0, tile_grid_size=(8, 8),)
# #         RandomBrightness(),
#          ], p=0.2),
    
#     OneOf([
#         ElasticTransform(alpha=120, sigma=120 * 0.045, alpha_affine=120 * 0.025),
#         GridDistortion(),
#         OpticalDistortion(distort_limit=2, shift_limit=0.5),
#         ], p=0.3),
    ShiftScaleRotate(
                    shift_limit=0,  # no resizing
                    scale_limit=0.1,
                    rotate_limit=15, # rotate
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT
                ),
    RandomSizedCrop(min_max_height=(int(h*0.85), h), height=h, width=w, p=0.3),

    ToFloat(max_value=1)
],p=1)


AUGMENTATIONS_TEST = Compose([
    ToFloat(max_value=1)
],p=1)