import argparse

class OptionsParser():
    def __init__(self):
        parser = argparse.ArgumentParser(description='Train semantic segmentation model.')
        # model and dataset settings
        # parser.add_argument('--model', type=str, required=True,
        #     help='Choose model: [encnet]'
        # )
        parser.add_argument('--model', type=str, required=True,
            help='Unet or EncNet'
        )
        parser.add_argument('--backbone', type=str, required=True,
            help='Choose backbone: [resnet18, resnet34, resnet50, resnet101, resnet152]'
        )
        parser.add_argument('--imagelist_path', type=str, required=True,
            help='Specify path to list with images'
        )
        parser.add_argument('--image_path', type=str, required=True,
            help='Specify path to images'
        )
        parser.add_argument('--masks_path', type=str, required=True,
            help='Specify path to masks'
        )
        parser.add_argument('--workers', type=int, default=16,
                            metavar='N', help='dataloader threads')
        parser.add_argument('--ckpt_name', type=str, required=True,
                            help='dataloader threads')
        parser.add_argument('--resume', type=str)


        # training hyper params
        parser.add_argument('--aux', action='store_true', default=False,
                            help='Auxilary Loss')
        parser.add_argument('--aux-weight', type=float, default=0.2,
                            help='Auxilary loss weight (default: 0.2)')
        parser.add_argument('--se-loss', action='store_true', default=False,
                            help='Semantic Encoding Loss SE-loss')
        parser.add_argument('--se-weight', type=float, default=0.2,
                            help='SE-loss weight (default: 0.2)')
        parser.add_argument('--epochs', type=int, required=True, metavar='N',
                            help='number of epochs to train (default: auto)')
        parser.add_argument('--batch-size', type=int, default=16,
                            metavar='N', help='input batch size for \
                            training (default: auto)')
        parser.add_argument('--size', type=int, default=512,
                            metavar='N', help='input size for \
                            image (default: auto)')
        parser.add_argument('--num-fold', type=int, default=0,
                            metavar='N', help='number of fold')
        parser.add_argument('--cls-weight', type=float, default=0.5,
                            help='Classification loss weight')
        parser.add_argument('--use-dice', action='store_true', default=False,
                            help='Use dice loss')

        # logger
        parser.add_argument('--logger-dir', type=str, required=True,
                            help='Logger directory for tensorboard'
        )
        parser.add_argument('--pred-path', type=str, default='/tmp',
                            help='Path for predictions')

        # optimizer params
        parser.add_argument('--lr', type=float, required=True,
            help='Choose LR'
        )
        parser.add_argument('--wd', type=float, required=True,
            help='Choose weight decay'
        )
        parser.add_argument('--lr-scheduler', type=str, default='poly',
                            help='learning rate scheduler (default: poly)')

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args