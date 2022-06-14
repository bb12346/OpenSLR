from data import transform as base_transform
import numpy as np

from utils import is_list, is_dict, get_valid_args


class NoOperation():
    def __call__(self, x):
        return x


class BaseSilTransform():
    def __init__(self, disvor=255.0, img_shape=None):
        self.disvor = disvor
        self.img_shape = img_shape

    def __call__(self, x):
        if self.img_shape is not None:
            s = x.shape[0]
            _ = [s] + [*self.img_shape]
            x = x.reshape(*_)
        return x / self.disvor


class BaseSilCuttingTransform():
    def __init__(self, img_w=64, disvor=255.0, cutting=None):
        self.img_w = img_w
        self.disvor = disvor
        self.cutting = cutting

    def __call__(self, x):
        if self.cutting is not None:
            cutting = self.cutting
        else:
            cutting = int(self.img_w // 64) * 10
        x = x[..., cutting:-cutting]
        return x / self.disvor


class BaseRgbTransform():
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = [0.485*255, 0.456*255, 0.406*255]
        if std is None:
            std = [0.229*255, 0.224*255, 0.225*255]
        self.mean = np.array(mean).reshape((1, 3, 1, 1))
        self.std = np.array(std).reshape((1, 3, 1, 1))

    def __call__(self, x):

        # 720*1280
        # 712*1280

        # return (x - self.mean) / self.std
        # print('------------BaseRgbTransform-------------')
        # print('-x1-',x.shape)
        if np.size(x,2)!=720 or np.size(x,3)!=1280:
            x = np.resize(x,(np.size(x,0),np.size(x,1),720,1280))  # 712*1280 -> 720*1280

        # x = x[:,:,:,220:940]
        x = x[:,:,:,160:880] # 720*720
        # print('-x2-',x.shape)

        input_size = 720
        output_size = 144
        bin_size = input_size // output_size
        # stest = x.reshape((len(x), 3, output_size, bin_size, 
        #                                 output_size, bin_size))
        # print('-stest-',stest.shape)
        # if len(x[0][0][0]) != 720 or len(x[0][0][0][0]) != 720:
            # print('-----------1-------------')
        small_image = x.reshape((len(x),3, output_size, bin_size, 
                                            output_size, bin_size)).max(5).max(3)

        # 3*144*144

        return (small_image - self.mean) / self.std


class BasePosejointTransform():
    def __init__(self,):
        self.selected = {
    '59': np.concatenate((np.arange(0,17), np.arange(91,133)), axis=0), #59
    '31': np.concatenate((np.arange(0,11), [91,95,96,99,100,103,104,107,108,111],[112,116,117,120,121,124,125,128,129,132]), axis=0), #31
    '27': np.concatenate(([0,5,6,7,8,9,10], 
                    [91,95,96,99,100,103,104,107,108,111],[112,116,117,120,121,124,125,128,129,132]), axis=0) #27
    }
    max_body_true = 1
    # max_frame = 150
    num_channels = 3
    def __call__(self, x):
        # 16, 133, 3
        # print('-x1-',x.shape)
        x = x[:,self.selected['27'],:]
        # 16, 27, 3
        # print('-x2-',x.shape)
        x = x[:,:,:,np.newaxis]
        # 16, 27, 3, 1

        # print('-x3-',x.shape)
        x = np.transpose(x,(2,0,1,3))
        # 3, 16, 27 ,1
        # C, frames, Key joints, M
        # print('-x4-',x.shape)
        return x 


def get_transform(trf_cfg=None):
    if is_dict(trf_cfg):
        transform = getattr(base_transform, trf_cfg['type'])
        valid_trf_arg = get_valid_args(transform, trf_cfg, ['type'])
        return transform(**valid_trf_arg)
    if trf_cfg is None:
        return lambda x: x
    if is_list(trf_cfg):
        transform = [get_transform(cfg) for cfg in trf_cfg]
        return transform
    raise "Error type for -Transform-Cfg-"
