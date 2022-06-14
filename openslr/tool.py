
from collections import OrderedDict, namedtuple
import yaml
import os 
import pickle
import cv2
import torch.nn as nn
def is_list(x):
    return isinstance(x, list) or isinstance(x, nn.ModuleList)

class Odict(OrderedDict):
    def append(self, odict):
        dst_keys = self.keys()
        for k, v in odict.items():
            if not is_list(v):
                v = [v]
            if k in dst_keys:
                if is_list(self[k]):
                    self[k] += v
                else:
                    self[k] = [self[k]] + v
            else:
                self[k] = v

def is_dict(x):
    return isinstance(x, dict) or isinstance(x, OrderedDict) or isinstance(x, Odict)

def MergeCfgsDict(src, dst):
    for k, v in src.items():
        if (k not in dst.keys()) or (type(v) != type(dict())):
            dst[k] = v
        else:
            if is_dict(src[k]) and is_dict(dst[k]):
                MergeCfgsDict(src[k], dst[k])
            else:
                dst[k] = v

def config_loader(path):
    with open(path, 'r') as stream:
        src_cfgs = yaml.safe_load(stream)
    with open("/root/SSL/OpenSLR/config/default.yaml", 'r') as stream:
        dst_cfgs = yaml.safe_load(stream)
    MergeCfgsDict(src_cfgs, dst_cfgs)
    return dst_cfgs

def generate_RGB_pkl(dir_path, save_path):
    file_list = sorted(os.listdir(dir_path))
    print(file_list)
    for name in file_list:
        dir_each = os.path.join(dir_path, name)
        frame_list = os.listdir(dir_each)
        frame_list.sort()
        print(name,len(frame_list))
        all_imgs = []
        for frame in frame_list:
            frame_each = os.path.join(dir_each, frame)
            img = cv2.imread(frame_each)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = img.transpose(2,0,1)
            if np.size(img,1)!=720 or np.size(img,2)!=1280:
                img = np.resize(img,(np.size(img,0),720,1280))
            all_imgs.append(torch.from_numpy(img))
        all_imgs = torch.stack(all_imgs).numpy()
        save_path_each = os.path.join(save_path, name)
        isExists = os.path.exists(save_path_each)
        print(all_imgs.shape)
        if not isExists:
            os.makedirs(save_path_each)
            all_imgs_pkl = os.path.join(save_path_each, '0.pkl')
            pickle.dump(all_imgs, open(all_imgs_pkl, 'wb')) 