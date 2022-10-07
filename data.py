# from semilearn.datasets.cv_datasets.cifar import get_cifar
import argparse
import os
import torchvision
from torchvision import transforms
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from semilearn.datasets.utils import split_ssl_data

import copy
import numpy as np 
from PIL import Image
from torch.utils.data import Dataset

from semilearn.datasets.augmentation import RandAugment
from semilearn.datasets.utils import get_onehot


NUM_CLASSES=10
args_d = {'dataset': 'cifar10',
         'num_classes': NUM_CLASSES,
         'train_sampler': 'RandomSampler',
         'num_workers': 0,
         'lb_imb_ratio': 1,
         'ulb_imb_ratio':1.0,
          'batch_size': 32,
         'ulb_num_labels': 150,
         'img_size': 32,
         'crop_ratio': 0.875,
         'num_labels': 30,
         'seed': 1,
         'epoch': 3,
         'num_train_iter':150,
         'net': 'wrn_28_8',
         'optim': 'SGD',
         'lr': 0.03,
         'momentum': 0.9,
         'weight_decay': 0.0005,
         'layer_decay': 0.75,
          'num_warmup_iter': 0,
         'algorithm': None,
         'data_dir': './data',
         'uratio': 3,
         'eval_batch_size': 64}

parser = argparse.ArgumentParser(description='Semi-Supervised Learning (USB semilearn package)')
args = parser.parse_args("")
# args
for k in args_d:
        setattr(args, k, args_d[k])

mean, std = {}, {}
mean['cifar10'] = [0.485, 0.456, 0.406]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]

std['cifar10'] = [0.229, 0.224, 0.225]
std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]

class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for Fixmatch,
    and return both weakly and strongly augmented images.
    """

    def __init__(self,
                 alg,
                 data,
                 targets=None,
                 num_classes=None,
                 transform=None,
                 is_ulb=False,
                 strong_transform=None,
                 onehot=False,
                 *args, 
                 **kwargs):
        """
        Args
            data: x_data
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            transform: basic transformation of data
            use_strong_transform: If True, this dataset returns both weakly and strongly augmented images.
            strong_transform: list of transformation functions for strong augmentation
            onehot: If True, label is converted into onehot vector.
        """
        super(BasicDataset, self).__init__()
        self.alg = alg
        self.data = data
        self.targets = targets

        self.num_classes = num_classes
        self.is_ulb = is_ulb
        self.onehot = onehot

        self.transform = transform
        self.strong_transform = strong_transform
        if self.strong_transform is None:
            if self.is_ulb:
                assert self.alg not in ['fullysupervised', 'supervised', 'pseudolabel', 'vat', 'pimodel', 'meanteacher', 'mixmatch'], f"alg {self.alg} requires strong augmentation"
    
    def __sample__(self, idx):
        """ dataset specific sample function """
        # set idx-th target
        if self.targets is None:
            target = None
        else:
            target_ = self.targets[idx]
            target = target_ if not self.onehot else get_onehot(self.num_classes, target_)

        # set augmented images
        img = self.data[idx]
        return img, target

    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """
        img, target = self.__sample__(idx)

        if self.transform is None:
            return  {'x_lb':  transforms.ToTensor()(img), 'y_lb': target}
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img_w = self.transform(img)
            print("self.alg: ", self.alg)

            if not self.is_ulb:
                return {'idx_lb': idx, 'x_lb': img_w, 'y_lb': target} 
            else:
                if self.alg == 'fullysupervised' or self.alg == 'supervised':
                    return {'idx_ulb': idx}
                elif self.alg == 'pseudolabel' or self.alg == 'vat':
                    return {'idx_ulb': idx, 'x_ulb_w':img_w} 
                elif self.alg == 'pimodel' or self.alg == 'meanteacher' or self.alg == 'mixmatch':
                    # NOTE x_ulb_s here is weak augmentation
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s': self.transform(img)}
                elif self.alg == 'remixmatch':
                    rotate_v_list = [0, 90, 180, 270]
                    rotate_v1 = np.random.choice(rotate_v_list, 1).item()
                    img_s1 = self.strong_transform(img)
                    img_s1_rot = torchvision.transforms.functional.rotate(img_s1, rotate_v1)
                    img_s2 = self.strong_transform(img)
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s_0': img_s1, 'x_ulb_s_1':img_s2, 'x_ulb_s_0_rot':img_s1_rot, 'rot_v':rotate_v_list.index(rotate_v1)}
                elif self.alg == 'comatch':
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s_0': self.strong_transform(img), 'x_ulb_s_1':self.strong_transform(img)} 
                else:
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s': self.strong_transform(img)} 


    def __len__(self):
        return len(self.data)
class BasicDataset2(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for Fixmatch,
    and return both weakly and strongly augmented images.
    """

    def __init__(self,
                 alg,
                 lb_data,
                 ulb_data,
                 lb_targets=None,                 
                 ulb_targets=None,
                 num_classes=None,
                 transform=None,
                 is_ulb=False,
                 strong_transform=None,
                 onehot=False,
                 *args, 
                 **kwargs):
        """
        Args
            data: x_data
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            transform: basic transformation of data
            use_strong_transform: If True, this dataset returns both weakly and strongly augmented images.
            strong_transform: list of transformation functions for strong augmentation
            onehot: If True, label is converted into onehot vector.
        """
        super(BasicDataset2, self).__init__()
        self.alg = alg
        self.lb_data = lb_data
        self.ulb_data = ulb_data
        self.lb_targets = lb_targets
        self.ulb_targets = ulb_targets

        self.num_classes = num_classes
        self.is_ulb = is_ulb
        self.onehot = onehot
        print("1-self.alg: ", self.alg)

        self.transform = transform
        self.strong_transform = strong_transform
        if self.strong_transform is None:
            if self.is_ulb:
                assert self.alg not in ['fullysupervised', 'supervised', 'pseudolabel', 'vat', 'pimodel', 'meanteacher', 'mixmatch'], f"alg {self.alg} requires strong augmentation"
    
    def __lb_sample__(self, idx):
        """ dataset specific sample function """
        # set idx-th target
        # if self.lb_targets is None:
        #     lb_target = None
        # else:
        lb_target_ = self.lb_targets[idx]
        lb_target = lb_target_ if not self.onehot else get_onehot(self.num_classes, lb_target_)

        # set augmented images
        img = self.lb_data[idx]
        # print("self.lb_data[idx]: ",self.lb_data[idx])
        return img, lb_target
    def __ulb_sample__(self, idx):
        """ dataset specific sample function """
        # set idx-th target
        # if self.lb_targets is None:
        #     lb_target = None
        # else:
        #     lb_target_ = self.lb_targets[idx]
        #     lb_target = target_ if not self.onehot else get_onehot(self.num_classes, target_)

        # set augmented images
        img = self.ulb_data[idx]
        return img

    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """
        lb_img, lb_target = self.__lb_sample__(idx%len(self.lb_data))
        ulb_img = self.__ulb_sample__(idx)

        if self.transform is None:
            return  {'x_lb':  transforms.ToTensor()(img), 'y_lb': lb_target}
        else:
            if isinstance(lb_img, np.ndarray):
                lb_img = Image.fromarray(lb_img)
            img_w = self.transform(lb_img)
            return {'idx_lb': idx, 
                    'x_lb': img_w, 
                    'y_lb': lb_target,
                    'idx_ulb': idx,
                    'x_ulb_w': img_w, 
                    'x_ulb_s': self.strong_transform( Image.fromarray(ulb_img))} 
            # if not self.is_ulb:
            #     return {'idx_lb': idx, 'x_lb': img_w, 'y_lb': target} 
            # else:
            #     if self.alg == 'fullysupervised' or self.alg == 'supervised':
            #         return {'idx_ulb': idx}
            #     elif self.alg == 'pseudolabel' or self.alg == 'vat':
            #         return {'idx_ulb': idx, 'x_ulb_w':img_w} 
            #     elif self.alg == 'pimodel' or self.alg == 'meanteacher' or self.alg == 'mixmatch':
            #         # NOTE x_ulb_s here is weak augmentation
            #         return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s': self.transform(img)}
            #     elif self.alg == 'remixmatch':
            #         rotate_v_list = [0, 90, 180, 270]
            #         rotate_v1 = np.random.choice(rotate_v_list, 1).item()
            #         img_s1 = self.strong_transform(img)
            #         img_s1_rot = torchvision.transforms.functional.rotate(img_s1, rotate_v1)
            #         img_s2 = self.strong_transform(img)
            #         return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s_0': img_s1, 'x_ulb_s_1':img_s2, 'x_ulb_s_0_rot':img_s1_rot, 'rot_v':rotate_v_list.index(rotate_v1)}
            #     elif self.alg == 'comatch':
            #         return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s_0': self.strong_transform(img), 'x_ulb_s_1':self.strong_transform(img)} 
            #     else:
            #         return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s': self.strong_transform(img)} 


    def __len__(self):
        return len(self.lb_data)

def get_cifar2(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    
    data_dir = os.path.join(data_dir, name.lower())
    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, train=True, download=True)
    data, targets = dset.data, dset.targets
    
    crop_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name],)
    ])

    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(args, data, targets, num_classes, 
                                                                lb_num_labels=num_labels,
                                                                ulb_num_labels=args.ulb_num_labels,
                                                                lb_imbalance_ratio=args.lb_imb_ratio,
                                                                ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                include_lb_to_ulb=include_lb_to_ulb)
    
    lb_count = [0 for _ in range(num_classes)]
    ulb_count = [0 for _ in range(num_classes)]
    for c in lb_targets:
        lb_count[c] += 1
    for c in ulb_targets:
        ulb_count[c] += 1
    # print("lb_targets: ",lb_targets)
    # print("ulb_targets: ",ulb_targets)
    print("lb count: {}".format(lb_count))
    print("ulb count: {}".format(ulb_count))
    # lb_count = lb_count / lb_count.sum()
    # ulb_count = ulb_count / ulb_count.sum()
    # args.lb_class_dist = lb_count
    # args.ulb_class_dist = ulb_count

    if alg == 'fullysupervised':
        lb_data = data
        lb_targets = targets
        # if len(ulb_data) == len(data):
        #     lb_data = ulb_data 
        #     lb_targets = ulb_targets
        # else:
        #     lb_data = np.concatenate([lb_data, ulb_data], axis=0)
        #     lb_targets = np.concatenate([lb_targets, ulb_targets], axis=0)
    
    # output the distribution of labeled data for remixmatch
    # count = [0 for _ in range(num_classes)]
    # for c in lb_targets:
    #     count[c] += 1
    # dist = np.array(count, dtype=float)
    # dist = dist / dist.sum()
    # dist = dist.tolist()
    # out = {"distribution": dist}
    # output_file = r"./data_statistics/"
    # output_path = output_file + str(name) + '_' + str(num_labels) + '.json'
    # if not os.path.exists(output_file):
    #     os.makedirs(output_file, exist_ok=True)
    # with open(output_path, 'w') as w:
    #     json.dump(out, w)

    lb_and_ulb_dset = BasicDataset2(alg,
                         lb_data = lb_data,
                         ulb_data=ulb_data, 
                         lb_targets = lb_targets,
                         ulb_targets=ulb_targets,
                         num_classes=num_classes,
                         transform=transform_weak, 
                         is_ulb=True, 
                         strong_transform=transform_strong,
                         onehot=False)
    '''
    transform=None,
     is_ulb=False,
     strong_transform=None,
     onehot=False,
     *args, 
     **kwargs
    '''

    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, train=False, download=True)
    test_data, test_targets = dset.data, dset.targets
    eval_dset = BasicDataset(alg, test_data, test_targets, num_classes, transform_val, False, None, False)

    return lb_and_ulb_dset, eval_dset