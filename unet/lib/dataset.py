import os
import numpy as np
import pandas as pd
import torch
import glob
from torch.utils.data import Dataset
from PIL import Image

class WSIDataset(Dataset):
    """ Dataset for plaques segmentation/classification """

    def __init__(self, meta_data, root_dir, normalization, cache_data=True, transform=None):
        """
        Args:
            root_dir (string): Directory with all the WSI. Each WSI has 4 folders:
            --root_dir
                --WSI_name
                    --macenko: all patches with macenko normalization.
                    --masks: all masks from patches.
                    --patches: original patches without normalization.
                    --vahadane: all patches with vahadane normalization.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        df = pd.read_csv(meta_data)
        self.wsi_list = df.wsi.to_list()
        self.root_dir = root_dir
        self.transform = transform
        self.normalization = normalization
        self.cache_data = cache_data

        if self.normalization == 'macenko':
            self.imgs_path = np.concatenate([glob.glob(os.path.join(self.root_dir, self.wsi_list[i], 'macenko','*.png')) for i in range(len(self.wsi_list))])
        elif self.normalization == 'vahadane':
            self.imgs_path = np.concatenate([glob.glob(os.path.join(self.root_dir, self.wsi_list[i], 'vahadane','*.png')) for i in range(len(self.wsi_list))])
        else:
            print(f'[ERROR] Normalization method is not recognized. Change the configuration file.')

        if cache_data:
            dataset_imgs = []
            dataset_gt = []
            for data in self.imgs_path:
                dataset_imgs.append(np.array(Image.open(data)))
                dataset_gt.append(np.array(Image.open(data.replace(self.normalization,'masks').replace('patch','mask')).convert('1')))
            self.dataset_imgs = dataset_imgs.copy()
            self.dataset_gt = dataset_gt.copy()

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        
        '''
        Load patch and mask arrays:
        - images come from a single WSI.
        - mask --> 1 for plaques, 0 for background.
        '''
        if self.cache_data:
            image = self.dataset_imgs[idx]
            gt_img = self.dataset_gt[idx]
        else:
            image = np.array(Image.open(self.imgs_path[idx]))
            gt_img = np.array(Image.open(self.imgs_path[idx].replace(self.normalization,'masks').replace('patch','mask')).convert('1'))

        sample = {'image': image, 'gt': gt_img}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample['image'], sample['gt']
