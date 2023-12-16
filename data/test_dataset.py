import os
from data.base_dataset import BaseDataset, get_params, get_transform
import torch.nn.functional as F

from PIL import Image,ImageFilter
import random
import numpy as np
import collections
import torch,cv2

class TestDataset(BaseDataset):

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        label_paths, image_paths = self.get_paths(opt)

        self.label_paths = label_paths
        self.image_paths = image_paths

        self.dataset_size = len(self.label_paths)

        self.ref_dict, self.train_test_folder = self.get_ref(opt)
    
    def get_paths(self, opt):
        image_paths = []
        label_paths = []
        dataroot = os.path.join(opt.dataroot,"content")

        lines = os.listdir(dataroot)
        for i in range(len(lines)):
            image_paths.append(os.path.join(dataroot, lines[i].replace('.png' , '.jpg')))
            label_paths.append(os.path.join(dataroot, lines[i]))
        return label_paths, image_paths

    def get_ref(self, opt):
        train_test_folder = ('style', 'content')
        return {}, train_test_folder
    
    def imgpath_to_labelpath(self, path):
        label_path = path[:-4] + '.png'
        return label_path

    def get_masks(self,labelmap,common_label):
        result = []
        for label in common_label:
            result.append(np.where(labelmap == label,1,0))
        return torch.from_numpy(np.array(result)).float()
            


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        label_path = self.label_paths[index]
        image_path = self.image_paths[index]
        # transform_image = get_transform(self.opt, params1)

        key = os.path.basename(image_path)
        path_ref = key
        if key in self.ref_dict:
            val = self.ref_dict[key]
            path_ref = val[0]
        
        path_ref = os.path.dirname(image_path).replace(self.train_test_folder[1], self.train_test_folder[0]) + '/' + path_ref

        image_ref = Image.open(path_ref).convert('RGB')
        path_ref_label = path_ref.replace('.jpg', '.png')

        style_label = cv2.imread(path_ref_label,0)
        style_label = cv2.resize(style_label,(self.opt.load_size, self.opt.load_size))
        
        content_label = cv2.imread(label_path,0)
        content_label = cv2.resize(content_label,(self.opt.load_size, self.opt.load_size))

        common_label = [label for label in np.unique(style_label) if label in np.unique(content_label)]
        A_mask = self.get_masks(style_label, common_label)
        A2_mask = self.get_masks(content_label,common_label)

        params = get_params(self.opt, image_ref.size)
        transform_image = get_transform(self.opt, params)
        A = transform_image(image_ref)
        

        return {'label':'','A_bondary':torch.zeros((1,256,256)),'A2_bondary':torch.zeros((1,256,256)),'A': A, 'A_mask':A_mask, 'A2': torch.zeros((1,256,256)), 'A2_mask':A2_mask ,\
         'A_paths': path_ref, 'A2_paths': label_path}
    
    def get_label_tensor(self, path):
        label = Image.open(path)
        
        params1 = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params1, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc
        return label_tensor, params1

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.dataset_size
