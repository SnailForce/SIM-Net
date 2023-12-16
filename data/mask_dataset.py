import os,yaml
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset_by_name
import torch.nn.functional as F

from PIL import Image,ImageFilter
import random
import numpy as np
import collections
import torch

class MaskDataset(BaseDataset):
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.root_dir = os.path.join(opt.dataroot,'class')

        self.phase = opt.phase
        self.img_mask_dict = {}
        self.img_names = {}
        self.data_size = {}
        self.label_list = os.listdir(os.path.join(self.root_dir))
        # The shape of the human face is more complex, so increase the training ratio
        if "face" in self.label_list:
            self.label_list.append("face")
        for label in self.label_list:
            label_dir = os.path.join(self.root_dir,label,"images")
            with open(os.path.join(self.root_dir,label,'list.yaml')) as f:
                self.img_mask_dict[label] = yaml.safe_load(f)
            self.img_names[label] = list(self.img_mask_dict[label].keys())
            self.data_size[label] = len(self.img_names[label])
        self.transform = get_transform(self.opt)
        self.A_name = None
        self.A2_name = None
        self.mask_name = None

    
    def get_mask(self,A_name,label,mask_name,root_dir):
        A_name = A_name[:-4] + mask_name + A_name[-4:]
        path = os.path.join(root_dir,label, 'mask',A_name) 
        if os.path.exists(path):
            mask = self.transform(Image.open(path).convert('L'))
        else:
            mask = torch.zeros(1,256,256)
            
        ct_path = os.path.join(root_dir,label, 'ct',A_name) 
        if os.path.exists(ct_path):
            ct = self.transform(Image.open(ct_path).convert('L'))
        else:
            ct = torch.zeros(1,256,256)
        return mask, ct
    
    def get_img(self,A_name,label,root_dir):
        path = os.path.join(root_dir,label,'images',A_name) 
        assert os.path.exists(path),'%s not exists.'%path
        return self.transform(Image.open(path).convert('RGB'))        




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
        label = random.choice(self.label_list)
        
        A_name = self.A_name if self.A_name is not None \
                            else self.img_names[label][index % self.data_size[label]]  # make sure index is within then range
        A = self.get_img(A_name,label,self.root_dir)

        A2_name = self.A2_name if self.A2_name is not None \
                                else self.img_names[label][random.randint(1,self.data_size[label]) % self.data_size[label]]
    
        A2 = self.get_img(A2_name,label,self.root_dir) if self.phase != 'test' else torch.zeros(1,256,256)

        if self.mask_name is not None:
            mask_name = self.mask_name
        else:
            common_mask = [mask for mask in self.img_mask_dict[label][A_name] if mask in self.img_mask_dict[label][A2_name]]
            assert len(common_mask) != 0, 'no common mask in same category %s,%s'%(A_name,A2_name)
            mask_name = random.choice(common_mask)
            
            if len(common_mask) == 0:
                print(A_name,A2_name)
                A_mask = torch.zeros(1,256,256)
                A_bondary = torch.zeros(1,256,256)
                A2_mask = torch.zeros(1,256,256)
                A2_bondary = torch.zeros(1,256,256)
    
        A_mask,A_bondary = self.get_mask(A_name,label,mask_name,self.root_dir)
        A2_mask,A2_bondary = self.get_mask(A2_name,label,mask_name,self.root_dir)
        
        A = A * A_mask[0:1] 
        A2 = A2 * A2_mask[0:1] 

        return {'label':label,'A_bondary':A_bondary,'A2_bondary':A2_bondary,'A': A, 'A_mask':A_mask, 'A2': A2, 'A2_mask':A2_mask ,\
         'A_paths': A_name, 'A2_paths': A2_name}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return sum(self.data_size.values())
