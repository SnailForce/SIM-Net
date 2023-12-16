from numpy.core.fromnumeric import mean
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

from ..spatial_model.util import Hourglass, make_coordinate_grid, AntiAliasInterpolation2d,kp2gaussian

class DenseNetwork(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_kp, num_channels,
                 scale_factor=1, kp_variance=0.01):
        super(DenseNetwork, self).__init__()
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp + 1) * (num_channels + 1),
                                   max_features=max_features, num_blocks=num_blocks)

        self.mask = nn.Conv2d(self.hourglass.out_filters, num_kp + 1, kernel_size=(7, 7), padding=(3, 3))

        self.num_kp = num_kp
        self.scale_factor = scale_factor
        self.kp_variance = kp_variance

        self.transform_matrix = torch.tensor([[0.257, 0.564, 0.098],
                                 [-0.148, -0.291, 0.439],
                                 [0.439, -0.368, -0.071]])
        self.shift_matrix = torch.tensor([16.0, 128.0, 128.0]) / 255
        self.shift_matrix.type(self.transform_matrix.type())

        self.transform_matrix_inv = self.transform_matrix.inverse()

        self.shift_matrix_inv =  torch.matmul(self.transform_matrix_inv,self.shift_matrix)

        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)
        
        self.weight = 0.2
    
    def create_heatmap_representations(self, spatial_size, kp):
        heatmap = kp2gaussian(kp, spatial_size=spatial_size, kp_variance=self.kp_variance)

        #adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1]).type(heatmap.type())
        heatmap = torch.cat([zeros, heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)
        return heatmap

    
    def rgb2ycbcr(self,rgb_image):
        bs,c,h,w = rgb_image.shape
        rgb_image = rgb_image.view(bs,c,-1)
        transform_matrix = self.transform_matrix.type(rgb_image.type())
        shift_matrix = self.shift_matrix.type(rgb_image.type()) 
        ycbcr_image =  torch.matmul(transform_matrix.unsqueeze(0),rgb_image) + shift_matrix.unsqueeze(0).unsqueeze(-1)
        return ycbcr_image.reshape(bs,c,h,w)
    
    def ycbcr2rgb(self,ycbcr_image):
        ycbcr_image[:,1:] = torch.where(ycbcr_image[:,1:] > 0.94118,torch.ones_like(ycbcr_image[:,1:]) * 0.94118,ycbcr_image[:,1:])
        ycbcr_image[:,0] = torch.where(ycbcr_image[:,0] > 0.92157,torch.ones_like(ycbcr_image[:,0]) * 0.92157,ycbcr_image[:,0])
        ycbcr_image = torch.where(ycbcr_image < 0.06275,torch.ones_like(ycbcr_image) * 0.06275,ycbcr_image)

        bs,c,h,w = ycbcr_image.shape
        ycbcr_image = ycbcr_image.view(bs,c,-1)
        transform_matrix = self.transform_matrix_inv.type(ycbcr_image.type())
        shift_matrix = self.shift_matrix_inv.type(ycbcr_image.type()) 
        rgb_image =  torch.matmul(transform_matrix.unsqueeze(0),ycbcr_image) - shift_matrix.unsqueeze(0).unsqueeze(-1)
        rgb_image = torch.where(rgb_image > 1,torch.ones_like(rgb_image),rgb_image)
        rgb_image = torch.where(rgb_image < 0,torch.zeros_like(rgb_image),rgb_image)
        return rgb_image.reshape(bs,c,h,w)
    
    def rgb2lab(self,rgb_image):
        transform_matrix = torch.tensor([[0.3811, 0.5783, 0.0402],
                                 [0.1967, 0.7244, 0.0782],
                                 [0.0241, 0.1288, 0.8444]])
        transform_matrix = transform_matrix.type(rgb_image.type())
        bs,c,h,w = rgb_image.shape
        rgb_image = rgb_image.view(bs,c,-1)
        lab_image =  torch.matmul(transform_matrix.unsqueeze(0),rgb_image)
        lab_image = torch.log(lab_image)

        matrix_1 = torch.tensor([[1 / np.sqrt(3),0,0],
                                 [0,1 / np.sqrt(6),0],
                                 [0,0,1/np.sqrt(2)]])

        matrix_2 = torch.tensor([[1.0,1,1],
                                 [1,1,-2],
                                 [1,-1,0]])
        matrix = torch.matmul(matrix_1,matrix_2)
        matrix = matrix.type(rgb_image.type())
        return torch.matmul(matrix.unsqueeze(0),lab_image).reshape(bs,c,h,w)
    
    def lab2rgb(self,lab_image):
        transform_matrix = torch.tensor([[4.4679 ,3.5873 ,0.1193],
                                 [-1.2186, 2.3809, 0.1624],
                                 [0.0497, 0.2439, 1.2045]])
        transform_matrix = transform_matrix.type(lab_image.type())
        matrix_1 = torch.tensor([[ np.sqrt(3) / 3,0,0],
                                 [0,np.sqrt(6) / 6,0],
                                 [0,0,np.sqrt(2) / 2]])

        matrix_2 = torch.tensor([[1.0,1,1],
                                 [1,1,-1],
                                 [1,-2,0]])
        matrix = torch.matmul(matrix_2,matrix_1)
        matrix = matrix.type(lab_image.type())

        bs,c,h,w = lab_image.shape
        lab_image = lab_image.view(bs,c,-1)

        rgb_image= torch.matmul(matrix.unsqueeze(0),lab_image)
        rgb_image = torch.pow(10,rgb_image)

        return  torch.matmul(transform_matrix.unsqueeze(0),rgb_image).reshape(bs,c,h,w)
    
    def weighted_mean(self,values,weighted,dim=-1):
        return torch.sum(values * weighted,dim) / (torch.sum(weighted,dim) +  1e-8)
        
    def weighted_mean_std(self,values,weighted,dim=-1):
        mean = self.weighted_mean(values,weighted)
        return mean,torch.sqrt(self.weighted_mean((values - mean.unsqueeze(-1))**2,weighted,dim) + 1e-8)
    
    def create_code(self,source_image,source_heatmaps):
        bs, c, h, w = source_image.shape
        source_repeat = source_image.unsqueeze(1).repeat(1, self.num_kp,  1, 1,1)
        source_repeat_flatten = source_repeat.view((bs,self.num_kp,c,-1))
        source_heatmaps_flatten = source_heatmaps.view((bs,self.num_kp,1,-1))
        source_mean,source_std = self.weighted_mean_std(source_repeat_flatten,source_heatmaps_flatten)
        source_std = source_std.unsqueeze(-1).unsqueeze(-1) +  1e-8
        source_mean = source_mean.unsqueeze(-1).unsqueeze(-1)
        source_image_code = (source_repeat - source_mean) / source_std
        return source_image_code,source_mean,source_std


    def create_transformed_source_image(self, source_image, target_image, source_heatmaps,target_heatmaps,common_heaatmaps):
        """
        Eq 7. in the paper \hat{T}_{s<-d}(z)
        """
        bs, c, h, w = source_image.shape
        source_image = source_image.clone()
        source_image[:,:3] = self.rgb2ycbcr(source_image[:,:3])

        target_image = target_image.clone()
        target_image[:,:3] = self.rgb2ycbcr(target_image[:,:3])
        
    
        source_image_code,_,_ = self.create_code(source_image,source_heatmaps)
        target_image_code,target_mean,target_std = self.create_code(target_image,target_heatmaps)

        target_image_code = self.create_deformed_source_image(target_image_code,self.sparse_motion)
   

        source_weight = self.weight * 1000
        target_weight = (1 - self.weight) * 1000

        transformed_image_code = target_image_code.clone()
        transformed_image_code[:,:,0]= (source_image_code[:,:,0]  *  source_weight + target_image_code[:,:,0] * target_weight) / (source_weight + target_weight +  1e-8)
     
        transformed_image = transformed_image_code * target_std + target_mean

        return transformed_image
    
    def create_deformed_source_image(self, source_image, sparse_motions):
        """
        Eq 7. in the paper \hat{T}_{s<-d}(z)
        """
        if len(source_image.shape) == 5:
            bs, _,_, h, w = source_image.shape
            source_repeat = source_image.clone()
        else:
            bs, _, h, w = source_image.shape
            source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp + 1, 1, 1, 1, 1)
        source_repeat = source_repeat.view(bs * (self.num_kp + 1), -1, h, w)
        sparse_motions = sparse_motions.contiguous().view((bs * (self.num_kp + 1), h, w, -1))
        sparse_deformed = F.grid_sample(source_repeat, sparse_motions, padding_mode = 'zeros')
        sparse_deformed = sparse_deformed.view((bs, self.num_kp + 1 , -1, h, w))
        return sparse_deformed

    def create_sparse_motions(self, source_image, kp_driving, kp_source):
        bs, _, h, w = source_image.shape
        identity_grid = make_coordinate_grid((h, w), type=kp_source['value'].type())
        identity_grid = identity_grid.view(1, 1, h, w, 2).to(kp_driving['value'].device)
        coordinate_grid = identity_grid - kp_driving['value'].view(bs, self.num_kp, 1, 1, 2)
        if 'jacobian' in kp_driving:
            jacobian = torch.matmul(kp_source['jacobian'], torch.inverse(kp_driving['jacobian']))
            jacobian = jacobian.unsqueeze(-3).unsqueeze(-3)
            jacobian = jacobian.repeat(1, 1, h, w, 1, 1)
            coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
            coordinate_grid = coordinate_grid.squeeze(-1)

        driving_to_source = coordinate_grid + kp_source['value'].view(bs, self.num_kp, 1, 1, 2)

        #adding background feature
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1)
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)
        return sparse_motions

    def forward(self, content_image, style_image, content_kp,style_kp,content_mask,style_mask,style_bd = None):
    
        bs, _, h, w = style_image.shape
        self.sparse_motion = self.create_sparse_motions(style_image,content_kp,style_kp)
        deformed_mask = self.create_deformed_source_image(style_mask,self.sparse_motion) 
  
        
        out_dict = dict()
        out_dict['sparse_motion'] = self.sparse_motion

        out_dict['deformed_mask'] = deformed_mask
        common_mask = deformed_mask - content_mask.unsqueeze(1)
        out_dict['common_mask'] = common_mask
        self.style_kp = style_kp
        self.content_kp = content_kp
        transformed_image = self.create_deformed_source_image( style_image, self.sparse_motion)
        out_dict['transformed_image'] = transformed_image

        bs,_, _,h, w = transformed_image.shape
        heatmap_representation = self.create_heatmap_representations((h,w),content_kp)
        input = torch.cat([heatmap_representation,common_mask, transformed_image], dim=2).to(style_image.device)
        input = input.view(bs, -1, h, w)


        prediction = self.hourglass(input)

        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1)
        out_dict['mask'] = mask
        mask = F.interpolate(mask, size=(h, w), mode='bilinear')
        mask = mask.unsqueeze(2)
        transformed_image = transformed_image * mask
        transformed_image = transformed_image.sum(dim = 1)

        predict_mask = (deformed_mask * mask).sum(dim = 1)
        out_dict['predict_mask'] = predict_mask 
        out_dict['prediction'] = transformed_image 

        if style_bd is not None:
            deformed_bd = self.create_deformed_source_image(style_bd,self.sparse_motion) 
            out_dict['predict_bd'] =  (deformed_bd * mask).sum(dim = 1)

        return out_dict
