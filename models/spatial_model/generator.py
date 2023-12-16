import torch
from torch import nn
import torch.nn.functional as F
from .util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from .dense_motion import DenseMotionNetwork


class OcclusionAwareGenerator(nn.Module):
    """
    Generator that given source image and and keypoints try to transform image according to movement trajectories
    induced by keypoints. Generator follows Johnson architecture.
    """

    def __init__(self, num_kp, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False):
        super(OcclusionAwareGenerator, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, 
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None


        self.estimate_occlusion_map = estimate_occlusion_map

    def deform_input(self, inp, deformation,occlusion_map):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear')
            deformation = deformation.permute(0, 2, 3, 1)
        if occlusion_map is not None:
                occlusion_map = F.interpolate(occlusion_map, size=(h, w), mode='bilinear')
                return F.grid_sample(inp, deformation) * occlusion_map
        return F.grid_sample(inp, deformation, padding_mode = 'zeros')

    def forward(self, source, kp_driving, kp_source,source_mask = None,source_bondary = None):
       
        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        if kp_driving is not None and self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(source_image=torch.cat([source_mask,source],1), kp_driving=kp_driving,
                                                     kp_source=kp_source)
            output_dict['mask'] = dense_motion['mask']
            output_dict['sparse_deformed'] = dense_motion['sparse_deformed']

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = occlusion_map
            else:
                occlusion_map = None
            deformation = dense_motion['deformation']
            output_dict['deformation'] = deformation
            output_dict["sparse_motion"] = dense_motion['sparse_motion']
            
            output_dict["sparse_deformed"] = dense_motion['sparse_deformed']
            output_dict["deformed"] = self.deform_input(source, deformation,None)
            if source_mask is not None:
                output_dict["deformed_mask"] = self.deform_input(source_mask, deformation,None)
            
            if source_bondary is not None:
                output_dict["deformed_bondary"] = self.deform_input(source_bondary, deformation,None)

        return output_dict
