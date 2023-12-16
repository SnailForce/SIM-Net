import torch
from options.test_options import TestOptions
from data import create_dataset
import os
import random
from PIL import Image
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np

from models import create_model

def get_im_tensor(path):
    img = Image.open(path).convert('RGB')
    img = np.array(img).astype(np.float32)
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda() / 255.0
    return img

def create_deformed_source_image(source_image, sparse_motions):
    num_kp = 10
    if len(source_image.shape) == 5:
        bs, _,_, h, w = source_image.shape
        source_repeat = source_image.clone()
    else:
        bs, _, h, w = source_image.shape
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1,num_kp + 1, 1, 1, 1, 1)
    _,_, h_old, w_old, _ = sparse_motions.shape
    if h_old != h or w_old != w:
        sparse_motions = sparse_motions[0].permute(0, 3, 1, 2)
        sparse_motions = F.interpolate(sparse_motions, size=(h, w), mode='bilinear')
        sparse_motions = sparse_motions.permute(0, 2, 3, 1).unsqueeze(0)

    source_repeat = source_repeat.view(bs * (num_kp + 1), -1, h, w)
    sparse_motions = sparse_motions.contiguous().view((bs * (num_kp + 1), h, w, -1))
    sparse_deformed = F.grid_sample(source_repeat, sparse_motions, padding_mode = 'reflection')
    sparse_deformed = sparse_deformed.view((bs, num_kp + 1 , -1, h, w))
    return sparse_deformed

def compute_result(model,img):
    sparse_motion = model.fake_A_dict['sparse_motion']
    
    transformed_image =  create_deformed_source_image(img.to(sparse_motion.device),sparse_motion)
    mask = F.interpolate(model.fake_A_dict['mask'], size=img.shape[2:], mode='bilinear')
    result = (transformed_image * mask .unsqueeze(2)).sum(dim = 1)
    return result

opt = TestOptions().parse()   # get training options
opt.config = os.path.join(opt.dataroot,opt.config)

model = create_model(opt)
opt.isTrain = False
model.setup(opt)
model.eval()

save_root = opt.results_dir
if not os.path.exists(save_root):
    os.makedirs(save_root)
dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

self = model
with torch.no_grad():
    for k,data in enumerate(dataset):
        model.set_input(data)
        A_ori = get_im_tensor(data["A_paths"][0])
        result = []
        for idx in range(model.real_A_mask.shape[1]):
            real_A_mask = self.real_A_mask[:,idx:idx + 1]
            real_A2_mask = self.real_A2_mask[:,idx:idx + 1]
            self.KPs['real_A'] = self.netG_KP(real_A_mask)
            self.KPs['real_A2'] = self.netG_KP(real_A2_mask)

            real_A = self.real_A * real_A_mask
            self.fake_A_dict = self.netG_A(content_image = self.real_A2,style_image = real_A ,content_kp = self.KPs['real_A2'],\
                style_kp = self.KPs['real_A'],content_mask = real_A2_mask,style_mask = real_A_mask)

            self.fake_A_mask = self.fake_A_dict['predict_mask']
            self.fake_A = self.fake_A_dict['prediction']
            
            tmp = compute_result(model,A_ori)
            fusion_mask = model.fake_A_mask * real_A2_mask
            bs,_,h,w = A_ori.shape
            fusion_mask = F.interpolate(fusion_mask, size=(h, w), mode='bilinear')
            if len(result) > 0:
                result.append(result[-1] * (1 - fusion_mask) + tmp * fusion_mask)
                fake_A_mask = torch.cat([fake_A_mask,model.fake_A_mask],1)
            else:
                result.append(tmp)
                fake_A_mask = model.fake_A_mask

        content_path_ = data['A2_paths'][0].split('/')[-1][:-4]
        style_path_ = data['A_paths'][0].split('/')[-1][:-4]

        vutils.save_image(result[-1][0], os.path.join(save_root ,style_path_ + '.jpg'),
                nrow=1, padding=0, normalize=False)