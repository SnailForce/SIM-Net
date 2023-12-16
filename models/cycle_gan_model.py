import torch
import itertools
import yaml
import torch.nn.functional as F

from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torch.autograd import Variable
from collections import OrderedDict
from .spatial_model.model import Transform, Vgg19
from .spatial_model.generator import OcclusionAwareGenerator
from .spatial_model.keypoint_detector import KPDetector
from .spatial_model.util import make_coordinate_grid

from .texture_model.heatmap_detector import DenseNetwork
import matplotlib.pyplot as plt


class IoULoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth)/(union + smooth)

        return 1 - IoU


class CycleGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(
            no_dropout=True)  # default CycleGAN did not use dropout
        return parser

    def get_current_visuals(self):
        def draw_kp(image, kp):
            colormap = plt.get_cmap('gist_rainbow')
            pos = (kp['value'][0] * 128 + 128).type(torch.int)
            for i in range(pos.shape[0]):
                color = colormap(i / pos.shape[0])
                for dim in range(image.shape[1]):
                    image[0, dim, pos[i, 1]-3:pos[i, 1] + 3,
                          pos[i, 0] - 3:pos[i, 0] + 3] = color[dim]
            return image

        def draw_mask(image, num=None):
            if num is None:
                num = image.shape[1]
            colormap = plt.get_cmap('gist_rainbow')
            _, _, h, w = image.shape
            mask = torch.zeros((1, 3, h, w)).to(image.device)
            for i in range(num):
                color = colormap(i / num)
                for dim in range(mask.shape[1]):
                    mask[0, dim] = mask[0, dim] + color[dim] * image[0, i]
            return mask

        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                im = getattr(self, name)
                if name[-4:] == "mask":
                    visual_ret[name] = im[:, :1].repeat(1, 3, 1, 1)
                else:
                    if im.shape[1] <= 2:
                        im = im[:, :1].repeat(1, 3, 1, 1)
                    else:
                        im = im[:, :3]
                    visual_ret[name] = im.clone()
                if self.KPs.get(name):
                    draw_kp(visual_ret[name], self.KPs[name])

        return visual_ret

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # 正向传播时：开启自动求导的异常侦测
        torch.autograd.set_detect_anomaly(True)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['dis_C', 'gen_C', 'cycle_A',
                           'dis_maskA', 'dis_bondaryA', 'dis_content_C']
        self.loss_names += ['cycle_C', 'dis_mask', 'dis_A',
                            'dis_bondary', 'dis_content', 'dis_C_L1']

        self.visual_names = ['real_A', 'real_A_mask', 'fake_motion_A', 'fake_motion_A_mask', 'fake_motion_A_bondary',
                             'rec_motion_A', 'real_A2', 'real_A2_mask', 'fake_A', 'fake_A_mask', 'fake_A_bondary']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_KP', 'G_A', 'G_C']
        else:  # during test time, only load Gs
            self.model_names = ['G_KP', 'G_A', 'G_C']

        # define G_C
        with open(opt.config) as f:
            config = yaml.safe_load(f)

        self.netG_KP = KPDetector(**config['model_params']['kp_detector_params'],
                                  **config['model_params']['common_params'])
        self.netG_A = DenseNetwork(**config['model_params']['generator_params']['dense_motion_params'],
                                   **config['model_params']['common_params'])

        self.netG_C = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                              **config['model_params']['common_params'])

        self.train_params = config['train_params']

        self.loss_weights = self.train_params['loss_weights']

        self.vgg = Vgg19()
        if len(self.gpu_ids) > 0:
            assert (torch.cuda.is_available())
            self.netG_KP.to(self.gpu_ids[0])
            self.netG_A.to(self.gpu_ids[0])
            self.netG_C.to(self.gpu_ids[0])
            self.vgg.to(self.gpu_ids[0])
            self.netG_C = torch.nn.DataParallel(
                self.netG_C, self.gpu_ids)  # multi-GPUs
            self.vgg = torch.nn.DataParallel(self.vgg, self.gpu_ids)
            self.netG_A = torch.nn.DataParallel(
                self.netG_A, self.gpu_ids)  # multi-GPUs
            self.netG_KP = torch.nn.DataParallel(
                self.netG_KP, self.gpu_ids)  # multi-GPUs
            self.KPs = {}

        if self.isTrain:
            # create image buffer to store previously generated images
            self.fake_A_pool = ImagePool(opt.pool_size)

            # define loss functions
            # define GAN loss.
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.MSELoss()
            self.criterionMask = IoULoss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G_A = torch.optim.Adam(itertools.chain(
                self.netG_A.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_C = torch.optim.Adam(self.netG_C.parameters(
            ), lr=self.train_params['lr_generator'], betas=(opt.beta1, 0.999))
            self.optimizer_G_KP = torch.optim.Adam(self.netG_KP.parameters(
            ), lr=self.train_params['lr_kp_detector'], betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G_A)
            self.optimizers.append(self.optimizer_G_C)
            self.optimizers.append(self.optimizer_G_KP)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        self.real_A = input['A'].to(self.device)
        self.real_A_mask = input['A_mask'].to(self.device)
        self.real_A2 = input['A2'].to(self.device)
        self.real_A2_mask = input['A2_mask'].to(self.device)

        self.real_A_bondary = input['A_bondary'].to(self.device)
        self.real_A2_bondary = input['A2_bondary'].to(self.device)

        self.label = input['label'][0]

    def flip(self, x, dim):
        xsize = x.size()
        dim = x.dim() + dim if dim < 0 else dim
        x = x.view(-1, *xsize[dim:])
        x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
                                                                     -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
        return x.view(xsize).detach()

    def combine(self, A, B, dim=1):
        return A
        # return torch.cat([A,B],dim)

    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    def styleLoss(self, x, y, mask):
        loss_weights = [40, 30, 20, 10, 5]
        x_vgg = self.vgg(x * mask)
        y_vgg = self.vgg(y)
        value_total = 0
        for i, weight in enumerate(loss_weights):
            x_vgg[i] *= F.interpolate(mask,
                                      size=x_vgg[i].shape[2:], mode='nearest')
            value = self.criterionCycle(self.gram_matrix(
                x_vgg[i]), self.gram_matrix(y_vgg[i].detach()))
            value_total += weight * value
        return value_total.mean() * 1e6

    def contentLoss(self, x, y, mask):
        loss_weights = [0, 0, 0, 1, 2]
        x_vgg = self.vgg(x * mask)
        y_vgg = self.vgg(y)
        value_total = 0
        for i, weight in enumerate(loss_weights):
            if weight < 0.001:
                continue
            value = self.criterionCycle(x_vgg[i], y_vgg[i].detach())
            value_total += weight * value
        return value_total.mean() / 10

    def perceptualLoss(self, x, y):
        loss_weights = [0.5, 0.5, 0.5, 0.5, 0.5]
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        value_total = 0
        for i, weight in enumerate(loss_weights):
            value = self.criterionCycle(x_vgg[i], y_vgg[i].detach())
            value_total += weight * value
        return value_total.mean() / 2

    def detach_kp(self, kp):
        return {key: value.detach() for key, value in kp.items()}

    def forward(self, mode=""):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        self.KPs['real_A'] = self.netG_KP(self.real_A_mask)
        self.KPs['real_A2'] = self.netG_KP(self.real_A2_mask)

        if mode == "motion":
            self.fake_motion_A_dict = self.netG_C(
                source=self.real_A, kp_driving=self.KPs['real_A2'], kp_source=self.KPs['real_A'], source_mask=self.real_A_mask, source_bondary=self.real_A_bondary)
            self.fake_motion_A = self.fake_motion_A_dict['deformed']
            self.fake_motion_A_mask = self.fake_motion_A_dict['deformed_mask']
            self.KPs['fake_motion_A'] = self.netG_KP(self.fake_motion_A_mask)
            self.fake_motion_A_bondary = self.fake_motion_A_dict['deformed_bondary']

        else:
            self.fake_A_dict = self.netG_A(content_image=self.real_A2, style_image=self.real_A, content_kp=self.KPs['real_A2'],
                                           style_kp=self.KPs['real_A'], content_mask=self.real_A2_mask, style_mask=self.real_A_mask, style_bd=self.real_A_bondary)

            self.fake_A_mask = self.fake_A_dict['predict_mask']
            self.fake_A_bondary = self.fake_A_dict['predict_bd']
            self.fake_A = self.fake_A_dict['prediction']

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5 * 2
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_A, fake_A)

    def backward_G_KP(self, mode="G_C"):
        transform = Transform(
            self.real_A.shape[0], **self.train_params['transform_params'])
        transformed_frame = transform.transform_frame(self.real_A)
        transformed_frame_mask = transform.transform_frame(self.real_A_mask)
        transformed_kp = self.netG_KP(transformed_frame_mask)

        loss_values = {}
        kp_driving = self.KPs['real_A']
        # Value loss part
        if self.loss_weights['equivariance_value'] != 0:
            value = torch.abs(
                kp_driving['value'] - transform.warp_coordinates(transformed_kp['value'])).mean()
            loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value
        self.loss_gen_C = loss_values['equivariance_value'].mean()

        jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp['value']),
                                            transformed_kp["jacobian"])

        normed_driving = torch.inverse(kp_driving["jacobian"])
        normed_transformed = jacobian_transformed
        value = torch.matmul(normed_driving, normed_transformed)

        eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())

        value = torch.abs(eye - value).mean()
        loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value
        self.loss_gen_C = (self.loss_gen_C +
                           loss_values['equivariance_jacobian'].mean())
        self.loss_gen = self.loss_gen_C

        self.loss_gen.backward(retain_graph=True)

    def backward_G_C(self):

        self.rec_motion_A_dict = self.netG_C(source=self.fake_motion_A, kp_driving=self.KPs['real_A'],
                                             kp_source=self.KPs['fake_motion_A'], source_mask=self.fake_motion_A_mask)
        self.rec_motion_A = self.rec_motion_A_dict['deformed']

        self.loss_dis_bondary = self.criterionMask(
            self.fake_motion_A_bondary, self.real_A2_bondary)

        self.loss_dis_mask = self.criterionMask(
            self.fake_motion_A_mask, self.real_A2_mask) * 4
        self.loss_dis_content_C = self.contentLoss(
            self.fake_motion_A, self.real_A2, self.real_A2_mask) / 5
        self.loss_cycle_C = self.criterionIdt(
            self.rec_motion_A, self.real_A) * 6

        self.loss_G_C = self.loss_cycle_C + self.loss_dis_mask + \
            self.loss_dis_bondary + self.loss_dis_content_C
        self.loss_G_C.backward(retain_graph=True)

    def backward_G(self):
        self.loss_dis_maskA = self.criterionMask(
            self.fake_A_mask, self.real_A2_mask) * 10
        self.loss_dis_bondaryA = self.criterionMask(
            self.fake_A_bondary, self.real_A2_bondary)

        # self.loss_dis_C = self.styleLoss(
        #     self.fake_A, self.fake_motion_A.detach(), self.real_A2_mask) / 10
        self.loss_dis_C = self.styleLoss(
            self.fake_A, self.real_A, self.real_A2_mask) / 10
        
        self.loss_dis_content = self.contentLoss(
            self.fake_A, self.real_A2, self.real_A2_mask) / 2
        self.rec_A = torch.zeros_like(self.real_A)
        self.loss_G = self.loss_dis_maskA + self.loss_dis_bondaryA + \
            self.loss_dis_content + self.loss_dis_C
        assert torch.isnan(self.loss_G).sum() == 0, print(self.loss_G)

        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""

        # forward
        # compute fake images and reconstruction images.
        self.forward("motion")

        # 反向传播时：在求导时开启侦测
        with torch.autograd.detect_anomaly():

            self.optimizer_G_KP.zero_grad()
            self.optimizer_G_C.zero_grad()
            self.backward_G_KP()
            self.backward_G_C()
            self.optimizer_G_C.step()
            self.optimizer_G_KP.step()

            # forward
            # compute fake images and reconstruction images.
            self.forward()
            self.optimizer_G_KP.zero_grad()
            self.optimizer_G_A.zero_grad()
            self.backward_G_KP()
            self.backward_G()             # calculate gradients for G_A and G_B
            self.optimizer_G_A.step()
            self.optimizer_G_KP.step()
