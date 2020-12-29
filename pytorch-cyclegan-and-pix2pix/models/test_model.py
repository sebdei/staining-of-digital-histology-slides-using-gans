from .base_model import BaseModel
from . import networks
import cv2
from PIL import Image

import numpy as np
import torch
import cv2
import time


class TestModel(BaseModel):
    """ This TesteModel can be used to generate CycleGAN results for only one direction.
    This model will automatically set '--dataset_mode single', which only loads the images from one collection.

    See the test instruction for more details.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        The model can only be used during test time. It requires '--dataset_mode single'.
        You need to specify the network using the option '--model_suffix'.
        """
        assert not is_train, 'TestModel cannot be used during training time'
        parser.set_defaults(dataset_mode='single')
        parser.add_argument('--model_suffix', type=str, default='',
                            help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        self.visual_names = ['real', 'fake']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        # only generator is needed.
        self.model_names = ['G' + opt.model_suffix]
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see <BaseModel.load_networks>
        # store netG in self.
        setattr(self, 'netG' + opt.model_suffix, self.netG)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.

        We need to use 'single_dataset' dataset mode. It only load images from one domain.
        """
        self.real = input['A'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass."""
        self.fake = self.netG(self.real)  # G(real)

    def optimize_parameters(self):
        """No optimization for test model."""
        pass

    def compute_visuals(self):
        if (self.opt.color_space_mode == 'LAB'):
            self.real = self.lab2rgb(self.real)
            self.fake = self.lab2rgb(self.fake)
        elif (self.opt.color_space_mode == 'HSV'):
            self.real = self.hsv2rgb(self.real)
            self.fake = self.hsv2rgb(self.fake, False)

    def hsv2rgb(self, hsv, real=True):
        image = hsv.squeeze(0).permute((1, 2, 0)).cpu().numpy()

        h, s, v = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        h = h * 255
        s = s * 255
        v = v * 255

        image = cv2.merge((h, s, v))
        image = image.astype(np.uint8)

        rgb = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

        if (real):
            cv2.imwrite("result.jpg", rgb)

        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

        r = r / 255
        g = g / 255
        b = b / 255

        rgb = cv2.merge((b, g, r))

        tensor = torch.from_numpy(rgb)
        result = tensor.permute(2, 0, 1).unsqueeze(0)

        return result

    def lab2rgb(self, tensor):
        # TODO
        image = tensor.cpu()[0].permute(1, 2, 0).numpy()

        image = Image.fromarray((image * 255).astype(np.uint8))
        image = np.array(image)

        image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)

        result = torch.from_numpy(image)

        return result.permute(2, 0, 1).unsqueeze(0)
