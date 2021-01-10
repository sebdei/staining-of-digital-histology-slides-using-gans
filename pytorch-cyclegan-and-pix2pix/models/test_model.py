from .base_model import BaseModel
from . import networks
import cv2
from PIL import Image
import numpy as np


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
        self.visual_names = ['real', 'real_B', 'fake']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        # only generator is needed.
        self.model_names = ['G' + opt.model_suffix]
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see <BaseModel.load_networks>
        # store netG in self.
        setattr(self, 'netG' + opt.model_suffix, self.netG)

    def evaluate(self):
        img_real_hsv = cv2.cvtColor(self.real_B, cv2.COLOR_BGR2HSV)
        img_fake_hsv = cv2.cvtColor(self.fake, cv2.COLOR_BGR2HSV)

        upper_brown = np.array([25, 255, 155], dtype=np.uint8)
        lower_brown = np.array([1, 1, 1], dtype=np.uint8)

        mask_real = cv2.inRange(img_real_hsv, lower_brown, upper_brown)
        mask_fake = cv2.inRange(img_fake_hsv, lower_brown, upper_brown)

        mask_real = mask_real.flatten()
        mask_fake = mask_fake.flatten()

        mask_real[mask_real > 0] = 1
        mask_fake[mask_fake > 0] = 1

        from sklearn.metrics import confusion_matrix
        confusion_matrix = confusion_matrix(
            mask_real, mask_fake, labels=[0, 1])

        TN, FP, FN, TP = confusion_matrix.ravel()

        return {'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN}

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.

        We need to use 'single_dataset' dataset mode. It only load images from one domain.
        """
        self.real = input['A'].to(self.device)
        self.real_B = input['B']
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass."""
        self.fake = self.netG(self.real)  # G(real)

    def optimize_parameters(self):
        """No optimization for test model."""
        pass
