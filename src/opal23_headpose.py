#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Alejandro Cobo'
__email__ = 'alejandro.cobo@upm.es'

import os
import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

from images_framework.src.alignment import Alignment
from images_framework.src.constants import Modes
from .irn_relu import IRN
from .task_headpose import PoseHead
from .utils import convert_rotation

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class Opal23Headpose(Alignment):
    """
    Object alignment using OPAL algorithm
    """
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.model = None
        self.gpu = None
        self.width = 128
        self.height = 128
        self.rotation_mode = 'euler'

    def parse_options(self, params):
        super().parse_options(params)
        import argparse
        parser = argparse.ArgumentParser(prog='Opal23Headpose', add_help=False)
        parser.add_argument('--rotation-mode', type=str, choices=['euler', 'quaternion', 'ortho6d'], default='euler',
                            help='Internal pose parameterization of the network (default: euler).')
        parser.add_argument('--gpu', dest='gpu', default=-1, type=int,
                            help='GPU ID (negative value indicates CPU).')
        args, unknown = parser.parse_known_args(params)
        print(parser.format_usage())
        self.rotation_mode = args.rotation_mode
        self.gpu = args.gpu if args.gpu >= 0 else 'cpu'

    def preprocess(self, image, bbox):
        bbox_width = bbox[2]-bbox[0]
        bbox_height = bbox[3]-bbox[1]
        # Squared bbox required
        max_size = max(bbox_width, bbox_height)
        shift = (float(max_size-bbox_width)/2.0, float(max_size-bbox_height)/2.0)
        bbox_squared = (bbox[0]-shift[0], bbox[1]-shift[1], bbox[2]+shift[0], bbox[3]+shift[1])
        # Enlarge bounding box
        bbox_scale = 0.3
        shift = max_size*bbox_scale
        bbox_enlarged = (bbox_squared[0]-shift, bbox_squared[1]-shift, bbox_squared[2]+shift, bbox_squared[3]+shift)
        # Project image
        T = np.zeros((2, 3), dtype=float)
        T[0, 0], T[0, 1], T[0, 2] = 1, 0, -bbox_enlarged[0]
        T[1, 0], T[1, 1], T[1, 2] = 0, 1, -bbox_enlarged[1]
        bbox_width = bbox_enlarged[2]-bbox_enlarged[0]
        bbox_height = bbox_enlarged[3]-bbox_enlarged[1]
        face_translated = cv2.warpAffine(image, T, (int(round(bbox_width)), int(round(bbox_height))))
        S = np.matrix([[self.width/bbox_width, 0, 0], [0, self.height/bbox_height, 0]], dtype=float)
        warped_image = cv2.warpAffine(face_translated, S, (self.width, self.height))

        # Image array (H x W x C) to tensor (C x H x W)
        tensor_image = to_tensor(warped_image).unsqueeze(0).to(self.gpu)
        return tensor_image

    def train(self, anns_train, anns_valid):
        print('Train model')

    def load(self, mode):
        print('Load model')
        setting = {
            'backbone':
            # t, c, n, s, l
            [[6, 1, 1, 1, -1],
             [6, 2, 1, 2, -1],
             [6, 4, 2, 2, -1],
             [6, 8, 3, 2, -1],
             [6, 8, 4, 2, -1]],
            'encoders':
            [[[6, 8, 1, 2, -1],
              [6, 8, 1, 2, -1],
              [6, 8, 1, 0, -1]]]}
        backbone = IRN(setting, inp_shape=128, in_planes=32)
        pose_head = PoseHead(backbone, rotation_mode=self.rotation_mode)
        self.model = torch.nn.Sequential(backbone, pose_head)
        model_file = self.path + 'data/' + self.database + '/' + self.database + '_' + self.rotation_mode + '.pth'
        state_dict = torch.load(model_file)
        self.model.load_state_dict(state_dict)
        self.model.train(mode is Modes.TRAIN)
        self.model.to(self.gpu)

    def process(self, ann, pred):
        from scipy.spatial.transform import Rotation
        for img_pred in pred.images:
            # Load image
            image = cv2.imread(img_pred.filename)
            for obj_pred in img_pred.objects:
                # Generate prediction
                tensor_image = self.preprocess(image, obj_pred.bb)
                # from matplotlib import pyplot as plt
                # aux = warped_image.copy()
                # plt.imshow(aux)
                # plt.show()
                with torch.set_grad_enabled(self.model.training):
                    output = self.model(tensor_image)
                    # headpose = self._pyr_to_ypr(output).detach().cpu().numpy()
                obj_pred.headpose = Rotation.from_euler('YZX', output[0], degrees=True).as_matrix()
                # obj_pred.headpose = Rotation.from_euler('ZYX', headpose[0], degrees=True).as_matrix()

    # def _pyr_to_ypr(self, pose):
    #     t_pose = torch.zeros(1, 3, dtype=torch.float32)
    #     t_pose_ypr = convert_rotation(t_pose, 'matrix', use_pyr_format=False)
    #     t_pose_pyr = convert_rotation(t_pose, 'matrix', use_pyr_format=True)
    #     delta = torch.bmm(t_pose_pyr.transpose(1, 2), t_pose_ypr)
    #
    #     pose = convert_rotation(pose, 'matrix', use_pyr_format=True)
    #     pose = torch.bmm(pose, delta)
    #
    #     return convert_rotation(pose, 'euler', use_pyr_format=False)
