#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Alejandro Cobo'
__email__ = 'alejandro.cobo@upm.es'

import os
import cv2
import torch
import numpy as np
from PIL import Image
from images_framework.src.alignment import Alignment
from images_framework.src.constants import Modes
from .irn_relu import IRN
from .task_headpose import PoseHead

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
        self.device = None
        self.width = 128
        self.height = 128
        self.rotation_mode = None
        self.target_dist = None

    def parse_options(self, params):
        super().parse_options(params)
        import argparse
        parser = argparse.ArgumentParser(prog='Opal23Headpose', add_help=False)
        parser.add_argument('--gpu', dest='gpu', default=-1, type=int,
                            help='GPU ID (negative value indicates CPU).')
        parser.add_argument('--rotation-mode', type=str, choices=['euler', 'quaternion', '6d', '6d_opal'], default='euler',
                            help='Internal pose parameterization of the network (default: euler).')
        parser.add_argument('--target-dist', type=float, default=1.0,
                            help='Target distance for each test data set (default: 1.0).')
        args, unknown = parser.parse_known_args(params)
        print(parser.format_usage())
        self.device = args.gpu if args.gpu >= 0 else 'cpu'
        self.rotation_mode = args.rotation_mode
        self.target_dist = args.target_dist

    def preprocess(self, image, bbox):
        x_min, y_min, x_max, y_max = bbox
        w = x_max - x_min
        h = y_max - y_min
        # we enlarge the area taken around the bounding box
        # it is neccesary to change the botton left point of the bounding box
        # according to the previous enlargement. Note this will NOT be the new
        # bounding box!
        # We return square images, which is neccesary since
        # all the images must have the same size in order to form batches
        side = max(w, h) * self.target_dist
        x_min -= (side - w) / 2
        y_min -= (side - h) / 2

        # center of the enlarged bounding box
        x0, y0 = x_min + side / 2, y_min + side / 2
        # homothety factor, chosen so the new horizontal dimension will
        # coincide with new_size
        mu_x = self.width / side
        mu_y = self.height / side

        new_w = self.width
        new_h = self.height
        new_x0, new_y0 = new_w / 2, new_h / 2

        # dilatation + translation
        affine_transf = np.array([[mu_x, 0, new_x0 - mu_x * x0],
                                  [0, mu_y, new_y0 - mu_y * y0]])
        inv_affine_transf = self._get_inverse_transf(affine_transf)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        warped_image = image.transform((self.width, self.height), Image.AFFINE, inv_affine_transf.flatten())
        warped_image = np.array(warped_image)
        warped_image = cv2.cvtColor(warped_image, cv2.COLOR_RGB2BGR)
        return warped_image

    def _get_inverse_transf(self, affine_transf):
        A = affine_transf[0:2, 0:2]
        b = affine_transf[:, 2]

        inv_A = np.linalg.inv(A)  # we assume A invertible!

        inv_affine = np.zeros((2, 3))
        inv_affine[0:2, 0:2] = inv_A
        inv_affine[:, 2] = -inv_A.dot(b)

        return inv_affine

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
        self.model.to(self.device)

    def process(self, ann, pred):
        from scipy.spatial.transform import Rotation
        for img_pred in pred.images:
            # Load image
            image = cv2.imread(img_pred.filename)
            for obj_pred in img_pred.objects:
                # Generate prediction
                warped_image = self.preprocess(image, obj_pred.bb)

                # Image array (H x W x C) to tensor (1 x C x H x W)
                tensor_image = torch.tensor(warped_image, dtype=torch.float)
                tensor_image = tensor_image.permute(2, 0, 1) / 255
                tensor_image = tensor_image.unsqueeze(0).to(self.device)

                with torch.set_grad_enabled(self.model.training):
                    out = self.model(tensor_image)[0].detach().cpu().numpy()
                    if self.rotation_mode == 'euler':
                        if self.database == 'panoptic':
                            out *= 180.0  # [-1, 1] -> [-180º, 180º]
                        yaw, pitch, roll = out
                        obj_pred.headpose = Rotation.from_euler('XYZ', [pitch, yaw, roll], degrees=True).as_matrix()
                    elif self.rotation_mode == 'quaternion':
                        w, x, y, z = out
                        obj_pred.headpose = Rotation.from_quat([x, y, z, w]).as_matrix().T
                    elif self.rotation_mode in ('6d', '6d_opal'):
                        matrix = out.reshape(3, 3).T
                        obj_pred.headpose = matrix
                    else:
                        raise NotImplementedError(f"Rotation mode '{self.rotation_mode}' is not implemented. "
                                                  "Possible values: 'euler', 'quaternion', '6d' or '6d_opal'")
