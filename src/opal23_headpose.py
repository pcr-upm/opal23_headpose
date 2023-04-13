#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Alejandro Cobo'
__email__ = 'alejandro.cobo@upm.es'

import os
import cv2
import numpy as np
from images_framework.src.alignment import Alignment
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)


class Opal23Headpose(Alignment):
    """
    Object alignment using OPAL algorithm
    """
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.model = None
        self.gpu = None
        self.width = 224
        self.height = 224

    def parse_options(self, params):
        super().parse_options(params)
        import argparse
        parser = argparse.ArgumentParser(prog='Opal23Headpose', add_help=False)
        parser.add_argument('--gpu', dest='gpu', type=int, action='append',
                            help='GPU ID (negative value indicates CPU).')
        args, unknown = parser.parse_known_args(params)
        print(parser.format_usage())
        self.gpu = args.gpu

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
        return warped_image

    def train(self, anns_train, anns_valid):
        print('Training model')

    def load(self, mode):
        print('Loading model')

    def process(self, ann, pred):
        for img_pred in pred.images:
            # Load image
            image = cv2.imread(img_pred.filename)
            for obj_pred in img_pred.objects:
                # Generate prediction
                warped_image = self.preprocess(image, obj_pred.bb)
                # from matplotlib import pyplot as plt
                # aux = warped_image.copy()
                # plt.imshow(aux)
                # plt.show()
                print('Predict image')
