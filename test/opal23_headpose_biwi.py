#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Alejandro Cobo'
__email__ = 'alejandro.cobo@upm.es'

import os
import sys
sys.path.append(os.getcwd())
import cv2
import copy
import argparse
import numpy as np
from tqdm import tqdm
# from scipy.spatial.transform import Rotation
from images_framework.src.datasets import Database
from images_framework.src.constants import Modes
from images_framework.src.composite import Composite
from images_framework.alignment.opal23_headpose.src.opal23_headpose import Opal23Headpose
import images_framework.alignment.opal23_headpose.test.utils as utils


def parse_options():
    """
    Parse options from command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--anns-file', '-a', dest='anns_file', required=True,
                        help='Ground truth annotations file.')
    parser.add_argument('--data-dir', '-d', dest='data_dir', required=True,
                        help='Images directory path.')
    args, unknown = parser.parse_known_args()
    print(parser.format_usage())
    anns_file = args.anns_file
    data_dir = args.data_dir
    return unknown, anns_file, data_dir


def load_annotations(anns_file, data_dir=None):
    """
    Load ground truth annotations according to each database.
    """
    print('Open annotations file: ' + str(anns_file))
    if os.path.isfile(anns_file):
        pos = anns_file.rfind('/') + 1
        # path = anns_file[:pos]
        file = anns_file[pos:]
        db = file[:file.find('_ann')]
        datasets = [subclass().get_names() for subclass in Database.__subclasses__()]
        with open(anns_file, 'r', encoding='utf-8') as ifs:
            lines = ifs.readlines()
            anns = []
            for i in tqdm(range(len(lines)), file=sys.stdout):
                parts = lines[i].strip().split(';')
                if parts[0] == '@':
                    db = parts[1]
                if parts[0] == '#' or parts[0] == '@':
                    continue
                idx = [datasets.index(subset) for subset in datasets if db in subset]
                if len(idx) != 1:
                    raise ValueError('Database does not exist')
                seq = Database.__subclasses__()[idx[0]]().load_filename(data_dir, db, lines[i])
                if len(seq.images) == 0:
                    continue
                anns.append(seq)
        ifs.close()
    else:
        raise ValueError('Annotations file does not exist')
    return anns


def main():
    """
    Opal 2023 using Biwi test set.
    """
    print('OpenCV ' + cv2.__version__)
    unknown, anns_file, data_dir = parse_options()

    # Load vision components
    composite = Composite()
    sr = Opal23Headpose('images_framework/alignment/opal23_headpose/')
    sr.target_dist = 1.0
    composite.add(sr)
    composite.parse_options(unknown)
    anns = load_annotations(anns_file, data_dir)
    # Process frame and show results
    print('Process annotations in ' + Modes.TEST.name + ' mode ...')
    composite.load(Modes.TEST)
    ann_matrix_array = []
    pred_matrix_array = []
    sequences = [[] for _ in range(24)]
    for i in tqdm(range(len(anns)), file=sys.stdout):
        pred = copy.deepcopy(anns[i])
        composite.process(anns[i], pred)
        for idx in range(len(pred.images[0].objects)):
            ann_rotm = utils.convert_rotation(anns[i].images[0].objects[idx].headpose, 'matrix', use_pyr_format=True)
            pred_rotm = utils.convert_rotation(pred.images[0].objects[idx].headpose, 'matrix', use_pyr_format=True)
            ann_matrix_array.append(ann_rotm)
            pred_matrix_array.append(pred_rotm)

            seq_id = int(anns[i].images[0].filename.split('/')[-2])
            sequences[seq_id - 1].append(i + idx)

    # Prediction alignment
    ann_matrix_array = np.array(ann_matrix_array)
    pred_matrix_array = np.array(pred_matrix_array)
    for seq in sequences:
        pred_matrix_array[seq] = utils.align_predictions(ann_matrix_array[seq], pred_matrix_array[seq])

    # Compute Euler angles from rotation matrices
    ann_euler_array = [utils.convert_rotation(rotm, 'euler', use_pyr_format=True) for rotm in ann_matrix_array]
    ann_euler_array = np.array(ann_euler_array)

    pred_euler_array = [utils.convert_rotation(rotm, 'euler', use_pyr_format=True) for rotm in pred_matrix_array]
    pred_euler_array = np.array(pred_euler_array)

    # Compute MAE and GE metrics
    mae = utils.compute_mae(np.array(ann_euler_array), np.array(pred_euler_array), use_pyr_format=True)
    mae = np.mean(mae, axis=0)
    ge = utils.compute_ge(np.array(ann_matrix_array), np.array(pred_matrix_array))
    print('MAE (yaw, pitch, roll): ' + str(mae))
    print('GE: ' + str(np.mean(ge)))
    print('End of opal23_headpose_biwi')


if __name__ == '__main__':
    main()
