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
from images_framework.src.datasets import Database
from images_framework.src.constants import Modes
from images_framework.src.composite import Composite
from images_framework.alignment.opal23_headpose.src.opal23_headpose import Opal23Headpose
from images_framework.alignment.opal23_headpose.test.evaluator import Evaluator


def parse_options():
    """
    Parse options from command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--anns-file', '-a', dest='anns_file', required=True,
                        help='Ground truth annotations file.')
    args, unknown = parser.parse_known_args()
    print(parser.format_usage())
    anns_file = args.anns_file
    return unknown, anns_file


def load_annotations(anns_file):
    """
    Load ground truth annotations according to each database.
    """
    print('Open annotations file: ' + str(anns_file))
    if os.path.isfile(anns_file):
        pos = anns_file.rfind('/') + 1
        path = anns_file[:pos]
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
                seq = Database.__subclasses__()[idx[0]]().load_filename(path, db, lines[i])
                if len(seq.images) == 0:
                    continue
                anns.append(seq)
        ifs.close()
    else:
        raise ValueError('Annotations file does not exist')
    return anns


def main():
    """
    Opal 2023 using CMU Panoptic test set.
    """
    print('OpenCV ' + cv2.__version__)
    unknown, anns_file = parse_options()

    # Load vision components
    composite = Composite()
    sr = Opal23Headpose('images_framework/alignment/opal23_headpose/')
    composite.add(sr)
    composite.parse_options(unknown)
    anns = load_annotations(anns_file)

    # Process frame and show results
    print('Process annotations in ' + Modes.TEST.name + ' mode ...')
    composite.load(Modes.TEST)
    sr.target_dist = 1.0  # Bounding box not enlarged
    anno_matrix_array, pred_matrix_array = [], []
    for i in tqdm(range(len(anns)), file=sys.stdout):
        pred = copy.deepcopy(anns[i])
        composite.process(anns[i], pred)
        for idx in range(len(pred.images[0].objects)):
            anno_matrix_array.append(anns[i].images[0].objects[idx].headpose)
            pred_matrix_array.append(pred.images[0].objects[idx].headpose)

    # Compute MAE and GE metrics
    evaluator = Evaluator(np.array(anno_matrix_array), np.array(pred_matrix_array))
    mae = np.mean(evaluator.compute_mae(), axis=0)
    ge = np.mean(evaluator.compute_ge())
    print('MAE (yaw, pitch, roll): ' + str(mae))
    print('MAE: ' + str(np.mean(mae)))
    print('GE: ' + str(ge))
    print('End of opal23_headpose_panoptic')


if __name__ == '__main__':
    main()
