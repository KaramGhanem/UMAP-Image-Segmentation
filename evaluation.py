from subprocess import call
import argparse
from PIL import Image
import glob
import os
import torch
import re
import numpy as np


def iou(pred, mask, label=1) -> float:
    """ A function that compute the Intersection over Union (IoU)
    for the pixels with a given label between the prediction and the mask
    """
    assert pred.shape == mask.shape
    pred_label = (pred == label).type(torch.int)
    mask_label = (mask == label).type(torch.int)

    intersection = pred_label * mask_label
    union = (pred_label + mask_label - intersection)

    iscore = intersection.sum().numpy()
    uscore = union.sum().numpy()

    assert uscore != 0, 'the label {} is not present in the pred and the mask'.format(label)

    return iscore / uscore


def compute_metrics(metric, pred_path: str, mask_path: str) -> float:
    """A function that applies the metric to compare
    each image and mask located in the pred_path and mask_path respectively.
    """
    pred_file_list = sorted(glob.glob(pred_path + "/*"))

    score = 0.

    for pred_file in pred_file_list:

        filename = re.split('[/.]', pred_file)[-2]

        mask = torch.from_numpy(np.array(Image.open(os.path.join(mask_path, filename)+'.bmp'), dtype=np.uint8))/255
        pred = np.array(Image.open(pred_file), dtype=np.uint8)/255
        mask_pred = np.zeros_like(mask, dtype=np.uint8)
        mask_pred[pred > 0.5] = 1

        score += metric(torch.from_numpy(mask_pred), mask)
    return score/len(pred_file_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to benchmark a model on different test sets')
    parser.add_argument('--script_path', type=str, default='./infer.py',
                        help='path of the script to perform inference with the model')
    parser.add_argument('--testset_path', type=str, default='./test/',
                        help='path of the test sets')
    parser.add_argument('--prediction_path', type=str, default='./test/',
                        help='path of the directory where the predictions are saved')
    args = parser.parse_args()

    test_dirs = os.listdir(args.testset_path)
    if not test_dirs:
        test_dirs.append('.')

    score_dict = {}

    script_dir = os.path.dirname(args.script_path)
    script_name = os.path.basename(args.script_path)
    interp_name = './'
    if os.path.splitext(script_name)[1] == '.py':
        interp_name = 'python'
    elif os.path.splitext(script_name)[1] == '.sh':
        interp_name = 'sh'

    for test_name in test_dirs:
        pred_path = os.path.join(args.prediction_path, test_name)
        if not os.path.isdir(pred_path):
            current_wd = os.getcwd()
            os.chdir(script_dir)
            print('Generating the predictions for {} set'.format(test_name))
            call([interp_name, script_name, os.path.join(args.testset_path, test_name, 'img'), pred_path])
            print('Generation done.')
            os.chdir(current_wd)

        test_mask_path = os.path.join(args.testset_path, test_name, 'mask')
        score = compute_metrics(iou, pred_path=pred_path, mask_path=test_mask_path)
        score_dict[test_name] = score

    if len(test_dirs) == 1:
        print('mean IOU on {}: {:.4f}'.format(test_name, score_dict[test_name]))
    else:
        for test_name in test_dirs:
            print('mean IOU on {}: {:.4f}'.format(test_name, score_dict[test_name]))
