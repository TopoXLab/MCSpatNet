import numpy as np;
from enum import Enum;
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNArchUtilsPyTorch:

    @staticmethod
    def crop_a_to_b(input_a, input_b):
        shape_a = input_a.size();
        shape_b = input_b.size();
        cropped = input_a[:, :, (shape_a[2]-shape_b[2])//2 : (shape_a[2]-shape_b[2])//2 + shape_b[2], (shape_a[3]-shape_b[3])//2 : (shape_a[3]-shape_b[3])//2 + shape_b[3]]

        return cropped;

