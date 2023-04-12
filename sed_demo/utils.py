#!/usr/bin python
# -*- coding:utf-8 -*-


"""
This module contains various utilities.

A substantial part has been borrowed from:
https://github.com/qiuqiangkong/audioset_tagging_cnn
"""


import os
import torch
import csv
import logging


# ##############################################################################
# # I/O
# ##############################################################################
def load_csv_labels(labels_csv_path):
    """
    Given the path to a 3-column CSV file ``(index, class_ID, class_name)``
    with comma-separated entries, this function ignores the first row and
    returns the triple ``(number_of_classes, IDs, names)``, in order of
    appearance.
    """
    with open(labels_csv_path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        lines = list(reader)
    idxs, ids, labels = zip(*lines[1:])
    num_classes = len(labels)
    return num_classes, ids, labels


# ##############################################################################
# # PYTORCH
# ##############################################################################
def move_data_to_device(x, device):
    """ """
    if "float" in str(x.dtype):
        x = torch.Tensor(x)
    elif "int" in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)


def interpolate(x, ratio):
    """Interpolate the prediction to compensate the downsampling operation in a
    CNN.

    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to upsample
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output, frames_num):
    """Pad framewise_output to the same length as input frames.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1:, :].repeat(
        1, frames_num - framewise_output.shape[1], 1
    )
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""

    return output


def do_mixup(x, mixup_lambda):
    """ """
    out = (
        x[0::2].transpose(0, -1) * mixup_lambda[0::2]
        + x[1::2].transpose(0, -1) * mixup_lambda[1::2]
    )
    return out.transpose(0, -1)


def print_args(name, opt):
    # Print argparser arguments
    LOGGER.info(
        colorstr(f"{name}: ") + ", ".join(f"{k}={v}" for k, v in vars(opt).items())
    )


def set_logging(name=None, verbose=True):
    # Sets level and returns logger
    rank = int(os.getenv("RANK", -1))  # rank in world for Multi-GPU trainings
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if (verbose and rank in (-1, 0)) else logging.WARNING,
    )
    return logging.getLogger(name)


LOGGER = set_logging(
    __name__
)  # define globally (used in train.py, val.py, detect.py, etc.)


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = (
        input if len(input) > 1 else ("blue", "bold", input[0])
    )  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]
