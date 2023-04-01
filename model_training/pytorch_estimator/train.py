################################################################################################################################################
######################################################## Import required modules ###############################################################
################################################################################################################################################

import argparse
import pprint
import json
import logging
import os
import sys
import pandas as pd
import random
import time
import glob
import numpy as np
from collections import defaultdict


import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader

from transformers import RobertaModel, RobertaConfig
from transformers import RobertaForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

################################################################################################################################################
###################################################### Parse input arguments ###################################################################
################################################################################################################################################

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_batch_size',
                        type=int,
                        default=64)

    parser.add_argument('--train_steps_per_epoch',
                        type=int,
                        default=64)

    parser.add_argument('--validation_batch_size',
                        type=int,
                        default=64)

    parser.add_argument('--validation_steps_per_epoch',
                        type=int,
                        default=64)

    parser.add_argument('--epochs',
                        type=int,
                        default=1)

    parser.add_argument('--freeze_bert_layer',
                        type=eval,
                        default=False)

    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.01)

    parser.add_argument('--momentum',
                        type=float,
                        default=0.5)

    parser.add_argument('--seed',
                        type=int,
                        default=42)

    parser.add_argument('--log_interval',
                        type=int,
                        default=100)

    parser.add_argument('--backend',
                        type=str,
                        default=None)

    parser.add_argument('--max_seq_length',
                        type=int,
                        default=128)

    parser.add_argument('--run_validation',
                        type=eval,
                        default=False)

    # Container environment

    parser.add_argument('--hosts',
                        type=list,
                        default=json.loads(os.environ['SM_HOSTS']))

    parser.add_argument('--current_host',
                        type=str,
                        default=os.environ['SM_CURRENT_HOST'])

    parser.add_argument('--model_dir',
                        type=str,
                        default=os.environ['SM_MODEL_DIR'])

    parser.add_argument('--train_data',
                        type=str,
                        default=os.environ['SM_CHANNEL_TRAIN'])

    parser.add_argument('--validation_data',
                        type=str,
                        default=os.environ['SM_CHANNEL_VALIDATION'])

    parser.add_argument('--output_dir',
                        type=str,
                        default=os.environ['SM_OUTPUT_DIR'])

    parser.add_argument('--num_gpus',
                        type=int,
                        default=os.environ['SM_NUM_GPUS'])

    # Debugger args

    parser.add_argument("--save-frequency",
                        type=int,
                        default=10,
                        help="frequency with which to save steps")

    parser.add_argument("--smdebug_path",
                        type=str,
                        help="output directory to save data in",
                        default="/opt/ml/output/tensors",)

    parser.add_argument("--hook-type",
                        type=str,
                        choices=["saveall", "module-input-output", "weights-bias-gradients"],
                        default="saveall",)

    return parser.parse_args()

################################################################################################################################################
########################################################### Tools and variables ################################################################
################################################################################################################################################
# Model name according to the PyTorch documentation:
# https://github.com/aws/sagemaker-pytorch-inference-toolkit/blob/6936c08581e26ff3bac26824b1e4946ec68ffc85/src/sagemaker_pytorch_serving_container/torchserve.py#L45
MODEL_NAME = 'model.pth'
# Hugging face list of models: https://huggingface.co/models
PRE_TRAINED_MODEL_NAME = 'roberta-base'




################################################################################################################################################
#################################################################### Main ######################################################################
################################################################################################################################################

if __name__ == '__main__':

    args = parse_args()




