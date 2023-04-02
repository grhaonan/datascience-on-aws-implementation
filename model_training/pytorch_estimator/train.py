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



def create_list_input_files(path):
    input_files = glob.glob('{}/*.tsv'.format(path))
    print(input_files)
    return input_files


def save_transformer_model(model, model_dir):
    path = '{}/transformer'.format(model_dir, exist_ok=True)
    os.makedirs(path, exist_ok=True)
    print('Saving model to: {}'.format(path))
    model.save_pretrained(path)

def save_pytorch_model(model, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    print('Saving model to: {}'.format(model_dir))
    save_path = os.path.join(model_dir, MODEL_NAME)
    # There are more than one approach to save a PyTorch model and below is the suggested one:
    torch.save(model.state_dict(), save_path)


################################################################################################################################################
########################################################### Configure the model ################################################################
################################################################################################################################################

def configure_model():
    classes = [-1, 0, 1]

    config = RobertaConfig.from_pretrained(
        PRE_TRAINED_MODEL_NAME,
        num_labels=len(classes),
        id2label={
            0: -1,
            1: 0,
            2: 1
        },
        label2id ={
            -1: 0,
            0: 1,
            1: 2
        }
    )
    config.output_attentions = True
    return config

################################################################################################################################################
####################################################### PyTorch Dataset and DataLoader #########################################################
################################################################################################################################################
# PyTorch dataset retrieves the dataset’s features and labels one sample at a time
# Create a custom Dataset class for the reviews

class ReviewDataset(Dataset):

    def __init__(self, input_ids_list, label_id_list):
        self.input_ids_list = input_ids_list
        self.label_id_list = label_id_list

    def __len__(self):
        return len(self.label_id_list)

    def __getitem__(self, item):
        input_ids = json.loads(self.label_id_list[item])
        label_id = self.label_id_list[item]

        input_ids_tensor = torch.LongTensor(input_ids)
        label_id_tensor = torch.tensor(label_id, dtype=torch.long)

        return input_ids_tensor, label_id_tensor

# PyTorch DataLoader helps to to organise the input training data in “minibatches” and reshuffle the data at every epoch
# It takes Dataset as an input

def create_data_loader(path, batch_size):
    print("Get data loader")

    df = pd.DataFrame(columns=['inputs_ids', 'label_id'])

    input_files = create_list_input_files(path)

    for file in input_files:
        df_temp = pd.read_csv(file,
                              sep='\t',
                              usecols=['inputs_ids', 'label_id'])
        df = df.append(df_temp)

    ds = ReviewDataset(
        input_ids_list=df.inputs_ids.to_numpy(),
        label_id_list=df.label_id.to_numpy()
    )

    return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
    ), df



################################################################################################################################################
################################################################ Train model ###################################################################
################################################################################################################################################



################################################################################################################################################
#################################################################### Main ######################################################################
################################################################################################################################################

if __name__ == '__main__':

    args = parse_args()




