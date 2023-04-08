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
        label2id={
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


def train_model(model,
                train_data_loader,
                df_train,
                validation_data_loader,
                df_val,
                args):

    loss_function = nn.CrossEntropyLoss()
    # it has to be a torch model
    optimiser = optim.Adam(params=model.parameters(), lr=args.learning_rate)
    if args.freeze_bert_laryer:
        print('Freezing BERT base layers...')
        for name, param in model.named_parameters():
            # If requires_grad is set to false, you are freezing the part of the model as no changes happen to its parameters
            if 'classifier' not in name:
                param.requires_grad = False
        print('Set classifier layers to `param.requires_grad=False`.')

    train_correct = 0
    train_total = 0

    for epoch in range(args.epochs):
        print('EPOCH -- {}'.format(epoch))

        for i, (sent, label) in enumerate(train_data_loader):
            if i < args.train_steps_per_epoch:
                model.train()
                optimiser.zero_grad()
                # squeeze(0) will squeeze the tensor to remove only the first dimension of size 1
                # squeeze() will squeeze all dimension of size 1
                sent = sent.squeeze(0)
                if torch.cuda.is_available():
                    sent = sent.cuda()
                    label = label.cuda()
                output = model(sent)[0]
                _, predicted = torch.max(output, 1)

                loss = loss_function(output, label)
                loss.backward()
                optimiser.step()

                if args.run_validation and i % args.validation_steps_per_epoch == 0:
                    print('RUNNING VALIDATION:')
                    correct = 0
                    total = 0
                    # When a model is in evaluation mode, certain layers and behaviors are changed to optimize the model for evaluation rather than training.
                    # For example, dropout layers will stop dropping out units, and batch normalization layers will use their running mean
                    # and variance statistics instead of computing them on the input batch.
                    model.eval()

                    for sent, label in validation_data_loader:
                        sent = sent.squeeze(0)
                        if torch.cuda.is_available():
                            # move to GPU memory
                            sent = sent.cuda()
                            label = label.cuda()
                        output = model(sent)[0]
                        _, predicted = torch.max(output, 1)

                        total += label.size(0)
                        correct += (predicted == label).sum().item()
                        accuracy = 100.00 * correct.numpy() / total
                        print('[epoch/step: {0}/{1}] val_loss: {2:.2f} - val_acc: {3:.2f}%'.format(epoch, i, loss.item(), accuracy))
            else:
                break
    print('TRAINING COMPLETED.')
    return model

    ################################################################################################################################################
#################################################################### Main ######################################################################
################################################################################################################################################

if __name__ == '__main__':

    # Parse args

    args = parse_args()
    print('Loaded arguments:')
    print(args)

    # Get environment variables
    env_var = os.environ
    print('Environment variables:')
    pprint.pprint(dict(env_var), width=1)

    # Check if distributed training
    is_distributed = len(args.hosts) > 1 and args.backend is not None

    print("Distributed training - {}".format(is_distributed))
    use_cuda = args.num_gpus > 0
    kwargs = {'num_workders': 1, 'pin_memory': True} if use_cuda else {}

    device = torch.device("cuda" if use_cuda else "cpu")
    # Initialize the distributed environment.

    # You can think of world as a group containing all the processes for your distributed training. Usually,
    # each GPU corresponds to one process. Processes in the world can communicate with each other, which is why you
    # can train your model distributedly and still get the correct gradient update. So world size is the number of
    # processes for your training, which is usually the number of GPUs you are using for distributed training.
    #
    # Rank is the unique ID given to a process, so that other processes know how to identify a particular process.
    # Local rank is the a unique local ID for processes running in a single node, this is where my view differs with
    # @zihaozhihao.
    #
    # Let's take a concrete example. Suppose we run our training in 2 servers (some articles also call them nodes)
    # and each server/node has 4 GPUs. The world size is 4*2=8. The ranks for the processes will be [0, 1, 2, 3, 4,
    # 5, 6, 7]. In each node, the local rank will be [0, 1, 2, 3].

    if is_distributed:
        world_size = len(args.hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ['RANK'] = str(host_rank)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
        print('Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
            args.backend, dist.get_world_size()) + 'Current host rank is {}. Number of gpus: {}'.format(
            dist.get_rank(), args.num_gpus))

        # Set the seed for generating random numbers
        torch.manual_seed(args.seed)
        if use_cuda:
            torch.cuda.manual_seed(args.seed)

        # Instantiate model
        config = None
        model = None

        successful_download = False
        retries = 0

    while (retries < 5 and not successful_download):
        try:
            # Configure model
            config = configure_model()
            model = RobertaForSequenceClassification.from_pretrained(
                'roberta-base',
                config=config
            )
            # pre-trained model was by GPU
            model.to(device)
            successful_download = True
            print('Sucessfully downloaded after {} retries.'.format(retries))

        except:
            retries += 1
            random_sleep = random.randit(1, 30)
            print('Retry #{}.  Sleeping for {} seconds'.format(retries, random_sleep))
            time.sleep(random_sleep)

    if not model:
        print('Not properly initialised')


    # Create data loaders

    train_data_loader, df_train = create_data_loader(args.train_data_path, args.batch_size)
    val_data_loader, df_val = create_data_loader(args.validation_data_path, args.batch_size)

    print("Processes {}/{} ({:.0f}%) of train data".format(
        len(train_data_loader.sampler), len(train_data_loader.dataset),
        100. * len(train_data_loader.sampler) / len(train_data_loader.dataset)
    ))

    print("Processes {}/{} ({:.0f}%) of validation data".format(
        len(val_data_loader.sampler), len(val_data_loader.dataset),
        100. * len(val_data_loader.sampler) / len(val_data_loader.dataset)
    ))

    print('model_dir: {}'.format(args.model_dir))
    print('model summary: {}'.format(model))

    print('model_dir: {}'.format(args.model_dir))
    print('model summary: {}'.format(model))

    callbacks = []
    initial_epoch_number = 0

    # Start training
    model = train_model(
        model,
        train_data_loader,
        df_train,
        val_data_loader,
        df_val,
        args
    )

    save_transformer_model(model, args.model_dir)
    save_pytorch_model(model, args.model_dir)

    # Prepare for inference which will be used in deployment
    # You will need three files for it: inference.py, requirements.txt, config.json

    inference_path  = os.path.join(args.model_dir, "code/")
    os.makedirs(inference_path, exist_ok=True)
    os.system("cp inference.py {}".format(inference_path))
    os.system("cp requirements.txt {}".format(inference_path))
    os.system("cp config.json {}".format(inference_path))















