from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import functools
import multiprocessing

from datetime import datetime
from time import gmtime, strftime, sleep

import pandas as pd
import argparse
import subprocess
import sys
import os
import re
import collections
import json
import csv
import glob
from pathlib import Path
import time
import boto3

subprocess.check_call([sys.executable, "-m", "conda", "install", "-c", "pytorch", "pytorch==1.6.0", "-y"])

subprocess.check_call([sys.executable, "-m", "conda", "install", "-c", "conda-forge", "transformers==3.5.1", "-y"])
from transformers import RobertaTokenizer

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sagemaker==2.35.0'])
import sagemaker

from sagemaker.session import Session
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.feature_definition import (
    FeatureDefinition,
    FeatureTypeEnum,
)


################################################################################################################################################
###################################################### Setup environmental variables ###########################################################
################################################################################################################################################

region = os.environ['AWS_DEFAULT_REGION']
sts = boto3.Session(region_name=region).client(service_name='sts', region_name=region)
iam = boto3.Session(region_name=region).client(service_name='iam', region_name=region)
featurestore_runtime = boto3.Session(region_name=region).client(service_name='sagemaker-featurestore-runtime', region_name=region)
sm = boto3.Session(region_name=region).client(service_name='sagemaker', region_name=region)

caller_identity = sts.get_caller_identity()
assumed_role_arn = caller_identity['Arn']
assumed_role_name = assumed_role_arn.split('/')[-2]
get_role_response = iam.get_role(RoleName=assumed_role_name)
role = get_role_response['Role']['Arn']
bucket = sagemaker.Session().default_bucket()

sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=region),
                                      sagemaker_client=sm,
                                      sagemaker_featurestore_runtime_client=featurestore_runtime)

# list of sentiment classes: -1 - negative; 0 - neutral; 1 - positive
classes = [-1, 0, 1]

# label IDs of the target class (sentiment) setup as a dictionary
classes_map = {
    -1: 0,
    0: 1,
    1: 2
}

# tokenization model
PRE_TRAINED_MODEL_NAME = 'roberta-base'

# create the tokenizer to use based on pre trained model
tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

################################################################################################################################################
################################################################# Tools ########################################################################
################################################################################################################################################
# functions which can be considered as tools

def cast_object_to_string(data_frame):
    for label in data_frame.columns:
        if data_frame.dtypes[label] == 'object':
            data_frame[label] = data_frame[label].astype("str").astype("string")
    return data_frame


def wait_for_feature_group_creation_complete(feature_group):
    try:
        status = feature_group.description().get("FeatureGroupStatus")
        print('Feature Group status: {}'.format(status))
        while status == "Creating":
            print('Waiting for Feature Group to be Created')
            time.sleep(10)
            status = feature_group.description().get("FeatureGroupStatus")
            print('Feature Group status: {}'.format(status))
        if status != "Created":
            raise RuntimeError("Error creating Feature Group")
    except:
        print('No feature group created yet')


def list_arg(raw_value):
    return raw_value.split(',')

def to_sentiment(star_rating):
    if star_rating in (1, 2):
        return -1
    if star_rating == 3:
        return 0
    if star_rating in (4, 5):
        return 1

################################################################################################################################################
################################################### Create or load Feature Group ###############################################################
################################################################################################################################################

def create_of_load_feature_group(prefix, feature_group_name):
    feature_defination = [
        FeatureDefinition(feature_name="review_id", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="date", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="sentiment", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="label_id", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name='input_ids', feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name='review_body', feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name='split_type', feature_type=FeatureTypeEnum.STRING)
    ]

# setup the Feature Group
feture_group = FeatureGroup(
    name=feature_group_name,
    feture_definition=feture_definition,
    sagemaker_session=sagemaker_session,
)

print('Feature Group: {}'.format(feature_group))

try:
    print('Waiting for existing Feature Group to become available if it is being created by another instance in our cluster...')
    wait_for_feature_group_creation_complete(feature_group)
except Exception as e:
    print('Before CREATE FG wait exeption: {}'.format(e))


try:
    record_identifier_feature_name = 'review_id'
    event_time_feature_name = 'date'
    feature_group.create(
        s3_uri=f"s3://{bucket}/{prefix}",
        record_identifier_name=record_identifier_feature_name,
        event_time_feature_name=event_time_feature_name,
        role_arn=role,
        enable_online_store=False
    )
    print('Creating Feature Group. Completed.')

    print('Waiting for new Feature Group to become available...')
    wait_for_feature_group_creation_complete(feature_group)
    print('Feature Group available.')
    feture_group.describe()

except Exception as e:
    print('Exception: {}'.format(e))
    print('Creating Feature Group with role {}...'.format(role))


return feature_group


################################################################################################################################################
################################################### Tokenization of the reviews ################################################################
################################################################################################################################################
def convert_to_bert_input_ids(review, max_seq_len):
    encoded_review = tokenizer.encode_plus(
        review,
        max_length=max_seq_len,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    return encoded_review['input_ids'].flatten().tolist()



################################################################################################################################################
###################################################### Parse input arguments ###################################################################
################################################################################################################################################
def parse_args():
    # Unlike SageMaker training jobs (which have `SM_HOSTS` and `SM_CURRENT_HOST` env vars), processing jobs to need to parse the resource config file directly
    resconfig = {}
    try:
        with open('/opt/ml/config/resourceconfig.json', 'r') as cfgfile:
            resconfig = json.load(cfgfile)
    except FileNotFoundError:
        print('/opt/ml/config/resourceconfig.json not found. current_host is unknown.')
        pass # Ignore

    # Local testing with CLI args
    parser = argparse.ArgumentParser(description='Process')

    parser.add_argument('--hosts', type=list_arg,
                        default=resconfig.get('hosts', ['unknown']),
                        help='Comma-separated list of host names running the job'
                        )
    parser.add_argument('--current-host', type=str,
                        default=resconfig.get('current_host', 'unknown'),
                        help='Name of this host running the job'
                        )
    parser.add_argument('--input-data', type=str,
                        default='/opt/ml/processing/input/data',
                        )
    parser.add_argument('--output-data', type=str,
                        default='/opt/ml/processing/output',
                        )
    parser.add_argument('--train-split-percentage', type=float,
                        default=0.90,
                        )
    parser.add_argument('--validation-split-percentage', type=float,
                        default=0.05,
                        )
    parser.add_argument('--test-split-percentage', type=float,
                        default=0.05,
                        )
    parser.add_argument('--balance-dataset', type=eval,
                        default=True
                        )
    parser.add_argument('--max-seq-length', type=int,
                        default=128
                        )
    parser.add_argument('--feature-store-offline-prefix', type=str,
                        default=None,
                        )
    parser.add_argument('--feature-group-name', type=str,
                        default=None,
                        )

    return parser.parse_args()


################################################################################################################################################
####################################################### Processing functions ###################################################################
################################################################################################################################################
def _preprocess_file(file,
                     balance_dataset,
                     max_seq_length,
                     prefix,
                     feature_group_name):
    print('file {}'.format(file))
    print('balance_dataset {}'.format(balance_dataset))
    print('max_seq_length {}'.format(max_seq_length))
    print('prefix {}'.format(prefix))
    print('feature_group_name {}'.format(feature_group_name))

    # Create a feature group
    # the Feature Group that was set in the main notebook cannot be passed here - it will be used later in the notebook for other purposes
    # you need to create a Feature Group with the same Feature Definitions within the processing job

    feature_group = create_of_load_feature_group(prefix, feature_group_name)
    # Note that if your file has multiple extensions, .stem will only remove the last extension
    filename_without_extension = Path(Path(file).stem).stem

    df = pd.read_csv(file, index_col=0)

    df.isna().values.any()
    df = df.reset_index(drop=True)
    print('Shape of dataframe {}'.format(df.shape))

    df['Sentiment'] = df['Rating'].apply(lambda star_rating: to_sentiment(star_rating = star_rating))
    print('Shape of dataframe with sentiment {}'.format(df.shape))

    df['label_id'] = df['Sentiment'].apply(lambda sentiment: classes_map[sentiment])

    df['input_ids'] = df['Review Text'].apply(lambda review: convert_to_bert_input_ids(review, max_seq_length))
    print('df[input_ids] after calling convert_to_bert_input_ids: {}'.format(df['input_ids']))

    # convert the index into a review_id
    df.reset_index(inplace=True)
    df = df.renae(columns = {'index': 'review_id',
                             'Review Text': 'review_body'})
    # drop all columns except the following:
    df = df[['review_id', 'sentiment', 'label_id', 'input_ids', 'review_body']]
    df = df.reset_index(drop=True)

    print('Shape of dataframe after dropping columns {}'.format(df.shape))
    # balance the dataset by sentiment down to the minority class
    if balance_dataset:
        df_unbalanced_group_by = df.groupby('sentiment')
        df_balanced = df_unbalanced_group_by.apply(lambda x: x.sample(df_unbalanced_group_by.size().min()).reset_index(drop=True))

################################################################################################################################################
#################################################################### Main ######################################################################
################################################################################################################################################


if __name__ == "__main__":
    args = parse_args()

