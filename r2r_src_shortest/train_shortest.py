import torch

import os
import time
import json
import random
import numpy as np
from collections import defaultdict

from utils import read_vocab, write_vocab, build_vocab, padding_idx, timeSince, read_img_features, print_progress, Tokenizer
import utils
from env import R2RBatch
from shortest_agent import ShortestCollectAgent
from eval import Evaluation
from param import args


import warnings
warnings.filterwarnings("ignore")
from tensorboardX import SummaryWriter

from vlnbert.vlnbert_init import get_tokenizer

log_dir = '/VL/space/zhan1624/VLN-speaker/snap/%s' % args.name
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
PLACE365_FEATURES = '/egr/research-hlr/joslin/img_features/ResNet-152-places365.tsv'
result_path = "/VL/space/zhan1624/VLN-speaker/result/"
experiment_time = time.strftime("%Y%m%d-%H%M%S", time.gmtime())

TRAIN_VOCAB = 'data/train_vocab.txt'
TRAINVAL_VOCAB = 'data/trainval_vocab.txt'

if args.features == 'imagenet':
    features = IMAGENET_FEATURES
elif args.features == 'places365':
    features = PLACE365_FEATURES

feedback_method = args.feedback  # teacher or sample

print(args); print('')




''' train the listener '''
def train(train_env, tok, n_iters, log_every=2000, val_envs={}, aug_env=None):
    listner = ShortestCollectAgent(train_env, "/VL/space/zhan1624/VLN-speaker/fake_data/", args.maxAction, name=train_env.name)
    interval = 130
    listner.env = train_env
    listner.train(interval, feedback=feedback_method)  # Train interval iters
    listner.collect()

       

def setup():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    random.seed(0)
    np.random.seed(0)

def train_val(test_only=False):
    ''' Train on the training set, and validate on seen and unseen splits. '''
    setup()
    tok = get_tokenizer(args)

    feat_dict = None
    
    navigable_locs_path = "/VL/space/zhan1624/VLN-speaker/generated_data/navigable_locs.json"
    with open(navigable_locs_path, 'r') as f:
        nav_graphs = json.load(f)
        
    train_env = R2RBatch(feat_dict, nav_graphs, batch_size=args.batchSize, splits=['train'], tokenizer=tok)

    train(train_env, tok, args.iters)

def train_val_augment(test_only=False):
    """
    Train the listener with the augmented data
    """
    setup()

    # Create a batch training environment that will also preprocess 
    tok = get_tokenizer(args)

    # Load the env img features
    feat_dict = read_img_features(features, test_only=test_only)
    if test_only:
        featurized_scans = None
        val_env_names = ['val_train_seen']
    else:
        featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
        val_env_names = ['val_seen', 'val_unseen']

    # Load the augmentation data
    aug_path = args.aug
    # Create the training environment
    train_env = R2RBatch(feat_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok)
    aug_env   = R2RBatch(feat_dict, batch_size=args.batchSize, splits=[aug_path], tokenizer=tok, name='aug')

    # Setup the validation data
    val_envs = {split: (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split], tokenizer=tok),
                Evaluation([split], featurized_scans, tok))
                for split in val_env_names}

    # Start training
    train(train_env, tok, args.iters, val_envs=val_envs, aug_env=aug_env)

if __name__ == "__main__":
    if args.train in ['listener', 'validlistener', 'speaker']:
        train_val(test_only=args.test_only)
    else:
        assert False