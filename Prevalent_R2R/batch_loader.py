import glob
import os, argparse, json
import time, copy, random, pickle
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from utils import read_vocab, write_vocab, build_vocab, Tokenizer, SplitTokenizer, padding_idx, \
    timeSince, boolean_string, preprocess_get_pano_states, current_best
from env import R2RBatch, EnvBatch
from model import BertAddEncoder
from agent import Seq2SeqAgent
from feature import Feature
from pytorch_transformers import BertTokenizer
import pprint
import pdb




angle_inc = np.pi / 6.
feature_store = 'img_features/ResNet-152-imagenet.tsv'
panoramic = True

def build_viewpoint_loc_embedding(viewIndex):
    """
    Position embedding:
    heading 64D + elevation 64D
    1) heading: [sin(heading) for _ in range(1, 33)] +
                [cos(heading) for _ in range(1, 33)]
    2) elevation: [sin(elevation) for _ in range(1, 33)] +
                  [cos(elevation) for _ in range(1, 33)]
    """
    embedding = np.zeros((36, 128), np.float32)
    for absViewIndex in range(36):
        relViewIndex = (absViewIndex - viewIndex) % 12 + (absViewIndex // 12) * 12
        rel_heading = (relViewIndex % 12) * angle_inc
        rel_elevation = (relViewIndex // 12 - 1) * angle_inc
        embedding[absViewIndex,  0:32] = np.sin(rel_heading)
        embedding[absViewIndex, 32:64] = np.cos(rel_heading)
        embedding[absViewIndex, 64:96] = np.sin(rel_elevation)
        embedding[absViewIndex,   96:] = np.cos(rel_elevation)
    return embedding

def target_loc_embedding(absViewIndex, viewIndex):
    relViewIndex = (absViewIndex - viewIndex) % 12 + (absViewIndex // 12) * 12
    rel_heading = (relViewIndex % 12) * angle_inc
    rel_elevation = (relViewIndex // 12 - 1) * angle_inc
    return np.array([np.sin(rel_heading), np.cos(rel_heading), np.sin(rel_elevation), np.cos(rel_elevation)], np.float32)


# pre-compute all the 36 possible paranoram location embeddings
_static_loc_embeddings = [
    build_viewpoint_loc_embedding(viewIndex) for viewIndex in range(36)]



class SingleQuery(object):
    """
    A single data example for pre-training
    """
    def __init__(self, instr_id, scan, viewpoint, viewIndex, teacher_action, absViewIndex, rel_heading, rel_elevation):
        self.instr_id = instr_id
        self.scan = scan
        self.viewpoint = viewpoint
        self.viewIndex = viewIndex
        self.teacher_action = teacher_action
        self.absViewIndex = absViewIndex
        self.rel_heading = rel_heading
        self.rel_elevation = rel_elevation
        self.next = None

def new_mask_tokens(inputs, tokenizer, args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.ByteTensor), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).type(torch.ByteTensor)
    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).type(torch.ByteTensor) & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).type(torch.ByteTensor) & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels




def mask_tokens(inputs, tokenizer, args):

    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    labels = inputs.clone()

    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [val in tokenizer.all_special_ids for val in labels.tolist()]
    att_mask = [val == tokenizer.pad_token_id for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.uint8), value=0.0)
    #masked_indices = torch.bernoulli(torch.full(labels.shape, args.mlm_probability)).type(torch.ByteTensor)
    masked_indices = torch.bernoulli(probability_matrix).type(torch.ByteTensor)

    attention_mask = torch.full(labels.shape, 1).masked_fill_(torch.tensor(att_mask, dtype=torch.uint8), value=0)
    labels[~masked_indices] = -1  # We only compute loss on masked tokens


    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])

    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).type(torch.ByteTensor) & masked_indices

    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)



    # 10% of the time, we replace masked input tokens with random word

    #indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).type(torch.ByteTensor) & masked_indices & ~indices_replaced

    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)

    inputs[indices_random] = random_words[indices_random]



    # The rest of the time (10% of the time) we keep the masked input tokens unchanged

    return inputs, labels,attention_mask


              

class NavDataset(data.Dataset):

    def __init__(self, json_dirs, tok, img_path, panoramic,args):

        # read all json files and create a list of query data
        self.json_dirs = json_dirs  #  a list of json files
        self.tok = tok    # should be a lang, vision, action aware tokenizer ['VCLS', 'ACLS']
        self.mask_index = tok._convert_token_to_id(tok.mask_token)
        self.feature_store = Feature(img_path, panoramic)
        self.args = args
 

        self.data = []
        self.instr_refer = dict()  # instr_id : instr_encoding
        for json_dir in self.json_dirs:
            with open(json_dir) as f:
                current_trajs = json.load(f)
                for traj in current_trajs:
                    self.data += self.disentangle_path(traj)


    def __getitem__(self, index):
        # you must return data and label pair tensor
        query = self.data[index]
        output = self.getQuery(query)
        return {key:torch.tensor(value) for key,value in output.items()}


    def __len__(self):
        return len(self.data)


    def disentangle_path(self, traj):
        query = list()
        instr_id = traj['instr_id']
        instruction = traj['instr_encoding']
        self.instr_refer[instr_id] = instruction

        path = traj['path']
        actions = list(list(zip(*path))[2])[1:]+[0]
        action_emds = traj['teacher_action_emd']
        for t in range(len(path)):
            scan = path[t][0]
            viewpoint = path[t][1]
            viewIndex = path[t][2]
            teacher_action = actions[t]
            absViewIndex, rel_heading, rel_elevation = action_emds[t]

            current_query = SingleQuery(instr_id, scan, viewpoint, viewIndex, teacher_action, absViewIndex, rel_heading,rel_elevation)
            if t <= len(path) - 2:
                next_scan = path[t+1][0]
                next_viewpoint = path[t+1][1]
                next_viewIndex = path[t+1][2]
                next_teacher_action = actions[t+1]
                
                next_query = SingleQuery(instr_id, next_scan, next_viewpoint, next_viewIndex, next_teacher_action, next_absViewIndex, next_rel_heading, next_rel_elevation)
            else:
                next_query = current_query


            current_query.next = next_query
            query.append(current_query)  # a list of (SASA)

        return query

    def getQuery(self, query):
        # prepare text tensor
        output = dict()
        text_seq = torch.LongTensor(self.instr_refer[query.instr_id])
        masked_text_seq, masked_text_label, attention_mask = mask_tokens(text_seq, self.tok,self.args)
        output['masked_text_seq'] = masked_text_seq
        output['masked_text_label'] = masked_text_label
        output['lang_attention_mask'] = attention_mask

        # prepare vision tensor
        scan, viewpoint, viewindex = query.scan, query.viewpoint, query.viewIndex
        feature_all, feature_1 = self.feature_store.rollout(scan, viewpoint, viewindex)
        feature_with_loc_all = np.concatenate((feature_all, _static_loc_embeddings[viewindex]), axis=-1)
        output['feature_all'] = feature_with_loc_all
        output['target_loc'] = torch.tensor(target_loc_embedding(viewindex, query.teacher_action))

        # prepare action
        if query.absViewIndex == -1:
            teacher_action_embedding = np.zeros(feature_all.shape[-1] + 128, np.float32)
        else:
            teacher_view = feature_all[query.absViewIndex, :]
            loc_embedding = np.zeros(128, np.float32)
            loc_embedding[0:32] = np.sin(query.rel_heading)
            loc_embedding[32:64] = np.cos(query.rel_heading)
            loc_embedding[64:96] = np.sin(query.rel_elevation)
            loc_embedding[96:] = np.cos(query.rel_elevation)
            teacher_action_embedding = np.concatenate((teacher_view, loc_embedding))
        output['teacher'] = query.teacher_action
        output['teacher_embedding'] = teacher_action_embedding


        # prepare next step info
        nscan, nviewpoint, nviewindex = query.next.scan, query.next.viewpoint, query.next.viewIndex
        nfeature_all, nfeature_1 = self.feature_store.rollout(nscan, nviewpoint, nviewindex)
        nfeature_with_loc_all = np.concatenate((nfeature_all, _static_loc_embeddings[nviewindex]), axis=-1)
        output['next_feature_all'] = nfeature_with_loc_all

        if query.next.absViewIndex == -1:
            nteacher_action_embedding = np.zeros(feature_all.shape[-1] + 128, np.float32)
        else:
            nteacher_view = nfeature_all[query.next.absViewIndex, :]
            nloc_embedding = np.zeros(128, np.float32)
            nloc_embedding[0:32] = np.sin(query.next.rel_heading)
            nloc_embedding[32:64] = np.cos(query.next.rel_heading)
            nloc_embedding[64:96] = np.sin(query.next.rel_elevation)
            nloc_embedding[96:] = np.cos(query.next.rel_elevation)
            nteacher_action_embedding = np.concatenate((nteacher_view, nloc_embedding))
        output['next_teacher'] = query.next.teacher_action
        output['next_teacher_embedding'] = nteacher_action_embedding


        # prepare random next step info
        prob = np.random.random()
        if prob <= 0.5:
            output['isnext'] = 1
            output['next_img'] = output['next_feature_all']
        else:
            output['isnext'] = 0
            candidates = list(range(36))
            candidates.remove(nviewindex)
            fake_nviewindex = np.random.choice(candidates)
            ffeature_all, ffeature_1 = self.feature_store.rollout(nscan, nviewpoint, fake_nviewindex)
            ffeature_with_loc_all = np.concatenate((ffeature_all, _static_loc_embeddings[fake_nviewindex]), axis=-1)
            output['next_img'] = ffeature_with_loc_all
        
        # prepare vision matching
        if prob <= 0.5:
            output['match'] = torch.tensor(1.0)
        else:
            import random
            output['match'] = torch.tensor(0.0)
            scan_view_list = list(self.feature_store.features.keys())
            scan_view_list = [i for i in scan_view_list if scan not in i]
            fake_img = random.choice(scan_view_list)
            fake_scan, fake_viewpoint = fake_img.split("_")
            fake_img_feat, _ = self.feature_store.rollout(fake_scan, fake_viewpoint, viewindex)
            fake_img_feat_loc = np.concatenate((fake_img_feat, _static_loc_embeddings[viewindex]), axis=-1)
            output['feature_all'] = fake_img_feat_loc
            output['teacher'] = -1
        
        # prepare orientation matching
        output['orient_target'] = _static_loc_embeddings[viewindex][viewindex]
        return output


    def random_word(self, text_seq):
        tokens = text_seq.copy()   # already be [cls t1 t2 sep]
        output_label = []

        for i, token in enumerate(tokens):
            if i ==0 or i == len(tokens) - 1:
                output_label.append(0)
                continue
            prob = np.random.random()
            if prob < 0.15:
                prob /= 0.15

                output_label.append(tokens[i])

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.tok))

                # 10% randomly change token to current token
                else:
                    tokens[i] = tokens[i]   # just keep it


            else:
                tokens[i] = tokens[i]   # just keep it
                output_label.append(0)

        return tokens, output_label




def Test():
    parser = argparse.ArgumentParser()


    ## Required parameters

    parser.add_argument("--train_data_file", default=None, type=str, required=True,

                        help="The input training data file (a text file).")

    parser.add_argument("--output_dir", default=None, type=str, required=True,

                        help="The output directory where the model predictions and checkpoints will be written.")



    ## Other parameters

    parser.add_argument("--eval_data_file", default=None, type=str,

                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")



    parser.add_argument("--model_type", default="bert", type=str,

                        help="The model architecture to be fine-tuned.")

    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str,

                        help="The model checkpoint for weights initialization.")



    parser.add_argument("--mlm", action='store_true',

                        help="Train with masked-language modeling loss instead of language modeling.")

    parser.add_argument("--mlm_probability", type=float, default=0.15,

                        help="Ratio of tokens to mask for masked language modeling loss")



    parser.add_argument("--config_name", default="", type=str,

                        help="Optional pretrained config name or path if not the same as model_name_or_path")

    parser.add_argument("--tokenizer_name", default="", type=str,

                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--cache_dir", default="", type=str,

                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")

    parser.add_argument("--block_size", default=-1, type=int,

                        help="Optional input sequence length after tokenization."

                             "The training dataset will be truncated in block of this size for training."

                             "Default to the model max input length for single sentence inputs (take into account special tokens).")

    parser.add_argument("--do_train", action='store_true',

                        help="Whether to run training.")

    parser.add_argument("--do_eval", action='store_true',

                        help="Whether to run eval on the dev set.")

    parser.add_argument("--evaluate_during_training", action='store_true',

                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--do_lower_case", action='store_true',

                        help="Set this flag if you are using an uncased model.")



    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,

                        help="Batch size per GPU/CPU for training.")

    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,

                        help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,

                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--learning_rate", default=5e-5, type=float,

                        help="The initial learning rate for Adam.")

    parser.add_argument("--weight_decay", default=0.0, type=float,

                        help="Weight deay if we apply some.")

    parser.add_argument("--adam_epsilon", default=1e-8, type=float,

                        help="Epsilon for Adam optimizer.")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,

                        help="Max gradient norm.")

    parser.add_argument("--num_train_epochs", default=1.0, type=float,

                        help="Total number of training epochs to perform.")

    parser.add_argument("--max_steps", default=-1, type=int,

                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")

    parser.add_argument("--warmup_steps", default=0, type=int,

                        help="Linear warmup over warmup_steps.")



    parser.add_argument('--logging_steps', type=int, default=50,

                        help="Log every X updates steps.")

    parser.add_argument('--save_steps', type=int, default=50,

                        help="Save checkpoint every X updates steps.")

    parser.add_argument("--eval_all_checkpoints", action='store_true',

                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")

    parser.add_argument("--no_cuda", action='store_true',

                        help="Avoid using CUDA when available")

    parser.add_argument('--overwrite_output_dir', action='store_true',

                        help="Overwrite the content of the output directory")

    parser.add_argument('--overwrite_cache', action='store_true',

                        help="Overwrite the cached training and evaluation sets")

    parser.add_argument('--seed', type=int, default=42,

                        help="random seed for initialization")


    parser.add_argument('--fp16', action='store_true',

                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")

    parser.add_argument('--fp16_opt_level', type=str, default='O1',

                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."

                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--local_rank", type=int, default=-1,

                        help="For distributed training: local_rank")

    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")

    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    args = parser.parse_args()



    if args.model_type in ["bert", "roberta", "distilbert"] and not args.mlm:

        raise ValueError("BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm "

                         "flag (masked language modeling).")

    if args.eval_data_file is None and args.do_eval:

        raise ValueError("Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "

                         "or remove the --do_eval argument.")


    import glob
    jfiles = glob.glob("./collect_traj" + "/*.json")
    params = {'batch_size':20, 'shuffle': False, 'num_workers': 1}
    tok = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = NavDataset(jfiles, tok, feature_store, panoramic,args)
    print("you have loaded %d  time steps" % (len(dataset)))
    #pdb.set_trace()
    data_gen = data.DataLoader(dataset, **params)
    obj  = next(iter(data_gen))
    #print(obj.keys())







#Test()







