from __future__ import absolute_import, division, print_function

import argparse
from ast import arg

import glob

import logging
import math

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pickle

import random

import pdb

import numpy as np

import torch

import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler

from torch.utils.data.distributed import DistributedSampler

from tensorboardX import SummaryWriter
from feature import Feature
from batch_loader import NavDataset

from tqdm import tqdm, trange
from pretrain_class import HugAddActionPreTrain, DicAddActionPreTrain



from pytorch_transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule,

                                  BertConfig, BertForMaskedLM, BertTokenizer)


logger = logging.getLogger(__name__)
tb_writer = SummaryWriter('/egr/research-hlr/joslin/pretrain/action_new/snap/')

MODEL_CLASSES = {

    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),

}


class TextDataset(Dataset):

    def __init__(self, tokenizer, file_path='train', block_size=512):

        assert os.path.isfile(file_path)

        directory, filename = os.path.split(file_path)

        cached_features_file = os.path.join(directory, 'cached_lm_{}_{}'.format(block_size, filename))



        if os.path.exists(cached_features_file):

            logger.info("Loading features from cached file %s", cached_features_file)

            with open(cached_features_file, 'rb') as handle:

                self.examples = pickle.load(handle)

        else:

            logger.info("Creating features from dataset file at %s", directory)


            self.examples = []

            with open(file_path, encoding="utf-8") as f:

                text = f.read()



            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
            for i in range(0, len(tokenized_text)-block_size+1, block_size): # Truncate in block of block_size

                self.examples.append(tokenizer.add_special_tokens_single_sequence(tokenized_text[i:i+block_size]))

            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)

            # If your dataset is small, first you should loook for a bigger one :-) and second you

            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)

            with open(cached_features_file, 'wb') as handle:

                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):

        return len(self.examples)

    def __getitem__(self, item):

        return torch.tensor(self.examples[item])



def load_and_cache_examples(args, tokenizer, evaluate=False):

    dataset = TextDataset(tokenizer, file_path=args.eval_data_file if evaluate else args.train_data_file, block_size=args.block_size)

    return dataset




def set_seed(args):

    random.seed(args.seed)

    np.random.seed(args.seed)

    torch.manual_seed(args.seed)

    if args.n_gpu > 0:

        torch.cuda.manual_seed_all(args.seed)



def mask_tokens(inputs, tokenizer, args):

    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    labels = inputs.clone()

    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

    masked_indices = torch.bernoulli(torch.full(labels.shape, args.mlm_probability)).bool()

    labels[~masked_indices] = -1  # We only compute loss on masked tokens



    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])

    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices

    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)



    # 10% of the time, we replace masked input tokens with random word

    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced

    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)

    inputs[indices_random] = random_words[indices_random]



    # The rest of the time (10% of the time) we keep the masked input tokens unchanged

    return inputs, labels



def train(args, train_dataset, model, tokenizer):

    """ Train the model """

    #if args.local_rank in [-1, 0]:

    #    tb_writer = SummaryWriter()



    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)



    if args.max_steps > 0:

        t_total = args.max_steps

        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1

    else:

        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    data_length = len(train_dataloader)

    # Prepare optimizer and schedule (linear warmup and decay)

    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [

        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},

        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}

        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:

        try:

            from apex import amp

        except ImportError:

            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)



    # multi-gpu training (should be after apex fp16 initialization)

    # if args.n_gpu > 1:

    #     model = torch.nn.DataParallel(model)



    # Distributed training (should be after apex fp16 initialization)

    if args.local_rank != -1:

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],

                                                          output_device=args.local_rank,

                                                          find_unused_parameters=True)



    # Train!

    logger.info("***** Running training *****")

    logger.info("  Num examples = %d", len(train_dataset))

    logger.info("  Num Epochs = %d", args.num_train_epochs)

    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)

    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",

                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))

    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0

    tr_loss, logging_loss = 0.0, 0.0
    total_mask_loss, total_action_loss = 0.0, 0.0

    model.zero_grad()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=True)

    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)

    for epo in train_iterator:

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)

        for step, batch in enumerate(epoch_iterator):

            #inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
   
            inputs, labels = batch['masked_text_seq'], batch['masked_text_label']
            lang_attention_mask = batch['lang_attention_mask']
            img_feats = batch['feature_all']
            #vis_mask = batch['img_mask']
            
            labels = labels.to(args.device)
            inputs = inputs.to(args.device)
            img_feats = img_feats.to(args.device)
            lang_attention_mask = lang_attention_mask.to(args.device)
            #vis_mask = vis_mask.to(args.device)
               
            model.train()

            #outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
            if args.include_next:
                actions = batch['teacher']
                actions = actions.to(args.device)
                ### vision match
                if args.vision_match:
                    match_label = batch['match'].to(args.device)
                else:
                    match_label = None

                ### orient match
                if args.orient_match:
                    orient_label = batch['target_loc'].to(args.device)
                else:
                    orient_label = None
                
                outputs = model(inputs,labels,actions,img_feats,lang_mask=lang_attention_mask, 
                                match_label=match_label, orient_label=orient_label)
            else:
                outputs = model(inputs,labels, None, img_feats,lang_mask=lang_attention_mask)


            loss, mask_loss, next_loss = outputs  # model outputs are always tuple in transformers (see doc)
            
        
            if args.n_gpu > 1:

                loss = loss.mean()  # mean() to average on multi-gpu parallel training

                # YZ
                mask_loss = mask_loss.mean()
                next_loss = next_loss.mean()

            if args.gradient_accumulation_steps > 1:

                loss = loss / args.gradient_accumulation_steps

            if args.fp16:

                with amp.scale_loss(loss, optimizer) as scaled_loss:

                    scaled_loss.backward()

            else:

                loss.backward()

            tr_loss += loss.item()

            # YZ
            total_mask_loss += mask_loss.item()
            total_action_loss += next_loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:

                if args.fp16:

                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)

                else:

                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()

                scheduler.step()  # Update learning rate schedule

                model.zero_grad()

                global_step += 1

                #if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % data_length == 0:

                    # Save model checkpoint

                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))

                    if not os.path.exists(output_dir):

                        os.makedirs(output_dir)

                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training

                    model_to_save.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))

                    logger.info("Saving model checkpoint to %s", output_dir)

            if step % 100 == 0:
                print("\n")
                print("PROGRESS: {}%".format(round((epo * len(train_dataloader) + step) * 100 / t_total, 4)))
                print("EVALERR: {}%".format(tr_loss / (global_step)))
                # YZ
                tb_writer.add_scalar('Loss/all_loss', tr_loss / global_step, (epo * len(train_dataloader) + step) * 100 / t_total)
                tb_writer.add_scalar('Loss/mask_loss', total_mask_loss / global_step, (epo * len(train_dataloader) + step) * 100 / t_total)
                tb_writer.add_scalar('Loss/action_loss', total_action_loss / global_step, (epo * len(train_dataloader) + step) * 100 / t_total)
            if args.max_steps > 0 and global_step > args.max_steps:

                epoch_iterator.close()

                break

        

        if args.max_steps > 0 and global_step > args.max_steps:

            train_iterator.close()

            break


    return global_step, tr_loss / global_step





def evaluate(args, model, tokenizer, prefix=""):

    # Loop to handle MNLI double evaluation (matched, mis-matched)

    eval_output_dir = args.output_dir



    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)



    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:

        os.makedirs(eval_output_dir)



    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly

    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)

    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)



    # Eval!

    logger.info("***** Running evaluation {} *****".format(prefix))

    logger.info("  Num examples = %d", len(eval_dataset))

    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0

    nb_eval_steps = 0

    model.eval()



    for batch in tqdm(eval_dataloader, desc="Evaluating"):

        batch = batch.to(args.device)



        with torch.no_grad():

            outputs = model(batch, masked_lm_labels=batch) if args.mlm else model(batch, labels=batch)

            lm_loss = outputs[0]

            eval_loss += lm_loss.mean().item()

        nb_eval_steps += 1



    eval_loss = eval_loss / nb_eval_steps

    perplexity = torch.exp(torch.tensor(eval_loss))



    result = {

        "perplexity": perplexity

    }



    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")

    with open(output_eval_file, "w") as writer:

        logger.info("***** Eval results {} *****".format(prefix))

        for key in sorted(result.keys()):

            logger.info("  %s = %s", key, str(result[key]))

            writer.write("%s = %s\n" % (key, str(result[key])))



    return result



feature_store = '/egr/research-hlr/joslin/img_features/ResNet-152-places365.tsv'
panoramic = True


def main():

    parser = argparse.ArgumentParser()


    ## Required parameters
    parser.add_argument("--train_data_file", default="/VL/space/zhan1624/PREVALENT_R2R/tasks/R2R/data/collect_traj/", type=str,

                        help="The input training data file (a text file).")
                        
    # action_models1: zy design
    parser.add_argument("--output_dir", default="/egr/research-hlr/joslin/pretrain/action_new/", type=str,

                        help="The output directory where the model predictions and checkpoints will be written.")



    ## Other parameters

    parser.add_argument("--eval_data_file", default=None, type=str,

                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")



    parser.add_argument("--model_type", default="bert", type=str,

                        help="The model architecture to be fine-tuned.")
    # /VL/space/zhan1624/PREVALENT_R2R/pretrained_hug_models/dicadd/checkpoint-12864/
    parser.add_argument("--model_name_or_path", default="/egr/research-hlr/joslin/pretrain/action_rxr/checkpoint-169858", type=str,

                        help="The model checkpoint for weights initialization.")



    parser.add_argument("--mlm", action='store_true', default=True,

                        help="Train with masked-language modeling loss instead of language modeling.")

    parser.add_argument("--mlm_probability", type=float, default=0.15,

                        help="Ratio of tokens to mask for masked language modeling loss")



    parser.add_argument("--config_name", default="bert-base-uncased", type=str,

                        help="Optional pretrained config name or path if not the same as model_name_or_path")

    parser.add_argument("--tokenizer_name", default="bert-base-uncased", type=str,

                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--cache_dir", default="", type=str,

                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")

    parser.add_argument("--block_size", default=-1, type=int,

                        help="Optional input sequence length after tokenization."

                             "The training dataset will be truncated in block of this size for training."

                             "Default to the model max input length for single sentence inputs (take into account special tokens).")

    parser.add_argument("--do_train", action='store_true', default=True,

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

    parser.add_argument("--num_train_epochs", default=20.0, type=float,

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

    parser.add_argument('--overwrite_output_dir', action='store_true', default=True,

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

    parser.add_argument("--vision_size", type=int, default=2048+128,help="imgaction size")
    parser.add_argument("--action_space", type=int, default=36,help="action space")
    parser.add_argument("--vl_layers", type=int, default=4,help="how many fusion layers")
    parser.add_argument("--la_layers", type=int, default=9,help="how many lang layers")

    parser.add_argument('--update', type=bool, default=True, help='update lang Bert')
    parser.add_argument('--update_add_layer', type=bool, default=True, help='update add layer')
    parser.add_argument('--include_next', type=bool, default=True, help='do action classification')

    #parser.add_argument('--result_dir', dest='result_dir', type=str, default='tasks/R2R/results/', help='path to the result_dir file')
    #parser.add_argument('--plot_dir', dest='plot_dir', type=str, default='tasks/R2R/plots/', help='path to the plot_dir file')
    #parser.add_argument('--snapshot_dir', dest='snapshot_dir', type=str, default='tasks/R2R/snapshots/', help='path to the snapshot_dir file')
    parser.add_argument('--result_dir', type=str, default='tasks/R2R/results/', help='path to the result_dir file')
    parser.add_argument('--plot_dir', type=str, default='tasks/R2R/plots/', help='path to the plot_dir file')
    parser.add_argument('--snapshot_dir', type=str, default='tasks/R2R/snapshots/', help='path to the snapshot_dir file')
    parser.add_argument('--philly', action='store_true', help='program runs on Philly, used to redirect `write_model_path`')

    #
    parser.add_argument("--vision_match", action='store_true', default=True)
    parser.add_argument("--orient_match", action='store_true', default=True)
    args = parser.parse_args()

    if args.philly: # use philly
        print('Info: Use Philly, all the output folders are reset.')
        RESULT_DIR = os.path.join(os.getenv('PT_OUTPUT_DIR'), args.result_dir)
        PLOT_DIR = os.path.join(os.getenv('PT_OUTPUT_DIR'), args.plot_dir)
        SNAPSHOT_DIR = os.path.join(os.getenv('PT_OUTPUT_DIR'), args.snapshot_dir)
        #TRAIN_VOCAB = os.path.join(SNAPSHOT_DIR, params['train_vocab'])
        #TRAINVAL_VOCAB = os.path.join(SNAPSHOT_DIR, params['trainval_vocab'])

        print('RESULT_DIR', RESULT_DIR)
        print('PLOT_DIR', PLOT_DIR)
        print('SNAPSHOT_DIR', SNAPSHOT_DIR)
        #print('TRAIN_VOC', TRAIN_VOCAB)


    if args.model_type in ["bert", "roberta", "distilbert"] and not args.mlm:

        raise ValueError("BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm "

                         "flag (masked language modeling).")

    if args.eval_data_file is None and args.do_eval:

        raise ValueError("Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "

                         "or remove the --do_eval argument.")



    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:

        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))



    # Setup distant debugging if needed

    if args.server_ip and args.server_port:

        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script

        import ptvsd

        print("Waiting for debugger attach")

        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)

        ptvsd.wait_for_attach()



    # Setup CUDA, GPU & distributed training

    if args.local_rank == -1 or args.no_cuda:

        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

        args.n_gpu = torch.cuda.device_count()
        #args.n_gpu = 1
        

        print("You are using %d GPUs to train!!" % (args.n_gpu))

    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs

        torch.cuda.set_device(args.local_rank)

        device = torch.device("cuda", args.local_rank)

        torch.distributed.init_process_group(backend='nccl')

        args.n_gpu = 1

    args.device = device



    # Setup logging

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',

                        datefmt = '%m/%d/%Y %H:%M:%S',

                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",

                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)



    # Set seed

    set_seed(args)



    # Load pretrained model and tokenizer

    if args.local_rank not in [-1, 0]:

        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab



    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)

    if args.block_size <= 0:

        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model

    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    
    config.img_feature_dim = args.vision_size
    config.img_visn_dim = 2048
    config.img_pos_dim = 128
    config.img_feature_type = ""
    config.update_lang_bert = args.update
    config.update_add_layer = args.update_add_layer
    config.vl_layers = args.vl_layers
    config.la_layers = args.la_layers
    config.action_space = args.action_space
    model_class = DicAddActionPreTrain
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    #model = DicAddActionPreTrain(config)

    model.to(args.device)


    if args.local_rank == 0:

        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab



    logger.info("Training/evaluation parameters %s", args)



    # Training
    if args.do_train:

        if args.local_rank not in [-1, 0]:

            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache



        #train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
        jfiles = glob.glob(args.train_data_file + "/*.json")
        train_dataset = NavDataset(jfiles, tokenizer, feature_store, panoramic,args)
        print("you have loaded %d  time steps" % (len(train_dataset)))


        if args.local_rank == 0:

            torch.distributed.barrier()


        global_step, tr_loss = train(args, train_dataset, model, tokenizer)

        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)





    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        # Create output directory if needed

        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:

            os.makedirs(args.output_dir)



        logger.info("Saving model checkpoint to %s", args.output_dir)

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.

        # They can then be reloaded using `from_pretrained()`

        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training

        model_to_save.save_pretrained(args.output_dir)

        tokenizer.save_pretrained(args.output_dir)



        # Good practice: save your training arguments together with the trained model

        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))



        # Load a trained model and vocabulary that you have fine-tuned

        model = DicAddActionPreTrain.from_pretrained(args.output_dir)

        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)

        model.to(args.device)

    results = {}


    return results



if __name__ == "__main__":
    main()