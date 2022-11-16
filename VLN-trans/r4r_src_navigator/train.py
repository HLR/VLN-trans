
#/localscratch/zhan1624/.conda/envs/vln
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
from agent import Seq2SeqAgent
from eval import Evaluation
from param import args


from speaker import Speaker

import warnings
warnings.filterwarnings("ignore")
from tensorboardX import SummaryWriter

from vlnbert.vlnbert_init import get_tokenizer

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = '7'

log_dir = '/localscratch/zhan1624/VLN-speaker/snap/%s' % args.name
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
PLACE365_FEATURES = '/egr/research-hlr/joslin/img_features/ResNet-152-places365.tsv'
result_path = "/localscratch/zhan1624/VLN-speaker/result/"
experiment_time = time.strftime("%Y%m%d-%H%M%S", time.gmtime())

TRAIN_VOCAB = 'data/train_vocab.txt'
TRAINVAL_VOCAB = 'data/trainval_vocab.txt'

if args.features == 'imagenet':
    features = IMAGENET_FEATURES
elif args.features == 'places365':
    features = PLACE365_FEATURES
feedback_method = args.feedback  # teacher or sample

print(args); print('')


'''train speaker'''
def train_speaker(train_env, tok, n_iters, log_every=500, val_envs={}):
    writer = SummaryWriter(logdir=log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)
    speaker = Speaker(train_env, listner, tok)

    if args.fast_train:
        log_every = 40

    best_bleu = defaultdict(lambda: 0)
    best_loss = defaultdict(lambda: 1232)
    for idx in range(0, n_iters, log_every):
        interval = min(log_every, n_iters - idx)

        # Train for log_every interval
        speaker.env = train_env
        speaker.train(interval)   # Train interval iters

        print()
        print("Iter: %d" % idx)

        # Evaluation
        for env_name, (env, evaluator) in val_envs.items():
            if 'train' in env_name: # Ignore the large training set for the efficiency
                continue

            print("............ Evaluating %s ............." % env_name)
            speaker.env = env
            path2inst, loss, word_accu, sent_accu = speaker.valid()
            path_id = next(iter(path2inst.keys()))
            print("Inference: ", speaker.decode_sentence(path2inst[path_id]))
            print("GT: ", evaluator.gt[str(path_id)]['instructions'])
            bleu_score, precisions = evaluator.bleu_score(path2inst)


            # Tensorboard log
            writer.add_scalar("bleu/%s" % (env_name), bleu_score, idx)
            writer.add_scalar("loss/%s" % (env_name), loss, idx)
            writer.add_scalar("word_accu/%s" % (env_name), word_accu, idx)
            writer.add_scalar("sent_accu/%s" % (env_name), sent_accu, idx)
            writer.add_scalar("bleu4/%s" % (env_name), precisions[3], idx)

            # Save the model according to the bleu score
            if bleu_score > best_bleu[env_name]:
                best_bleu[env_name] = bleu_score
                print('Save the model with %s BEST env bleu %0.4f' % (env_name, bleu_score))
                speaker.save(idx, os.path.join(log_dir, 'state_dict', 'best_%s_bleu' % env_name))

            if loss < best_loss[env_name]:
                best_loss[env_name] = loss
                print('Save the model with %s BEST env loss %0.4f' % (env_name, loss))
                speaker.save(idx, os.path.join(log_dir, 'state_dict', 'best_%s_loss' % env_name))

            # Screen print out
            print("Bleu 1: %0.4f Bleu 2: %0.4f, Bleu 3 :%0.4f,  Bleu 4: %0.4f" % tuple(precisions))



''' train the listener '''
def train(train_env, tok, n_iters, log_every=2000, val_envs={}, aug_env=None):
    writer = SummaryWriter(log_dir=log_dir)
    
    # speaker
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    if args.speaker is not None:
        speaker = Speaker(train_env, listner, train_env.tok)
        print("Load the speaker from %s." % args.speaker)
        speaker.load(args.speaker)
    else:
        speaker = None

    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)
    record_file = open('./logs/' + args.name + '.txt', 'a')
    record_file.write(str(args) + '\n\n')
    record_file.close()

    start_iter = 0
    if args.load is not None:
        if args.aug is None:
            start_iter = listner.load(os.path.join(args.load))
            print("\nLOAD the model from {}, iteration ".format(args.load, start_iter))
        else:
            load_iter = listner.load(os.path.join(args.load))
            print("\nLOAD the model from {}, iteration ".format(args.load, load_iter))

    start = time.time()
    print('\nListener training starts, start iteration: %s' % str(start_iter))

    best_val = {'val_unseen': {"spl": 0., "sr": 0., "state":"", 'update':False}}

    for idx in range(start_iter, start_iter+n_iters, log_every):
        listner.logs = defaultdict(list)
        interval = min(log_every, n_iters-idx)
        iter = idx + interval

        # Train for log_every interval
        
        if aug_env is None:
            listner.env = train_env
            listner.train(interval, feedback=feedback_method, speaker=speaker)  # Train interval iters
        else:
            jdx_length = len(range(interval // 2))
            for jdx in range(interval // 2):
                # Train with GT data
                listner.env = train_env
                listner.train(1, feedback=feedback_method, speaker=speaker)

                # Train with Augmented data
                listner.env = aug_env
                listner.train(1, feedback=feedback_method, speaker=speaker)

                print_progress(jdx, jdx_length, prefix='Progress:', suffix='Complete', bar_length=50)

        # Log the training stats to tensorboard
        total = max(sum(listner.logs['total']), 1)
        length = max(len(listner.logs['critic_loss']), 1)
        critic_loss = sum(listner.logs['critic_loss']) / total
        RL_loss = sum(listner.logs['RL_loss']) / max(len(listner.logs['RL_loss']), 1)
        IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)
        sub_loss = sum(listner.logs['sub_loss']) / max(len(listner.logs['sub_loss']), 1)
        split_loss = sum(listner.logs['split_loss']) / max(len(listner.logs['split_loss']), 1)
        entropy = sum(listner.logs['entropy']) / total
        writer.add_scalar("loss/critic", critic_loss, idx)
        writer.add_scalar("policy_entropy", entropy, idx)
        writer.add_scalar("loss/RL_loss", RL_loss, idx)
        writer.add_scalar("loss/IL_loss", IL_loss, idx)
        writer.add_scalar("loss/sub_loss", sub_loss, idx)
        writer.add_scalar("loss/split_loss", split_loss, idx)
        writer.add_scalar("total_actions", total, idx)
        writer.add_scalar("max_length", length, idx)
        # print("total_actions", total, ", max_length", length)
        
        # Run validation
        loss_str = "iter {}".format(iter)
        for env_name, (env, evaluator) in val_envs.items():
            listner.env = env

            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None, speaker=speaker)
            result = listner.get_results()
            
            score_summary, _ = evaluator.score(result)
            loss_str += ", %s " % env_name
            for metric, val in score_summary.items():
                if metric in ['spl']:
                    writer.add_scalar("spl/%s" % env_name, val, idx)
                    if env_name in best_val:
                        if val > best_val[env_name]['spl']:
                            best_val[env_name]['spl'] = val
                            best_val[env_name]['update'] = True
                        elif (val == best_val[env_name]['spl']) and (score_summary['success_rate'] > best_val[env_name]['sr']):
                            best_val[env_name]['spl'] = val
                            best_val[env_name]['update'] = True
                loss_str += ', %s: %.4f' % (metric, val)

        record_file = open('./logs/' + args.name + '.txt', 'a')
        record_file.write(loss_str + '\n')
        record_file.close()

        for env_name in best_val:
            if best_val[env_name]['update']:
                best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                best_val[env_name]['update'] = False
                listner.save(idx, os.path.join("snap", args.name, "state_dict", "best_%s" % (env_name)))
               
            else:
                listner.save(idx, os.path.join("snap", args.name, "state_dict", "latest_dict"))
            
            

        print(('%s (%d %d%%) %s' % (timeSince(start, float(iter)/n_iters),
                                             iter, float(iter)/n_iters*100, loss_str)))

        with open(result_path+str(experiment_time)+".txt", "a") as f_result:
            f_result.write(('%s (%d %d%%) %s' % (timeSince(start, float(iter)/n_iters),
                                             iter, float(iter)/n_iters*100, loss_str)))
            f_result.write('\n')

        if iter % 1000 == 0:
            print("BEST RESULT TILL NOW")
            for env_name in best_val:
                print(env_name, best_val[env_name]['state'])

                record_file = open('./logs/' + args.name + '.txt', 'a')
                record_file.write('BEST RESULT TILL NOW: ' + env_name + ' | ' + best_val[env_name]['state'] + '\n')
                record_file.close()
    
    listner.save(idx, os.path.join("snap", args.name, "state_dict", "LAST_iter%d" % (idx)))


def valid(train_env, tok, val_envs={}):
    agent = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    print("Loaded the listener model at iter %d from %s" % (agent.load(args.load), args.load))

    for env_name, (env, evaluator) in val_envs.items():
       
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        agent.test(use_dropout=False, feedback='argmax', iters=iters)
        result = agent.get_results()

        if env_name != '':
            score_summary, _ = evaluator.score(result)
            loss_str = "Env name: %s" % env_name
            for metric,val in score_summary.items():
                loss_str += ', %s: %.4f' % (metric, val)
            print(loss_str)

        # if args.submit:
            json.dump(
                result,
                open(os.path.join(log_dir, "submit_%s.json" % env_name), 'w'),
                sort_keys=True, indent=4, separators=(',', ': ')
            )
        

def setup():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    random.seed(0)
    np.random.seed(0)

def train_val(test_only=False):
    ''' Train on the training set, and validate on seen and unseen splits. '''
    setup()
    tok = get_tokenizer(args)

    feat_dict = read_img_features(features, test_only=test_only)
    #feat_dict = None

    train_env = R2RBatch(feat_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok)
    
    if test_only:
        featurized_scans = None
        val_env_names = ['val_train_seen']
    else:
        featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
        #val_env_names = ['val_train_seen', 'val_seen', 'val_unseen']
        val_env_names = ['val_seen', 'val_unseen']

    from collections import OrderedDict

    val_envs = OrderedDict(
        ((split,
          (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split], tokenizer=tok),
           Evaluation([split], featurized_scans, tok))
          )
         for split in val_env_names
         )
    )

    if args.train == 'listener':
        train(train_env, tok, args.iters, val_envs=val_envs)
    elif args.train == 'validlistener':
        valid(train_env, tok, val_envs=val_envs)
    elif args.train == 'speaker':
        train_speaker(train_env, tok, args.iters, val_envs=val_envs)
    else:
        assert False

def train_val_augment(test_only=False):
    """
    Train the listener with the augmented data
    """
    setup()

    # Create a batch training environment that will also preprocess 
    tok = get_tokenizer(args)

    # Load the env img features
    feat_dict = read_img_features(features, test_only=test_only)
    #feat_dict = None
    # Load the augmentation data
    aug_path = args.aug
    # Create the training environment
    
    train_env = R2RBatch(feat_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok)
    aug_env   = R2RBatch(feat_dict, batch_size=args.batchSize, splits=[aug_path], tokenizer=tok, name='aug')
    
    if test_only:
        featurized_scans = None
        val_env_names = ['val_train_seen']
    else:
        featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
        val_env_names = ['val_seen', 'val_unseen']

    
    # Setup the validation data
    val_envs = {split: (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split], tokenizer=tok),
                Evaluation([split], featurized_scans, tok))
                for split in val_env_names}

    # Start training
    train(train_env, tok, args.iters, val_envs=val_envs, aug_env=aug_env)

if __name__ == "__main__":
    if args.train in ['listener', 'validlistener', 'speaker']:
        train_val(test_only=args.test_only)
    elif args.train == 'auglistener':
        train_val_augment(test_only=args.test_only)
    else:
        assert False