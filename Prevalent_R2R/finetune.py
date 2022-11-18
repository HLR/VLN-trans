
import os, argparse, json
import time, copy, random, pickle
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from utils import read_vocab, write_vocab, build_vocab, Tokenizer, SplitTokenizer, padding_idx, \
    timeSince, boolean_string, preprocess_get_pano_states, current_best
from env import R2RBatch, EnvBatch
from model import EncoderLSTM, EncoderMultiLSTM, BertEncoder, MultiBertEncoder, GptEncoder, MultiGptEncoder,\
    TransformerEncoder, MultiTransformerEncoder, BertImgEncoder, BertAddEncoder,MultiVilBertEncoder, MultiVilAddEncoder, MultiAddLoadEncoder, AttnDecoderLSTM
from pytorch_transformers import BertForMaskedLM,BertTokenizer
from agent import Seq2SeqAgent
from eval import Evaluation
from feature import Feature
import pprint
import pdb

class CustomDataParallel(nn.Module):
    def __init__(self, model):
        super(CustomDataParallel, self).__init__()
        self.model = nn.DataParallel(model).cuda()
        print(type(self.model))

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model.module, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_vocab', dest='train_vocab', type=str, default='train_vocab.txt', help='train_vocab filename (in snapshots folder)')
    parser.add_argument('--trainval_vocab', dest='trainval_vocab', type=str, default='trainval_vocab.txt', help='trainval_vocab filename (in snapshots folder)')
    parser.add_argument('--use_glove', dest='use_glove', type=boolean_string, default=False, help='whether use glove')
    parser.add_argument('--glove_path', dest='glove_path', type=str, default='tasks/R2R/data/train_glove.npy', help='path to the glove file')

    parser.add_argument('--result_dir', dest='result_dir', type=str, default='tasks/R2R/results/', help='path to the result_dir file')
    parser.add_argument('--snapshot_dir', dest='snapshot_dir', type=str, default='tasks/R2R/snapshots/', help='path to the snapshot_dir file')
    parser.add_argument('--plot_dir', dest='plot_dir', type=str, default='tasks/R2R/plots/', help='path to the plot_dir file')

    parser.add_argument('--min_count', dest='min_count', type=int, default=5, help='word min_count')
    parser.add_argument('--max_input_length', dest='max_input_length', default=80, type=int, help='max_input_length')

    parser.add_argument('--batch_size', dest='batch_size', default=10, type=int, help='batch size')
    parser.add_argument('--max_episode_len', dest='max_episode_len', default=8, type=int, help='max_episode_len')
    parser.add_argument('--word_embedding_size', dest='word_embedding_size', default=256, type=int, help='word_embedding_size')
    parser.add_argument('--action_embedding_size', dest='action_embedding_size', default=64, type=int, help='action_embedding_size')
    parser.add_argument('--hidden_size', dest='hidden_size', default=1024, type=int, help='decoder hidden_size')
    parser.add_argument('--enc_hidden_size', dest='enc_hidden_size', default=1024, type=int, help='encoder hidden_size')

    parser.add_argument('--feature_store', dest='feature_store', type=str, default='img_features/ResNet-152-imagenet.tsv', help='path to the image feature file')
    parser.add_argument('--feature_size', dest='feature_size', default=2048, type=int, help='feature_size')
    parser.add_argument('--feature_all_size', dest='feature_all_size', default=2176, type=int, help='imgaction_size')

    parser.add_argument('--n_iters', dest='n_iters', default=70000, type=int, help='n_iters')
    parser.add_argument('--n_iters_resume', dest='n_iters_resume', default=0, type=int, help='n_iters_resume')
    parser.add_argument('--n_iters_pretrain_resume', dest='n_iters_pretrain_resume', default=0, type=int, help='n_iters_pretrain_resume')
    parser.add_argument('--ss_n_pretrain_iters', dest='ss_n_pretrain_iters', default=-1, type=int, help='scheduled sampling n_iters in pretrain')
    parser.add_argument('--ss_n_iters', dest='ss_n_iters', default=65000, type=int, help='scheduled sampling n_iters')
    parser.add_argument('--finetune_iters', dest='finetune_iters', default=-1, type=int, help='finetune_iters for BERT')
    parser.add_argument('--finetune_batchsize', dest='finetune_batchsize', default=-1, type=int, help='finetune_batchsize for BERT')
    parser.add_argument('--sc_after', dest='sc_after', default=-1, type=int, help='SELF_CRITICAL_AFTER')

    parser.add_argument('--pretrain_model_path', dest='pretrain_model_path', type=str, default='tasks/R2R/snapshots/', help='the path of pretrained model')
    parser.add_argument('--pretrain_lm_model_path', dest='pretrain_lm_model_path', type=str, default='pretrained_models/', help='the path of pretrained lm model')
    parser.add_argument('--pretrain_model_name', dest='pretrain_model_name', type=str, default=None, help='the name of pretrained model')
    parser.add_argument('--pretrain_decoder_name', dest='pretrain_decoder_name', type=str, default=None, help='the name of decoder model')

    parser.add_argument('--log_every', dest='log_every', default=20, type=int, help='log_every')
    parser.add_argument('--save_ckpt', dest='save_ckpt', default=48, type=int, help='dump model checkpoint, default -1')

    parser.add_argument('--schedule_ratio', dest='schedule_ratio', default=0.2, type=float, help='ratio for sample or teacher')
    parser.add_argument('--schedule_anneal', dest='schedule_anneal', action='store_true', help='schedule_ratio is annealling or not')
    parser.add_argument('--dropout_ratio', dest='dropout_ratio', default=0.4, type=float, help='dropout_ratio')
    parser.add_argument('--temp_alpha', dest='temp_alpha', default=1.0, type=float, help='temperate alpha for softmax')
    parser.add_argument('--learning_rate', dest='learning_rate', default=5e-05, type=float, help='learning_rate')
    parser.add_argument('--sc_learning_rate', dest='sc_learning_rate', default=2e-05, type=float, help='sc_learning_rate')
    parser.add_argument('--weight_decay', dest='weight_decay', default=0.0005, type=float, help='weight_decay')
    parser.add_argument('--optm', dest='optm', default='Adamax', type=str, help='Adam, Adamax, RMSprop')

    parser.add_argument('--sc_reward_scale', dest='sc_reward_scale', default=1., type=float, help='sc_reward_scale')
    parser.add_argument('--sc_discouted_immediate_r_scale', dest='sc_discouted_immediate_r_scale', default=0., type=float, help='sc_discouted_immediate_r_scale')
    parser.add_argument('--sc_length_scale', dest='sc_length_scale', default=0., type=float, help='sc_length_scale')

    parser.add_argument('--feedback_method', dest='feedback_method', type=str, default='teacher', help='sample or teacher')
    parser.add_argument('--bidirectional', dest='bidirectional', type=boolean_string, default=True, help='bidirectional')

    parser.add_argument('--monotonic_sc', dest='monotonic_sc', type=boolean_string, default=False, help='monotonic self-critic')
    parser.add_argument('--panoramic', dest='panoramic', type=boolean_string, default=True, help='panoramic img')
    parser.add_argument('--action_space', dest='action_space', type=int, default=-1, help='6 or -1(navigable viewpoints)')
    parser.add_argument('--ctrl_feature', dest='ctrl_feature', type=boolean_string, default=False, help='ctrl_feature')
    parser.add_argument('--ctrl_f_net', dest='ctrl_f_net', type=str, default='linear', help='imglinear, linear, nonlinear, imgnl or deconv')
    parser.add_argument('--aux_n_iters', dest='aux_n_iters', type=int, help='update auxiliary net after aux_n_iters')
    parser.add_argument('--aux_ratio', dest='aux_ratio', type=float, help='aux_ratio')
    parser.add_argument('--accu_n_iters', dest='accu_n_iters', type=int, default=0, help='gradient accumulation')

    parser.add_argument('--att_ctx_merge', dest='att_ctx_merge', type=str, default='mean', help='mean cat mean sum (to merge attention)')
    parser.add_argument('--ctx_dropout_ratio', dest='ctx_dropout_ratio', type=float, default=0.0, help='ctx_dropout_ratio')
    parser.add_argument('--clip_gradient', dest='clip_gradient', type=float, default=0.1, help='clip_gradient')
    parser.add_argument('--clip_gradient_norm', dest='clip_gradient_norm', type=float, default=0.0, help='clip gradient norm')

    parser.add_argument('--multi_share', dest='multi_share', type=boolean_string, default=True, help='share encoders in EncoderMultiLSTM')
    parser.add_argument('--decoder_init', dest='decoder_init', type=boolean_string, default=True, help='init decoder with lstm output')
    parser.add_argument('--dec_h_init', dest='dec_h_init', type=str, default='tanh', help='linear, tanh, none')
    parser.add_argument('--dec_c_init', dest='dec_c_init', type=str, default='none', help='linear, tanh, none')
    parser.add_argument('--dec_h_type', dest='dec_h_type', type=str, default='vc', help='none or vc')

    parser.add_argument('--encoder_type', dest='encoder_type', type=str, default='bert', help='lstm transformer bert or gpt')
    parser.add_argument('--top_lstm', dest='top_lstm', type=boolean_string, default=True, help='add lstm to the top of transformers')
    parser.add_argument('--transformer_update', dest='transformer_update', type=boolean_string, default=False, help='update Bert')
    parser.add_argument('--bert_n_layers', dest='bert_n_layers', type=int, default=1, help='bert_n_layers')
    parser.add_argument('--bert_type', dest='bert_type', type=str, default="small", help='small or large')

    parser.add_argument('--heads', dest='heads', type=int, default=4, help='heads in transformer')
    parser.add_argument('--transformer_emb_size', dest='transformer_emb_size', type=int, default=512, help='transformer_emb_size')
    parser.add_argument('--transformer_d_ff', dest='transformer_d_ff', type=int, default=1024, help='transformer_d_ff')
    parser.add_argument('--transformer_num_layers', dest='transformer_num_layers', type=int, default=1, help='transformer_num_layers')
    parser.add_argument('--vl_layers', dest='vl_layers', type=int, default=1, help='vl_layers')

    parser.add_argument('--use_cuda', dest='use_cuda', type=boolean_string, default=True, help='use_cuda')
    parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=0, help='gpu_id')
    parser.add_argument('--train', dest='train', type=boolean_string, default=True, help='train or test')

    parser.add_argument('--pretrain_score_name', dest='pretrain_score_name', type=str, default='sr_sum', help='sr_sum spl_sum sr_unseen spl_unseen')
    parser.add_argument('--train_score_name', dest='train_score_name', type=str, default='sr_sum', help='sr_sum spl_sum sr_unseen spl_unseen')
    parser.add_argument('--sc_score_name', dest='sc_score_name', type=str, default='sr_sum', help='sr_sum spl_sum sr_unseen spl_unseen')

    parser.add_argument('--beam_size', dest='beam_size', type=int, default=1, help='beam_size for inference')
    parser.add_argument('--use_speaker', dest='use_speaker', type=boolean_string, default=False, help='use speaker for inference')
    parser.add_argument('--speaker_weight', dest='speaker_weight', type=str, default='0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95', help='speaker weight for inference')
    parser.add_argument('--speaker_prefix', dest='speaker_prefix', type=str, default='tasks/R2R/snapshots/release/speaker_final_release', help='speaker enc dec prefix')
    parser.add_argument('--speaker_merge', dest='speaker_merge', type=str, default='mean', help='how speaker score for multiple sentences')
    parser.add_argument('--state_factored', type=boolean_string, default=False, help='state factored beam search')
    parser.add_argument('--successors', dest='successors', type=int, default=1, help='successors for state_factored_search inference')

    parser.add_argument('--use_pretrain', action='store_true', help='pretrain or not')
    parser.add_argument('--train_in_pretrain', action='store_true', help='pretrain train or not')

    parser.add_argument('--pretrain_splits', type=str, default="literal_speaker_data_augmentation_paths", help="pretrain dataset")
    parser.add_argument('--pretrain_n_iters', dest='pretrain_n_iters', type=int, default=0, help='pretrain_n_iters')
    parser.add_argument('--pretrain_n_sentences', dest='pretrain_n_sentences', type=int, default=3,
                        help='This is only for pretraining when using EncoderMultiLSTM. In normal train/test/val, it will be reset to 3')
    parser.add_argument('--single_sentence_test', dest='single_sentence_test', type=boolean_string, default=False,
                        help='run additional test for single sentence as input')

    parser.add_argument('--val_splits', dest='val_splits', type=str, default='val_seen,val_unseen', help='test can be added')
    parser.add_argument('--warmup_iters', dest='warmup_iters', type=int, default=0, help='warmup iterations for BertImgEncoder')

    parser.add_argument('--reward_func', dest='reward_func', type=str, default='spl', help='reward function: sr_sc, spl, spl_sc, spl_last, spl_last_sc')

    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--philly', action='store_true', help='program runs on Philly, used to redirect `write_model_path`')
    parser.add_argument('--dump_result', action='store_true', help='dump result file')

    parser.add_argument('--test_A', action='store_true', help='testing_settingA')

    # args = '--panoramic True ' \
    #        '--action_space -1 ' \
    #        '--result_dir /home/xiul/Programs/exps/tmp/test/results/ ' \
    #        '--snapshot_dir /home/xiul/Programs/exps/tmp/test/snapshots/ ' \
    #        '--plot_dir /home/xiul/Programs/exps/tmp/test/plots/ ' \
    #        '--max_episode_len 8 ' \
    #        '--att_ctx_merge mean ' \
    #        '--n_iters 1500 ' \
    #        '--batch_size 64 --log_every 64 --feedback_method teacher ' \
    #        '--enc_hidden_size 1024 ' \
    #        '--hidden_size 1024 '
    # args = parser.parse_args(args.split())

    # args = '--action_space -1 ' \
    #        '--result_dir /home/xiul/Programs/exps/tmp/test/results/ ' \
    #        '--snapshot_dir /home/xiul/Programs/exps/tmp/test/snapshots/ ' \
    #        '--plot_dir /home/xiul/Programs/exps/tmp/test/plots/ ' \
    #        '--att_ctx_merge mean --batch_size 64 --log_every 64 --feedback_method teacher --clip_gradient_norm 0 ' \
    #        '--ss_n_pretrain_iters 400 --pretrain_n_iters 500 --ss_n_iters 500 --n_iters 600 ' \
    #        '--use_pretrain --pretrain_n_sentences 4 '\
    #        '--pretrain_splits sample_seed10_20_30_40_50_data_aug_paths '\
    #        '--enc_hidden_size 1024 --hidden_size 1024 '

    # args = '--feedback_method teacher ' \
    #        '--result_dir /home/xql/Source/Subgoal/tasks/R2R/exps/test_trans_mean/results/ ' \
    #        '--snapshot_dir /home/xql/Source/Subgoal/tasks/R2R/exps/test_trans_mean/snapshots/ ' \
    #        '--plot_dir /home/xql/Source/Subgoal/tasks/R2R/exps/test_trans_mean/plots/ ' \
    #        '--ss_n_iters 20000 ' \
    #        '--dropout_ratio 0.4 ' \
    #        '--dec_h_type vc --schedule_ratio 0.3 ' \
    #        '--optm Adam --clip_gradient_norm 0 --log_every 64 ' \
    #        '--action_space -1 ' \
    #        '--train_score_name sr_unseen ' \
    #        '--n_iters 40000 ' \
    #        '--enc_hidden_size 1024 --hidden_size 1024 ' \
    #        '--bidirectional True ' \
    #        '--batch_size 10 ' \
    #        '--encoder_type transformer ' \
    #        '--transformer_emb_size 512 --top_lstm True ' \
    #        '--att_ctx_merge mean '
    # args = parser.parse_args(args.split())

    args = parser.parse_args()
    params = vars(args)


assert params['panoramic'] or (params['panoramic'] == False and params['action_space'] == 6)
RESULT_DIR = params['result_dir'] #'tasks/R2R/results/'
SNAPSHOT_DIR = params['snapshot_dir'] #'tasks/R2R/snapshots/'
PLOT_DIR = params['plot_dir'] #'tasks/R2R/plots/'

TRAIN_VOCAB = os.path.join(SNAPSHOT_DIR, params['train_vocab']) #'tasks/R2R/data/train_vocab.txt'
TRAINVAL_VOCAB = os.path.join(SNAPSHOT_DIR, params['trainval_vocab'])
MIN_COUNT = params['min_count']
use_glove = params['use_glove']
glove_path = params['glove_path']

batch_size = params['batch_size'] #100
encoder_type = params['encoder_type']  # lstm
top_lstm = params['top_lstm']
transformer_update = params['transformer_update']
bert_n_layers = params['bert_n_layers']  # 1
reverse_input = True

MAX_INPUT_LENGTH = params['max_input_length'] #80
max_episode_len = params['max_episode_len'] #20

word_embedding_size = params['word_embedding_size'] #256
action_embedding_size = params['action_embedding_size'] # 32
hidden_size = params['hidden_size'] #512
enc_hidden_size = params['enc_hidden_size'] # 512

heads = params['heads']
transformer_emb_size = params['transformer_emb_size']
transformer_d_ff = params['transformer_d_ff']
transformer_num_layers = params['transformer_num_layers']
vl_layers = params['vl_layers']

bidirectional = params['bidirectional'] #False
feedback_method = params['feedback_method'] #'sample'  # teacher or sample

schedule_ratio = params['schedule_ratio']
schedule_anneal = params['schedule_anneal']
dropout_ratio = params['dropout_ratio'] # 0.5
learning_rate = params['learning_rate'] # 0.0001
weight_decay = params['weight_decay'] # 0.0005
sc_learning_rate = params['sc_learning_rate']
optm = params['optm']

sc_reward_scale = params['sc_reward_scale']
sc_discouted_immediate_r_scale = params['sc_discouted_immediate_r_scale']
sc_length_scale = params['sc_length_scale']


#n_iters = 5000 if feedback_method == 'teacher' else 20000
n_iters = params['n_iters'] # 60000  # jolin
ss_n_iters = params['ss_n_iters']
ss_n_pretrain_iters = params['ss_n_pretrain_iters']
finetune_iters = params['finetune_iters']
finetune_batchsize = params['finetune_batchsize']
log_every = params['log_every']
save_ckpt = params['save_ckpt']

monotonic = params['monotonic_sc'] #False  # jolin
panoramic = params['panoramic']
action_space = params['action_space']
ctrl_feature = params['ctrl_feature']  # False
ctrl_f_net = params['ctrl_f_net']  # linear
aux_n_iters = params['aux_n_iters']
aux_ratio = params['aux_ratio']
accu_n_iters = params['accu_n_iters']
att_ctx_merge = params['att_ctx_merge']
ctx_dropout_ratio = params['ctx_dropout_ratio']
multi_share = params['multi_share']
decoder_init = params['decoder_init']

pretrain_score_name = params['pretrain_score_name']
train_score_name = params['train_score_name']
sc_score_name = params['sc_score_name']

use_pretrain = params['use_pretrain']
train_in_pretrain = params['train_in_pretrain']

pretrain_splits = params['pretrain_splits'].split(',')
pretrain_n_iters = params['pretrain_n_iters']
pretrain_n_sentences = params['pretrain_n_sentences']
assert multi_share or pretrain_n_sentences==3

train_splits= ['train']
val_splits= params['val_splits'].split(',')

from env import debug_beam
#if debug_beam:
# train_splits=['debug1']
# val_splits=['debug1']

beam_size = params['beam_size']
use_speaker = params['use_speaker']
speaker_weight = params['speaker_weight'].split(',')
speaker_prefix = params['speaker_prefix']
state_factored = params['state_factored']
speaker_merge = params['speaker_merge']
successors = params['successors']

dump_result = params['dump_result']
if dump_result:
    print('Info: Temporary result files will be dumped!')
else:
    print('Info: Save space mode ON. All previous best models and temporary result files will be deleted!')

if ctrl_feature: assert aux_n_iters>1

model_prefix = 'seq2seq_%s_imagenet' % (feedback_method)


# controlled by env variable CUDA_AVAILABLE_DEVICES. If =4 and #(real devices)<4, cpu
#gpu_id = str(params['gpu_id'])
gpuid = 'cpu'
if params['use_cuda']:
    if torch.cuda.is_available():
        gpuid='cuda:0'
        torch.backends.cudnn.deterministic = True
device = torch.device(gpuid)
is_train = params['train']

print('------ Check CUDA Info ------')
print('cuda:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu num:', torch.cuda.device_count())
    print('gpu IDs:', torch.cuda.current_device())
    print('gpu name:', torch.cuda.get_device_name(0))
print('-----------------------------')


FEATURE_STORE = params['feature_store']
FEATURE_SIZE = params['feature_size']
FEATURE_ALL_SIZE = params['feature_all_size']

N_ITERS_RESUME = params['n_iters_resume'] # 0 #45000
n_iters_pretrain_resume = params['n_iters_pretrain_resume']
SELF_CRITICAL_AFTER = params['sc_after'] #-1 #320  # print('copied from 4snapshots')
single_sentence_test = params['single_sentence_test']

features = FEATURE_STORE
model_prefix = 'seq2seq_%s_imagenet' % (feedback_method)

use_bert = (encoder_type in ['bert', 'gpt','vlbert'])  # for tokenizer and dataloader
ctx_hidden_size = enc_hidden_size * (2 if bidirectional else 1)
if (use_bert and not top_lstm):
    ctx_hidden_size = 768
elif encoder_type=='transformer':
    if not top_lstm:
        ctx_hidden_size = transformer_emb_size

bert_pre_best_model_iter = 0  # for space reason, delete previous best model
bert_type = params['bert_type']

glove = np.load(glove_path) if use_glove else None
if use_glove and (glove.shape[1] != word_embedding_size):
    print('Warning: reset word_embedding_size according to glove (dim=%d)' % glove.shape[1])
    params['word_embedding_size'] = glove.shape[1]
    word_embedding_size = glove.shape[1]


submit_splits = train_splits+val_splits

nav_graphs = None  # navigable loc cache

philly = params['philly']
seed = params['seed']


agent_params = {}
agent_params['clip_gradient'] = params['clip_gradient']
agent_params['clip_gradient_norm'] = params['clip_gradient_norm']
agent_params['reward_func'] = params['reward_func']
#agent_params['schedule_ratio'] = params['schedule_ratio']
agent_params['schedule_ratio'] = 0.3 #params['schedule_ratio']
agent_params['temp_alpha'] = params['temp_alpha']

agent_params['test_A'] = params['test_A']

encoder_params = {}
encoder_params['dec_h_init'] = params['dec_h_init']
encoder_params['dec_c_init'] = params['dec_c_init']

decoder_params = {}
decoder_params['dec_h_type'] = params['dec_h_type']


navigable_locs_path = "tasks/R2R/data"
pretrain_model_path = params['pretrain_model_path']
pretrain_lm_model_path = params['pretrain_lm_model_path']
pretrain_model_name = params['pretrain_model_name'] #pretrained_lm_00000.model.ep0  for test
pretrain_decoder_name = params['pretrain_decoder_name'] #pretrained_lm_00000.model.ep0  for test
warmup_iters = params['warmup_iters']

if philly: # use philly
    print('Info: Use Philly, all the output folders are reset.')
    RESULT_DIR = os.path.join(os.getenv('PT_OUTPUT_DIR'), params['result_dir'])
    PLOT_DIR = os.path.join(os.getenv('PT_OUTPUT_DIR'), params['plot_dir'])
    SNAPSHOT_DIR = os.path.join(os.getenv('PT_OUTPUT_DIR'), params['snapshot_dir'])
    TRAIN_VOCAB = os.path.join(SNAPSHOT_DIR, params['train_vocab'])
    TRAINVAL_VOCAB = os.path.join(SNAPSHOT_DIR, params['trainval_vocab'])
    navigable_locs_path = os.path.join(os.getenv('PT_OUTPUT_DIR'), "tasks/R2R/data")

    print('RESULT_DIR', RESULT_DIR)
    print('PLOT_DIR', PLOT_DIR)
    print('SNAPSHOT_DIR', SNAPSHOT_DIR)
    print('TRAIN_VOC', TRAIN_VOCAB)


def train(train_env, finetunes, train_Eval, encoder, decoder, monotonic, n_iters, resume_split_arr, cur_split_arr, score_name, jump_iters, log_every=log_every, val_envs={}, n_iters_resume=0,warmup_iters=0):
    ''' Train on training set, validating on both seen and unseen. '''

    data_log = defaultdict(list)

    agent = Seq2SeqAgent(train_env, "", encoder, decoder, 'resume' if n_iters_resume>0 else 10, aux_ratio, decoder_init, agent_params, monotonic, max_episode_len, accu_n_iters = accu_n_iters)
    if hasattr(encoder, 'n_sentences'): encoder_n_sentences = encoder.n_sentences

    best_model_iter = n_iters_resume
    best_sr, best_spl, best_score = 0, 0, 0
    best_eval = 0
    idx = 0

    if n_iters_resume>0:
        split_string = "-".join(resume_split_arr)
        print('Resuming from', n_iters_resume, 'on', split_string)
        if len(resume_split_arr) == 0:
            enc_path = '%s%s_%s_enc_iter_%d' % (pretrain_model_path, model_prefix, split_string, n_iters_resume)
            dec_path = '%s%s_%s_dec_iter_%d' % (pretrain_model_path, model_prefix, split_string, n_iters_resume)
        else:
            enc_path = '%s%s_%s_enc_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, n_iters_resume)
            dec_path = '%s%s_%s_dec_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, n_iters_resume)

        agent.load(enc_path, dec_path)
        loss_str = ''

        if hasattr(encoder, 'n_sentences'): encoder.set_n_sentences(3)
        for env_name, (env, evaluator) in val_envs.items():
            result_path = '%s%s_%s_iter_%d.json' % (RESULT_DIR, model_prefix, env_name, n_iters_resume)
            score_summary, env_loss_str, data_log = test_env(env_name, env, evaluator, agent, result_path, feedback_method, data_log, 1)
            loss_str += env_loss_str
            best_sr += score_summary['success_rate']
            best_spl += score_summary['spl']

            agent.env = train_env
        print('\n'.join([str((k, round(v[0], 4))) for k, v in sorted(data_log.items())]))
        best_score = current_best(data_log, -1, score_name)
        best_eval = current_best(data_log, -1, 'spl_unseen')
        print('Resumed', score_name, 'score:', best_score)
        data_log = defaultdict(list)

        if jump_iters>0:
            idx = jump_iters
            print('Jump to pretrain_n_iters',  jump_iters)
        try:
            df_path = '%s%s_log.csv' % (PLOT_DIR, model_prefix)
            df = pd.read_csv(df_path)
            data_log = {key: list(df[key]) for key in df.keys() if key != 'Unnamed: 0'}
            new_best = False
            for v_id, v in enumerate(df['iteration']):
                best_score_old = current_best(df, v_id, score_name)
                if best_score < best_score_old:
                    best_score = best_score_old
                    best_model_iter = v
                    new_best = True
            if new_best:
                print('Best score found in plot file at ', best_model_iter,', best_score/best_model_iter reseted (model won\'t be reseted).')
        except:
            pass

        if hasattr(encoder, 'n_sentences'): encoder.set_n_sentences(encoder_n_sentences)

    if 0 <agent_params['schedule_ratio'] < 1.0:
        print('Training with Scheduled Sampling, sampling ratio %.1f' % (agent_params['schedule_ratio']))
    else:
        print('Training with %s feedback' % feedback_method)

    if optm == 'RMSprop':
        optim_func = optim.RMSprop
    elif optm == 'Adamax':
        optim_func = optim.Adamax
    else: # default: Adam
        optim_func = optim.Adam

    #encoder_param_lst = list()
    #lst = list()
    #for name, pa in encoder.named_parameters():
    #    if pa.requires_grad:
    #        #lst.append(name)
    #        encoder_param_lst.append(pa)

    #encoder_optimizer = optim_func(encoder_param_lst, lr=learning_rate, weight_decay=weight_decay)
    encoder_optimizer = optim_func(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)

    decoder_optimizer = optim_func(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = time.time()

    epoch = 0
    idx = idx-log_every
    sc_started = False
    finetune_start = False
    best_model = {
        'iter': -1,
        'encoder': copy.deepcopy(agent.encoder.state_dict()),
        'decoder': copy.deepcopy(agent.decoder.state_dict()),
        'torch_cuda_rn': copy.deepcopy(torch.cuda.random.get_rng_state()),
        'torch_rn': copy.deepcopy(torch.random.get_rng_state()),
        'random': copy.deepcopy(random.getstate())
    }

    myidx = 0
    while idx+log_every < n_iters:
        idx += log_every
        interval = min(log_every, n_iters - idx)
        iter = idx + interval
        myidx += interval
        print("PROGRESS: {}%".format(round((myidx) * 100 / n_iters, 4)))

        # scheduled
        if schedule_anneal:
            agent_params['schedule_ratio'] = max(params['schedule_ratio'], (1.0-float(iter)/n_iters))
        if iter <= n_iters_resume:
            epo_inc = agent.rollout_notrain(interval)
            epoch += epo_inc
            continue

        # debug, add finetune for BERT
        if (not finetune_start) and encoder_type == 'bert' and (0 < finetune_iters <= iter):
            print("------start BERT finetune on iter %d------" % (iter))
            finetune_start = True
            agent.encoder.update = True
            #learning_rate = 5e-5

        if encoder_type == 'vlbert' and agent.encoder.__class__.__name__ in ['BertImgEncoder']:
            if warmup_iters > 0 and myidx > warmup_iters:
                agent.encoder.update = False
                warmup_iters = -1

        if (not finetune_start) and encoder_type == 'vlbert' and (0 < finetune_iters <= iter):
            print("------start VLBERT finetune on iter %d------" % (iter))
            finetune_start = True
            agent.encoder.update = True
            agent.env = finetunes[0]
            del train_env
            del val_envs
            train_env = finetunes[0]
            val_envs = finetunes[1]

            #agent.env.batch_size = finetune_batchsize  # change batchsize accordingly
            #agent.env.reset(False)
            #text_bert_param = agent.encoder.flip_text_bert_params(agent.encoder.update)
            #encoder_optimizer.add_param_group({'params': text_bert_param})
            #learning_rate = 5e-5

        if sc_started or (SELF_CRITICAL_AFTER != -1 and iter > SELF_CRITICAL_AFTER):
            if score_name!=sc_score_name:
                print('score_name changed in SC', score_name, '->',sc_score_name)
                temp_data_log = defaultdict(list)
                for env_name, (env, evaluator) in val_envs.items():
                    _, _, temp_data_log = test_env(env_name, env, evaluator, agent,
                                                   '%s%s_%s_iter_%d.json' % (RESULT_DIR, model_prefix, env_name, best_model_iter), feedback_method, temp_data_log, 1)
                    agent.env = train_env
                best_score = current_best(temp_data_log, -1, sc_score_name)
                print('Best', sc_score_name, 'score:', best_score)
            score_name = sc_score_name

            if agent.decoder.ctrl_feature:
                agent.decoder.ctrl_feature = False
                print('Auxiliary task turned off.')

            # jolin: self-critical
            if (not sc_started) and iter == SELF_CRITICAL_AFTER + log_every:
                print('SC step')
                sc_started = True
                print('Loading best model for SC from iter', best_model_iter)

                #split_string = "-".join(train_env.splits)
                split_string = "-".join(cur_split_arr)
                #enc_path = '%s%s_%s_enc_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, best_model_iter)
                #dec_path = '%s%s_%s_dec_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, best_model_iter)
                if len(resume_split_arr) == 0:
                    enc_path = '%s%s_%s_enc_iter_%d' % (pretrain_model_path, model_prefix, split_string, best_model_iter)
                    dec_path = '%s%s_%s_dec_iter_%d' % (pretrain_model_path, model_prefix, split_string, best_model_iter)
                else:
                    enc_path = '%s%s_%s_enc_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, best_model_iter)
                    dec_path = '%s%s_%s_dec_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, best_model_iter)
                agent.load(enc_path, dec_path)

                encoder_optimizer = optim.Adam(agent.encoder.parameters(), lr=sc_learning_rate, weight_decay=weight_decay)
                decoder_optimizer = optim.Adam(agent.decoder.parameters(), lr=sc_learning_rate, weight_decay=weight_decay)
                # idx = best_model_iter
                agent.env.reset_epoch()
                print('Using',sc_score_name,'for saving SC best model')
            epo_inc = agent.rl_train(train_Eval, encoder_optimizer, decoder_optimizer, interval, sc_reward_scale, sc_discouted_immediate_r_scale, sc_length_scale)
        else:
            # Train for log_every interval
            epo_inc = agent.train(encoder_optimizer, decoder_optimizer, interval, aux_n_iters, feedback=feedback_method)

        # jolin: returned from self.env._next_minibatch(R2RBatch)
        epoch += epo_inc
        data_log['iteration'].append(iter)

        train_losses = np.array(agent.losses)
        assert len(train_losses) == interval
        train_loss_avg = np.average(train_losses)
        data_log['train loss'].append(train_loss_avg)
        loss_str = 'train loss: %.4f' % train_loss_avg

        if ctrl_feature:
            if agent.decoder.ctrl_feature:
                train_loss_ctrl_f_avg = np.average(np.array(agent.losses_ctrl_f))
                data_log['loss_ctrl_f'].append(train_loss_ctrl_f_avg)
                loss_str += ' loss_ctrl_f: %.4f' % train_loss_ctrl_f_avg
            else:
                data_log['loss_ctrl_f'].append(0.)

        #split_string = "-".join(train_env.splits)
        split_string = '-'.join(cur_split_arr)
        enc_path = '%s%s_%s_enc_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, iter)
        dec_path = '%s%s_%s_dec_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, iter)
        #agent.save(enc_path, dec_path) # write or not

        # Run validation
        success_rate, spl = 0, 0
        if hasattr(encoder, 'n_sentences'): encoder.set_n_sentences(3)

        for env_name, (env, evaluator) in val_envs.items():
            score_summary, env_loss_str, data_log = test_env(env_name, env, evaluator, agent,
                                              '%s%s_%s_iter_%d.json' % (RESULT_DIR, model_prefix, env_name, iter), feedback_method, data_log, 1)
            loss_str += "," + env_name
            loss_str += env_loss_str
            success_rate += score_summary['success_rate']
            spl += score_summary['spl']

        candidate_score = current_best(data_log, -1, score_name)
        eval_candidate = current_best(data_log, -1, 'spl_unseen')

        if candidate_score>best_score:
            bert_pre_best_model_iter = best_model_iter
            best_model_iter = iter
            best_score = candidate_score
            loss_str+=' best'+score_name.upper()
            if monotonic:
                agent.copy_seq2seq()

            # best one
            best_model['iter'] = iter
            best_model['encoder'] = copy.deepcopy(agent.encoder.state_dict())
            best_model['decoder'] = copy.deepcopy(agent.decoder.state_dict())
            best_model['torch_cuda_rn'] = copy.deepcopy(torch.cuda.random.get_rng_state())
            best_model['torch_rn'] = copy.deepcopy(torch.random.get_rng_state())
            best_model['random'] = copy.deepcopy(random.getstate())

        if spl>best_spl:
            best_spl=spl
            loss_str+=' bestSPL'

        if success_rate>best_sr:
            best_sr=success_rate
            loss_str+=' bestSR'

        if hasattr(encoder, 'n_sentences'):
            encoder.set_n_sentences(encoder_n_sentences)

        if finetune_start:
            agent.env = finetunes[0]
        else:
            agent.env = train_env

        ss_str = "ss_ratio %.2f" % (agent_params['schedule_ratio'])
        print(('%s (%d %d %d%%) %s %s' % (timeSince(start, float(iter) / n_iters), iter, epoch, float(iter) / n_iters * 100, loss_str, ss_str)))

        if eval_candidate > best_eval:
            best_eval = eval_candidate
        print("EVALERR: {}%".format(best_eval))
        if save_ckpt != -1 and iter%save_ckpt == 0:
            save_best_model(best_model, SNAPSHOT_DIR, model_prefix, split_string, best_model_iter)

        df = pd.DataFrame(data_log)
        df.set_index('iteration')
        df_path = '%s%s_log.csv' % (PLOT_DIR, model_prefix)
        write_num = 0
        while (write_num < 10):
            try:
                df.to_csv(df_path)
                break
            except:
                write_num += 1

    # debug: torch save best
    save_best_model(best_model, SNAPSHOT_DIR, model_prefix, split_string, best_model_iter)
    return best_model_iter


def create_folders(path):
    """ recursively create folders """
    if not os.path.isdir(path):
        while True:
            try:
                os.makedirs(path)
            except:
                pass
                time.sleep(1)
            else:
                break

def save_best_model(best_model, SNAPSHOT_DIR, model_prefix, split_string, best_model_iter):
    """ Save the current best model """
    enc_path = '%s%s_%s_enc_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, best_model_iter)
    dec_path = '%s%s_%s_dec_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, best_model_iter)

    torch.save(best_model['encoder'], enc_path)
    torch.save(best_model['decoder'], dec_path)
    if torch.cuda.is_available():
        torch.save(best_model['torch_cuda_rn'], dec_path + '.rng.gpu')
    torch.save(best_model['torch_rn'], dec_path + '.rng')
    with open(dec_path + '.rng2', 'wb') as f:
        pickle.dump(best_model['random'], f)


def setup():
    ''' Build+Dump vocabulary for models other than bert.
    Create folders. Dump parameters    '''

    global navigable_locs_path

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Check for vocabs
    if not os.path.exists(RESULT_DIR):
        #os.mkdir(RESULT_DIR)
        create_folders(RESULT_DIR)
    if not os.path.exists(PLOT_DIR):
        #os.mkdir(PLOT_DIR)
        create_folders(PLOT_DIR)
    if not os.path.exists(SNAPSHOT_DIR):
        #os.mkdir(SNAPSHOT_DIR)
        create_folders(SNAPSHOT_DIR)
    if not os.path.exists(navigable_locs_path):
        create_folders(navigable_locs_path)

    if encoder_type!='bert' or encoder_type!='gpt':
        if not os.path.exists(TRAIN_VOCAB):
            write_vocab(build_vocab(splits=['train'], min_count=MIN_COUNT), TRAIN_VOCAB)
        if not os.path.exists(TRAINVAL_VOCAB):
            write_vocab(build_vocab(splits=['train', 'val_seen', 'val_unseen'], min_count=0), TRAINVAL_VOCAB)

    navigable_locs_path += '/navigable_locs.json'
    print('navigable_locs_path', navigable_locs_path)
    preprocess_get_pano_states(navigable_locs_path)

    global nav_graphs
    if action_space == -1:  # load navigable location cache
        with open(navigable_locs_path, 'r') as f:
            nav_graphs = json.load(f)

    print('Parameters: ')
    print(json.dumps(params, indent=2))

    if is_train:
        with open(SNAPSHOT_DIR+'params.json', 'w') as fp:
            json.dump(params, fp)


def test_submission():
    ''' Train on combined training and validation sets, and generate test submission. '''
    # TODO: think how to add pretraining here
    setup()
    # Create a batch training environment that will also preprocess text
    if use_bert:
        tok = SplitTokenizer(0, MAX_INPUT_LENGTH)
    else:
        vocab = read_vocab(TRAINVAL_VOCAB)
        tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    feature_store = Feature(features, panoramic)  # jolin
    train_env = R2RBatch(feature_store, nav_graphs, panoramic, action_space, ctrl_feature, encoder_type, batch_size=batch_size,
                         splits= submit_splits, tokenizer=tok, att_ctx_merge=att_ctx_merge) #, subgoal
    train_Eval = Evaluation(train_splits, encoder_type)  #, subgoal)  # jolin: add Evaluation() for reward calculation

    # Build models and train
    # enc_hidden_size = hidden_size // 2 if bidirectional else hidden_size
    if encoder_type == 'bert':
        encoder = BertEncoder(hidden_size, dropout_ratio, bidirectional, transformer_update, bert_n_layers, reverse_input, top_lstm).to(device)
    elif encoder_type == 'gpt':
        encoder = GptEncoder(hidden_size, dropout_ratio, bidirectional, transformer_update, bert_n_layers, reverse_input, top_lstm).to(device)
    elif encoder_type == 'transformer':
        if att_ctx_merge in ['mean','cat','max','sum']:
            encoder = MultiTransformerEncoder(len(vocab), transformer_emb_size, padding_idx, dropout_ratio,
                                              multi_share, pretrain_n_sentences, glove, heads, transformer_d_ff,
                                              hidden_size, num_layers=transformer_num_layers).to(device)
        else:
            assert not use_glove
            encoder = TransformerEncoder(len(vocab), transformer_emb_size, padding_idx, dropout_ratio,
                                         glove, heads, transformer_d_ff, hidden_size, num_layers=transformer_num_layers).to(device)
    else:
        if att_ctx_merge in ['mean','cat','max','sum']:
            encoder = EncoderMultiLSTM(len(vocab), word_embedding_size, enc_hidden_size, hidden_size, padding_idx,
                              dropout_ratio, multi_share, pretrain_n_sentences, glove, encoder_params, bidirectional=bidirectional).to(device)
        else:
            encoder = EncoderLSTM(len(vocab), word_embedding_size, enc_hidden_size, hidden_size, padding_idx,
                              dropout_ratio, glove=glove, bidirectional=bidirectional).to(device)
    decoder = AttnDecoderLSTM(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                              action_embedding_size, ctx_hidden_size, hidden_size, dropout_ratio,
                              FEATURE_SIZE, panoramic, action_space, ctrl_feature, ctrl_f_net, att_ctx_merge, ctx_dropout_ratio, decoder_params).to(device)

    if att_ctx_merge in ['mean','cat','max','sum']: encoder.set_n_sentences(3)
    train(train_env, finetunes, train_Eval, encoder, 3, decoder, monotonic, n_iters, None, -1)

    # Generate test submission
    test_env = R2RBatch(feature_store, nav_graphs, panoramic, action_space, ctrl_feature, encoder_type, batch_size=batch_size, splits=['test'], tokenizer=tok, att_ctx_merge=att_ctx_merge)  # , subgoal
    agent = Seq2SeqAgent(test_env, "", encoder, decoder,'resume', aux_ratio, decoder_init, agent_params, monotonic=monotonic,
                         episode_len=max_episode_len)  # , subgoal
    agent.results_path = '%s%s_%s_iter_%d.json' % (RESULT_DIR, model_prefix, 'test', 20000)
    agent.test(use_dropout=False, feedback='argmax', beam_size=beam_size)
    agent.write_results(dump_result)


def train_val(n_iters_resume=0):
    ''' Train on the training set, and validate on seen and unseen splits. '''

    global n_iters_pretrain_resume

    setup()
    # Create a batch training environment that will also preprocess text
    if use_bert:
        tok = SplitTokenizer(0, MAX_INPUT_LENGTH)
    else:
        vocab = read_vocab(TRAIN_VOCAB)
        tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    feature_store = Feature(features, panoramic)


    # Build models and train
    #enc_hidden_size = hidden_size // 2 if bidirectional else hidden_size
    if encoder_type == 'vlbert':
        if att_ctx_merge in ['mean','cat','max','sum']:
            if pretrain_model_name is not None:
                #encoder = MultiAddLoadEncoder(FEATURE_ALL_SIZE, enc_hidden_size, hidden_size, dropout_ratio, bidirectional, transformer_update, bert_n_layers,reverse_input, top_lstm, multi_share, pretrain_n_sentences, vl_layers, pretrain_lm_model_path + pretrain_model_name, bert_type)
                encoder = MultiVilAddEncoder(FEATURE_ALL_SIZE, enc_hidden_size, hidden_size, dropout_ratio, bidirectional, transformer_update, bert_n_layers,reverse_input, top_lstm, multi_share, pretrain_n_sentences, vl_layers, bert_type)
                encoder.load_state_dict(torch.load(pretrain_lm_model_path + pretrain_model_name))
            else:
                encoder = MultiVilAddEncoder(FEATURE_ALL_SIZE, enc_hidden_size, hidden_size, dropout_ratio, bidirectional, transformer_update, bert_n_layers,reverse_input, top_lstm, multi_share, pretrain_n_sentences, vl_layers, bert_type)

        else:
            if pretrain_model_name is not None:
                print("Using the pretrained lm model from %s" %(pretrain_model_name))
                encoder = torch.load(pretrain_lm_model_path + pretrain_model_name)
                encoder.dropout_ratio = dropout_ratio
                encoder.drop = nn.Dropout(p=dropout_ratio)
                encoder.update = transformer_update
                encoder.reverse_input = reverse_input
                encoder.top_lstm = top_lstm
                encoder.pretrain = False
            else:
                #encoder = BertAddEncoder(FEATURE_ALL_SIZE,enc_hidden_size, hidden_size, dropout_ratio, bidirectional, transformer_update, bert_n_layers, reverse_input, top_lstm, vl_layers,bert_type)
                encoder = BertImgEncoder(FEATURE_ALL_SIZE,enc_hidden_size, hidden_size, dropout_ratio, bidirectional, transformer_update, bert_n_layers, reverse_input, top_lstm,bert_type)
                #encoder = VicEncoder(FEATURE_ALL_SIZE,enc_hidden_size, hidden_size, dropout_ratio, bidirectional, transformer_update, bert_n_layers, reverse_input, top_lstm, False,bert_type)
                encoder.pretrain = False


    elif encoder_type == 'bert':
        if att_ctx_merge in ['mean','cat','max','sum']:
            encoder = MultiBertEncoder(enc_hidden_size, hidden_size, dropout_ratio, bidirectional, transformer_update, bert_n_layers,
                                       reverse_input, top_lstm, multi_share, pretrain_n_sentences, bert_type).to(device)
        else:

            if pretrain_model_name is not None:
                encoder = HugLangEncoder(enc_hidden_size, hidden_size, dropout_ratio, bidirectional, transformer_update, bert_n_layers, reverse_input, top_lstm, bert_type).to(device)
                print("Using the pretrained lm model from %s" %(pretrain_model_name))
                premodel = BertForMaskedLM.from_pretrained(pretrain_model_name)
                encoder.bert = premodel.bert
            else:
                encoder = BertEncoder(enc_hidden_size, hidden_size, dropout_ratio, bidirectional, transformer_update, bert_n_layers, reverse_input, top_lstm, bert_type).to(device)
    elif encoder_type=='gpt':
        if att_ctx_merge in ['mean','cat','max','sum']:
            encoder = MultiGptEncoder(enc_hidden_size, hidden_size, dropout_ratio, bidirectional, transformer_update, bert_n_layers, reverse_input, top_lstm, multi_share, pretrain_n_sentences).to(device)
        else:
            encoder = GptEncoder(enc_hidden_size, hidden_size, dropout_ratio, bidirectional, transformer_update, bert_n_layers, reverse_input, top_lstm).to(device)
    elif encoder_type == 'transformer':
        if att_ctx_merge in ['mean','cat','max','sum']:
            encoder = MultiTransformerEncoder(len(vocab), transformer_emb_size, padding_idx, dropout_ratio,
                                              multi_share, pretrain_n_sentences, glove, heads, transformer_d_ff, enc_hidden_size, hidden_size, top_lstm, bidirectional, num_layers=transformer_num_layers).to(device)
        else:
            assert not use_glove
            encoder = TransformerEncoder(len(vocab), transformer_emb_size, padding_idx, dropout_ratio, glove, heads,
                                         transformer_d_ff, enc_hidden_size, hidden_size, top_lstm, bidirectional, num_layers=transformer_num_layers).to(device)
    else:
        if att_ctx_merge in ['mean','cat','max','sum']:
            encoder = EncoderMultiLSTM(len(vocab), word_embedding_size, enc_hidden_size, hidden_size, padding_idx,
                              dropout_ratio, multi_share, pretrain_n_sentences, glove, encoder_params, bidirectional=bidirectional).to(device)
        else:
            encoder = EncoderLSTM(len(vocab), word_embedding_size, enc_hidden_size, hidden_size, padding_idx,
                              dropout_ratio, glove=glove, bidirectional=bidirectional).to(device)

    decoder = AttnDecoderLSTM(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                              action_embedding_size, ctx_hidden_size, hidden_size, dropout_ratio,
                              FEATURE_SIZE, panoramic, action_space, ctrl_feature, ctrl_f_net, att_ctx_merge, ctx_dropout_ratio, decoder_params).to(device)

    if pretrain_decoder_name is not None:
        decoder.load_state_dict(torch.load(pretrain_lm_model_path + pretrain_decoder_name))
    #envs = EnvBatch(feature_store=feature_store, batch_size=batch_size)
    #if torch.cuda.device_count() > 1:
    #    encoder = CustomDataParallel(encoder)
    #    decoder = CustomDataParallel(decoder)
    encoder.cuda()
    decoder.cuda()


    # Create validation environments
    val_envs = {split: (R2RBatch(feature_store, nav_graphs, panoramic, action_space, ctrl_feature, encoder_type, batch_size=batch_size,
             splits=[split], tokenizer=tok, att_ctx_merge=att_ctx_merge), Evaluation([split], encoder_type)) for split in val_splits}  # subgoal

    # Create training env
    train_env = R2RBatch(feature_store, nav_graphs, panoramic, action_space, ctrl_feature, encoder_type, batch_size=batch_size,
                          splits=train_splits, tokenizer=tok, att_ctx_merge=att_ctx_merge)

    train_Eval = Evaluation(train_splits, encoder_type)# jolin: add Evaluation() for reward calculation
    finetunes = None

    # Create pretraining env
    resume_splits = train_env.splits

    resume_splits_array = []
    cur_splits_array = []

    last_train_iters = -1

    if use_pretrain:
        print('Pretraining Step')
        pretrain_env = R2RBatch(feature_store, nav_graphs, panoramic, action_space, ctrl_feature, encoder_type, batch_size=batch_size,
                          splits=pretrain_splits, tokenizer=tok, att_ctx_merge=att_ctx_merge, min_n_sentences=pretrain_n_sentences)

        # pre-trained scheduled sampling
        if (agent_params['schedule_ratio'] > 0 and agent_params['schedule_ratio'] < 1.0) and ss_n_pretrain_iters > 0:
            print('Pretraining with Scheduled Sampling, sampling ratio %.1f' % (agent_params['schedule_ratio']))
            #cur_splits_array = pretrain_env.splits
            cur_splits_array = ['ss', 'pretrain']

            #best_model_ss_pretrain_iter = train(pretrain_env, None, encoder, decoder, None, ss_n_iters, pretrain_env.splits, pretrain_score_name, -1, log_every, val_envs=val_envs, n_iters_resume=n_iters_pretrain_resume)
            best_model_ss_pretrain_iter = train(pretrain_env, None, None, encoder, decoder, None, ss_n_pretrain_iters, resume_splits_array, cur_splits_array,
                                                pretrain_score_name, last_train_iters, log_every, val_envs=val_envs, n_iters_resume=n_iters_pretrain_resume,warmup_iters=warmup_iters)
            n_iters_pretrain_resume = best_model_ss_pretrain_iter
            agent_params['schedule_ratio'] = -1.0
            print('Changing to %s Pretraining with the best model at iteration %d' % (feedback_method, n_iters_pretrain_resume))
            resume_splits_array = ['ss', 'pretrain']
            last_train_iters = ss_n_pretrain_iters

        if train_in_pretrain:
            # training; may need to cancel training for pre-train
            cur_splits_array = ['pretrain']
            n_iters_resume = train(pretrain_env, None, None, encoder, decoder, None, pretrain_n_iters, resume_splits_array, cur_splits_array, #pretrain_env.splits,
                               pretrain_score_name, last_train_iters, val_envs=val_envs, n_iters_resume=n_iters_pretrain_resume,warmup_iters=warmup_iters)
            resume_splits = pretrain_splits
            resume_splits_array = ['pretrain']
            last_train_iters = pretrain_n_iters
        else:
            print('Skip Train in pre-training!')
            n_iters_resume = n_iters_pretrain_resume
            resume_splits_array = ['ss', 'pretrain']
            # agent_params['schedule_ratio'] = params['schedule_ratio']
            agent_params['schedule_ratio'] = params['schedule_ratio']
    else:
        if n_iters_resume>0:  # use train splits
            pass
        elif n_iters_pretrain_resume>0:
            print('Skip pretraining but use pretrained model')
            n_iters_resume = n_iters_pretrain_resume
            resume_splits = pretrain_splits
        agent_params['schedule_ratio'] = params['schedule_ratio']

    if att_ctx_merge in ['mean','cat','max','sum']: encoder.set_n_sentences(3)
    print('Training Step')

    # scheduled sampling
    if 0 < agent_params['schedule_ratio'] < 1.0 and ss_n_iters > 0:
        print('Training with Scheduled Sampling, sampling ratio %.1f' % (agent_params['schedule_ratio']))

        # train_splits_array = resume_splits
        #resume_splits_array = ['pretrain']
        cur_splits_array = ['ss', 'train']
        best_model_ss_iter = train(train_env, finetunes, train_Eval, encoder, decoder, monotonic, ss_n_iters, resume_splits_array, cur_splits_array, #resume_splits,
                                   train_score_name, last_train_iters if use_pretrain else -1, val_envs=val_envs, n_iters_resume=n_iters_resume,warmup_iters=warmup_iters)
        n_iters_resume = best_model_ss_iter
        agent_params['schedule_ratio'] = -1.0
        print('Changing to %s training with the best model at iteration %d' % (feedback_method, n_iters_resume))
        resume_splits_array = ['ss', 'train']
        last_train_iters = ss_n_iters

    # training
    #resume_splits_array = ['ss', 'train']
    cur_splits_array = ['train']
    best_model_iter = train(train_env, finetunes, train_Eval, encoder, decoder, monotonic, n_iters, resume_splits_array, cur_splits_array, #resume_splits,
                            train_score_name, last_train_iters if use_pretrain else -1, val_envs=val_envs, n_iters_resume=n_iters_resume,warmup_iters=warmup_iters)
    print("The best model iter is %d" % (best_model_iter))
    return best_model_iter


def load_test(n_iters_resume):

    setup()
    # Create a batch training environment that will also preprocess text
    if use_bert:
        tok = SplitTokenizer(0, MAX_INPUT_LENGTH)
    else:
        vocab = read_vocab(TRAIN_VOCAB)
        tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    feature_store = Feature(features, panoramic)  # jolin

    #envs = EnvBatch(feature_store=feature_store, batch_size=batch_size, beam_size=beam_size)

    train_env = R2RBatch(feature_store, nav_graphs, panoramic, action_space, ctrl_feature, encoder_type, beam_size, batch_size=batch_size,
                          splits=train_splits, tokenizer=tok, att_ctx_merge=att_ctx_merge) # , subgoal

    # Creat validation environments
    val_envs = {split: (R2RBatch(feature_store, nav_graphs, panoramic, action_space, ctrl_feature, encoder_type, beam_size, batch_size=batch_size, splits=[split],
                                 tokenizer=tok, att_ctx_merge=att_ctx_merge), Evaluation([split], encoder_type)) for split in val_splits}  # , subgoal

    # Build models and train
    # enc_hidden_size = hidden_size // 2 if bidirectional else hidden_size
    if encoder_type =='bert':
        if att_ctx_merge in ['mean','cat','max','sum']:
            encoder = MultiBertEncoder(enc_hidden_size, hidden_size, dropout_ratio, bidirectional, transformer_update, bert_n_layers,
                                       reverse_input, top_lstm, multi_share, pretrain_n_sentences, bert_type).to(device)
        else:
            encoder = BertEncoder(enc_hidden_size, hidden_size, dropout_ratio, bidirectional, transformer_update, bert_n_layers, reverse_input, top_lstm, bert_type).to(device)
        #encoder = BertEncoder(hidden_size, dropout_ratio, bidirectional, transformer_update, bert_n_layers, reverse_input, top_lstm).to(device)
    elif encoder_type == 'gpt':
        if att_ctx_merge in ['mean','cat','max','sum']:
            encoder = MultiGptEncoder(enc_hidden_size, hidden_size, dropout_ratio, bidirectional, transformer_update, bert_n_layers,
                                      reverse_input, top_lstm, multi_share, pretrain_n_sentences).to(device)
        else:
            encoder = GptEncoder(enc_hidden_size, hidden_size, dropout_ratio, bidirectional, transformer_update, bert_n_layers, reverse_input, top_lstm).to(device)
        #encoder = GptEncoder(hidden_size, dropout_ratio, bidirectional, transformer_update, bert_n_layers, reverse_input, top_lstm).to(device)
    elif encoder_type == 'transformer':
        if att_ctx_merge in ['mean','cat','max','sum']:
            encoder = MultiTransformerEncoder(len(vocab), transformer_emb_size, padding_idx, dropout_ratio,
                                              multi_share, 3, glove, heads, transformer_d_ff, hidden_size, num_layers=transformer_num_layers).to(device)
        else:
            assert not use_glove
            encoder = TransformerEncoder(len(vocab), transformer_emb_size, padding_idx, dropout_ratio,
                                         glove, heads, transformer_d_ff, hidden_size, num_layers=transformer_num_layers).to(device)
    else:
        if att_ctx_merge in ['mean','cat','max','sum']:
            encoder = EncoderMultiLSTM(len(vocab), word_embedding_size, enc_hidden_size, hidden_size, padding_idx,
                              dropout_ratio, multi_share, 3, glove, encoder_params, bidirectional=bidirectional).to(device)
        else:
            encoder = EncoderLSTM(len(vocab), word_embedding_size, enc_hidden_size, hidden_size, padding_idx,
                              dropout_ratio, glove=glove, bidirectional=bidirectional).to(device)

    decoder = AttnDecoderLSTM(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                              action_embedding_size, ctx_hidden_size, hidden_size, dropout_ratio,
                              FEATURE_SIZE, panoramic, action_space, ctrl_feature, ctrl_f_net, att_ctx_merge, ctx_dropout_ratio, decoder_params).to(device)

    agent = Seq2SeqAgent(train_env, "", encoder, decoder, 'resume', aux_ratio, decoder_init, agent_params, monotonic, max_episode_len, state_factored=state_factored) # , subgoal

    #split_string = "-".join(train_splits)
    cur_split_arr = ['train']
    split_string = "-".join(cur_split_arr)

    #enc_path = '%s%s_%s_enc_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, n_iters_resume)
    #dec_path = '%s%s_%s_dec_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, n_iters_resume)
    enc_path = '%s%s_%s_enc_iter_%d' % (pretrain_model_path, model_prefix, split_string, n_iters_resume)
    dec_path = '%s%s_%s_dec_iter_%d' % (pretrain_model_path, model_prefix, split_string, n_iters_resume)
    agent.load(enc_path, dec_path)

    ###############################load speaker#################################
    if use_speaker:

        print('Loading speaker for inference...')
        from collections import namedtuple
        SpeakerArgs = namedtuple('SpeakerArgs',['use_train_subset', 'n_iters', 'no_save','result_dir','snapshot_dir','plot_dir','seed','image_feature_type','image_feature_datasets'])
        from speaker import train_speaker
        speaker_args = SpeakerArgs(use_train_subset=False, n_iters=20000,no_save=False,result_dir=train_speaker.RESULT_DIR, snapshot_dir=train_speaker.SNAPSHOT_DIR, plot_dir=train_speaker.PLOT_DIR,seed=10,image_feature_datasets=['imagenet'],image_feature_type=['mean_pooled'])
        speaker, _, _ = train_speaker.train_setup(speaker_args)
        load_args = {'map_location':device}
        speaker.load(speaker_prefix, **load_args)

        print('Speaker loaded')
    else:
        speaker = None
    ############################################################################
    loss_str, data_log = '', defaultdict(list)
    for env_name, (env, evaluator) in val_envs.items():
        json_path = '%s%s_%s_iter_%d.json' % (RESULT_DIR, model_prefix, env_name, n_iters_resume)
        score_summary, env_loss_str, data_log = test_env(env_name, env, evaluator, agent, json_path, feedback_method, data_log, beam_size, speaker)
        loss_str += env_loss_str
    print('Result of Iteration ', n_iters_resume)
    print('\n'.join([str((k, round(v[0], 4))) for k, v in sorted(data_log.items())]))
    print('=========================================================')

    if hasattr(encoder, 'n_sentences') and single_sentence_test:
        encoder.set_n_sentences(1)
        print('====Additional test for single instruction performance============')
        # Creat validation environments
        val_envs = {split: (
        R2RBatch(feature_store, nav_graphs, panoramic, action_space, ctrl_feature, encoder_type, beam_size=beam_size, batch_size=batch_size,
                 splits=[split], tokenizer=tok, att_ctx_merge='Eval'), Evaluation([split], encoder_type)) for split in val_splits if split!='test'}  # , subgoal
        loss_str, data_log = '', defaultdict(list)

        for env_name, (env, evaluator) in val_envs.items():
            json_path = '%s%s_%s_iter_%d_add.json' % (RESULT_DIR, model_prefix, env_name, n_iters_resume)
            score_summary, env_loss_str, data_log = test_env(env_name, env, evaluator, agent, json_path, feedback_method, data_log, beam_size)
            loss_str += env_loss_str
        print('Result of Iteration ', n_iters_resume)
        print('\n'.join([str((k, round(v[0], 4))) for k, v in sorted(data_log.items())]))
        print('====Additional test for single instruction performance finished===')


def test_env(env_name, env, evaluator, agent, results_path, feedback, data_log, beam_size, speaker=None):
    save_env = agent.env  # for restore
    save_results_path = agent.results_path  # for restore
    agent.env = env
    agent.results_path = results_path
    env_loss_str = ""

    """
    if env_name!='test' and beam_size==1:
        # Get validation loss under the same conditions as training
        agent.test(use_dropout=True, feedback=feedback, allow_cheat=True, beam_size=beam_size, successors=successors, speaker=(None, None, None, None))
        val_losses = np.array(agent.losses)
        val_loss_avg = np.average(val_losses)
        data_log['%s loss' % env_name].append(val_loss_avg)
        if ctrl_feature:
            if agent.decoder.ctrl_feature:
                val_losses_ctrl_f = np.array(agent.losses_ctrl_f)
                val_loss_ctrl_f_avg = np.average(val_losses_ctrl_f)
                data_log['%s loss_ctrl_f' % env_name].append(val_loss_ctrl_f_avg)
            else:
                data_log['%s loss_ctrl_f' % env_name].append(0.)
    """
    # Get validation distance from goal under test evaluation conditions
    agent.test(use_dropout=False, feedback='argmax', beam_size=beam_size, successors=successors, speaker=(speaker,speaker_weight,speaker_merge,evaluator))
    output = agent.write_results(dump_result)
    score_summary = None

    if env_name!='test':
        if dump_result:
            score_summary, _ = evaluator.score(agent.results_path)
        else: score_summary, _ = evaluator.score_output(output)

        if beam_size == 1:
            #env_loss_str += ', %s loss: %.4f' % (env_name, val_loss_avg)
            if ctrl_feature:
                if agent.decoder.ctrl_feature:
                    env_loss_str += ', %s loss_ctrl_f: %.4f' % (env_name, val_loss_ctrl_f_avg)
        for metric, val in score_summary.items():
            data_log['%s %s' % (env_name, metric)].append(val)
            if metric in ['success_rate','spl']:
                env_loss_str += ', %s: %.3f' % (metric, val)
    # restore
    agent.env = save_env
    agent.results_path = save_results_path
    return score_summary, env_loss_str, data_log


if is_train:
    assert beam_size==1
    N_ITERS_RESUME = train_val(N_ITERS_RESUME)  # resume from iter
else:
    load_test(N_ITERS_RESUME) # test iter
# test_submission()
