
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
from model import BertAddEncoder
from pretrain_class import BertAddPreTrain
from pytorch_transformers import BertTokenizer
from batch_loader import NavBertDataset
from feature import Feature
import tqdm
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



class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


def sort_seq(seq_tensor):
    seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
    seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]  # Full length

    seq_tensor = torch.from_numpy(seq_tensor)
    seq_lengths = torch.from_numpy(seq_lengths)

    # Sort sequences by lengths
    seq_lengths, perm_idx = seq_lengths.sort(0, True)
    sorted_tensor = seq_tensor[perm_idx]
    mask = (sorted_tensor == padding_idx)[:, :seq_lengths[0]]

    return sorted_tensor, mask, seq_lengths, perm_idx



class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.
        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction
    """

    def __init__(self, model, vocab_size,
                 train_dataloader, test_dataloader= None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, include_next = False, include_vision = True, total_epochs = 1):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        n_gpu = torch.cuda.device_count()
        print("device", device, "n_gpu", n_gpu)

        # Initialize the BERT Language Model, with BERT model
        self.model = model.to(self.device)
        self.bert = self.model.bert
        self.padding_idx = 0
        self.include_next = include_next
        self.include_vision = include_vision

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = optim.Adamax(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        if self.model.__class__.__name__ == 'DataParallel':
            self.optim_schedule = ScheduledOptim(self.optim, self.model.module.bert.transformer_hidden_size, n_warmup_steps=warmup_steps)
        else:
            self.optim_schedule = ScheduledOptim(self.optim, self.model.bert.transformer_hidden_size, n_warmup_steps=warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        #self.criterion = nn.NLLLoss(ignore_index=0)

        self.log_freq = log_freq
        self.total_iters = total_epochs * len(train_dataloader)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch
        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}",
                              disable=False)

        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        for i, data in data_iter: 
                if self.include_next:
                    next_sent_output, mask_lm_output, loss = self.model.forward(sorted_tensor.cuda(), mask.cuda(),seq_lengths.cuda(),labels[:,:seq_lengths[0]].cuda(), isnext.cuda(), f_t_all.cuda())
                else:
                    next_sent_output, mask_lm_output, loss = self.model.forward(sorted_tensor.cuda(), mask.cuda(),seq_lengths.cuda(),labels[:,:seq_lengths[0]].cuda(), None, f_t_all.cuda())
            else:
                next_sent_output, mask_lm_output, loss = self.model.forward(sorted_tensor.cuda(), mask.cuda(),seq_lengths.cuda(),labels[:,:seq_lengths[0]].cuda(), None, None)

            # 2-1. NLL(negative log likelihood) loss of is_next classification result
            #next_loss = 0
            #if self.include_vision and self.include_next:
            #    next_loss = self.criterion(next_sent_output, isnext.cuda())

            # 2-2. NLLLoss of predicting masked token word
            #mask_loss = self.criterion(mask_lm_output.transpose(1, 2), labels[:,:seq_lengths[0]].cuda())

            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            #loss = next_loss + mask_loss

            # 3. backward and optimization only in train
                loss = loss.mean()
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            # next vision prediction accuracy
            if self.include_next:
                correct = next_sent_output.argmax(dim=-1).eq(isnext.cuda()).sum().item()
                total_correct += correct
                total_element += data["isnext"].nelement()
            avg_loss += loss.item()

            if self.include_next:
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "avg_acc": total_correct / total_element * 100,
                    "loss": loss.item()
                }
            else:
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "loss": loss.item()
                }

            #if i % self.log_freq == 0:
            #    data_iter.write(str(post_fix))


            if i % 100 == 0:
                #print("PROGRESS: {}%".format(round((myidx) * 100 / n_iters, 4)))
                print("\n")
                print("PROGRESS: {}%".format(round((epoch * len(data_loader) + i) * 100 / self.total_iters, 4)))
                print("EVALERR: {}%".format(avg_loss / (i+1)))

        #print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter))

    def save(self, epoch, file_path="pretrained_models/addbert_trained.model"):
        """
        Saving the current BERT model on file_path
        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_vocab', dest='train_vocab', type=str, default='train_vocab.txt', help='train_vocab filename (in snapshots folder)')
    parser.add_argument('--trainval_vocab', dest='trainval_vocab', type=str, default='trainval_vocab.txt', help='trainval_vocab filename (in snapshots folder)')
    parser.add_argument('--result_dir', dest='result_dir', type=str, default='tasks/R2R/results/', help='path to the result_dir file')
    parser.add_argument('--snapshot_dir', dest='snapshot_dir', type=str, default='tasks/R2R/snapshots/', help='path to the snapshot_dir file')
    parser.add_argument('--plot_dir', dest='plot_dir', type=str, default='tasks/R2R/plots/', help='path to the plot_dir file')

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

    parser.add_argument('--log_every', dest='log_every', default=20, type=int, help='log_every')
    parser.add_argument('--save_ckpt', dest='save_ckpt', default=48, type=int, help='dump model checkpoint, default -1')

    parser.add_argument('--schedule_ratio', dest='schedule_ratio', default=0.2, type=float, help='ratio for sample or teacher')
    parser.add_argument('--schedule_anneal', dest='schedule_anneal', action='store_true', help='schedule_ratio is annealling or not')
    parser.add_argument('--dropout_ratio', dest='dropout_ratio', default=0.4, type=float, help='dropout_ratio')
    parser.add_argument('--temp_alpha', dest='temp_alpha', default=1.0, type=float, help='temperate alpha for softmax')
    parser.add_argument('--learning_rate', dest='learning_rate', default=5e-05, type=float, help='learning_rate')
    parser.add_argument('--sc_learning_rate', dest='sc_learning_rate', default=2e-05, type=float, help='sc_learning_rate')
    parser.add_argument('--weight_decay', dest='weight_decay', default=0.01, type=float, help='weight_decay')
    parser.add_argument('--optm', dest='optm', default='Adamax', type=str, help='Adam, Adamax, RMSprop')

    parser.add_argument('--sc_reward_scale', dest='sc_reward_scale', default=1., type=float, help='sc_reward_scale')
    parser.add_argument('--sc_discouted_immediate_r_scale', dest='sc_discouted_immediate_r_scale', default=0., type=float, help='sc_discouted_immediate_r_scale')
    parser.add_argument('--sc_length_scale', dest='sc_length_scale', default=0., type=float, help='sc_length_scale')

    parser.add_argument('--feedback_method', dest='feedback_method', type=str, default='teacher', help='sample or teacher')
    parser.add_argument('--bidirectional', dest='bidirectional', type=boolean_string, default=True, help='bidirectional')
    parser.add_argument('--include_next', dest='include_next', type=boolean_string, default=False, help='next vision prediction')
    parser.add_argument('--include_vision', dest='include_vision', type=boolean_string, default=True, help='include vision feature or not')

    parser.add_argument('--monotonic_sc', dest='monotonic_sc', type=boolean_string, default=False, help='monotonic self-critic')
    parser.add_argument('--panoramic', dest='panoramic', type=boolean_string, default=True, help='panoramic img')
    parser.add_argument('--action_space', dest='action_space', type=int, default=-1, help='6 or -1(navigable viewpoints)')
    parser.add_argument('--ctrl_feature', dest='ctrl_feature', type=boolean_string, default=False, help='ctrl_feature')
    parser.add_argument('--ctrl_f_net', dest='ctrl_f_net', type=str, default='linear', help='imglinear, linear, nonlinear, imgnl or deconv')
    parser.add_argument('--aux_n_iters', dest='aux_n_iters', type=int, help='update auxiliary net after aux_n_iters')
    parser.add_argument('--aux_ratio', dest='aux_ratio', type=float, help='aux_ratio')

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

    parser.add_argument('--use_cuda', dest='use_cuda', type=boolean_string, default=True, help='use_cuda')
    parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=0, help='gpu_id')
    parser.add_argument('--train', dest='train', type=boolean_string, default=True, help='train or test')

    parser.add_argument('--pretrain_score_name', dest='pretrain_score_name', type=str, default='sr_sum', help='sr_sum spl_sum sr_unseen spl_unseen')
    parser.add_argument('--train_score_name', dest='train_score_name', type=str, default='spl_unseen', help='sr_sum spl_sum sr_unseen spl_unseen')
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

    parser.add_argument('--reward_func', dest='reward_func', type=str, default='spl', help='reward function: sr_sc, spl, spl_sc, spl_last, spl_last_sc')

    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--philly', action='store_true', help='program runs on Philly, used to redirect `write_model_path`')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train')
    parser.add_argument('--vl_layers', type=int, default=1, help='number of vl layers')


    args = parser.parse_args()
    params = vars(args)


assert params['panoramic'] or (params['panoramic'] == False and params['action_space'] == 6)
RESULT_DIR = params['result_dir'] #'tasks/R2R/results/'
SNAPSHOT_DIR = params['snapshot_dir'] #'tasks/R2R/snapshots/'
PLOT_DIR = params['plot_dir'] #'tasks/R2R/plots/'

TRAIN_VOCAB = os.path.join(SNAPSHOT_DIR, params['train_vocab']) #'tasks/R2R/data/train_vocab.txt'
TRAINVAL_VOCAB = os.path.join(SNAPSHOT_DIR, params['trainval_vocab'])

batch_size = params['batch_size'] #100
top_lstm = params['top_lstm']
transformer_update = params['transformer_update']
bert_n_layers = params['bert_n_layers']  # 1
reverse_input = False

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

bidirectional = params['bidirectional'] #False
include_vision = params['include_vision'] #True
include_next = params['include_next'] #False

dropout_ratio = params['dropout_ratio'] # 0.5
learning_rate = params['learning_rate'] # 0.0001
weight_decay = params['weight_decay'] # 0.0005
optm = params['optm']

panoramic = params['panoramic']

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


features = FEATURE_STORE

use_bert = True  # for tokenizer and dataloader
ctx_hidden_size = enc_hidden_size * (2 if bidirectional else 1)
if (use_bert and not top_lstm):
    ctx_hidden_size = 768

bert_type = params['bert_type']
philly = params['philly']
seed = params['seed']
epochs = params['epochs']
vl_layers = params['vl_layers']



if philly: # use philly
    print('Info: Use Philly, all the output folders are reset.')
    RESULT_DIR = os.path.join(os.getenv('PT_OUTPUT_DIR'), params['result_dir'])
    PLOT_DIR = os.path.join(os.getenv('PT_OUTPUT_DIR'), params['plot_dir'])
    SNAPSHOT_DIR = os.path.join(os.getenv('PT_OUTPUT_DIR'), params['snapshot_dir'])
    TRAIN_VOCAB = os.path.join(SNAPSHOT_DIR, params['train_vocab'])
    TRAINVAL_VOCAB = os.path.join(SNAPSHOT_DIR, params['trainval_vocab'])
    JFILE = os.path.join(os.getenv('PT_OUTPUT_DIR'), 'collect_traj')

    print('RESULT_DIR', RESULT_DIR)
    print('PLOT_DIR', PLOT_DIR)
    print('SNAPSHOT_DIR', SNAPSHOT_DIR)
    print('TRAIN_VOC', TRAIN_VOCAB)
    print('JFILE', JFILE)

feature_store = 'img_features/ResNet-152-imagenet.tsv'
panoramic = True
print(params)

def train():

    import glob
    jfiles = glob.glob("./collect_traj" + "/*.json")
    bparams = {'batch_size':batch_size, 'shuffle': True, 'num_workers':4}
    tok = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = NavBertDataset(jfiles, tok, feature_store, panoramic)
    print("you have loaded %d  time steps" % (len(dataset)))
    data_gen = data.DataLoader(dataset, **bparams)

    model = BertAddPreTrain(FEATURE_ALL_SIZE, enc_hidden_size, hidden_size, dropout_ratio, bidirectional, transformer_update, bert_n_layers, reverse_input, top_lstm, vl_layers,bert_type, len(tok))
    trainer = BERTTrainer(model, len(tok), train_dataloader=data_gen, lr = learning_rate, include_next=include_next, include_vision = include_vision,total_epochs=epochs)
    #(model, vocab_size: int,train_dataloader: DataLoader, test_dataloader: DataLoader = None,lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,with_cuda: bool = True,cuda_devices=None, log_freq: int = 10, include_next = False)
    modelname = "pretrained_iv{:d}_in{:d}_bt{:03d}_lr{:.5f}_dr{:.2f}".format(include_vision, include_next,batch_size,learning_rate, dropout_ratio)
    for epoch in range(epochs):
        trainer.train(epoch)
        #if epoch>99 and epoch % 10 == 0:
        if epoch == 5:
            trainer.save(epoch, SNAPSHOT_DIR+ modelname + "_ep%05d.model" % (epoch))
        #trainer.save(epoch, 'pretrained_models/'+ modelname + "_ep%05d.model" % (epoch))

train()

