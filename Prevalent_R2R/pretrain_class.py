
from numpy import cos
from pytorch_transformers import BertPreTrainedModel,BertConfig
from pytorch_transformers.modeling_bert import BertOnlyMLMHead, BertPreTrainingHeads
from model import  BertAddEncoder, BertImgEncoder,BertLangEncoder,BertAddSepEncoder
from vilmodel import BertAddModel,VicModel,DicModel
import torch
import torch.nn as nn
import pdb

class HugAddActionPreTrain(BertPreTrainedModel):
    def __init__(self,config):
        super(HugAddActionPreTrain, self).__init__(config)

        self.config = config
        self.bert = BertAddModel(config)


        self.next_action = NextActionPrediction(self.config.hidden_size, self.config.action_space)


        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.mlmhead = BertOnlyMLMHead(self.config)

        self.init_weights()
        self.tie_weights()


    def tie_weights(self):
        self._tie_or_clone_weights(self.mlmhead.predictions.decoder,self.bert.embeddings.word_embeddings)


    def forward(self, seq,labels, isnext=None, f_t_all = None):

        ctx, pooled_out = self.bert(seq, img_feats=f_t_all)

        cls_part = pooled_out

        if f_t_all is not None:
            vision_len = f_t_all.shape[1]
        else:
            vision_len = 0
        lang_part = ctx[:, vision_len:, :]

        prediction_scores = self.mlmhead(lang_part)
        mask_loss = self.criterion(prediction_scores.view(-1,self.config.vocab_size), labels[:,:lang_part.shape[1]].contiguous().view(-1))


        action_scores = self.next_action(cls_part)
        next_loss = 0
        if isnext is not None:
            next_loss = self.criterion(action_scores, isnext)
        loss = mask_loss + next_loss

        return loss,prediction_scores, action_scores

class VicAddActionPreTrain(BertPreTrainedModel):
    def __init__(self,config):
        super(VicAddActionPreTrain, self).__init__(config)

        self.config = config
        self.bert = VicModel(config)


        self.next_action = NextActionPrediction(self.config.hidden_size, self.config.action_space)


        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.mlmhead = BertOnlyMLMHead(self.config)

        self.init_weights()
        self.tie_weights()


    def tie_weights(self):
        self._tie_or_clone_weights(self.mlmhead.predictions.decoder, self.bert.embeddings.word_embeddings)


    def forward(self, seq,labels, isnext=None, f_t_all = None,lang_mask=None):

        ctx, pooled_out = self.bert(seq, attention_mask=lang_mask,img_feats=f_t_all)

        cls_part = pooled_out
        lang_part = ctx

        prediction_scores = self.mlmhead(lang_part)
        mask_loss = self.criterion(prediction_scores.view(-1,self.config.vocab_size), labels.view(-1))


        action_scores = self.next_action(cls_part)
        next_loss = 0
        if isnext is not None:
            next_loss = self.criterion(action_scores, isnext)
        loss = mask_loss + next_loss

        return loss,prediction_scores, action_scores


class DicAddActionPreTrain(BertPreTrainedModel):
    def __init__(self,config):
        super(DicAddActionPreTrain, self).__init__(config)

        self.config = config
        self.bert = DicModel(config)

        self.matching = MatchPrediction(self.config.hidden_size, 1)
        self.orient_matching = OrientMatchPrediction(self.config.hidden_size, 4)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.orient_criterion = nn.MSELoss()
        self.match_criterion = nn.BCELoss()
        self.alignment = AlignmentMatrix()
        self.mlmhead = BertOnlyMLMHead(self.config)
        #self.cls = BertPreTrainingHeads(self.config)
        self.cosin_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)

        self.init_weights()
        self.tie_weights()


    def tie_weights(self):
        self._tie_or_clone_weights(self.mlmhead.predictions.decoder,self.bert.embeddings.word_embeddings)


    def forward(self, seq,labels,isnext=None, f_t_all = None,lang_mask=None, vis_mask=None, \
                match_label=None, orient_label=None):

        ctx, pooled_out, vision_score, only_vision_pooled_output,  pos_pooled_output = self.bert(seq, attention_mask=lang_mask,img_feats=f_t_all)


        cls_part = pooled_out
        lang_part = ctx

        prediction_scores = self.mlmhead(lang_part)
        mask_loss = self.criterion(prediction_scores.view(-1,self.config.vocab_size), labels.view(-1))

        
        #action_scores = self.next_action(cls_part)
        action_scores = vision_score

        next_loss = 0
        if isnext is not None:
            next_loss = self.criterion(action_scores, isnext)
        

        loss = mask_loss + next_loss

        if match_label is not None:
            predict_match = self.matching(only_vision_pooled_output)
            match_loss = self.match_criterion(predict_match, match_label)
            loss += match_loss
        
        if orient_label is not None:
            predict_orient_match = self.orient_matching(pos_pooled_output)
            orient_match_loss = self.orient_criterion(predict_orient_match, orient_label)
            loss += orient_match_loss

        return loss, mask_loss, next_loss


class BertAddPreTrain(nn.Module):
    def __init__(self,vision_size, hidden_size, dec_hidden_size,dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, vl_layers, bert_type, vocab_size):
        super(BertAddPreTrain, self).__init__()

        self.bert = BertAddEncoder(vision_size, hidden_size, dec_hidden_size,dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, vl_layers, bert_type)

        self.dropout = nn.Dropout(dropout_ratio)
        if not self.bert.top_lstm:
            self.next_vision = NextImgPrediction(self.bert.transformer_hidden_size)
            self.mask_lm = MaskedLanguageModel(self.bert.transformer_hidden_size, vocab_size)
        else:
            self.num_directions = 2 if bidirectional else 1
            in_size = hidden_size * self.num_directions
            self.next_vision = NextImgPrediction(in_size)
            self.mask_lm = MaskedLanguageModel(in_size, vocab_size)

        self.criterion = nn.NLLLoss(ignore_index=0)




    def forward(self, seq, seq_mask, seq_lengths, labels, isnext=None, f_t_all = None):

        ctx, ht, ct, vl_mask = self.bert(seq, seq_mask, seq_lengths, f_t_all=f_t_all)

        if f_t_all is not None:
            vision_len = f_t_all.shape[1]
        else:
            vision_len = 0
        cls_part = ctx[:,vision_len, :]
        lang_part = ctx[:, vision_len+1:, :]

        #lang_part = self.dropout(lang_part)   # necessary to dropout==-p here?
        next_vision_output = self.next_vision(cls_part)
        mask_lm_output = self.mask_lm(lang_part)

        # calculate the loss
        mask_loss = self.criterion(mask_lm_output.transpose(1, 2), labels[:, 1:])
        next_loss = 0
        if isnext is not None:
            next_loss = self.criterion(next_vision_output, isnext)
        loss = mask_loss + next_loss


        return next_vision_output, mask_lm_output, loss

class BertAddActionPreTrain(nn.Module):
    def __init__(self,vision_size, hidden_size, dec_hidden_size,dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, vl_layers, bert_type, vocab_size, action_space):
        super(BertAddActionPreTrain, self).__init__()

        self.bert = BertAddEncoder(vision_size, hidden_size, dec_hidden_size,dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, vl_layers, bert_type)
        self.config = self.bert.config

        self.dropout = nn.Dropout(dropout_ratio)
        if not self.bert.top_lstm:
            self.next_action = NextActionPrediction(self.bert.transformer_hidden_size, action_space)
        else:
            self.num_directions = 2 if bidirectional else 1
            self.in_size = hidden_size * self.num_directions
            self.next_action = NextActionPrediction(self.in_size, action_space)
            self.config.hidden_size = self.in_size

        self.lang_criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.act_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.mlmhead = BertOnlyMLMHead(self.config)



    def forward(self, seq, seq_mask, seq_lengths, labels, isnext=None, f_t_all = None):

        ctx, ht, ct, vl_mask,pooled_out = self.bert(seq, seq_mask, seq_lengths, f_t_all=f_t_all)

        if not self.bert.top_lstm:
            cls_part = pooled_out
        else:
            cls_part = torch.cat((ht,ct), dim=1)

        if f_t_all is not None:
            vision_len = f_t_all.shape[1]
        else:
            vision_len = 0
        lang_part = ctx[:, vision_len:, :]

        prediction_scores = self.mlmhead(lang_part)
        mask_loss = self.lang_criterion(prediction_scores.view(-1,self.config.vocab_size), labels[:,:lang_part.shape[1]].contiguous().view(-1))


        action_scores = self.next_action(cls_part)
        next_loss = 0
        if isnext is not None:
            next_loss = self.act_criterion(action_scores, isnext)
        loss = mask_loss + next_loss

        return action_scores, prediction_scores, loss


class BertAddActionSepPreTrain(nn.Module):
    def __init__(self,vision_size, hidden_size, dec_hidden_size,dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, vl_layers, bert_type, vocab_size, action_space):
        super(BertAddActionSepPreTrain, self).__init__()

        self.bert = BertAddSepEncoder(vision_size, hidden_size, dec_hidden_size,dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, vl_layers, bert_type)
        self.config = self.bert.config

        self.dropout = nn.Dropout(dropout_ratio)
        if not self.bert.top_lstm:
            self.next_action = NextActionPrediction(self.bert.transformer_hidden_size, action_space)
        else:
            self.num_directions = 2 if bidirectional else 1
            self.in_size = hidden_size * self.num_directions
            self.next_action = NextActionPrediction(self.in_size, action_space)
            self.config.hidden_size = self.in_size

        self.lang_criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.act_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.mlmhead = BertOnlyMLMHead(self.config)


    def forward(self, seq, seq_mask, seq_lengths, labels, isnext=None, f_t_all = None):

        ctx, ht, ct, vl_mask,pooled_out = self.bert(seq, seq_mask, seq_lengths, f_t_all=f_t_all)

        if not self.bert.top_lstm:
            cls_part = pooled_out
        else:
            cls_part = torch.cat((ht,ct), dim=1)
        lang_part = ctx

        prediction_scores = self.mlmhead(lang_part)
        mask_loss = self.lang_criterion(prediction_scores.view(-1,self.config.vocab_size), labels[:,:lang_part.shape[1]].contiguous().view(-1))

        action_scores = self.next_action(cls_part)
        next_loss = 0
        if isnext is not None:
            next_loss = self.act_criterion(action_scores, isnext)
        loss = mask_loss + next_loss

        return action_scores, prediction_scores, loss



class BertImgPreTrain(nn.Module):
    def __init__(self,vision_size, hidden_size, dec_hidden_size,dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, vl_layers, bert_type, vocab_size):
        super(BertImgPreTrain, self).__init__()

        self.bert = BertImgEncoder(vision_size, hidden_size, dec_hidden_size,dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, bert_type)

        self.dropout = nn.Dropout(dropout_ratio)
        if not self.bert.top_lstm:
            self.next_vision = NextImgPrediction(self.bert.transformer_hidden_size)
            self.mask_lm = MaskedLanguageModel(self.bert.transformer_hidden_size, vocab_size)
        else:
            self.num_directions = 2 if bidirectional else 1
            in_size = hidden_size * self.num_directions
            self.next_vision = NextImgPrediction(in_size)
            self.mask_lm = MaskedLanguageModel(in_size, vocab_size)

        self.criterion = nn.NLLLoss(ignore_index=0)


    def forward(self, seq, seq_mask, seq_lengths, labels, isnext=None, f_t_all = None):

        ctx, ht, ct, vl_mask = self.bert(seq, seq_mask, seq_lengths, f_t_all=f_t_all)

        if f_t_all is not None:
            vision_len = f_t_all.shape[1]
        else:
            vision_len = 0
        cls_part = ctx[:,vision_len, :]
        lang_part = ctx[:, vision_len+1:, :]

        next_vision_output = self.next_vision(cls_part)
        mask_lm_output = self.mask_lm(lang_part)

        # calculate the loss
        mask_loss = self.criterion(mask_lm_output.transpose(1, 2), labels[:, 1:])
        next_loss = 0
        if isnext is not None:
            next_loss = self.criterion(next_vision_output, isnext)
        loss = mask_loss + next_loss


        return next_vision_output, mask_lm_output, loss

class BertImgActionPreTrain(nn.Module):
    def __init__(self,vision_size, hidden_size, dec_hidden_size,dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, vl_layers, bert_type, vocab_size, action_space):
        super(BertImgActionPreTrain, self).__init__()

        self.bert = BertImgEncoder(vision_size, hidden_size, dec_hidden_size,dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, bert_type)

        self.dropout = nn.Dropout(dropout_ratio)
        if not self.bert.top_lstm:
            self.next_action = NextActionPrediction(self.bert.transformer_hidden_size, action_space)
            self.mask_lm = MaskedLanguageModel(self.bert.transformer_hidden_size, vocab_size)
        else:
            self.num_directions = 2 if bidirectional else 1
            in_size = hidden_size * self.num_directions
            self.next_action = NextActionPrediction(in_size,action_space)
            self.mask_lm = MaskedLanguageModel(in_size, vocab_size)

        self.criterion = nn.NLLLoss(ignore_index=0)


    def forward(self, seq, seq_mask, seq_lengths, labels, isnext=None, f_t_all = None):

        ctx, ht, ct, vl_mask = self.bert(seq, seq_mask, seq_lengths, f_t_all=f_t_all)

        if f_t_all is not None:
            vision_len = f_t_all.shape[1]
        else:
            vision_len = 0
        cls_part = ctx[:,vision_len, :]
        lang_part = ctx[:, vision_len+1:, :]

        next_action_output = self.next_action(cls_part)
        mask_lm_output = self.mask_lm(lang_part)

        # calculate the loss
        mask_loss = self.criterion(mask_lm_output.transpose(1, 2), labels[:, 1:])
        next_loss = 0
        if isnext is not None:
            next_loss = self.criterion(next_action_output, isnext)
        loss = mask_loss + next_loss


        return next_action_output, mask_lm_output, loss

class BertImgActionSepPreTrain(nn.Module):
    def __init__(self,vision_size, hidden_size, dec_hidden_size,dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, vl_layers, bert_type, vocab_size, action_space):
        super(BertImgActionSepPreTrain, self).__init__()

        self.bert = BertImgEncoder(vision_size, hidden_size, dec_hidden_size,dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, bert_type)

        self.dropout = nn.Dropout(dropout_ratio)
        if not self.bert.top_lstm:
            self.next_action = NextActionPrediction(self.bert.transformer_hidden_size, action_space)
            self.mask_lm = MaskedLanguageModel(self.bert.transformer_hidden_size, vocab_size)
        else:
            self.num_directions = 2 if bidirectional else 1
            in_size = hidden_size * self.num_directions
            self.next_action = NextActionPrediction(in_size, action_space)
            self.mask_lm = MaskedLanguageModel(in_size, vocab_size)

        self.criterion = nn.NLLLoss(ignore_index=0)



    def forward(self, seq, seq_mask, seq_lengths, labels, isnext=None, f_t_all = None):

        ctx, ht, ct, vl_mask = self.bert(seq, seq_mask, seq_lengths, f_t_all=f_t_all)

        if f_t_all is not None:
            vision_len = f_t_all.shape[1]
        else:
            vision_len = 0
        cls_part = ctx[:,vision_len, :]
        lang_part = ctx[:, vision_len+1:, :]

        next_action_output = self.next_action(cls_part)
        mask_lm_output = self.mask_lm(lang_part)

        # calculate the loss
        mask_loss = self.criterion(mask_lm_output.transpose(1, 2), labels[:, 1:])
        next_loss = 0
        if isnext is not None:
            next_loss = self.criterion(next_action_output, isnext)
        loss = next_loss

        return next_action_output, mask_lm_output, loss



class BertLangPreTrain(nn.Module):
    def __init__(self,vision_size, hidden_size, dec_hidden_size,dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, vl_layers, bert_type, vocab_size):
        super(BertLangPreTrain, self).__init__()

        self.bert = BertLangEncoder(vision_size, hidden_size, dec_hidden_size,dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, vl_layers, bert_type)
        self.config = self.bert.config

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        self.mlmhead = BertOnlyMLMHead(self.config)




    def forward(self, seq, seq_mask, seq_lengths, labels, isnext=None, f_t_all = None):

        ctx, ht, ct, vl_mask = self.bert(seq, seq_mask, seq_lengths, f_t_all=f_t_all)
        prediction_scores = self.mlmhead(ctx)
        mask_loss = self.criterion(prediction_scores.view(-1,self.config.vocab_size), labels.view(-1))
        next_loss = 0

        loss = mask_loss + next_loss


        return [], [], loss



class BertAddPaPreTrain(nn.Module):
    def __init__(self,vision_size, hidden_size, dec_hidden_size,dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, vl_layers, bert_type, vocab_size):
        super(BertAddPaPreTrain, self).__init__()

        self.bert = BertAddEncoder(vision_size, hidden_size, dec_hidden_size,dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, vl_layers, bert_type)

        self.dropout = nn.Dropout(dropout_ratio)
        if not self.bert.top_lstm:
            self.next_vision = NextImgPrediction(self.bert.transformer_hidden_size)
            self.mask_lm = MaskedLanguageModel(self.bert.transformer_hidden_size, vocab_size)
        else:
            self.num_directions = 2 if bidirectional else 1
            in_size = hidden_size * self.num_directions
            self.next_vision = NextImgPrediction(in_size)
            self.mask_lm = MaskedLanguageModel(in_size, vocab_size)



    def forward(self, seq, seq_mask, seq_lengths, f_t_all = None):

        ctx, ht, ct, vl_mask = self.bert(seq, seq_mask, seq_lengths, f_t_all=f_t_all)

        if f_t_all is not None:
            vision_len = f_t_all.shape[1]
        else:
            vision_len = 0
        cls_part = ctx[:,vision_len, :]
        lang_part = ctx[:, vision_len:, :]

        next_vision_output = self.next_vision(cls_part)
        mask_lm_output = self.mask_lm(lang_part)


        return next_vision_output, mask_lm_output



class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))




class NextImgPrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))  # the 0-35 is the vision, 36th is the CLS token


class NextActionPrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden, actionspace):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, actionspace)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))  # the 0-35 is the vision, 36th is the CLS token


class MatchPrediction(nn.Module):
    def __init__(self, hidden, actionspace):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, actionspace)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))  


class OrientMatchPrediction(nn.Module):
    def __init__(self, hidden, actionspace):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, actionspace)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x)) 

class ProgressMonitor(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden, subnum):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, subnum)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))  



class AlignmentMatrix(nn.Module):

    def __init__(self):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, lang_part, visn_part, lang_mask, visn_mask):
        #return self.softmax(self.linear(x))  # the 0-35 is the vision, 36th is the CLS token
        matrix_score = torch.matmul(visn_part, lang_part.transpose(-1, -2))
        matrix_score = (matrix_score * visn_mask.unsqueeze(-1))*lang_mask.unsqueeze(1)
        return matrix_score