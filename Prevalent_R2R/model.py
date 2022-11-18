
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pytorch_pretrained_bert import BertModel, OpenAIGPTModel
from vilmodel import BertImgModel,BertLayer,BertLayerNorm,BertPooler,BertAddModel,VicModel,LXRTXLayer,DicModel
from pytorch_transformers import BertConfig
import math
import random
import pdb

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# region Transformer
def generate_relative_positions_matrix(length, max_relative_positions,
                                       cache=False):
    """Generate the clipped relative positions matrix
       for a given length and maximum relative positions

generate_relative_positions_matrix(4, 4)
tensor([[4, 5, 6, 7],
        [3, 4, 5, 6],
        [2, 3, 4, 5],
        [1, 2, 3, 4]])
       """
    if cache:
        distance_mat = torch.arange(-length+1, 1, 1).unsqueeze(0)
    else:
        range_vec = torch.arange(length)
        range_mat = range_vec.unsqueeze(-1).expand(-1, length).transpose(0, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
    distance_mat_clipped = torch.clamp(distance_mat,
                                       min=-max_relative_positions,
                                       max=max_relative_positions)
    # Shift values to be >= 0
    final_mat = distance_mat_clipped + max_relative_positions
    return final_mat


def relative_matmul(x, z, transpose):
    """Helper function for relative positions attention.
    x: batch_size x heads * length * dim
    z: length x dim x M

    x = torch.Tensor(2,4,3,5)
    print(x.shape)  #2435
    batch_size = x.shape[0]
    heads = x.shape[1]
    length = x.shape[2]
    dim_per_head = x.shape[3]
    print(batch_size, heads, length, dim_per_head)
    x_t = x.permute(2, 0, 1, 3)
    print(x_t.shape)  # 3245
    x_t_r = x_t.reshape(length, heads * batch_size, -1)
    print(x_t_r.shape)  # 3,8,5

    z = torch.Tensor(length,dim_per_head,2)  # 3,5,2
    print(z.shape)
    x_tz_matmul = torch.matmul(x_t_r, z)
    print(x_tz_matmul.shape)
    """
    batch_size = x.shape[0]
    heads = x.shape[1]
    length = x.shape[2]
    x_t = x.permute(2, 0, 1, 3)
    x_t_r = x_t.reshape(length, heads * batch_size, -1)
    if transpose:
        z_t = z.transpose(1, 2)
        x_tz_matmul = torch.matmul(x_t_r, z_t)
    else:
        x_tz_matmul = torch.matmul(x_t_r, z)
    x_tz_matmul_r = x_tz_matmul.reshape(length, batch_size, heads, -1)
    x_tz_matmul_r_t = x_tz_matmul_r.permute(1, 2, 0, 3)
    return x_tz_matmul_r_t


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention module from "Attention is All You Need"
        :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.
        Similar to standard `dot` attention but uses
        multiple attention distributions simulataneously
        to select relevant items.
        .. mermaid::
           graph BT
              A[key]
              B[value]
              C[query]
              O[output]
              subgraph Attn
                D[Attn 1]
                E[Attn 2]
                F[Attn N]
              end
              A --> D
              C --> D
              A --> E
              C --> E
              A --> F
              C --> F
              D --> O
              E --> O
              F --> O
              B --> O
        Also includes several additional tricks.
        Args:
           head_count (int): number of parallel heads
           model_dim (int): the dimension of keys/values/queries,
               must be divisible by head_count
           dropout (float): dropout parameter
        """

    def __init__(self, head_count, model_dim, dropout=0.1,
                 max_relative_positions=0):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

        self.max_relative_positions = max_relative_positions

        if max_relative_positions > 0:
            vocab_size = max_relative_positions * 2 + 1
            self.relative_positions_embeddings = nn.Embedding(
                vocab_size, self.dim_per_head)

    def forward(self, key, value, query, mask= None,
            layer_cache=None, type=None):
        """
        Compute the context vector and the attention vectors.
        Args:
           key (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (FloatTensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (FloatTensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask indicating which keys have
               non-zero attention ``(batch, query_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):
           * output context vectors ``(batch, query_len, dim)``
           * one of the attention vectors ``(batch, query_len, key_len)``
        """

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)
        device = key.device

        def shape(x):
            """Projection.
            batch_size, length, head_count * dim_per_head
            -->batch_size, heads, length, dim_per_head"""
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if type == "self":
                query, key, value = self.linear_query(query), \
                                    self.linear_keys(query), \
                                    self.linear_values(query)
                key = shape(key)
                value = shape(value)
                if layer_cache["self_keys"] is not None:
                    key = torch.cat(
                        (layer_cache["self_keys"].to(device), key),
                        dim=2)  # length +1? x2?
                if layer_cache["self_values"] is not None:
                    value = torch.cat(
                        (layer_cache["self_values"].to(device), value),
                        dim=2)
                layer_cache["self_keys"] = key
                layer_cache["self_values"] = value
            elif type == "context":
                query = self.linear_query(query)
                if layer_cache["memory_keys"] is None:
                    key, value = self.linear_keys(key), \
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
                else:
                    key, value = layer_cache["memory_keys"], \
                                 layer_cache["memory_values"]
                layer_cache["memory_keys"] = key
                layer_cache["memory_values"] = value
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        if self.max_relative_positions > 0 and type == "self":
            key_len = key.size(2)  # length
            # 1 or key_len x key_len
            relative_positions_matrix = generate_relative_positions_matrix(
                key_len, self.max_relative_positions, cache=True if layer_cache is not None else False)
            #  1 or key_len x key_len x dim_per_head
            relations_keys = self.relative_positions_embeddings(
                relative_positions_matrix.to(device))
            #  1 or key_len x key_len x dim_per_head
            relations_values = self.relative_positions_embeddings(
                relative_positions_matrix.to(device))

        query = shape(query)  # batch_size, heads, length, dim_per_head

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        # batch x num_heads x query_len x key_len
        query_key = torch.matmul(query, key.transpose(2, 3))
        # batch x num_heads x query_len x query_len

        if self.max_relative_positions > 0 and type == "self":
            scores = query_key + relative_matmul(query, relations_keys, True)
        else:
            scores = query_key
        scores = scores.float()  # [batch_size, head_count, len, len]

        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch_size, 1, 1, T_values] can be broadcast
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores).to(query.dtype)
        drop_attn = self.dropout(attn)

        # weighted context?
        context_original = torch.matmul(drop_attn, value)

        if self.max_relative_positions > 0 and type == "self":
            context = unshape(context_original
                              + relative_matmul(drop_attn,
                                                relations_values,
                                                False))
        else:
            context = unshape(context_original)

        output = self.final_linear(context)
        # CHECK
        # batch_, q_len_, d_ = output.size()
        # aeq(q_len, q_len_)
        # aeq(batch, batch_)
        # aeq(d, d_)

        # Return one attn
        top_attn = attn \
            .view(batch_size, head_count,
                  query_len, key_len)[:, 0, :, :] \
            .contiguous()

        return output, top_attn


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.
    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """Layer definition.
        Args:
            x: ``(batch_size, input_len, model_dim)``
        Returns:
            (FloatTensor): Output ``(batch_size, input_len, model_dim)``.
        """

        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.
    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout,
                 max_relative_positions=0):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout,
            max_relative_positions=max_relative_positions)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, src_len, src_len)``
        Returns:
            (FloatTensor):
            * outputs ``(batch_size, src_len, model_dim)``
        """
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask, type="self")
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerEncoder(nn.Module):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    .. mermaid::
       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O
    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings
    Returns:
        (torch.FloatTensor, torch.FloatTensor):
        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """
    lstm_num_layers = 1

    def __init__(self, vocab_size, emb_hidden_size, padding_idx,
                 dropout_ratio, glove, heads, d_ff, hidden_size, dec_hidden_size, top_lstm, bidirectional, max_relative_positions=0, num_layers = 1):
        super(TransformerEncoder, self).__init__()

        self.embedding_size = emb_hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, emb_hidden_size, padding_idx)
        self.use_glove = glove is not None
        self.dec_hidden_size = dec_hidden_size
        self.hidden_size = hidden_size
        self.dropout_ratio = dropout_ratio
        self.drop = nn.Dropout(p=dropout_ratio)
        # if self.use_glove:
        #     print('Using GloVe embedding')
        #     self.embedding.weight.data[...] = torch.from_numpy(glove)
        #     self.embedding.weight.requires_grad = False
        #
        self.top_lstm = top_lstm
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                emb_hidden_size, heads, d_ff, dropout_ratio,
                max_relative_positions=max_relative_positions)
             for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(emb_hidden_size, eps=1e-6)

        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.lstm_num_layers,
                            batch_first=True,
                            dropout=dropout_ratio if self.lstm_num_layers > 1 else 0, bidirectional=bidirectional)

        self.linear_n_in = self.embedding_size if not self.top_lstm else hidden_size * self.num_directions
        self.encoder2decoder_ht = nn.Linear(self.linear_n_in, dec_hidden_size)
        self.encoder2decoder_ct = nn.Linear(self.linear_n_in, dec_hidden_size)

    def init_state(self, batch_size):
        ''' Initialize to zero cell states and hidden states.'''
        h0 = Variable(torch.zeros(batch_size, self.dec_hidden_size), requires_grad=False)
        c0 = Variable(torch.zeros(batch_size, self.dec_hidden_size), requires_grad=False)
        return h0.to(device), c0.to(device)

    def init_state_lstm(self, batch_size):
        ''' Initialize to zero cell states and hidden states.'''
        h0 = Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size), requires_grad=False)
        c0 = Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size), requires_grad=False)
        return h0.to(device), c0.to(device)

    def forward(self, inputs, mask, lengths):

        emb = self.embedding(inputs[:,:mask.shape[1]])
        # Run the forward pass of every layer of the tranformer.
        for layer in self.transformer:
            out = layer(emb, mask.unsqueeze(1))
        out = self.layer_norm(out)

        if not self.top_lstm:
            h0, c0 = self.init_state(inputs.size(0))
            return out, h0, c0, mask

        batch_size = inputs.size(0)
        embeds = out
        h0, c0 = self.init_state_lstm(batch_size)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))
        if self.num_directions == 2:
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1] # (batch, hidden_size)

        decoder_init = nn.Tanh()(self.encoder2decoder_ht(h_t))
        if self.hidden_size * self.num_directions != self.dec_hidden_size:
            c_t = self.encoder2decoder_ct(c_t)

        ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)
        ctx = self.drop(ctx)
        return ctx, decoder_init, c_t, mask


class MultiTransformerEncoder(TransformerEncoder):
    def __init__(self, vocab_size, emb_hidden_size, padding_idx,
                 dropout_ratio, multi_share, n_sentences, glove, heads, d_ff, hidden_size, dec_hidden_size, top_lstm, bidirectional, max_relative_positions=0, num_layers = 1):
        super(MultiTransformerEncoder, self).__init__(vocab_size, emb_hidden_size, padding_idx,
                                                      dropout_ratio, glove, heads, d_ff, hidden_size, dec_hidden_size, top_lstm, bidirectional, max_relative_positions, num_layers)
        self.n_sentences = n_sentences
        self.multi_share = multi_share

        for i in range(max(self.n_sentences-1, 2)):
            setattr(self, 'transformer'+str(i+1), None)
            setattr(self, 'layer_norm'+str(i+1), None)
            setattr(self, 'lstm'+str(i+1), None)
        if not multi_share:
            for i in range(self.n_sentences - 1):
                setattr(self, 'transformer' + str(i + 1),nn.ModuleList( [TransformerEncoderLayer(
                    emb_hidden_size, heads, d_ff, dropout_ratio,
                    max_relative_positions=max_relative_positions)
                    for l in range(num_layers)]))
                setattr(self, 'layer_norm' + str(i + 1),nn.LayerNorm(emb_hidden_size, eps=1e-6))
                setattr(self, 'lstm' + str(i + 1),nn.LSTM(self.embedding_size, self.hidden_size, self.lstm_num_layers,
                            batch_first=True,
                            dropout=dropout_ratio if self.lstm_num_layers > 1 else 0,
                                                          bidirectional=bidirectional))

    def set_n_sentences(self, n_sentences):
        self.n_sentences = n_sentences

    def forward(self, inputs, masks, lengths):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        ctxs, decoder_inits, c_ts, maskss = [], None, None ,[]

        transformers = [getattr(self, 'transformer'+(str(i) if i!=0 else '')) for i in range(self.n_sentences)]
        layer_norms = [getattr(self, 'layer_norm'+(str(i) if i!=0 else '')) for i in range(self.n_sentences)]  #  lstms = [self.lstm, self.lstm2, self.lstm3]
        lstms = [getattr(self, 'lstm' + (str(i) if i != 0 else '')) for i in
                 range(self.n_sentences)]  # lstms = [self.lstm, self.lstm2, self.lstm3]

        if not self.multi_share:
            temp=[i for i in range(self.n_sentences)]
            random.shuffle(temp)
            transformers = [transformers[i] for i in temp]
            layer_norms = [layer_norms[i] for i in temp]

        for si,(input, mask, (length, perm_idx), transformerN, layer_normN, lstmN) in enumerate(zip(inputs, masks, lengths, transformers, layer_norms, lstms)):
            emb = self.embedding(input[:, :mask.shape[1]])
            # Run the forward pass of every layer of the tranformer.
            for layer in (self.transformer if self.multi_share else transformerN):
                out = layer(emb, mask.unsqueeze(1))
            out = self.layer_norm(out) if self.multi_share else layer_normN(out)

            if not self.top_lstm:
                decoder_init, c_t = self.init_state(input.size(0))
                ctx = out
            else:
                batch_size = input.size(0)
                embeds = out
                h0, c0 = self.init_state_lstm(batch_size)
                packed_embeds = pack_padded_sequence(embeds, length, batch_first=True)
                enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0)) if self.multi_share \
                    else lstmN(packed_embeds, (h0, c0))
                if self.num_directions == 2:
                    h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
                    c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
                else:
                    h_t = enc_h_t[-1]
                    c_t = enc_c_t[-1]  # (batch, hidden_size)

                decoder_init = nn.Tanh()(self.encoder2decoder_ht(h_t))
                if self.hidden_size * self.num_directions != self.dec_hidden_size:
                    c_t = self.encoder2decoder_ct(c_t)

                ctx, length = pad_packed_sequence(enc_h, batch_first=True)
                ctx = self.drop(ctx)
                #return ctx, decoder_init, c_t, mask

            perm_idx = list(perm_idx.numpy())
            unperm_idx = [perm_idx.index(i) for i, _ in enumerate(perm_idx)]
            ctx = ctx[unperm_idx]
            mask = mask[unperm_idx]
            decoder_init = decoder_init[unperm_idx]
            c_t = c_t[unperm_idx]

            ctxs.append(ctx)
            maskss.append(mask)

            if decoder_inits is None:
                decoder_inits = decoder_init
                c_ts = c_t
            else:
                decoder_inits = decoder_inits + decoder_init
                c_ts = c_ts + c_t

        decoder_inits = decoder_inits / self.n_sentences
        c_ts = c_ts / self.n_sentences
        return ctxs, decoder_inits, c_ts, maskss  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)


# endregion


class GptEncoder(nn.Module):

    transformer_hidden_size = 768
    lstm_num_layers = 1

    def __init__(self, hidden_size, dec_hidden_size, dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm):
        super(GptEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.dropout_ratio = dropout_ratio
        self.drop = nn. Dropout(p=dropout_ratio)
        self.update = update
        self.bert_n_layers = bert_n_layers
        self.reverse_input = reverse_input
        self.top_lstm = top_lstm
        self.gpt = OpenAIGPTModel.from_pretrained('openai-gpt').to(device)

        self.lstm = nn.LSTM(self.transformer_hidden_size*self.bert_n_layers, self.hidden_size, self.lstm_num_layers, batch_first=True,
                             dropout=dropout_ratio if self.lstm_num_layers>1 else 0, bidirectional=bidirectional)
        #
        self.num_directions = 2 if bidirectional else 1

        self.linear_n_in = self.transformer_hidden_size * self.bert_n_layers if not self.top_lstm else hidden_size * self.num_directions
        self.encoder2decoder_ht = nn.Linear(self.linear_n_in, dec_hidden_size)
        self.encoder2decoder_ct = nn.Linear(self.linear_n_in, dec_hidden_size)


    def init_state(self, batch_size):
        ''' Initialize to zero cell states and hidden states.'''
        h0 = Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size), requires_grad=False)
        c0 = Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size), requires_grad=False)
        return h0.to(device), c0.to(device)


    def forward(self, inputs, mask, lengths):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        batch_size = inputs.size(0)
        seq_max_len = mask.size(1)
        att_mask = ~mask
        encoder_layer = self.gpt(inputs[:, :seq_max_len])# token_type_ids=None, attention_mask=att_mask.to(device))
        embeds = encoder_layer

        if self.reverse_input:
            reversed_embeds = torch.zeros(embeds.size()).to(device)
            reverse_idx = torch.arange(seq_max_len-1,-1,-1)
            reversed_embeds[att_mask] = embeds[:, reverse_idx][att_mask[:,reverse_idx]]
            embeds = reversed_embeds

        if not self.update:
            embeds = embeds.detach()  # batch_size x input_len x bert_hidden_size*n

        if not self.top_lstm:
            ctx = embeds
            decoder_init, c_t = self.init_state(batch_size)
        else:
            h0, c0 = self.init_state(batch_size)
            packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
            enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

            if self.num_directions == 2:
                h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
                c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
            else:
                h_t = enc_h_t[-1]
                c_t = enc_c_t[-1] # (batch, hidden_size)

            decoder_init = nn.Tanh()(self.encoder2decoder_ht(h_t))
            if self.hidden_size * self.num_directions != self.dec_hidden_size:
                c_t = self.encoder2decoder_ct(c_t)

            ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)
        ctx = self.drop(ctx)
        return ctx,decoder_init,c_t, mask  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)


class MultiGptEncoder(GptEncoder):

    def __init__(self, hidden_size, dec_hidden_size, dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, multi_share, n_sentences):
        super(MultiGptEncoder, self).__init__(hidden_size, dec_hidden_size, dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm)

        self.n_sentences = n_sentences
        self.multi_share = multi_share

        for i in range(max(self.n_sentences - 1, 2)):
            setattr(self, 'gpt' + str(i + 1), None)
            setattr(self, 'lstm' + str(i + 1), None)
        if not multi_share:
            raise NotImplementedError()

    def set_n_sentences(self, n_sentences):
        self.n_sentences = n_sentences

    def forward(self, inputs, masks, lengths):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        ctxs, decoder_inits, c_ts, maskss = [], None, None, []

        lstms = [getattr(self, 'lstm' + (str(i) if i != 0 else '')) for i in
                 range(self.n_sentences)]  # lstms = [self.lstm, self.lstm2, self.lstm3]
        gpts = [getattr(self, 'gpt' + (str(i) if i != 0 else '')) for i in
                 range(self.n_sentences)]  # lstms = [self.lstm, self.lstm2, self.lstm3]
        if not self.multi_share:
            raise NotImplementedError()

        for si, (input, mask, (length, perm_idx), lstmN, gptN) in enumerate(zip(inputs, masks, lengths, lstms, gpts)):
            batch_size = input.size(0)
            seq_max_len = mask.size(1)
            att_mask = ~mask
            if self.multi_share:
                encoder_layer = self.gpt(
                    input[:, :seq_max_len])  # token_type_ids=None, attention_mask=att_mask.to(device))
                embeds = encoder_layer
            #else:
            #    raise

            if self.reverse_input:
                reversed_embeds = torch.zeros(embeds.size()).to(device)
                reverse_idx = torch.arange(seq_max_len - 1, -1, -1)
                reversed_embeds[att_mask] = embeds[:, reverse_idx][att_mask[:, reverse_idx]]
                embeds = reversed_embeds

            if not self.update:
                embeds = embeds.detach()  # batch_size x input_len x bert_hidden_size*n

            if not self.top_lstm:
                ctx = embeds
                decoder_init, c_t = self.init_state(batch_size)
            else:
                h0, c0 = self.init_state(batch_size)
                packed_embeds = pack_padded_sequence(embeds, length, batch_first=True)
                enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

                if self.num_directions == 2:
                    h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
                    c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
                else:
                    h_t = enc_h_t[-1]
                    c_t = enc_c_t[-1]  # (batch, hidden_size)

                decoder_init = nn.Tanh()(self.encoder2decoder_ht(h_t))
                if self.hidden_size * self.num_directions != self.dec_hidden_size:
                    c_t = self.encoder2decoder_ct(c_t)

                ctx, length = pad_packed_sequence(enc_h, batch_first=True)
            ctx = self.drop(ctx)

            perm_idx = list(perm_idx.numpy())
            unperm_idx = [perm_idx.index(i) for i, _ in enumerate(perm_idx)]
            ctx = ctx[unperm_idx]
            mask = mask[unperm_idx]
            decoder_init = decoder_init[unperm_idx]
            c_t = c_t[unperm_idx]

            ctxs.append(ctx)
            maskss.append(mask)

            if decoder_inits is None:
                decoder_inits = decoder_init
                c_ts = c_t
            else:
                decoder_inits = decoder_inits + decoder_init
                c_ts = c_ts + c_t

            # debug: no divide
        decoder_inits = decoder_inits / self.n_sentences
        c_ts = c_ts / self.n_sentences
        return ctxs, decoder_inits, c_ts, maskss  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)


class BertEncoder(nn.Module):

    #transformer_hidden_size = 768
    lstm_num_layers = 1

    def __init__(self, hidden_size, dec_hidden_size, dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, bert_type="small"):
        super(BertEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.dropout_ratio = dropout_ratio
        self.drop = nn.Dropout(p=dropout_ratio)
        self.update = update
        self.bert_n_layers = bert_n_layers
        self.reverse_input = reverse_input
        self.top_lstm = top_lstm
        self.bert_type = bert_type
        if self.bert_type == "large":
            self.bert = BertModel.from_pretrained('bert-large-uncased').to(device)
            self.transformer_hidden_size = 1024
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased').to(device)
            self.transformer_hidden_size = 768

        self.lstm = nn.LSTM(self.transformer_hidden_size*self.bert_n_layers, self.hidden_size, self.lstm_num_layers, batch_first=True,
                             dropout=dropout_ratio if self.lstm_num_layers>1 else 0, bidirectional=bidirectional)
        #
        self.num_directions = 2 if bidirectional else 1

        self.linear_n_in = self.transformer_hidden_size * self.bert_n_layers if not self.top_lstm else hidden_size * self.num_directions
        self.encoder2decoder_ht = nn.Linear(self.linear_n_in, self.dec_hidden_size)
        self.encoder2decoder_ct = nn.Linear(self.linear_n_in, self.dec_hidden_size)


    def init_state(self, batch_size):
        ''' Initialize to zero cell states and hidden states.'''
        h0 = Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size ), requires_grad=False)
        c0 = Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size ), requires_grad=False)
        return h0.to(device), c0.to(device)

    def forward(self, inputs, mask, lengths):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        batch_size = inputs.size(0)
        seq_max_len = mask.size(1)
        att_mask = ~mask
        all_encoder_layers, pooled_output = self.bert(inputs[:, :seq_max_len], token_type_ids=None,
                                          attention_mask=att_mask.to(device))
        embeds = torch.cat(all_encoder_layers[-self.bert_n_layers:],-1)

        if self.reverse_input:
            reversed_embeds = torch.zeros(embeds.size()).to(device)
            reverse_idx = torch.arange(seq_max_len-1,-1,-1)
            reversed_embeds[att_mask] = embeds[:, reverse_idx][att_mask[:,reverse_idx]]
            embeds = reversed_embeds

        if not self.update:
            embeds = embeds.detach()  # batch_size x input_len x bert_hidden_size*n

        if not self.top_lstm:
            ctx = embeds
            c_t = self.encoder2decoder_ct(ctx[:,-1])
            decoder_init = nn.Tanh()(self.encoder2decoder_ht(pooled_output))
        else:
            h0, c0 = self.init_state(batch_size)
            packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
            enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

            if self.num_directions == 2:
                h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
                c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
            else:
                h_t = enc_h_t[-1]
                c_t = enc_c_t[-1] # (batch, hidden_size)

            decoder_init = nn.Tanh()(self.encoder2decoder_ht(h_t))
            if self.hidden_size * self.num_directions != self.dec_hidden_size:
                c_t = self.encoder2decoder_ct(c_t)

            ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)
        ctx = self.drop(ctx)
        return ctx,decoder_init,c_t, mask  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)

class HugLangEncoder(nn.Module):

    #transformer_hidden_size = 768
    lstm_num_layers = 1

    def __init__(self, hidden_size, dec_hidden_size, dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, bert_type="small"):
        super(HugLangEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.dropout_ratio = dropout_ratio
        self.drop = nn.Dropout(p=dropout_ratio)
        self.update = update
        self.bert_n_layers = bert_n_layers
        self.reverse_input = reverse_input
        self.top_lstm = top_lstm
        self.bert_type = bert_type
        if self.bert_type == "large":
            self.bert = BertModel.from_pretrained('bert-large-uncased').to(device)
            self.transformer_hidden_size = 1024
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased').to(device)
            self.transformer_hidden_size = 768

        self.lstm = nn.LSTM(self.transformer_hidden_size*self.bert_n_layers, self.hidden_size, self.lstm_num_layers, batch_first=True,
                             dropout=dropout_ratio if self.lstm_num_layers>1 else 0, bidirectional=bidirectional)
        #
        self.num_directions = 2 if bidirectional else 1

        self.linear_n_in = self.transformer_hidden_size * self.bert_n_layers if not self.top_lstm else hidden_size * self.num_directions
        self.encoder2decoder_ht = nn.Linear(self.linear_n_in, self.dec_hidden_size)
        self.encoder2decoder_ct = nn.Linear(self.linear_n_in, self.dec_hidden_size)


    def init_state(self, batch_size):
        ''' Initialize to zero cell states and hidden states.'''
        h0 = Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size ), requires_grad=False)
        c0 = Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size ), requires_grad=False)
        return h0.to(device), c0.to(device)

    def forward(self, inputs, mask, lengths):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        batch_size = inputs.size(0)
        seq_max_len = mask.size(1)
        att_mask = ~mask
        last_encoder_layers, pooled_output = self.bert(inputs[:, :seq_max_len], token_type_ids=None,
                                          attention_mask=att_mask.to(device))
        embeds = last_encoder_layers

        if self.reverse_input:
            reversed_embeds = torch.zeros(embeds.size()).to(device)
            reverse_idx = torch.arange(seq_max_len-1,-1,-1)
            reversed_embeds[att_mask] = embeds[:, reverse_idx][att_mask[:,reverse_idx]]
            embeds = reversed_embeds

        if not self.update:
            embeds = embeds.detach()  # batch_size x input_len x bert_hidden_size*n

        if not self.top_lstm:
            ctx = embeds
            c_t = self.encoder2decoder_ct(ctx[:,-1])
            decoder_init = nn.Tanh()(self.encoder2decoder_ht(pooled_output))
        else:
            h0, c0 = self.init_state(batch_size)
            packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
            enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

            if self.num_directions == 2:
                h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
                c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
            else:
                h_t = enc_h_t[-1]
                c_t = enc_c_t[-1] # (batch, hidden_size)

            decoder_init = nn.Tanh()(self.encoder2decoder_ht(h_t))
            if self.hidden_size * self.num_directions != self.dec_hidden_size:
                c_t = self.encoder2decoder_ct(c_t)

            ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)
        ctx = self.drop(ctx)
        return ctx,decoder_init,c_t, mask  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)




class MultiBertEncoder(BertEncoder):

    def __init__(self, hidden_size, dec_hidden_size, dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, multi_share, n_sentences, bert_type="small"):
        super(MultiBertEncoder, self).__init__(hidden_size, dec_hidden_size, dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, bert_type)

        self.n_sentences = n_sentences
        self.multi_share = multi_share

        for i in range(max(self.n_sentences - 1, 2)):
            setattr(self, 'bert' + str(i + 1), None)
            setattr(self, 'lstm' + str(i + 1), None)
        if not multi_share:
            raise NotImplementedError()

    def set_n_sentences(self, n_sentences):
        self.n_sentences = n_sentences

    def forward(self, inputs, masks, lengths):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        ctxs, decoder_inits, c_ts, maskss = [], None, None, []

        lstms = [getattr(self, 'lstm' + (str(i) if i != 0 else '')) for i in
                 range(self.n_sentences)]  # lstms = [self.lstm, self.lstm2, self.lstm3]
        berts = [getattr(self, 'bert' + (str(i) if i != 0 else '')) for i in
                 range(self.n_sentences)]  # lstms = [self.lstm, self.lstm2, self.lstm3]
        if not self.multi_share:
            raise NotImplementedError()

        for si, (input, mask, (length, perm_idx), lstmN,bertN) in enumerate(zip(inputs, masks, lengths, lstms,berts)):
            batch_size = input.size(0)
            seq_max_len = mask.size(1)
            att_mask = ~mask
            if self.multi_share:
                all_encoder_layers, pooled_output = self.bert(input[:, :seq_max_len], token_type_ids=None,
                                                              attention_mask=att_mask.to(device))
                # Victor add here
                if isinstance(all_encoder_layers, list):
                    embeds = torch.cat(all_encoder_layers[-self.bert_n_layers:], -1)
                else:
                    embeds = all_encoder_layers
            #else:
            #    raise

            if self.reverse_input:
                reversed_embeds = torch.zeros(embeds.size()).to(device)
                reverse_idx = torch.arange(seq_max_len-1,-1,-1)
                reversed_embeds[att_mask] = embeds[:, reverse_idx][att_mask[:,reverse_idx]]
                embeds = reversed_embeds

            if not self.update:
                embeds = embeds.detach()  # batch_size x input_len x bert_hidden_size*n

            if not self.top_lstm:
                ctx = embeds
                c_t = self.encoder2decoder_ct(ctx[:,-1])
                decoder_init = nn.Tanh()(self.encoder2decoder_ht(pooled_output))
            else:
                h0, c0 = self.init_state(batch_size)
                packed_embeds = pack_padded_sequence(embeds, length, batch_first=True)
                enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

                if self.num_directions == 2:
                    h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
                    c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
                else:
                    h_t = enc_h_t[-1]
                    c_t = enc_c_t[-1] # (batch, hidden_size)

                decoder_init = nn.Tanh()(self.encoder2decoder_ht(h_t))
                if self.hidden_size * self.num_directions != self.dec_hidden_size:
                    c_t = self.encoder2decoder_ct(c_t)

                ctx, length = pad_packed_sequence(enc_h, batch_first=True)
            ctx = self.drop(ctx)

            perm_idx = list(perm_idx.numpy())
            unperm_idx = [perm_idx.index(i) for i, _ in enumerate(perm_idx)]
            ctx = ctx[unperm_idx]
            mask = mask[unperm_idx]
            decoder_init = decoder_init[unperm_idx]
            c_t = c_t[unperm_idx]

            ctxs.append(ctx)
            maskss.append(mask)

            if decoder_inits is None:
                decoder_inits = decoder_init
                c_ts = c_t
            else:
                decoder_inits = decoder_inits + decoder_init
                c_ts = c_ts + c_t

            # debug: no divide
        decoder_inits = decoder_inits / self.n_sentences
        c_ts = c_ts / self.n_sentences
        return ctxs, decoder_inits, c_ts, maskss  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)

class BertImgEncoder(nn.Module):

    #transformer_hidden_size = 768
    lstm_num_layers = 1

    def __init__(self, vision_size, hidden_size, dec_hidden_size, dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, bert_type="small"):
        super(BertImgEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.dropout_ratio = dropout_ratio
        self.drop = nn.Dropout(p=dropout_ratio)
        self.update = update
        self.bert_n_layers = bert_n_layers
        self.reverse_input = reverse_input
        self.top_lstm = top_lstm
        self.bert_type = bert_type
        self.pretrain = True
        if self.bert_type == "large":
            self.config = BertConfig.from_pretrained('bert-large-uncased')
            self.config.img_feature_dim = vision_size
            self.config.img_feature_type = ""
            self.bert = BertImgModel(self.config)
            self.transformer_hidden_size = 1024
        else:
            self.config = BertConfig.from_pretrained('bert-base-uncased')
            self.config.img_feature_dim = vision_size
            self.config.img_feature_type = ""
            self.bert = BertImgModel(self.config)
            self.transformer_hidden_size = 768

        self.lstm = nn.LSTM(self.transformer_hidden_size*self.bert_n_layers, self.hidden_size, self.lstm_num_layers, batch_first=True,
                             dropout=dropout_ratio if self.lstm_num_layers>1 else 0, bidirectional=bidirectional)
        #
        self.num_directions = 2 if bidirectional else 1

        self.linear_n_in = self.transformer_hidden_size * self.bert_n_layers if not self.top_lstm else hidden_size * self.num_directions
        self.encoder2decoder_ht = nn.Linear(self.linear_n_in, self.dec_hidden_size)
        self.encoder2decoder_ct = nn.Linear(self.linear_n_in, self.dec_hidden_size)

        self.encoder_lstm2decoder_ht = nn.Linear(hidden_size * self.num_directions, self.dec_hidden_size)
        self.encoder_lstm2decoder_ct = nn.Linear(hidden_size * self.num_directions, self.dec_hidden_size)
        #self.ctx2decoder = nn.Linear(self.transformer_hidden_size, 2 * self.hidden_size)
        #_ = self.flip_text_bert_params(False)  # default is to fix the text bert params



    def init_state(self, batch_size):
        ''' Initialize to zero cell states and hidden states.'''
        h0 = Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size ), requires_grad=False)
        c0 = Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size ), requires_grad=False)
        return h0.to(device), c0.to(device)

    def flip_text_bert_params(self, bool_input):
        result = list()
        for name, param in self.bert.named_parameters():
            if name in self.text_bert_params:
                if param.requires_grad != bool_input:
                    param.requires_grad = bool_input
                    result.append(param)
        return result


    def forward(self, inputs, mask, lengths,f_t_all=None):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        batch_size = inputs.size(0)
        seq_max_len = mask.size(1)
        att_mask = ~mask
        # need to extend mask for image features
        if f_t_all is not None:
            img_seq_len = f_t_all.shape[1]
            img_seq_mask = torch.ones(batch_size, img_seq_len, dtype=att_mask.dtype).to(att_mask.device)
            mask = torch.cat((~img_seq_mask, mask),dim=1)
            #lengths = list(map(lambda x: x + img_seq_len, lengths))
            lengths = lengths + img_seq_len
            att_mask = torch.cat((img_seq_mask, att_mask), dim=1)
        #all_encoder_layers, pooled_output = self.bert(inputs[:, :seq_max_len], token_type_ids=None,
        #                                  attention_mask=att_mask.to(device), img_feats = f_t_all)
        #embeds = torch.cat(all_encoder_layers[-self.bert_n_layers:],-1)
        last_encoder_layers, pooled_output = self.bert(inputs[:, :seq_max_len], token_type_ids=None,
                                          attention_mask=att_mask, img_feats = f_t_all)
        embeds = last_encoder_layers

        if self.reverse_input:
            reversed_embeds = torch.zeros(embeds.size()).to(embeds.device)
            if f_t_all is not None:
                img_seq_len = f_t_all.shape[1]
                reverse_idx = torch.arange(img_seq_len + seq_max_len-1,-1,-1)
            else:
                reverse_idx = torch.arange(seq_max_len-1,-1,-1)
            reversed_embeds[att_mask] = embeds[:, reverse_idx][att_mask[:,reverse_idx]]
            embeds = reversed_embeds

        if not self.update:
            embeds = embeds.detach()  # batch_size x input_len x bert_hidden_size*n

        if not self.top_lstm:
            #ctx = self.ctx2decoder(embeds)
            ctx = embeds
            c_t = self.encoder2decoder_ct(embeds[:,-1])
            decoder_init = nn.Tanh()(self.encoder2decoder_ht(pooled_output))
        else:
            h0, c0 = self.init_state(batch_size)
            packed_embeds = pack_padded_sequence(embeds, list(lengths), batch_first=True)
            enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

            if self.num_directions == 2:
                h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
                c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
            else:
                h_t = enc_h_t[-1]
                c_t = enc_c_t[-1] # (batch, hidden_size)

            decoder_init = nn.Tanh()(self.encoder_lstm2decoder_ht(h_t))
            if self.hidden_size * self.num_directions != self.dec_hidden_size:
                c_t = self.encoder_lstm2decoder_ct(c_t)

            ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)

        if not self.pretrain:
            ctx = self.drop(ctx)
        return ctx,decoder_init,c_t, mask  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)


class AddEncoder(nn.Module):
    def __init__(self, config):
        super(AddEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.vl_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if head_mask is not None:
                layer_outputs = layer_module(hidden_states, extended_attention_mask, head_mask[i])
            else:
                layer_outputs = layer_module(hidden_states, extended_attention_mask)

            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)




class BertAddEncoder(nn.Module):

    #transformer_hidden_size = 768
    lstm_num_layers = 1

    def __init__(self, vision_size, hidden_size, dec_hidden_size, dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, vl_layers, bert_type="small"):
        super(BertAddEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.dropout_ratio = dropout_ratio
        self.drop = nn.Dropout(p=dropout_ratio)
        self.update = update
        self.bert_n_layers = bert_n_layers
        self.reverse_input = reverse_input
        self.top_lstm = top_lstm
        self.bert_type = bert_type
        self.vl_layers = vl_layers
        self.pretrain = True
        if self.bert_type == "large":
            self.config = BertConfig.from_pretrained('bert-large-uncased')
            self.config.vl_layers = self.vl_layers
            self.bert = BertModel.from_pretrained('bert-large-uncased').cuda()
            self.transformer_hidden_size = 1024
        else:
            self.config = BertConfig.from_pretrained('bert-base-uncased')
            self.config.vl_layers = self.vl_layers
            self.bert = BertModel.from_pretrained('bert-base-uncased').cuda()
            self.transformer_hidden_size = 768

        #if self.bert_type == "large":
        #    self.config = BertConfig.from_pretrained('bert-large-uncased')
        #    self.config.img_feature_dim = vision_size
        #    self.config.img_feature_type = ""
        #    self.bert = BertModel(self.config)
        #    self.transformer_hidden_size = 1024
        #else:
        #    self.config = BertConfig.from_pretrained('bert-base-uncased')
        #    self.config.img_feature_dim = vision_size
        #    self.config.img_feature_type = ""
        #    self.bert = BertImgModel(self.config)
        #    self.transformer_hidden_size = 768

        self.img_embedding = nn.Linear(vision_size, self.config.hidden_size)
        self.addlayer = AddEncoder(self.config)
        self.lstm = nn.LSTM(self.transformer_hidden_size*self.bert_n_layers, self.hidden_size, self.lstm_num_layers, batch_first=True,
                             dropout=dropout_ratio if self.lstm_num_layers>1 else 0, bidirectional=bidirectional)
        #
        self.num_directions = 2 if bidirectional else 1

        self.linear_n_in = self.transformer_hidden_size * self.bert_n_layers if not self.top_lstm else hidden_size * self.num_directions
        self.encoder2decoder_ht = nn.Linear(self.linear_n_in, self.dec_hidden_size)
        self.encoder2decoder_ct = nn.Linear(self.linear_n_in, self.dec_hidden_size)

        self.encoder_lstm2decoder_ht = nn.Linear(hidden_size * self.num_directions, self.dec_hidden_size)
        self.encoder_lstm2decoder_ct = nn.Linear(hidden_size * self.num_directions, self.dec_hidden_size)
        #self.ctx2decoder = nn.Linear(self.transformer_hidden_size, 2 * self.hidden_size)
        #_ = self.flip_text_bert_params(False)  # default is to fix the text bert params



    def init_state(self, batch_size):
        ''' Initialize to zero cell states and hidden states.'''
        h0 = Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size ), requires_grad=False)
        c0 = Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size ), requires_grad=False)
        return h0, c0

    def flip_text_bert_params(self, bool_input):
        result = list()
        for name, param in self.bert.named_parameters():
            if name in self.text_bert_params:
                if param.requires_grad != bool_input:
                    param.requires_grad = bool_input
                    result.append(param)
        return result


    def forward(self, inputs, mask, lengths,f_t_all=None):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        batch_size = inputs.size(0)
        seq_max_len = mask.size(1)
        att_mask = ~mask
        #all_encoder_layers, pooled_output = self.bert(inputs[:, :seq_max_len], token_type_ids=None,
        #                                  attention_mask=att_mask.to(device))
        all_encoder_layers, pooled_output = self.bert(inputs[:, :seq_max_len], token_type_ids=None,
                                          attention_mask=att_mask)
        text_embeds = torch.cat(all_encoder_layers[-self.bert_n_layers:],-1)
        #last_encoder_layers, pooled_output = self.bert(inputs[:, :seq_max_len], token_type_ids=None,
        #                                  attention_mask=att_mask.to(device))
        #embeds = last_encoder_layers

        if not self.update:
            text_embeds = text_embeds.detach()  # batch_size x input_len x bert_hidden_size*n


        # Now we have pure text embeddings. Then need to concat vision emed and extend mask for image features
        if f_t_all is not None:
            img_embedding_output = self.img_embedding(f_t_all)
            img_seq_len = f_t_all.shape[1]
            #img_seq_mask = torch.ones(batch_size, img_seq_len, dtype=att_mask.dtype).to(device)
            img_seq_mask = torch.ones(batch_size, img_seq_len, dtype=att_mask.dtype).to(att_mask.device)
            mask = torch.cat((~img_seq_mask, mask),dim=1)
            #lengths = list(map(lambda x: x + img_seq_len, lengths))
            lengths = lengths + img_seq_len
            att_mask = torch.cat((img_seq_mask, att_mask), dim=1)
            input_embeds = torch.cat((img_embedding_output, text_embeds), 1)
            outputs = self.addlayer(input_embeds, att_mask)
            embeds = outputs[0]
        else:
            embeds = text_embeds

        #if not self.update:
        #    embeds = embeds.detach()

        if self.reverse_input:
            #reversed_embeds = torch.zeros(embeds.size()).to(device)
            reversed_embeds = torch.zeros(embeds.size()).to(embeds.device)
            if f_t_all is not None:
                img_seq_len = f_t_all.shape[1]
                reverse_idx = torch.arange(img_seq_len + seq_max_len-1,-1,-1)
            else:
                reverse_idx = torch.arange(seq_max_len-1,-1,-1)
            reversed_embeds[att_mask] = embeds[:, reverse_idx][att_mask[:,reverse_idx]]
            embeds = reversed_embeds


        if not self.top_lstm:
            #ctx = self.ctx2decoder(embeds)
            ctx = embeds
            c_t = self.encoder2decoder_ct(embeds[:,-1])
            decoder_init = nn.Tanh()(self.encoder2decoder_ht(pooled_output))
        else:
            h0, c0 = self.init_state(batch_size)
            packed_embeds = pack_padded_sequence(embeds, list(lengths), batch_first=True)
            enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0.to(embeds.device), c0.to(embeds.device)))

            if self.num_directions == 2:
                h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
                c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
            else:
                h_t = enc_h_t[-1]
                c_t = enc_c_t[-1] # (batch, hidden_size)

            decoder_init = nn.Tanh()(self.encoder_lstm2decoder_ht(h_t))
            if self.hidden_size * self.num_directions != self.dec_hidden_size:
                c_t = self.encoder_lstm2decoder_ct(c_t)

            ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)

        if self.pretrain:
            origin_length = mask.shape[1]
            padded_ctx = torch.zeros(ctx.shape[0], origin_length-lengths[0],ctx.shape[2]).to(ctx.device)
            ctx = torch.cat((ctx, padded_ctx), dim=1)
            return ctx,decoder_init,c_t,mask,pooled_output # (batch, seq_len, hidden_size*num_directions)
        else:
            ctx = self.drop(ctx)
            return ctx,decoder_init,c_t, mask  # (batch, seq_len, hidden_size*num_directions)


class HugAddEncoder(nn.Module):

    #transformer_hidden_size = 768
    lstm_num_layers = 1

    def __init__(self, vision_size, hidden_size, dec_hidden_size, dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, vl_layers,bert_type="small",update_add_layer=True):
        super(HugAddEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.dropout_ratio = dropout_ratio
        self.drop = nn.Dropout(p=dropout_ratio)
        self.update = update
        self.bert_n_layers = bert_n_layers
        self.reverse_input = reverse_input
        self.top_lstm = top_lstm
        self.bert_type = bert_type
        self.pretrain = True
        if self.bert_type == "large":
            self.config = BertConfig.from_pretrained('bert-large-uncased')
            self.config.img_feature_dim = vision_size
            self.config.img_feature_type = ""
            self.config.update_lang_bert = update
            self.config.update_add_layer = update_add_layer
            self.config.vl_layers = vl_layers
            self.bert = BertAddModel(self.config)
            self.transformer_hidden_size = 1024
        else:
            self.config = BertConfig.from_pretrained('bert-base-uncased')
            self.config.img_feature_dim = vision_size
            self.config.img_feature_type = ""
            self.config.update_lang_bert = update
            self.config.update_add_layer = update_add_layer
            self.config.vl_layers = vl_layers
            self.bert = BertAddModel(self.config)
            self.transformer_hidden_size = 768

        self.lstm = nn.LSTM(self.transformer_hidden_size*self.bert_n_layers, self.hidden_size, self.lstm_num_layers, batch_first=True,
                             dropout=dropout_ratio if self.lstm_num_layers>1 else 0, bidirectional=bidirectional)
        #
        self.num_directions = 2 if bidirectional else 1

        self.linear_n_in = self.transformer_hidden_size * self.bert_n_layers if not self.top_lstm else hidden_size * self.num_directions
        self.encoder2decoder_ht = nn.Linear(self.linear_n_in, self.dec_hidden_size)
        self.encoder2decoder_ct = nn.Linear(self.linear_n_in, self.dec_hidden_size)

        self.encoder_lstm2decoder_ht = nn.Linear(hidden_size * self.num_directions, self.dec_hidden_size)
        self.encoder_lstm2decoder_ct = nn.Linear(hidden_size * self.num_directions, self.dec_hidden_size)



    def init_state(self, batch_size):
        ''' Initialize to zero cell states and hidden states.'''
        h0 = Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size ), requires_grad=False)
        c0 = Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size ), requires_grad=False)
        return h0, c0

    def flip_text_bert_params(self, bool_input):
        result = list()
        for name, param in self.bert.named_parameters():
            if name in self.text_bert_params:
                if param.requires_grad != bool_input:
                    param.requires_grad = bool_input
                    result.append(param)
        return result


    def forward(self, inputs, mask, lengths,f_t_all=None):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        batch_size = inputs.size(0)
        seq_max_len = mask.size(1)
        att_mask = ~mask
        # need to extend mask for image features
        if f_t_all is not None:
            img_seq_len = f_t_all.shape[1]
            img_seq_mask = torch.ones(batch_size, img_seq_len, dtype=att_mask.dtype).to(att_mask.device)
            mask = torch.cat((~img_seq_mask, mask),dim=1)
        #    #lengths = list(map(lambda x: x + img_seq_len, lengths))
            lengths = lengths + img_seq_len
        #    att_mask = torch.cat((img_seq_mask, att_mask), dim=1)
        #all_encoder_layers, pooled_output = self.bert(inputs[:, :seq_max_len], token_type_ids=None,
        #                                  attention_mask=att_mask.to(device), img_feats = f_t_all)
        #embeds = torch.cat(all_encoder_layers[-self.bert_n_layers:],-1)
        last_encoder_layers, pooled_output = self.bert(inputs[:, :seq_max_len], token_type_ids=None,
                                          attention_mask=att_mask, img_feats = f_t_all)
        embeds = last_encoder_layers

        if self.reverse_input:
            reversed_embeds = torch.zeros(embeds.size()).to(embeds.device)
            if f_t_all is not None:
                img_seq_len = f_t_all.shape[1]
                reverse_idx = torch.arange(img_seq_len + seq_max_len-1,-1,-1)
                idmask = torch.cat((img_seq_mask, att_mask), dim=1)
                reversed_embeds[idmask] = embeds[:, reverse_idx][idmask[:,reverse_idx]]
            else:
                reverse_idx = torch.arange(seq_max_len-1,-1,-1)
                reversed_embeds[att_mask] = embeds[:, reverse_idx][att_mask[:,reverse_idx]]
            embeds = reversed_embeds

        if not self.config.update_add_layer:
            embeds = embeds.detach()  # batch_size x input_len x bert_hidden_size*n

        if not self.top_lstm:
            #ctx = self.ctx2decoder(embeds)
            ctx = embeds
            c_t = self.encoder2decoder_ct(embeds[:,-1])
            decoder_init = nn.Tanh()(self.encoder2decoder_ht(pooled_output))
        else:
            h0, c0 = self.init_state(batch_size)
            packed_embeds = pack_padded_sequence(embeds, list(lengths), batch_first=True)
            enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0.to(embeds.device), c0.to(embeds.device)))

            if self.num_directions == 2:
                h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
                c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
            else:
                h_t = enc_h_t[-1]
                c_t = enc_c_t[-1] # (batch, hidden_size)

            decoder_init = nn.Tanh()(self.encoder_lstm2decoder_ht(h_t))
            if self.hidden_size * self.num_directions != self.dec_hidden_size:
                c_t = self.encoder_lstm2decoder_ct(c_t)

            ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)

        if not self.pretrain:
            ctx = self.drop(ctx)
        return ctx,decoder_init,c_t, mask  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)

#class VicEncoder(nn.Module):
#
#    #transformer_hidden_size = 768
#    lstm_num_layers = 1
#
#    def __init__(self, vision_size, hidden_size, dec_hidden_size, dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, vl_layers,bert_type="small",update_add_layer=True):
#        super(VicEncoder, self).__init__()
#        self.hidden_size = hidden_size
#        self.dec_hidden_size = dec_hidden_size
#        self.dropout_ratio = dropout_ratio
#        self.drop = nn.Dropout(p=dropout_ratio)
#        self.update = update
#        self.bert_n_layers = bert_n_layers
#        self.reverse_input = reverse_input
#        self.top_lstm = top_lstm
#        self.bert_type = bert_type
#        self.pretrain = True
#        if self.bert_type == "large":
#            self.config = BertConfig.from_pretrained('bert-large-uncased')
#            self.config.img_feature_dim = vision_size
#            self.config.img_feature_type = ""
#            self.config.update_lang_bert = update
#            self.config.update_add_layer = update_add_layer
#            self.config.vl_layers = vl_layers
#            self.bert = VicModel(self.config)
#            self.transformer_hidden_size = 1024
#        else:
#            self.config = BertConfig.from_pretrained('bert-base-uncased')
#            self.config.img_feature_dim = vision_size
#            self.config.img_feature_type = ""
#            self.config.update_lang_bert = update
#            self.config.update_add_layer = update_add_layer
#            self.config.vl_layers = vl_layers
#            self.bert = VicModel(self.config)
#            self.transformer_hidden_size = 768
#
#        self.lstm = nn.LSTM(self.transformer_hidden_size*self.bert_n_layers, self.hidden_size, self.lstm_num_layers, batch_first=True,
#                             dropout=dropout_ratio if self.lstm_num_layers>1 else 0, bidirectional=bidirectional)
#        #
#        self.num_directions = 2 if bidirectional else 1
#
#        self.linear_n_in = self.transformer_hidden_size * self.bert_n_layers if not self.top_lstm else hidden_size * self.num_directions
#        self.encoder2decoder_ht = nn.Linear(self.linear_n_in, self.dec_hidden_size)
#        self.encoder2decoder_ct = nn.Linear(self.linear_n_in, self.dec_hidden_size)
#
#        self.encoder_lstm2decoder_ht = nn.Linear(hidden_size * self.num_directions, self.dec_hidden_size)
#        self.encoder_lstm2decoder_ct = nn.Linear(hidden_size * self.num_directions, self.dec_hidden_size)
#
#
#
#    def init_state(self, batch_size):
#        ''' Initialize to zero cell states and hidden states.'''
#        h0 = Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size ), requires_grad=False)
#        c0 = Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size ), requires_grad=False)
#        return h0, c0
#
#    def flip_text_bert_params(self, bool_input):
#        result = list()
#        for name, param in self.bert.named_parameters():
#            if name in self.text_bert_params:
#                if param.requires_grad != bool_input:
#                    param.requires_grad = bool_input
#                    result.append(param)
#        return result
#
#
#    def forward(self, inputs, mask, lengths,f_t_all=None):
#        ''' Expects input vocab indices as (batch, seq_len). Also requires a
#            list of lengths for dynamic batching. '''
#        batch_size = inputs.size(0)
#        seq_max_len = mask.size(1)
#        att_mask = ~mask
#        # need to extend mask for image features
#        #if f_t_all is not None:
#        #    img_seq_len = f_t_all.shape[1]
#        #    img_seq_mask = torch.ones(batch_size, img_seq_len, dtype=att_mask.dtype).to(att_mask.device)
#        #    mask = torch.cat((~img_seq_mask, mask),dim=1)
#        #    lengths = lengths + img_seq_len
#
#        last_encoder_layers, pooled_output = self.bert(inputs[:, :seq_max_len], token_type_ids=None,
#                                          attention_mask=att_mask, img_feats = f_t_all)
#        embeds = last_encoder_layers
#
#        if self.reverse_input:
#            reversed_embeds = torch.zeros(embeds.size()).to(embeds.device)
#            #if f_t_all is not None:
#            #    img_seq_len = f_t_all.shape[1]
#            #    reverse_idx = torch.arange(img_seq_len + seq_max_len-1,-1,-1)
#            #    idmask = torch.cat((img_seq_mask, att_mask), dim=1)
#            #    reversed_embeds[idmask] = embeds[:, reverse_idx][idmask[:,reverse_idx]]
#            #else:
#            reverse_idx = torch.arange(seq_max_len-1,-1,-1)
#            reversed_embeds[att_mask] = embeds[:, reverse_idx][att_mask[:,reverse_idx]]
#            embeds = reversed_embeds
#
#        if not self.config.update_add_layer:
#            embeds = embeds.detach()  # batch_size x input_len x bert_hidden_size*n
#
#        if not self.top_lstm:
#            #ctx = self.ctx2decoder(embeds)
#            ctx = embeds
#            c_t = self.encoder2decoder_ct(embeds[:,-1])
#            decoder_init = nn.Tanh()(self.encoder2decoder_ht(pooled_output))
#        else:
#            h0, c0 = self.init_state(batch_size)
#            packed_embeds = pack_padded_sequence(embeds, list(lengths), batch_first=True)
#            enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0.to(embeds.device), c0.to(embeds.device)))
#
#            if self.num_directions == 2:
#                h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
#                c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
#            else:
#                h_t = enc_h_t[-1]
#                c_t = enc_c_t[-1] # (batch, hidden_size)
#
#            decoder_init = nn.Tanh()(self.encoder_lstm2decoder_ht(h_t))
#            if self.hidden_size * self.num_directions != self.dec_hidden_size:
#                c_t = self.encoder_lstm2decoder_ct(c_t)
#
#            ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)
#
#        if not self.pretrain:
#            ctx = self.drop(ctx)
#        return ctx,decoder_init,c_t, mask  # (batch, seq_len, hidden_size*num_directions)
#                                 # (batch, hidden_size)



class VisionEncoder(nn.Module):
    def __init__(self, vision_size, config):
        super(VisionEncoder, self).__init__()
        self.vision_size = vision_size
        self.config = config
        self.img_fc = nn.Linear(vision_size, self.config.hidden_size)
        self.img_layer_norm = BertLayerNorm(self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

    def forward(self, img_feat):
        x = self.img_fc(img_feat)
        x = self.img_layer_norm(x)
        x = self.dropout(x)
        return x

class BertAddSepEncoder(nn.Module):

    #transformer_hidden_size = 768
    lstm_num_layers = 1

    def __init__(self, vision_size, hidden_size, dec_hidden_size, dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, vl_layers, bert_type="small"):
        super(BertAddSepEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.dropout_ratio = dropout_ratio
        self.drop = nn.Dropout(p=dropout_ratio)
        self.update = update
        self.bert_n_layers = bert_n_layers
        self.reverse_input = reverse_input
        self.top_lstm = top_lstm
        self.bert_type = bert_type
        self.vl_layers = vl_layers
        self.pretrain = True
        if self.bert_type == "large":
            self.config = BertConfig.from_pretrained('bert-large-uncased')
            self.config.vl_layers = self.vl_layers
            self.bert = BertModel.from_pretrained('bert-large-uncased').cuda()
            self.transformer_hidden_size = 1024
        else:
            self.config = BertConfig.from_pretrained('bert-base-uncased')
            self.config.vl_layers = self.vl_layers
            self.bert = BertModel.from_pretrained('bert-base-uncased').cuda()
            self.transformer_hidden_size = 768


        self.vision_encoder = VisionEncoder(vision_size, self.config)
        self.addlayer = AddEncoder(self.config)
        self.pooler = BertPooler(self.config)
        self.lstm = nn.LSTM(self.transformer_hidden_size*self.bert_n_layers, self.hidden_size, self.lstm_num_layers, batch_first=True,
                             dropout=dropout_ratio if self.lstm_num_layers>1 else 0, bidirectional=bidirectional)
        #
        self.num_directions = 2 if bidirectional else 1

        self.linear_n_in = self.transformer_hidden_size * self.bert_n_layers if not self.top_lstm else hidden_size * self.num_directions
        self.encoder2decoder_ht = nn.Linear(self.linear_n_in, self.dec_hidden_size)
        self.encoder2decoder_ct = nn.Linear(self.linear_n_in, self.dec_hidden_size)

        self.encoder_lstm2decoder_ht = nn.Linear(hidden_size * self.num_directions, self.dec_hidden_size)
        self.encoder_lstm2decoder_ct = nn.Linear(hidden_size * self.num_directions, self.dec_hidden_size)
        #self.ctx2decoder = nn.Linear(self.transformer_hidden_size, 2 * self.hidden_size)
        #_ = self.flip_text_bert_params(False)  # default is to fix the text bert params



    def init_state(self, batch_size):
        ''' Initialize to zero cell states and hidden states.'''
        h0 = Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size ), requires_grad=False)
        c0 = Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size ), requires_grad=False)
        return h0, c0

    def flip_text_bert_params(self, bool_input):
        result = list()
        for name, param in self.bert.named_parameters():
            if name in self.text_bert_params:
                if param.requires_grad != bool_input:
                    param.requires_grad = bool_input
                    result.append(param)
        return result


    def forward(self, inputs, mask, lengths,f_t_all=None):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        batch_size = inputs.size(0)
        seq_max_len = mask.size(1)
        att_mask = ~mask
        #all_encoder_layers, pooled_output = self.bert(inputs[:, :seq_max_len], token_type_ids=None,
        #                                  attention_mask=att_mask.to(device))
        all_encoder_layers, pooled_output = self.bert(inputs[:, :seq_max_len], token_type_ids=None,
                                          attention_mask=att_mask)
        text_embeds = torch.cat(all_encoder_layers[-self.bert_n_layers:],-1)
        #last_encoder_layers, pooled_output = self.bert(inputs[:, :seq_max_len], token_type_ids=None,
        #                                  attention_mask=att_mask.to(device))
        #embeds = last_encoder_layers

        if not self.update:
            text_embeds = text_embeds.detach()  # batch_size x input_len x bert_hidden_size*n


        # Now we have pure text embeddings. Then need to concat vision emed and extend mask for image features
        if f_t_all is not None:
            img_embedding_output = self.vision_encoder(f_t_all)
            img_seq_len = f_t_all.shape[1]
            img_seq_mask = torch.ones(batch_size, img_seq_len, dtype=att_mask.dtype).to(att_mask.device)
            #mask = torch.cat((~img_seq_mask, mask),dim=1)
            #lengths = list(map(lambda x: x + img_seq_len, lengths))
            #lengths = lengths + img_seq_len
            fusion_att_mask = torch.cat((img_seq_mask, att_mask), dim=1)
            input_embeds = torch.cat((img_embedding_output, text_embeds), 1)
            outputs = self.addlayer(input_embeds, fusion_att_mask)
            #embeds = outputs[0]
            embeds = outputs[0][:,img_seq_len:,:]
            pooled_output = self.pooler(embeds)
        else:
            embeds = text_embeds

        #if not self.update:
        #    embeds = embeds.detach()

        if self.reverse_input:
            reversed_embeds = torch.zeros(embeds.size()).to(embeds.device)
            #if f_t_all is not None:
            #    img_seq_len = f_t_all.shape[1]
            #    reverse_idx = torch.arange(img_seq_len + seq_max_len-1,-1,-1)
            #else:
            #    reverse_idx = torch.arange(seq_max_len-1,-1,-1)
            reverse_idx = torch.arange(seq_max_len-1,-1,-1)
            reversed_embeds[att_mask] = embeds[:, reverse_idx][att_mask[:,reverse_idx]]
            embeds = reversed_embeds


        if not self.top_lstm:
            #ctx = self.ctx2decoder(embeds)
            ctx = embeds
            c_t = self.encoder2decoder_ct(embeds[:,-1])
            decoder_init = nn.Tanh()(self.encoder2decoder_ht(pooled_output))
        else:
            h0, c0 = self.init_state(batch_size)
            packed_embeds = pack_padded_sequence(embeds, list(lengths), batch_first=True)
            enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0.to(embeds.device), c0.to(embeds.device)))

            if self.num_directions == 2:
                h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
                c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
            else:
                h_t = enc_h_t[-1]
                c_t = enc_c_t[-1] # (batch, hidden_size)

            decoder_init = nn.Tanh()(self.encoder_lstm2decoder_ht(h_t))
            if self.hidden_size * self.num_directions != self.dec_hidden_size:
                c_t = self.encoder_lstm2decoder_ct(c_t)

            ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)

        if self.pretrain:
            origin_length = mask.shape[1]
            padded_ctx = torch.zeros(ctx.shape[0],origin_length-lengths[0],ctx.shape[2]).to(ctx.device)
            ctx = torch.cat((ctx, padded_ctx), dim=1)

            return ctx,decoder_init,c_t,mask,pooled_output # (batch, seq_len, hidden_size*num_directions)
        else:
            ctx = self.drop(ctx)
            return ctx,decoder_init,c_t, mask  # (batch, seq_len, hidden_size*num_directions)



class BertMixEncoder(nn.Module):

    #transformer_hidden_size = 768
    lstm_num_layers = 1

    def __init__(self, vision_size, hidden_size, dec_hidden_size, dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, vl_layers, bert_type="small",pretrained_encoder=None):
        super(BertMixEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.dropout_ratio = dropout_ratio
        self.drop = nn.Dropout(p=dropout_ratio)
        self.update = update
        self.bert_n_layers = bert_n_layers
        self.reverse_input = reverse_input
        self.top_lstm = top_lstm
        self.bert_type = bert_type
        self.vl_layers = vl_layers
        self.pretrain = True
        if self.bert_type == "large":
            self.config = BertConfig.from_pretrained('bert-large-uncased')
            self.config.vl_layers = self.vl_layers
            self.bert = BertModel.from_pretrained('bert-large-uncased').cuda()
            self.transformer_hidden_size = 1024
        else:
            self.config = BertConfig.from_pretrained('bert-base-uncased')
            self.config.vl_layers = self.vl_layers
            self.bert = BertModel.from_pretrained('bert-base-uncased').cuda()
            self.transformer_hidden_size = 768


        self.vision_encoder = VisionEncoder(vision_size, self.config)
        self.addlayer = AddEncoder(self.config)
        self.pooler = BertPooler(self.config)
        self.lstm = nn.LSTM(self.transformer_hidden_size*self.bert_n_layers, self.hidden_size, self.lstm_num_layers, batch_first=True,
                             dropout=dropout_ratio if self.lstm_num_layers>1 else 0, bidirectional=bidirectional)
        #
        self.num_directions = 2 if bidirectional else 1

        self.linear_n_in = self.transformer_hidden_size * self.bert_n_layers if not self.top_lstm else hidden_size * self.num_directions
        self.encoder2decoder_ht = nn.Linear(self.linear_n_in, self.dec_hidden_size)
        self.encoder2decoder_ct = nn.Linear(self.linear_n_in, self.dec_hidden_size)

        self.encoder_lstm2decoder_ht = nn.Linear(hidden_size * self.num_directions, self.dec_hidden_size)
        self.encoder_lstm2decoder_ct = nn.Linear(hidden_size * self.num_directions, self.dec_hidden_size)
        #self.ctx2decoder = nn.Linear(self.transformer_hidden_size, 2 * self.hidden_size)
        #_ = self.flip_text_bert_params(False)  # default is to fix the text bert params



    def init_state(self, batch_size):
        ''' Initialize to zero cell states and hidden states.'''
        h0 = Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size ), requires_grad=False)
        c0 = Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size ), requires_grad=False)
        return h0, c0

    def flip_text_bert_params(self, bool_input):
        result = list()
        for name, param in self.bert.named_parameters():
            if name in self.text_bert_params:
                if param.requires_grad != bool_input:
                    param.requires_grad = bool_input
                    result.append(param)
        return result


    def forward(self, inputs, mask, lengths,f_t_all=None):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        batch_size = inputs.size(0)
        seq_max_len = mask.size(1)
        att_mask = ~mask
        #all_encoder_layers, pooled_output = self.bert(inputs[:, :seq_max_len], token_type_ids=None,
        #                                  attention_mask=att_mask.to(device))
        all_encoder_layers, pooled_output = self.bert(inputs[:, :seq_max_len], token_type_ids=None,
                                          attention_mask=att_mask)
        text_embeds = torch.cat(all_encoder_layers[-self.bert_n_layers:],-1)
        #last_encoder_layers, pooled_output = self.bert(inputs[:, :seq_max_len], token_type_ids=None,
        #                                  attention_mask=att_mask.to(device))
        #embeds = last_encoder_layers

        if not self.update:
            text_embeds = text_embeds.detach()  # batch_size x input_len x bert_hidden_size*n


        # Now we have pure text embeddings. Then need to concat vision emed and extend mask for image features
        if f_t_all is not None:
            img_embedding_output = self.vision_encoder(f_t_all)
            img_seq_len = f_t_all.shape[1]
            img_seq_mask = torch.ones(batch_size, img_seq_len, dtype=att_mask.dtype).to(att_mask.device)
            #mask = torch.cat((~img_seq_mask, mask),dim=1)
            #lengths = list(map(lambda x: x + img_seq_len, lengths))
            #lengths = lengths + img_seq_len
            fusion_att_mask = torch.cat((img_seq_mask, att_mask), dim=1)
            input_embeds = torch.cat((img_embedding_output, text_embeds), 1)
            outputs = self.addlayer(input_embeds, fusion_att_mask)
            #embeds = outputs[0]
            embeds = outputs[0][:,img_seq_len:,:]
            pooled_output = self.pooler(embeds)
        else:
            embeds = text_embeds

        #if not self.update:
        #    embeds = embeds.detach()

        if self.reverse_input:
            reversed_embeds = torch.zeros(embeds.size()).to(embeds.device)
            #if f_t_all is not None:
            #    img_seq_len = f_t_all.shape[1]
            #    reverse_idx = torch.arange(img_seq_len + seq_max_len-1,-1,-1)
            #else:
            #    reverse_idx = torch.arange(seq_max_len-1,-1,-1)
            reverse_idx = torch.arange(seq_max_len-1,-1,-1)
            reversed_embeds[att_mask] = embeds[:, reverse_idx][att_mask[:,reverse_idx]]
            embeds = reversed_embeds


        if not self.top_lstm:
            #ctx = self.ctx2decoder(embeds)
            ctx = embeds
            c_t = self.encoder2decoder_ct(embeds[:,-1])
            decoder_init = nn.Tanh()(self.encoder2decoder_ht(pooled_output))
        else:
            h0, c0 = self.init_state(batch_size)
            packed_embeds = pack_padded_sequence(embeds, list(lengths), batch_first=True)
            enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0.to(embeds.device), c0.to(embeds.device)))

            if self.num_directions == 2:
                h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
                c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
            else:
                h_t = enc_h_t[-1]
                c_t = enc_c_t[-1] # (batch, hidden_size)

            decoder_init = nn.Tanh()(self.encoder_lstm2decoder_ht(h_t))
            if self.hidden_size * self.num_directions != self.dec_hidden_size:
                c_t = self.encoder_lstm2decoder_ct(c_t)

            ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)

        if self.pretrain:
            origin_length = mask.shape[1]
            padded_ctx = torch.zeros(ctx.shape[0],origin_length-lengths[0],ctx.shape[2]).to(ctx.device)
            ctx = torch.cat((ctx, padded_ctx), dim=1)

            return ctx,decoder_init,c_t,mask,pooled_output # (batch, seq_len, hidden_size*num_directions)
        else:
            ctx = self.drop(ctx)
            return ctx,decoder_init,c_t, mask  # (batch, seq_len, hidden_size*num_directions)



class BertLangEncoder(nn.Module):

    #transformer_hidden_size = 768
    lstm_num_layers = 1

    def __init__(self, vision_size, hidden_size, dec_hidden_size, dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, vl_layers, bert_type="small"):
        super(BertLangEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.dropout_ratio = dropout_ratio
        self.drop = nn.Dropout(p=dropout_ratio)
        self.update = update
        self.bert_n_layers = bert_n_layers
        self.reverse_input = reverse_input
        self.top_lstm = top_lstm
        self.bert_type = bert_type
        self.vl_layers = vl_layers
        self.pretrain = True
        if self.bert_type == "large":
            self.config = BertConfig.from_pretrained('bert-large-uncased')
            self.config.vl_layers = self.vl_layers
            self.bert = BertModel.from_pretrained('bert-large-uncased').cuda()
            self.transformer_hidden_size = 1024
        else:
            self.config = BertConfig.from_pretrained('bert-base-uncased')
            self.config.vl_layers = self.vl_layers
            self.bert = BertModel.from_pretrained('bert-base-uncased').cuda()
            self.transformer_hidden_size = 768

        #if self.bert_type == "large":
        #    self.config = BertConfig.from_pretrained('bert-large-uncased')
        #    self.config.img_feature_dim = vision_size
        #    self.config.img_feature_type = ""
        #    self.bert = BertModel(self.config)
        #    self.transformer_hidden_size = 1024
        #else:
        #    self.config = BertConfig.from_pretrained('bert-base-uncased')
        #    self.config.img_feature_dim = vision_size
        #    self.config.img_feature_type = ""
        #    self.bert = BertImgModel(self.config)
        #    self.transformer_hidden_size = 768

        self.img_embedding = nn.Linear(vision_size, self.config.hidden_size)
        self.addlayer = AddEncoder(self.config)
        self.lstm = nn.LSTM(self.transformer_hidden_size*self.bert_n_layers, self.hidden_size, self.lstm_num_layers, batch_first=True,
                             dropout=dropout_ratio if self.lstm_num_layers>1 else 0, bidirectional=bidirectional)
        #
        self.num_directions = 2 if bidirectional else 1

        self.linear_n_in = self.transformer_hidden_size * self.bert_n_layers if not self.top_lstm else hidden_size * self.num_directions
        self.encoder2decoder_ht = nn.Linear(self.linear_n_in, self.dec_hidden_size)
        self.encoder2decoder_ct = nn.Linear(self.linear_n_in, self.dec_hidden_size)

        self.encoder_lstm2decoder_ht = nn.Linear(hidden_size * self.num_directions, self.dec_hidden_size)
        self.encoder_lstm2decoder_ct = nn.Linear(hidden_size * self.num_directions, self.dec_hidden_size)
        #self.ctx2decoder = nn.Linear(self.transformer_hidden_size, 2 * self.hidden_size)
        #_ = self.flip_text_bert_params(False)  # default is to fix the text bert params



    def init_state(self, batch_size):
        ''' Initialize to zero cell states and hidden states.'''
        h0 = Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size ), requires_grad=False)
        c0 = Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size ), requires_grad=False)
        return h0.to(device), c0.to(device)

    def flip_text_bert_params(self, bool_input):
        result = list()
        for name, param in self.bert.named_parameters():
            if name in self.text_bert_params:
                if param.requires_grad != bool_input:
                    param.requires_grad = bool_input
                    result.append(param)
        return result


    def forward(self, inputs, mask, lengths,f_t_all=None):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        batch_size = inputs.size(0)
        seq_max_len = mask.size(1)
        att_mask = ~mask
        #all_encoder_layers, pooled_output = self.bert(inputs[:, :seq_max_len], token_type_ids=None,
        #                                  attention_mask=att_mask.to(device))
        all_encoder_layers, pooled_output = self.bert(inputs[:, :seq_max_len], token_type_ids=None,
                                          attention_mask=att_mask)
        text_embeds = torch.cat(all_encoder_layers[-self.bert_n_layers:],-1)
        #last_encoder_layers, pooled_output = self.bert(inputs[:, :seq_max_len], token_type_ids=None,
        #                                  attention_mask=att_mask.to(device))
        #embeds = last_encoder_layers

        #if not self.update:
        #    text_embeds = text_embeds.detach()  # batch_size x input_len x bert_hidden_size*n


        # Now we have pure text embeddings. Then need to concat vision emed and extend mask for image features
        if f_t_all is not None:
            img_embedding_output = self.img_embedding(f_t_all)
            img_seq_len = f_t_all.shape[1]
            #img_seq_mask = torch.ones(batch_size, img_seq_len, dtype=att_mask.dtype).to(device)
            img_seq_mask = torch.ones(batch_size, img_seq_len, dtype=att_mask.dtype).to(att_mask.device)
            mask = torch.cat((~img_seq_mask, mask),dim=1)
            #lengths = list(map(lambda x: x + img_seq_len, lengths))
            lengths = lengths + img_seq_len
            att_mask = torch.cat((img_seq_mask, att_mask), dim=1)
            input_embeds = torch.cat((img_embedding_output, text_embeds), 1)
            outputs = self.addlayer(input_embeds, att_mask)
            embeds = outputs[0]
        else:
            embeds = text_embeds

        if not self.update:
            embeds = embeds.detach()

        if self.reverse_input:
            #reversed_embeds = torch.zeros(embeds.size()).to(device)
            reversed_embeds = torch.zeros(embeds.size()).to(embeds.device)
            if f_t_all is not None:
                img_seq_len = f_t_all.shape[1]
                reverse_idx = torch.arange(img_seq_len + seq_max_len-1,-1,-1)
            else:
                reverse_idx = torch.arange(seq_max_len-1,-1,-1)
            reversed_embeds[att_mask] = embeds[:, reverse_idx][att_mask[:,reverse_idx]]
            embeds = reversed_embeds


        if not self.top_lstm:
            #ctx = self.ctx2decoder(embeds)
            ctx = embeds
            c_t = self.encoder2decoder_ct(embeds[:,-1])
            decoder_init = nn.Tanh()(self.encoder2decoder_ht(pooled_output))
        else:
            h0, c0 = self.init_state(batch_size)
            packed_embeds = pack_padded_sequence(embeds, list(lengths), batch_first=True)
            enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

            if self.num_directions == 2:
                h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
                c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
            else:
                h_t = enc_h_t[-1]
                c_t = enc_c_t[-1] # (batch, hidden_size)

            decoder_init = nn.Tanh()(self.encoder_lstm2decoder_ht(h_t))
            if self.hidden_size * self.num_directions != self.dec_hidden_size:
                c_t = self.encoder_lstm2decoder_ct(c_t)

            ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)

        if not self.pretrain:
            ctx = self.drop(ctx)
        return ctx,decoder_init,c_t, mask  # (batch, seq_len, hidden_size*num_directions)



class VicEncoder(nn.Module):

    #transformer_hidden_size = 768
    lstm_num_layers = 1

    def __init__(self, vision_size, hidden_size, dec_hidden_size, dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, vl_layers, pretrain=True, bert_type="small",update_add_layer=True):
        super(VicEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.dropout_ratio = dropout_ratio
        self.drop = nn.Dropout(p=dropout_ratio)
        self.update = update
        self.bert_n_layers = bert_n_layers
        self.reverse_input = reverse_input
        self.top_lstm = top_lstm
        self.bert_type = bert_type
        self.vl_layers = vl_layers
        self.pretrain = pretrain
#        if self.bert_type == "large":
#            self.config = BertConfig.from_pretrained('bert-large-uncased')
#            self.config.vl_layers = self.vl_layers
#            self.bert = BertModel.from_pretrained('bert-large-uncased').cuda()
#            self.transformer_hidden_size = 1024
#        else:
#            self.config = BertConfig.from_pretrained('bert-base-uncased')
#            self.config.vl_layers = self.vl_layers
#            self.bert = BertModel.from_pretrained('bert-base-uncased').cuda()
#            self.transformer_hidden_size = 768

        if self.bert_type == "large":
            self.config = BertConfig.from_pretrained('bert-large-uncased')
            self.config.img_feature_dim = vision_size
            self.config.img_feature_type = ""
            self.config.update_lang_bert = update
            self.config.update_add_layer = update_add_layer
            self.config.vl_layers = vl_layers
            self.bert = VicModel(self.config)
            self.transformer_hidden_size = 1024
        else:
            self.config = BertConfig.from_pretrained('bert-base-uncased')
            self.config.img_feature_dim = vision_size
            self.config.img_feature_type = ""
            self.config.update_lang_bert = update
            self.config.update_add_layer = update_add_layer
            self.config.vl_layers = vl_layers
            self.bert = VicModel(self.config)
            self.transformer_hidden_size = 768


        #self.img_embedding = VisionEncoder(vision_size, self.config)
        #self.addlayer = LXRTXLayer(self.config)
        self.lstm = nn.LSTM(self.transformer_hidden_size*self.bert_n_layers, self.hidden_size, self.lstm_num_layers, batch_first=True,
                             dropout=dropout_ratio if self.lstm_num_layers>1 else 0, bidirectional=bidirectional)
        #
        self.num_directions = 2 if bidirectional else 1

        self.linear_n_in = self.transformer_hidden_size * self.bert_n_layers if not self.top_lstm else hidden_size * self.num_directions
        self.encoder2decoder_ht = nn.Linear(self.linear_n_in, self.dec_hidden_size)
        self.encoder2decoder_ct = nn.Linear(self.linear_n_in, self.dec_hidden_size)

        self.encoder_lstm2decoder_ht = nn.Linear(hidden_size * self.num_directions, self.dec_hidden_size)
        self.encoder_lstm2decoder_ct = nn.Linear(hidden_size * self.num_directions, self.dec_hidden_size)



    def init_state(self, batch_size):
        ''' Initialize to zero cell states and hidden states.'''
        h0 = Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size ), requires_grad=False)
        c0 = Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size ), requires_grad=False)
        return h0, c0

    def flip_text_bert_params(self, bool_input):
        result = list()
        for name, param in self.bert.named_parameters():
            if name in self.text_bert_params:
                if param.requires_grad != bool_input:
                    param.requires_grad = bool_input
                    result.append(param)
        return result


    def forward(self, inputs, mask, lengths,f_t_all=None):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        batch_size = inputs.size(0)
        seq_max_len = mask.size(1)
        att_mask = ~mask

        #all_encoder_layers, pooled_output = self.bert(inputs[:, :seq_max_len], token_type_ids=None,
        #                                  attention_mask=att_mask)
        #text_embeds = torch.cat(all_encoder_layers[-self.bert_n_layers:],-1)
        outputs = self.bert(inputs[:, :seq_max_len], token_type_ids=None,
                                          attention_mask=att_mask, img_feats=f_t_all)
        embeds, pooled_output = outputs[0], outputs[1]

        #if not self.config.update_lang_bert:
        #    text_embeds = text_embeds.detach()  # batch_size x input_len x bert_hidden_size*n


        # Now we have pure text embeddings. Then need to concat vision emed and extend mask for image features
        if not self.config.update_add_layer:
            embeds = embeds.detach()

        if self.reverse_input:
            reversed_embeds = torch.zeros(embeds.size()).to(embeds.device)
            reverse_idx = torch.arange(seq_max_len-1,-1,-1)
            reversed_embeds[att_mask] = embeds[:, reverse_idx][att_mask[:,reverse_idx]]
            embeds = reversed_embeds


        if not self.top_lstm:
            #ctx = self.ctx2decoder(embeds)
            ctx = embeds
            c_t = self.encoder2decoder_ct(embeds[:,-1])
            decoder_init = nn.Tanh()(self.encoder2decoder_ht(pooled_output))
        else:
            h0, c0 = self.init_state(batch_size)
            packed_embeds = pack_padded_sequence(embeds, list(lengths), batch_first=True)
            enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0.to(embeds.device), c0.to(embeds.device)))

            if self.num_directions == 2:
                h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
                c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
            else:
                h_t = enc_h_t[-1]
                c_t = enc_c_t[-1] # (batch, hidden_size)

            decoder_init = nn.Tanh()(self.encoder_lstm2decoder_ht(h_t))
            if self.hidden_size * self.num_directions != self.dec_hidden_size:
                c_t = self.encoder_lstm2decoder_ct(c_t)

            ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)

        if not self.pretrain:
            ctx = self.drop(ctx)
        return ctx,decoder_init,c_t, mask  # (batch, seq_len, hidden_size*num_directions)


class DicEncoder(nn.Module):

    #transformer_hidden_size = 768
    lstm_num_layers = 1

    def __init__(self, vision_size, hidden_size, dec_hidden_size, dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, vl_layers, la_layers, bert_type="small",update_add_layer=True):
        super(DicEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.dropout_ratio = dropout_ratio
        self.drop = nn.Dropout(p=dropout_ratio)
        self.update = update
        self.bert_n_layers = bert_n_layers
        self.reverse_input = reverse_input
        self.top_lstm = top_lstm
        self.bert_type = bert_type
        self.vl_layers = vl_layers
        self.la_layers = la_layers

        if self.bert_type == "large":
            self.config = BertConfig.from_pretrained('bert-large-uncased')
            self.config.img_feature_dim = vision_size
            self.config.img_feature_type = ""
            self.config.update_lang_bert = update
            self.config.update_add_layer = update_add_layer
            self.config.vl_layers = vl_layers
            self.config.la_layers = la_layers
            self.bert = DicModel(self.config)
            self.transformer_hidden_size = 1024
        else:
            self.config = BertConfig.from_pretrained('bert-base-uncased')
            self.config.img_feature_dim = vision_size
            self.config.img_feature_type = ""
            self.config.update_lang_bert = update
            self.config.update_add_layer = update_add_layer
            self.config.vl_layers = vl_layers
            self.config.la_layers = la_layers
            self.bert = DicModel(self.config)
            self.transformer_hidden_size = 768

        self.lstm = nn.LSTM(self.transformer_hidden_size*self.bert_n_layers, self.hidden_size, self.lstm_num_layers, batch_first=True,
                             dropout=dropout_ratio if self.lstm_num_layers>1 else 0, bidirectional=bidirectional)
        #
        self.num_directions = 2 if bidirectional else 1

        self.linear_n_in = self.transformer_hidden_size * self.bert_n_layers if not self.top_lstm else hidden_size * self.num_directions
        self.encoder2decoder_ht = nn.Linear(self.linear_n_in, self.dec_hidden_size)
        self.encoder2decoder_ct = nn.Linear(self.linear_n_in, self.dec_hidden_size)

        self.encoder_lstm2decoder_ht = nn.Linear(hidden_size * self.num_directions, self.dec_hidden_size)
        self.encoder_lstm2decoder_ct = nn.Linear(hidden_size * self.num_directions, self.dec_hidden_size)



    def init_state(self, batch_size):
        ''' Initialize to zero cell states and hidden states.'''
        h0 = Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size ), requires_grad=False)
        c0 = Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size ), requires_grad=False)
        return h0, c0

    def flip_text_bert_params(self, bool_input):
        result = list()
        for name, param in self.bert.named_parameters():
            if name in self.text_bert_params:
                if param.requires_grad != bool_input:
                    param.requires_grad = bool_input
                    result.append(param)
        return result


    def forward(self, inputs, mask, lengths,f_t_all=None):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        batch_size = inputs.size(0)
        seq_max_len = mask.size(1)
        att_mask = ~mask

        #all_encoder_layers, pooled_output = self.bert(inputs[:, :seq_max_len], token_type_ids=None,
        #                                  attention_mask=att_mask)
        #text_embeds = torch.cat(all_encoder_layers[-self.bert_n_layers:],-1)
        outputs = self.bert(inputs[:, :seq_max_len], token_type_ids=None,
                                          attention_mask=att_mask, img_feats=f_t_all)
        embeds, pooled_output = outputs[0], outputs[1]

        #if not self.config.update_lang_bert:
        #    text_embeds = text_embeds.detach()  # batch_size x input_len x bert_hidden_size*n


        # Now we have pure text embeddings. Then need to concat vision emed and extend mask for image features
        if not self.config.update_add_layer:
            embeds = embeds.detach()

        if self.reverse_input:
            reversed_embeds = torch.zeros(embeds.size()).to(embeds.device)
            reverse_idx = torch.arange(seq_max_len-1,-1,-1)
            reversed_embeds[att_mask] = embeds[:, reverse_idx][att_mask[:,reverse_idx]]
            embeds = reversed_embeds


        if not self.top_lstm:
            #ctx = self.ctx2decoder(embeds)
            ctx = embeds
            c_t = self.encoder2decoder_ct(embeds[:,-1])
            decoder_init = nn.Tanh()(self.encoder2decoder_ht(pooled_output))
        else:
            h0, c0 = self.init_state(batch_size)
            packed_embeds = pack_padded_sequence(embeds, list(lengths), batch_first=True)
            enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0.to(embeds.device), c0.to(embeds.device)))

            if self.num_directions == 2:
                h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
                c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
            else:
                h_t = enc_h_t[-1]
                c_t = enc_c_t[-1] # (batch, hidden_size)

            decoder_init = nn.Tanh()(self.encoder_lstm2decoder_ht(h_t))
            if self.hidden_size * self.num_directions != self.dec_hidden_size:
                c_t = self.encoder_lstm2decoder_ct(c_t)

            ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)

        ctx = self.drop(ctx)
        return ctx,decoder_init,c_t, mask  # (batch, seq_len, hidden_size*num_directions)


class MultiVilBertEncoder(BertImgEncoder):

    def __init__(self, vision_size,hidden_size, dec_hidden_size, dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, multi_share, n_sentences, bert_type="small"):
        super(MultiVilBertEncoder, self).__init__(vision_size, hidden_size, dec_hidden_size, dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, bert_type)

        self.n_sentences = n_sentences
        self.multi_share = multi_share

        for i in range(max(self.n_sentences - 1, 2)):
            setattr(self, 'bert' + str(i + 1), None)
            setattr(self, 'lstm' + str(i + 1), None)
        if not multi_share:
            raise NotImplementedError()

    def set_n_sentences(self, n_sentences):
        self.n_sentences = n_sentences

    def forward(self, inputs, masks, lengths, f_t_all=None):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        """
        suppose f_t is the vision feature and f_t_all is the vision + action feature
        """
        ctxs, decoder_inits, c_ts, maskss = [], None, None, []

        lstms = [getattr(self, 'lstm' + (str(i) if i != 0 else '')) for i in
                 range(self.n_sentences)]  # lstms = [self.lstm, self.lstm2, self.lstm3]
        berts = [getattr(self, 'bert' + (str(i) if i != 0 else '')) for i in
                 range(self.n_sentences)]  # lstms = [self.lstm, self.lstm2, self.lstm3]

        if not self.multi_share:
            raise NotImplementedError()

        for si, (input, mask, (length, perm_idx), lstmN,bertN) in enumerate(zip(inputs, masks, lengths, lstms,berts)):
            batch_size = input.size(0)
            seq_max_len = mask.size(1)
            att_mask = ~mask

            # need to extend mask for image features
            if f_t_all is not None:
                img_seq_len = f_t_all.shape[1]
                img_seq_mask = torch.ones(batch_size, img_seq_len, dtype=att_mask.dtype).to(device)
                mask = torch.cat((~img_seq_mask, mask),dim=1)
                length = list(map(lambda x: x + img_seq_len, length))
                att_mask = torch.cat((img_seq_mask, att_mask), dim=1)

            if self.multi_share:
                last_encoder_layers, pooled_output = self.bert(input[:, :seq_max_len], token_type_ids=None,
                                                              attention_mask=att_mask.to(device), img_feats=f_t_all)
                #all_encoder_layers, pooled_output = Super(MultiVilBertEncoder,self).forward(input[:, :seq_max_len], token_type_ids=None,
                #                                              attention_mask=att_mask.to(device),img_feats=f_t_all)
                #embeds = torch.cat(all_encoder_layers[-self.bert_n_layers:], -1)
                embeds = last_encoder_layers
            #else:
            #    raise

            if self.reverse_input:
                reversed_embeds = torch.zeros(embeds.size()).to(device)
                if f_t_all is not None:
                    img_seq_len = f_t_all.shape[1]
                    reverse_idx = torch.arange(img_seq_len + seq_max_len-1,-1,-1)
                else:
                    reverse_idx = torch.arange(seq_max_len-1,-1,-1)
                reversed_embeds[att_mask] = embeds[:, reverse_idx][att_mask[:,reverse_idx]]
                embeds = reversed_embeds

            if not self.update:
                embeds = embeds.detach()  # batch_size x input_len x bert_hidden_size*n

            if not self.top_lstm:
                ctx = embeds
                c_t = self.encoder2decoder_ct(ctx[:,-1])
                decoder_init = nn.Tanh()(self.encoder2decoder_ht(pooled_output))
            else:
                h0, c0 = self.init_state(batch_size)
                packed_embeds = pack_padded_sequence(embeds, length, batch_first=True)
                enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

                if self.num_directions == 2:
                    h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
                    c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
                else:
                    h_t = enc_h_t[-1]
                    c_t = enc_c_t[-1] # (batch, hidden_size)

                decoder_init = nn.Tanh()(self.encoder2decoder_ht(h_t))
                if self.hidden_size * self.num_directions != self.dec_hidden_size:
                    c_t = self.encoder2decoder_ct(c_t)

                ctx, length = pad_packed_sequence(enc_h, batch_first=True)
            ctx = self.drop(ctx)

            perm_idx = list(perm_idx.numpy())
            unperm_idx = [perm_idx.index(i) for i, _ in enumerate(perm_idx)]
            ctx = ctx[unperm_idx]
            mask = mask[unperm_idx]
            decoder_init = decoder_init[unperm_idx]
            c_t = c_t[unperm_idx]

            ctxs.append(ctx)
            maskss.append(mask)

            if decoder_inits is None:
                decoder_inits = decoder_init
                c_ts = c_t
            else:
                decoder_inits = decoder_inits + decoder_init
                c_ts = c_ts + c_t

            # debug: no divide
        decoder_inits = decoder_inits / self.n_sentences
        c_ts = c_ts / self.n_sentences
        return ctxs, decoder_inits, c_ts, maskss  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)


class MultiHugAddEncoder(HugAddEncoder):

    def __init__(self, vision_size,hidden_size, dec_hidden_size, dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, multi_share, n_sentences, vl_layers,bert_type="small"):
        super(MultiHugAddEncoder, self).__init__(vision_size, hidden_size, dec_hidden_size, dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, vl_layers,bert_type)

        self.n_sentences = n_sentences
        self.multi_share = multi_share

        for i in range(max(self.n_sentences - 1, 2)):
            setattr(self, 'bert' + str(i + 1), None)
            setattr(self, 'lstm' + str(i + 1), None)
        if not multi_share:
            raise NotImplementedError()

    def set_n_sentences(self, n_sentences):
        self.n_sentences = n_sentences

    def forward(self, inputs, masks, lengths, f_t_all=None):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        """
        suppose f_t is the vision feature and f_t_all is the vision + action feature
        """
        ctxs, decoder_inits, c_ts, maskss = [], None, None, []

        lstms = [getattr(self, 'lstm' + (str(i) if i != 0 else '')) for i in
                 range(self.n_sentences)]  # lstms = [self.lstm, self.lstm2, self.lstm3]
        berts = [getattr(self, 'bert' + (str(i) if i != 0 else '')) for i in
                 range(self.n_sentences)]  # lstms = [self.lstm, self.lstm2, self.lstm3]

        if not self.multi_share:
            raise NotImplementedError()

        for si, (input, mask, (length, perm_idx), lstmN,bertN) in enumerate(zip(inputs, masks, lengths, lstms,berts)):
            batch_size = input.size(0)
            seq_max_len = mask.size(1)
            att_mask = ~mask

            # need to extend mask for image features
            if f_t_all is not None:
                img_seq_len = f_t_all.shape[1]
                img_seq_mask = torch.ones(batch_size, img_seq_len, dtype=att_mask.dtype).to(device)
                mask = torch.cat((~img_seq_mask, mask),dim=1)
                #length = list(map(lambda x: x + img_seq_len, length))
                length = length + img_seq_len
            #    att_mask = torch.cat((img_seq_mask, att_mask), dim=1)

            if self.multi_share:
                last_encoder_layers, pooled_output = self.bert(input[:, :seq_max_len], token_type_ids=None,
                                                              attention_mask=att_mask.to(device), img_feats=f_t_all)
                #embeds = torch.cat(all_encoder_layers[-self.bert_n_layers:], -1)
                embeds = last_encoder_layers

            if self.reverse_input:
                reversed_embeds = torch.zeros(embeds.size()).to(device)
                if f_t_all is not None:
                    img_seq_len = f_t_all.shape[1]
                    reverse_idx = torch.arange(img_seq_len + seq_max_len-1,-1,-1)
                    idmask = torch.cat((img_seq_mask, att_mask), dim=1)
                    reversed_embeds[idmask] = embeds[:, reverse_idx][idmask[:,reverse_idx]]
                else:
                    reverse_idx = torch.arange(seq_max_len-1,-1,-1)
                    reversed_embeds[att_mask] = embeds[:, reverse_idx][att_mask[:,reverse_idx]]
                embeds = reversed_embeds

            if not self.update:
                embeds = embeds.detach()  # batch_size x input_len x bert_hidden_size*n

            if not self.top_lstm:
                ctx = embeds
                c_t = self.encoder2decoder_ct(ctx[:,-1])
                decoder_init = nn.Tanh()(self.encoder2decoder_ht(pooled_output))
            else:
                h0, c0 = self.init_state(batch_size)
                packed_embeds = pack_padded_sequence(embeds, list(length), batch_first=True)
                enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0.to(embeds.device), c0.to(embeds.device)))

                if self.num_directions == 2:
                    h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
                    c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
                else:
                    h_t = enc_h_t[-1]
                    c_t = enc_c_t[-1] # (batch, hidden_size)

                decoder_init = nn.Tanh()(self.encoder2decoder_ht(h_t))
                if self.hidden_size * self.num_directions != self.dec_hidden_size:
                    c_t = self.encoder2decoder_ct(c_t)

                ctx, length = pad_packed_sequence(enc_h, batch_first=True)
            ctx = self.drop(ctx)

            perm_idx = list(perm_idx.numpy())
            unperm_idx = [perm_idx.index(i) for i, _ in enumerate(perm_idx)]
            ctx = ctx[unperm_idx]
            mask = mask[unperm_idx]
            decoder_init = decoder_init[unperm_idx]
            c_t = c_t[unperm_idx]

            ctxs.append(ctx)
            maskss.append(mask)

            if decoder_inits is None:
                decoder_inits = decoder_init
                c_ts = c_t
            else:
                decoder_inits = decoder_inits + decoder_init
                c_ts = c_ts + c_t

            # debug: no divide
        decoder_inits = decoder_inits / self.n_sentences
        c_ts = c_ts / self.n_sentences
        return ctxs, decoder_inits, c_ts, maskss  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)


class MultiVicEncoder(VicEncoder):

    def __init__(self, vision_size,hidden_size, dec_hidden_size, dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, multi_share, n_sentences, vl_layers,bert_type="small"):
        super(MultiVicEncoder, self).__init__(vision_size, hidden_size, dec_hidden_size, dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, vl_layers,bert_type)

        self.n_sentences = n_sentences
        self.multi_share = multi_share

        for i in range(max(self.n_sentences - 1, 2)):
            setattr(self, 'bert' + str(i + 1), None)
            setattr(self, 'lstm' + str(i + 1), None)
        if not multi_share:
            raise NotImplementedError()

    def set_n_sentences(self, n_sentences):
        self.n_sentences = n_sentences

    def forward(self, inputs, masks, lengths, f_t_all=None):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        """
        suppose f_t is the vision feature and f_t_all is the vision + action feature
        """
        ctxs, decoder_inits, c_ts, maskss = [], None, None, []

        lstms = [getattr(self, 'lstm' + (str(i) if i != 0 else '')) for i in
                 range(self.n_sentences)]  # lstms = [self.lstm, self.lstm2, self.lstm3]
        berts = [getattr(self, 'bert' + (str(i) if i != 0 else '')) for i in
                 range(self.n_sentences)]  # lstms = [self.lstm, self.lstm2, self.lstm3]

        if not self.multi_share:
            raise NotImplementedError()

        for si, (input, mask, (length, perm_idx), lstmN,bertN) in enumerate(zip(inputs, masks, lengths, lstms,berts)):
            batch_size = input.size(0)
            seq_max_len = mask.size(1)
            att_mask = ~mask

            # need to extend mask for image features
            """
            if f_t_all is not None:
                img_seq_len = f_t_all.shape[1]
                img_seq_mask = torch.ones(batch_size, img_seq_len, dtype=att_mask.dtype).to(device)
                mask = torch.cat((~img_seq_mask, mask),dim=1)
                #length = list(map(lambda x: x + img_seq_len, length))
                length = length + img_seq_len
            #    att_mask = torch.cat((img_seq_mask, att_mask), dim=1)
            """

            if self.multi_share:
                last_encoder_layers, pooled_output = self.bert(input[:, :seq_max_len], token_type_ids=None,
                                                              attention_mask=att_mask.to(device), img_feats=f_t_all)
                #embeds = torch.cat(all_encoder_layers[-self.bert_n_layers:], -1)
                embeds = last_encoder_layers

            if self.reverse_input:
                reversed_embeds = torch.zeros(embeds.size()).to(device)
                reverse_idx = torch.arange(seq_max_len-1,-1,-1)
                reversed_embeds[att_mask] = embeds[:, reverse_idx][att_mask[:,reverse_idx]]
                embeds = reversed_embeds

            if not self.update:
                embeds = embeds.detach()  # batch_size x input_len x bert_hidden_size*n

            if not self.top_lstm:
                ctx = embeds
                c_t = self.encoder2decoder_ct(ctx[:,-1])
                decoder_init = nn.Tanh()(self.encoder2decoder_ht(pooled_output))
            else:
                h0, c0 = self.init_state(batch_size)
                packed_embeds = pack_padded_sequence(embeds, list(length), batch_first=True)
                enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0.to(embeds.device), c0.to(embeds.device)))

                if self.num_directions == 2:
                    h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
                    c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
                else:
                    h_t = enc_h_t[-1]
                    c_t = enc_c_t[-1] # (batch, hidden_size)

                decoder_init = nn.Tanh()(self.encoder2decoder_ht(h_t))
                if self.hidden_size * self.num_directions != self.dec_hidden_size:
                    c_t = self.encoder2decoder_ct(c_t)

                ctx, length = pad_packed_sequence(enc_h, batch_first=True)
            ctx = self.drop(ctx)

            perm_idx = list(perm_idx.numpy())
            unperm_idx = [perm_idx.index(i) for i, _ in enumerate(perm_idx)]
            ctx = ctx[unperm_idx]
            mask = mask[unperm_idx]
            decoder_init = decoder_init[unperm_idx]
            c_t = c_t[unperm_idx]

            ctxs.append(ctx)
            maskss.append(mask)

            if decoder_inits is None:
                decoder_inits = decoder_init
                c_ts = c_t
            else:
                decoder_inits = decoder_inits + decoder_init
                c_ts = c_ts + c_t

            # debug: no divide
        decoder_inits = decoder_inits / self.n_sentences
        c_ts = c_ts / self.n_sentences
        return ctxs, decoder_inits, c_ts, maskss  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)

class MultiDicEncoder(DicEncoder):

    def __init__(self, vision_size,hidden_size, dec_hidden_size, dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, multi_share, n_sentences, vl_layers,la_layers,bert_type="small"):
        super(MultiDicEncoder, self).__init__(vision_size, hidden_size, dec_hidden_size, dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, vl_layers,la_layers,bert_type)

        self.n_sentences = n_sentences
        self.multi_share = multi_share

        for i in range(max(self.n_sentences - 1, 2)):
            setattr(self, 'bert' + str(i + 1), None)
            setattr(self, 'lstm' + str(i + 1), None)
        if not multi_share:
            raise NotImplementedError()

    def set_n_sentences(self, n_sentences):
        self.n_sentences = n_sentences

    def forward(self, inputs, masks, lengths, f_t_all=None):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        """
        suppose f_t is the vision feature and f_t_all is the vision + action feature
        """
        ctxs, decoder_inits, c_ts, maskss = [], None, None, []

        lstms = [getattr(self, 'lstm' + (str(i) if i != 0 else '')) for i in
                 range(self.n_sentences)]  # lstms = [self.lstm, self.lstm2, self.lstm3]
        berts = [getattr(self, 'bert' + (str(i) if i != 0 else '')) for i in
                 range(self.n_sentences)]  # lstms = [self.lstm, self.lstm2, self.lstm3]

        if not self.multi_share:
            raise NotImplementedError()

        for si, (input, mask, (length, perm_idx), lstmN,bertN) in enumerate(zip(inputs, masks, lengths, lstms,berts)):
            batch_size = input.size(0)
            seq_max_len = mask.size(1)
            att_mask = ~mask

            # need to extend mask for image features
            """
            if f_t_all is not None:
                img_seq_len = f_t_all.shape[1]
                img_seq_mask = torch.ones(batch_size, img_seq_len, dtype=att_mask.dtype).to(device)
                mask = torch.cat((~img_seq_mask, mask),dim=1)
                #length = list(map(lambda x: x + img_seq_len, length))
                length = length + img_seq_len
            #    att_mask = torch.cat((img_seq_mask, att_mask), dim=1)
            """

            if self.multi_share:
                last_encoder_layers, pooled_output = self.bert(input[:, :seq_max_len], token_type_ids=None,
                                                              attention_mask=att_mask.to(device), img_feats=f_t_all)
                #embeds = torch.cat(all_encoder_layers[-self.bert_n_layers:], -1)
                embeds = last_encoder_layers

            if self.reverse_input:
                reversed_embeds = torch.zeros(embeds.size()).to(device)
                reverse_idx = torch.arange(seq_max_len-1,-1,-1)
                reversed_embeds[att_mask] = embeds[:, reverse_idx][att_mask[:,reverse_idx]]
                embeds = reversed_embeds

            if not self.update:
                embeds = embeds.detach()  # batch_size x input_len x bert_hidden_size*n

            if not self.top_lstm:
                ctx = embeds
                c_t = self.encoder2decoder_ct(ctx[:,-1])
                decoder_init = nn.Tanh()(self.encoder2decoder_ht(pooled_output))
            else:
                h0, c0 = self.init_state(batch_size)
                packed_embeds = pack_padded_sequence(embeds, list(length), batch_first=True)
                enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0.to(embeds.device), c0.to(embeds.device)))

                if self.num_directions == 2:
                    h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
                    c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
                else:
                    h_t = enc_h_t[-1]
                    c_t = enc_c_t[-1] # (batch, hidden_size)

                decoder_init = nn.Tanh()(self.encoder2decoder_ht(h_t))
                if self.hidden_size * self.num_directions != self.dec_hidden_size:
                    c_t = self.encoder2decoder_ct(c_t)

                ctx, length = pad_packed_sequence(enc_h, batch_first=True)
            ctx = self.drop(ctx)

            perm_idx = list(perm_idx.numpy())
            unperm_idx = [perm_idx.index(i) for i, _ in enumerate(perm_idx)]
            ctx = ctx[unperm_idx]
            mask = mask[unperm_idx]
            decoder_init = decoder_init[unperm_idx]
            c_t = c_t[unperm_idx]

            ctxs.append(ctx)
            maskss.append(mask)

            if decoder_inits is None:
                decoder_inits = decoder_init
                c_ts = c_t
            else:
                decoder_inits = decoder_inits + decoder_init
                c_ts = c_ts + c_t

            # debug: no divide
        decoder_inits = decoder_inits / self.n_sentences
        c_ts = c_ts / self.n_sentences
        return ctxs, decoder_inits, c_ts, maskss  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)




class MultiVilAddEncoder(BertAddEncoder):

    def __init__(self, vision_size,hidden_size, dec_hidden_size, dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, multi_share, n_sentences, vl_layers, bert_type="small"):
        super(MultiVilAddEncoder, self).__init__(vision_size, hidden_size, dec_hidden_size,dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, vl_layers, bert_type)

        self.n_sentences = n_sentences
        self.multi_share = multi_share

        for i in range(max(self.n_sentences - 1, 2)):
            setattr(self, 'bert' + str(i + 1), None)
            setattr(self, 'lstm' + str(i + 1), None)
        if not multi_share:
            raise NotImplementedError()

    def set_n_sentences(self, n_sentences):
        self.n_sentences = n_sentences

    def forward(self, inputs, masks, lengths, f_t_all=None):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        """
        suppose f_t is the vision feature and f_t_all is the vision + action feature
        """
        ctxs, decoder_inits, c_ts, maskss = [], None, None, []

        lstms = [getattr(self, 'lstm' + (str(i) if i != 0 else '')) for i in
                 range(self.n_sentences)]  # lstms = [self.lstm, self.lstm2, self.lstm3]
        berts = [getattr(self, 'bert' + (str(i) if i != 0 else '')) for i in
                 range(self.n_sentences)]  # lstms = [self.lstm, self.lstm2, self.lstm3]

        if not self.multi_share:
            raise NotImplementedError()

        for si, (input, mask, (length, perm_idx), lstmN,bertN) in enumerate(zip(inputs, masks, lengths, lstms,berts)):
            batch_size = input.size(0)
            seq_max_len = mask.size(1)
            att_mask = ~mask

            if self.multi_share:
                all_encoder_layers, pooled_output = self.bert(input[:, :seq_max_len], token_type_ids=None,
                                                              attention_mask=att_mask.to(mask.device))
                text_embeds = torch.cat(all_encoder_layers[-self.bert_n_layers:], -1)

            if not self.update:
                text_embeds = text_embeds.detach()  # batch_size x input_len x bert_hidden_size*n


            # Now we have pure text embeddings. Then need to concat vision emed and extend mask for image features
            if f_t_all is not None:
                img_embedding_output = self.img_embedding(f_t_all)
                img_seq_len = f_t_all.shape[1]
                img_seq_mask = torch.ones(batch_size, img_seq_len, dtype=att_mask.dtype).to(att_mask.device)
                mask = torch.cat((~img_seq_mask, mask),dim=1)
                #length = list(map(lambda x: x + img_seq_len, length))
                length = length + img_seq_len
                att_mask = torch.cat((img_seq_mask, att_mask), dim=1)
                input_embeds = torch.cat((img_embedding_output, text_embeds), 1)
                outputs = self.addlayer(input_embeds, att_mask)
                embeds = outputs[0]
            else:
                embeds = text_embeds

            if self.reverse_input:
                reversed_embeds = torch.zeros(embeds.size()).to(embeds.device)
                if f_t_all is not None:
                    img_seq_len = f_t_all.shape[1]
                    reverse_idx = torch.arange(img_seq_len + seq_max_len-1,-1,-1)
                else:
                    reverse_idx = torch.arange(seq_max_len-1,-1,-1)
                reversed_embeds[att_mask] = embeds[:, reverse_idx][att_mask[:,reverse_idx]]
                embeds = reversed_embeds

            if not self.top_lstm:
                ctx = embeds
                c_t = self.encoder2decoder_ct(ctx[:,-1])
                decoder_init = nn.Tanh()(self.encoder2decoder_ht(pooled_output))
            else:
                h0, c0 = self.init_state(batch_size)
                packed_embeds = pack_padded_sequence(embeds, list(length), batch_first=True)
                enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0.to(embeds.device), c0.to(embeds.device)))

                if self.num_directions == 2:
                    h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
                    c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
                else:
                    h_t = enc_h_t[-1]
                    c_t = enc_c_t[-1] # (batch, hidden_size)

                decoder_init = nn.Tanh()(self.encoder_lstm2decoder_ht(h_t))
                if self.hidden_size * self.num_directions != self.dec_hidden_size:
                    c_t = self.encoder_lstm2decoder_ct(c_t)

                ctx, length = pad_packed_sequence(enc_h, batch_first=True)
            ctx = self.drop(ctx)

            perm_idx = list(perm_idx.cpu().numpy())
            unperm_idx = [perm_idx.index(i) for i, _ in enumerate(perm_idx)]
            ctx = ctx[unperm_idx]
            mask = mask[unperm_idx]
            decoder_init = decoder_init[unperm_idx]
            c_t = c_t[unperm_idx]

            ctxs.append(ctx)
            maskss.append(mask)

            if decoder_inits is None:
                decoder_inits = decoder_init
                c_ts = c_t
            else:
                decoder_inits = decoder_inits + decoder_init
                c_ts = c_ts + c_t

            # debug: no divide
        decoder_inits = decoder_inits / self.n_sentences
        c_ts = c_ts / self.n_sentences
        return ctxs, decoder_inits, c_ts, maskss  # (batch, seq_len, hidden_size*num_directions)


class MultiAddLoadEncoder(BertAddEncoder):

    def __init__(self, vision_size,hidden_size, dec_hidden_size, dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, multi_share, n_sentences, vl_layers, pretrained_path = None, bert_type="small"):
        super(MultiAddLoadEncoder, self).__init__(vision_size, hidden_size, dec_hidden_size,dropout_ratio, bidirectional, update, bert_n_layers, reverse_input, top_lstm, vl_layers, bert_type)

        self.n_sentences = n_sentences
        self.multi_share = multi_share
        self.pretrained_path = pretrained_path
        if pretrained_path is not None: # Note that the forward implmentation is not the same
            print("Using the pretrained lm model from %s" %(pretrained_path))
            self.pretrained_model = torch.load(pretrained_path)
            self.pretrained_model.dropout_ratio = dropout_ratio
            self.pretrained_model.drop = nn.Dropout(p=dropout_ratio)
            self.pretrained_model.update = update
            #self.pretrained_model.reverse_input = reverse_input
            #self.pretrained_model.top_lstm = top_lstm
            self.pretrained_model.pretrain = False
            self.vision_encoder = VisionEncoder(vision_size, self.pretrained_model.config)
            self.addlayer = AddEncoder(self.pretrained_model.config)
            self.pooler = BertPooler(self.pretrained_model.config)

        for i in range(max(self.n_sentences - 1, 2)):
            setattr(self, 'bert' + str(i + 1), None)
            setattr(self, 'lstm' + str(i + 1), None)
        if not multi_share:
            raise NotImplementedError()

    def set_n_sentences(self, n_sentences):
        self.n_sentences = n_sentences

    def forward(self, inputs, masks, lengths, f_t_all=None):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        """
        suppose f_t is the vision feature and f_t_all is the vision + action feature
        """
        ctxs, decoder_inits, c_ts, maskss = [], None, None, []

        lstms = [getattr(self, 'lstm' + (str(i) if i != 0 else '')) for i in
                 range(self.n_sentences)]  # lstms = [self.lstm, self.lstm2, self.lstm3]
        berts = [getattr(self, 'bert' + (str(i) if i != 0 else '')) for i in
                 range(self.n_sentences)]  # lstms = [self.lstm, self.lstm2, self.lstm3]

        if not self.multi_share:
            raise NotImplementedError()

        if self.pretrained_path is None:
            for si, (input, mask, (length, perm_idx), lstmN,bertN) in enumerate(zip(inputs, masks, lengths, lstms,berts)):
                batch_size = input.size(0)
                seq_max_len = mask.size(1)
                att_mask = ~mask

                if self.multi_share:
                    all_encoder_layers, pooled_output = self.bert(input[:, :seq_max_len], token_type_ids=None,
                                                                  attention_mask=att_mask.to(mask.device))
                    text_embeds = torch.cat(all_encoder_layers[-self.bert_n_layers:], -1)

                if not self.update:
                    text_embeds = text_embeds.detach()  # batch_size x input_len x bert_hidden_size*n


                # Now we have pure text embeddings. Then need to concat vision emed and extend mask for image features
                if f_t_all is not None:
                    img_embedding_output = self.img_embedding(f_t_all)
                    img_seq_len = f_t_all.shape[1]
                    img_seq_mask = torch.ones(batch_size, img_seq_len, dtype=att_mask.dtype).to(att_mask.device)
                    mask = torch.cat((~img_seq_mask, mask),dim=1)
                    #length = list(map(lambda x: x + img_seq_len, length))
                    length = length + img_seq_len
                    att_mask = torch.cat((img_seq_mask, att_mask), dim=1)
                    input_embeds = torch.cat((img_embedding_output, text_embeds), 1)
                    outputs = self.addlayer(input_embeds, att_mask)
                    embeds = outputs[0]
                else:
                    embeds = text_embeds

                if self.reverse_input:
                    reversed_embeds = torch.zeros(embeds.size()).to(embeds.device)
                    if f_t_all is not None:
                        img_seq_len = f_t_all.shape[1]
                        reverse_idx = torch.arange(img_seq_len + seq_max_len-1,-1,-1)
                    else:
                        reverse_idx = torch.arange(seq_max_len-1,-1,-1)
                    reversed_embeds[att_mask] = embeds[:, reverse_idx][att_mask[:,reverse_idx]]
                    embeds = reversed_embeds

                if not self.top_lstm:
                    ctx = embeds
                    c_t = self.encoder2decoder_ct(ctx[:,-1])
                    decoder_init = nn.Tanh()(self.encoder2decoder_ht(pooled_output))
                else:
                    h0, c0 = self.init_state(batch_size)
                    packed_embeds = pack_padded_sequence(embeds, list(length), batch_first=True)
                    enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

                    if self.num_directions == 2:
                        h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
                        c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
                    else:
                        h_t = enc_h_t[-1]
                        c_t = enc_c_t[-1] # (batch, hidden_size)

                    decoder_init = nn.Tanh()(self.encoder_lstm2decoder_ht(h_t))
                    if self.hidden_size * self.num_directions != self.dec_hidden_size:
                        c_t = self.encoder_lstm2decoder_ct(c_t)

                    ctx, length = pad_packed_sequence(enc_h, batch_first=True)
                ctx = self.drop(ctx)

                perm_idx = list(perm_idx.cpu().numpy())
                unperm_idx = [perm_idx.index(i) for i, _ in enumerate(perm_idx)]
                ctx = ctx[unperm_idx]
                mask = mask[unperm_idx]
                decoder_init = decoder_init[unperm_idx]
                c_t = c_t[unperm_idx]

                ctxs.append(ctx)
                maskss.append(mask)

                if decoder_inits is None:
                    decoder_inits = decoder_init
                    c_ts = c_t
                else:
                    decoder_inits = decoder_inits + decoder_init
                    c_ts = c_ts + c_t

                # debug: no divide
            decoder_inits = decoder_inits / self.n_sentences
            c_ts = c_ts / self.n_sentences
            return ctxs, decoder_inits, c_ts, maskss  # (batch, seq_len, hidden_size*num_directions)
        else:
            #for si, (input, mask, (length, perm_idx), lstmN,bertN) in enumerate(zip(inputs, masks, lengths, lstms,berts)):
            #    ctx, en_ht, en_ct, vl_mask = self.pretrained_model(input, mask, length, f_t_all=f_t_all)

            #    perm_idx = list(perm_idx.cpu().numpy())
            #    unperm_idx = [perm_idx.index(i) for i, _ in enumerate(perm_idx)]
            #    ctx = ctx[unperm_idx]
            #    mask = vl_mask[unperm_idx]
            #    decoder_init = en_ht[unperm_idx]
            #    c_t = en_ct[unperm_idx]

            #    ctxs.append(ctx)
            #    maskss.append(mask)

            #    if decoder_inits is None:
            #        decoder_inits = decoder_init
            #        c_ts = c_t
            #    else:
            #        decoder_inits = decoder_inits + decoder_init
            #        c_ts = c_ts + c_t

            #    # debug: no divide
            #decoder_inits = decoder_inits / self.n_sentences
            #c_ts = c_ts / self.n_sentences
            #return ctxs, decoder_inits, c_ts, maskss  # (batch, seq_len, hidden_size*num_directions)
            for si, (input, mask, (length, perm_idx), lstmN,bertN) in enumerate(zip(inputs, masks, lengths, lstms,berts)):
                seq_max_len = mask.size(1)
                ctx, en_ht, en_ct, vl_mask = self.pretrained_model(input, mask, length, f_t_all=f_t_all)
                img_seq_len = f_t_all.shape[1]
                text_embeds = ctx[:,img_seq_len:,:]
                text_embeds = text_embeds.detach()

                img_embedding_output = self.vision_encoder(f_t_all)
                input_embeds = torch.cat((img_embedding_output, text_embeds), 1)
                outputs = self.addlayer(input_embeds, vl_mask)
                embeds = outputs[0]

                if self.reverse_input:
                    reversed_embeds = torch.zeros(embeds.size()).to(embeds.device)
                    reverse_idx = torch.arange(img_seq_len + seq_max_len-1,-1,-1)
                    reversed_embeds[vl_mask] = embeds[:, reverse_idx][vl_mask[:,reverse_idx]]
                    embeds = reversed_embeds

                if not self.top_lstm:
                    ctx = embeds
                    c_t = self.encoder2decoder_ct(ctx[:,-1])
                    decoder_init = nn.Tanh()(self.encoder2decoder_ht(pooled_output))
                else:
                    h0, c0 = self.init_state(batch_size)
                    packed_embeds = pack_padded_sequence(embeds, list(length), batch_first=True)
                    enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

                    if self.num_directions == 2:
                        h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
                        c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
                    else:
                        h_t = enc_h_t[-1]
                        c_t = enc_c_t[-1] # (batch, hidden_size)

                    decoder_init = nn.Tanh()(self.encoder_lstm2decoder_ht(h_t))
                    if self.hidden_size * self.num_directions != self.dec_hidden_size:
                        c_t = self.encoder_lstm2decoder_ct(c_t)

                    ctx, length = pad_packed_sequence(enc_h, batch_first=True)
                ctx = self.drop(ctx)


                perm_idx = list(perm_idx.cpu().numpy())
                unperm_idx = [perm_idx.index(i) for i, _ in enumerate(perm_idx)]
                ctx = ctx[unperm_idx]
                mask = vl_mask[unperm_idx]
                decoder_init = en_ht[unperm_idx]
                c_t = en_ct[unperm_idx]

                ctxs.append(ctx)
                maskss.append(mask)

                if decoder_inits is None:
                    decoder_inits = decoder_init
                    c_ts = c_t
                else:
                    decoder_inits = decoder_inits + decoder_init
                    c_ts = c_ts + c_t

                # debug: no divide
            decoder_inits = decoder_inits / self.n_sentences
            c_ts = c_ts / self.n_sentences
            return ctxs, decoder_inits, c_ts, maskss  # (batch, seq_len, hidden_size*num_directions)





class EncoderLSTM(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''

    def __init__(self, vocab_size, embedding_size, hidden_size, dec_hidden_size, padding_idx,
                            dropout_ratio, glove, bidirectional=False, num_layers=1):
        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.use_glove = glove is not None

        if self.use_glove:
            print('Using GloVe embedding')
            self.embedding.weight.data[...] = torch.from_numpy(glove)
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_size, hidden_size, self.num_layers, batch_first=True,
                            dropout=dropout_ratio if self.num_layers>1 else 0, bidirectional=bidirectional)
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions, dec_hidden_size)
        self.encoder2decoder_ct = nn.Linear(hidden_size * self.num_directions, dec_hidden_size)
        #if hidden_size * self.num_directions != dec_hidden_size:
        #    self.encoder2decoder_ct = nn.Linear(hidden_size * self.num_directions, dec_hidden_size)

    def init_state(self, batch_size):
        ''' Initialize to zero cell states and hidden states.'''
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        return h0.to(device), c0.to(device)

    def forward(self, inputs, mask, lengths):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        batch_size = inputs.size(0)
        embeds = self.embedding(inputs)   # (batch, seq_len, embedding_size)
        if not self.use_glove:
            embeds = self.drop(embeds)
        h0, c0 = self.init_state(batch_size)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

        if self.num_directions == 2:
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1] # (batch, hidden_size)

        decoder_init = nn.Tanh()(self.encoder2decoder(h_t))
        if self.hidden_size * self.num_directions != self.dec_hidden_size:
            c_t = self.encoder2decoder_ct(c_t)
        ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)
        ctx = self.drop(ctx)
        return ctx,decoder_init,c_t, mask  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)


class EncoderMultiLSTM(EncoderLSTM):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''

    def __init__(self, vocab_size, embedding_size, hidden_size, dec_hidden_size, padding_idx,
                            dropout_ratio, multi_share, n_sentences, glove, params=None, bidirectional=False, num_layers=1):
        super(EncoderMultiLSTM, self).__init__(vocab_size, embedding_size, hidden_size, dec_hidden_size, padding_idx, dropout_ratio, glove, bidirectional, num_layers)

        self.n_sentences = n_sentences
        self.multi_share = multi_share

        self.dec_h_init = params['dec_h_init']
        self.dec_c_init = params['dec_c_init']

        for i in range(max(self.n_sentences-1, 2)):
            setattr(self, 'lstm'+str(i+1), None)
        if not multi_share:
            for i in range(self.n_sentences - 1):
                setattr(self, 'lstm' + str(i + 1), nn.LSTM(embedding_size, hidden_size, self.num_layers,
                                    batch_first=True, dropout=dropout_ratio if self.num_layers > 1 else 0, bidirectional=bidirectional))

    def set_n_sentences(self, n_sentences):
        self.n_sentences = n_sentences

    def forward(self, inputs, masks, lengths):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a list of lengths for dynamic batching. '''
        ctxs, decoder_inits, c_ts, maskss = [], None, None ,[]

        lstms = [getattr(self, 'lstm'+(str(i) if i!=0 else '')) for i in range(self.n_sentences)]  #  lstms = [self.lstm, self.lstm2, self.lstm3]
        if not self.multi_share:
            temp=[i for i in range(self.n_sentences)]
            random.shuffle(temp)
            lstms = [lstms[i] for i in temp]

        for si,(input, mask, (length, perm_idx), lstmN) in enumerate(zip(inputs, masks, lengths, lstms)):
            batch_size = input.size(0)  # batch, n_sentences, seq_len
            embeds = self.embedding(input)   # (batch, seq_len, embedding_size)
            if not self.use_glove:
                embeds = self.drop(embeds)
            h0, c0 = self.init_state(batch_size)

            packed_embeds = pack_padded_sequence(embeds, length, batch_first=True)
            if self.multi_share:
                enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))
            else:
                enc_h, (enc_h_t, enc_c_t) = lstmN(packed_embeds, (h0, c0))

            if self.num_directions == 2:
                h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
                c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
            else:
                h_t = enc_h_t[-1]
                c_t = enc_c_t[-1] # (batch, hidden_size)

            # best: dec_h_init (tanh), c_t (origin)
            if self.dec_h_init == 'linear':
                decoder_init = self.encoder2decoder(h_t)
            elif self.dec_h_init == 'tanh':
                decoder_init = nn.Tanh()(self.encoder2decoder(h_t))
            else: # bidirectional has no None
                decoder_init = h_t

            # best: dec_c_init: none
            if self.dec_c_init == 'linear':
                c_t = self.encoder2decoder_ct(c_t)
            elif self.dec_c_init == 'tanh':
                c_t = nn.Tanh()(self.encoder2decoder_ct(c_t))
            else:
                if self.hidden_size * self.num_directions != self.dec_hidden_size:
                    c_t = self.encoder2decoder_ct(c_t)

            # debug
            #if self.hidden_size * self.num_directions != self.dec_hidden_size:
            #    if self.dec_c_init == 'linear':
            #        c_t = self.encoder2decoder_ct(c_t)
            #    else:
            #        c_t = nn.Tanh()(self.encoder2decoder_ct(c_t))

            ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)
            ctx = self.drop(ctx)

            perm_idx = list(perm_idx.numpy())
            unperm_idx = [perm_idx.index(i) for i, _ in enumerate(perm_idx)]
            ctx = ctx[unperm_idx]
            mask = mask[unperm_idx]
            decoder_init = decoder_init[unperm_idx]
            c_t = c_t[unperm_idx]

            ctxs.append(ctx)
            maskss.append(mask)

            if decoder_inits is None:
                decoder_inits = decoder_init
                c_ts = c_t
            else:
                decoder_inits = decoder_inits + decoder_init
                c_ts = c_ts + c_t

        # debug: no divide
        decoder_inits = decoder_inits/self.n_sentences
        c_ts = c_ts/self.n_sentences
        return ctxs, decoder_inits, c_ts, maskss  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)


class SoftDotAttention(nn.Module):
    '''Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, ctx_hidden_size, dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, ctx_hidden_size, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim + ctx_hidden_size, dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None, ctx_drop=None):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim

        if ctx_drop is not None:
            weighted_context = ctx_drop(weighted_context)

        h_tilde = torch.cat((weighted_context, h), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class EltwiseProdScoring(nn.Module):
    '''
    Linearly mapping h and v to the same dimension, and do an elementwise
    multiplication and a linear scoring
    '''

    def __init__(self, h_dim, a_dim, dot_dim=256):
        '''Initialize layer.'''
        super(EltwiseProdScoring, self).__init__()
        self.linear_in_h = nn.Linear(h_dim, dot_dim, bias=True)
        self.linear_in_a = nn.Linear(a_dim, dot_dim, bias=True)
        self.linear_out = nn.Linear(dot_dim, 1, bias=True)

    def forward(self, h, all_u_t, mask=None):
        '''Propagate h through the network.

        h: batch x h_dim
        all_u_t: batch x a_num x a_dim
        '''
        target = self.linear_in_h(h).unsqueeze(1)  # batch x 1 x dot_dim
        context = self.linear_in_a(all_u_t)  # batch x a_num x dot_dim
        eltprod = torch.mul(target, context)  # batch x a_num x dot_dim
        logits = self.linear_out(eltprod).squeeze(2)  # batch x a_num
        return logits


# Speaker-Follower
class VisualSoftDotAttention(nn.Module):
    ''' Visual Dot Attention Layer. '''

    def __init__(self, h_dim, v_dim, dot_dim=256):
        '''Initialize layer.'''
        super(VisualSoftDotAttention, self).__init__()
        self.linear_in_h = nn.Linear(h_dim, dot_dim, bias=True)
        self.linear_in_v = nn.Linear(v_dim, dot_dim, bias=True)
        self.sm = nn.Softmax(dim=1)

    def forward(self, h, visual_context, mask=None):
        '''Propagate h through the network.

        h: batch x h_dim
        visual_context: batch x v_num x v_dim 100x36x2048
        '''
        target = self.linear_in_h(h).unsqueeze(2)  # batch x dot_dim x 1
        context = self.linear_in_v(visual_context)  # batch x v_num x dot_dim

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x v_num
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x v_num

        weighted_context = torch.bmm(attn3, visual_context).squeeze(1)  # batch x v_dim
        return weighted_context, attn


class LinearFeature(nn.Module):
    '''
    Linear mapping h and v to the dimension of the image feature
    '''
    # TODO: tune this structure
    def __init__(self, h_dim, f_dim):
        '''Initialize layer.'''
        super(LinearFeature, self).__init__()
        self.linear_in_h = nn.Linear(h_dim, f_dim, bias=True)
        #self.linear_in_a = nn.Linear(a_dim, f_dim, bias=True)

    def forward(self, h, mask=None):  #, u_t
        '''Propagate h through the network.

        h: batch x h_dim
        all_u_t: batch x a_num x a_dim
        u_t: batch x a_dim
        '''
        target = self.linear_in_h(h) #.unsqueeze(1)  # batch x f_dim
        #context = self.linear_in_a(u_t)  # batch x f_dim
        #eltprod = torch.mul(target, context)  # batch x f_dim
        pred_f = F.relu(target)  # batch x f_dim
        #pred_f = F.relu(eltprod)  # batch x f_dim
        return pred_f

class NonLinearFeature(nn.Module):
    '''
    Non-Linear mapping hiddens to the dimension of the image feature
    better: h_dim > f_dim
    '''
    # TODO: tune this structure
    def __init__(self, i_dim, h_dim, f_dim):
        ''' Initialize layer. '''
        super(NonLinearFeature, self).__init__()
        self.linear_in_h = nn.Linear(i_dim, h_dim, bias=True)
        self.linear_h_o = nn.Linear(h_dim, f_dim, bias=True)

        #self.linear_in_a = nn.Linear(a_dim, f_dim, bias=True)

    def forward(self, h, mask=None):
        '''Propagate h through the network.
        h: batch x h_dim
        all_u_t: batch x a_num x a_dim
        u_t: batch x a_dim
        '''

        h_ins = self.linear_in_h(h)
        h_outs = F.relu(h_ins)
        o_ins = self.linear_h_o(h_outs)
        pred_f = F.relu(o_ins)

        return pred_f


# https://github.com/vdumoulin/conv_arithmetic
class DeconvFeature(nn.Module):
    def __init__(self, h_dim, f_dim):
        super(DeconvFeature, self).__init__()
        self.fc = nn.Linear(h_dim, 16*15*15)
        # self.setting = setting
        # if setting==1:
        #     self.deconv = nn.ConvTranspose2d(16, 4, kernel_size=4,stride=2,padding=0)
        #     self.fc2 = nn.Linear(4096, f_dim)
        # if setting==2:
        self.deconv = nn.ConvTranspose2d(16, 2, kernel_size=4,stride=2,padding=0)

    def forward(self, x):
        x = self.fc(x).view([-1, 16, 15, 15])
        # if self.setting==1:
        #     x = self.deconv(x).view([-1, 4096])
        #     x = self.fc2(x)
        # if self.setting==2:
        x = self.deconv(x).view([-1, 2048])
        pred_f = F.relu(x, inplace=True)
        return pred_f


def torch_max_not_return_idx(input, axis):
    return torch.max(input, axis)[0]


class AttnDecoderLSTM(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, input_action_size, output_action_size, embedding_size, ctx_hidden_size, hidden_size,
                      dropout_ratio, feature_size, panoramic, action_space, ctrl_feature, ctrl_f_net, att_ctx_merge, ctx_dropout_ratio, params=None):
        super(AttnDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size + (128 if panoramic else 0)
        self.ctx_hidden_size = ctx_hidden_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        self.panoramic = panoramic
        self.action_space = action_space
        self.ctrl_feature = ctrl_feature
        self.ctrl_f_net = ctrl_f_net
        self.ctx_drop = nn.Dropout(p=ctx_dropout_ratio)

        self.dec_h_type = params['dec_h_type']

        action_hidden_size = hidden_size
        self.att_ctx_merge = None
        if att_ctx_merge in ['mean', 'cat', 'max', 'sum']:
            if att_ctx_merge == 'cat':
                self.att_ctx_merge = torch.flatten
                action_hidden_size = 3 * hidden_size
            elif att_ctx_merge == 'mean':
                self.att_ctx_merge = torch.mean
            elif att_ctx_merge == 'max':
                self.att_ctx_merge = torch_max_not_return_idx
            elif att_ctx_merge == 'sum':
                self.att_ctx_merge = torch.sum

        if not self.panoramic:
            LSTM_n_in = embedding_size + feature_size  # action_embedding + original single view feature
        else:
            if self.action_space == 6:
                LSTM_n_in = embedding_size + feature_size + self.feature_size  # attented multi-view feature (single feature +128)
            else:  # if self.action_space == -1:
                LSTM_n_in = self.feature_size * 2  # action feature + attented multi-view feature
                #LSTM_n_in = feature_size + self.feature_size * 2  # img feature + action feature + attented multi-view feature

        self.lstm = nn.LSTMCell(LSTM_n_in, hidden_size)
        self.attention_layer = SoftDotAttention(ctx_hidden_size, hidden_size)
        # panoramic feature
        if self.panoramic:
            self.visual_attention_layer = VisualSoftDotAttention(hidden_size, self.feature_size)
        else:
            self.visual_attention_layer = None

        # panoramic action space
        self.u_begin, self.embedding = None, None
        if self.action_space == 6:
            self.embedding = nn.Embedding(input_action_size, embedding_size)
            self.decoder2action = nn.Linear(action_hidden_size, output_action_size)
        else:
            self.u_begin = Variable(torch.zeros(self.feature_size), requires_grad=False).to(device)
            self.decoder2action = EltwiseProdScoring(action_hidden_size, self.feature_size)
            #self.decoder2action = EltwiseProdScoring(action_hidden_size+self.feature_size, self.feature_size) # debug

        if self.ctrl_feature:
            if self.ctrl_f_net =='deconv':
                self.decoder2feature = DeconvFeature(action_hidden_size, feature_size)
            elif self.ctrl_f_net == 'linear':  # linear
                self.decoder2feature = LinearFeature(action_hidden_size, feature_size)
            elif self.ctrl_f_net == 'imglinear':
                self.decoder2feature = LinearFeature(action_hidden_size + feature_size, feature_size)
            elif self.ctrl_f_net == 'nonlinear':
                self.decoder2feature = NonLinearFeature(action_hidden_size, 1024, feature_size)
            elif self.ctrl_f_net == 'imgnl':
                self.decoder2feature = NonLinearFeature(action_hidden_size + feature_size, 1024, feature_size)
        else:
            self.decoder2feature = None

    def forward(self, action_prev, u_prev, u_features,  # teacher_u_feature,
                feature, feature_all, h_0, c_0, ctx, ctx_mask=None):  #, action_prev_feature
        ''' Takes a single step in the decoder LSTM (allowing sampling).
        action: batch x 1
        feature: batch x feature_size
        h_0: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        '''

        if self.panoramic:  # feature.dim()==3:
            feature2, alpha_v = self.visual_attention_layer(h_0, feature_all)

        if self.action_space == 6:
            if self.panoramic:
                feature = torch.cat((feature, feature2), 1)
            action_embeds = self.embedding(action_prev.view(-1, 1))  # (batch, 1, embedding_size)
            action_embeds = action_embeds.squeeze()
        else: # bug: todo
            #feature = feature2
            action_embeds = u_prev

        concat_input = torch.cat((action_embeds, feature2), 1)
        #concat_input = torch.cat((feature, action_embeds, feature2), 1)
        drop = self.drop(concat_input)

        h_1, c_1 = self.lstm(drop, (h_0, c_0))
        h_1_drop = self.drop(h_1)

        if self.att_ctx_merge is not None:
            temp, alpha = torch.zeros(h_1_drop.size(0), len(ctx), self.hidden_size).to(device), []
            for si in range(len(ctx)):
                h_tilde, alpha_si = self.attention_layer(h_1_drop, ctx[si], ctx_mask[si], self.ctx_drop)
                temp[:, si, :] = h_tilde
                alpha.append(alpha_si)  # for plot

            h_tilde = self.att_ctx_merge(temp, 1)
        else:
            h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)

        if self.action_space == 6:
            logit = self.decoder2action(h_tilde)  # (100, 6)
        else:
            logit = self.decoder2action(h_tilde, u_features)
            #logit = self.decoder2action(action_input, u_features)

        pred_f = None
        if self.ctrl_feature:
            if self.ctrl_f_net == 'imglinear' or self.ctrl_f_net == 'imgnl': #
                #aux_input = torch.cat((h_1_drop, feature), 1) # with img feature h_1_drop
                aux_input = torch.cat((h_tilde, feature), 1)  # with img feature h_tilde
                pred_f = self.decoder2feature(aux_input)  # , teacher_u_feature)
            else:
                pred_f = self.decoder2feature(h_tilde) #, teacher_u_feature)

        if self.dec_h_type == 'vc':
            return h_tilde, c_1, alpha, logit, pred_f  # h_tilde
        else:
            return h_1, c_1, alpha, logit, pred_f # old verion


class AttnDecoderCLS(nn.Module):
    ''' A simple classifier with attention over instructions for decoding navigation actions. '''

    def __init__(self, input_action_size, output_action_size, embedding_size, ctx_hidden_size, hidden_size,
                      dropout_ratio, feature_size, panoramic, action_space, ctrl_feature, ctrl_f_net, att_ctx_merge, ctx_dropout_ratio, params=None):
        super(AttnDecoderCLS, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size + (128 if panoramic else 0)
        self.ctx_hidden_size = ctx_hidden_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        self.panoramic = panoramic
        self.action_space = action_space
        self.ctrl_feature = ctrl_feature
        self.ctrl_f_net = ctrl_f_net
        self.ctx_drop = nn.Dropout(p=ctx_dropout_ratio)

        self.dec_h_type = params['dec_h_type']

        action_hidden_size = hidden_size
        self.att_ctx_merge = None
        if att_ctx_merge in ['mean', 'cat', 'max', 'sum']:
            if att_ctx_merge == 'cat':
                self.att_ctx_merge = torch.flatten
                action_hidden_size = 3 * hidden_size
            elif att_ctx_merge == 'mean':
                self.att_ctx_merge = torch.mean
            elif att_ctx_merge == 'max':
                self.att_ctx_merge = torch_max_not_return_idx
            elif att_ctx_merge == 'sum':
                self.att_ctx_merge = torch.sum

        if not self.panoramic:
            LSTM_n_in = embedding_size + feature_size  # action_embedding + original single view feature
        else:
            if self.action_space == 6:
                LSTM_n_in = embedding_size + feature_size + self.feature_size  # attented multi-view feature (single feature +128)
            else:  # if self.action_space == -1:
                LSTM_n_in = self.feature_size * 2  # action feature + attented multi-view feature
                #LSTM_n_in = feature_size + self.feature_size * 2  # img feature + action feature + attented multi-view feature

        #self.lstm = nn.LSTMCell(LSTM_n_in, hidden_size)
        self.affine_h = nn.Linear(LSTM_n_in, hidden_size)
        self.affine_c = nn.Linear(LSTM_n_in, hidden_size)
        self.attention_layer = SoftDotAttention(ctx_hidden_size, hidden_size)
        # panoramic feature
        if self.panoramic:
            self.visual_attention_layer = VisualSoftDotAttention(hidden_size, self.feature_size)
        else:
            self.visual_attention_layer = None

        # panoramic action space
        self.u_begin, self.embedding = None, None
        if self.action_space == 6:
            self.embedding = nn.Embedding(input_action_size, embedding_size)
            self.decoder2action = nn.Linear(action_hidden_size, output_action_size)
        else:
            self.u_begin = Variable(torch.zeros(self.feature_size), requires_grad=False).to(device)
            self.decoder2action = EltwiseProdScoring(action_hidden_size, self.feature_size)
            #self.decoder2action = EltwiseProdScoring(action_hidden_size+self.feature_size, self.feature_size) # debug

        if self.ctrl_feature:
            if self.ctrl_f_net =='deconv':
                self.decoder2feature = DeconvFeature(action_hidden_size, feature_size)
            elif self.ctrl_f_net == 'linear':  # linear
                self.decoder2feature = LinearFeature(action_hidden_size, feature_size)
            elif self.ctrl_f_net == 'imglinear':
                self.decoder2feature = LinearFeature(action_hidden_size + feature_size, feature_size)
            elif self.ctrl_f_net == 'nonlinear':
                self.decoder2feature = NonLinearFeature(action_hidden_size, 1024, feature_size)
            elif self.ctrl_f_net == 'imgnl':
                self.decoder2feature = NonLinearFeature(action_hidden_size + feature_size, 1024, feature_size)
        else:
            self.decoder2feature = None

    def forward(self, action_prev, u_prev, u_features,  # teacher_u_feature,
                feature, feature_all, h_0, c_0, ctx, ctx_mask=None):  #, action_prev_feature
        ''' Takes a single step in the decoder classifier (allowing sampling).
        action: batch x 1
        feature: batch x feature_size
        h_0: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        '''

        if self.panoramic:  # feature.dim()==3:
            feature2, alpha_v = self.visual_attention_layer(h_0, feature_all)

        if self.action_space == 6:
            if self.panoramic:
                feature = torch.cat((feature, feature2), 1)
            action_embeds = self.embedding(action_prev.view(-1, 1))  # (batch, 1, embedding_size)
            action_embeds = action_embeds.squeeze()
        else: # bug: todo
            #feature = feature2
            action_embeds = u_prev

        concat_input = torch.cat((action_embeds, feature2), 1)
        #concat_input = torch.cat((feature, action_embeds, feature2), 1)
        drop = self.drop(concat_input)

        #h_1, c_1 = self.lstm(drop, (h_0, c_0))
        h_1, c_1 = self.affine_h(h_0), self.affine_c(c_0)
        h_1_drop = self.drop(h_1)

        if self.att_ctx_merge is not None:
            temp, alpha = torch.zeros(h_1_drop.size(0), len(ctx), self.hidden_size).to(device), []
            for si in range(len(ctx)):
                h_tilde, alpha_si = self.attention_layer(h_1_drop, ctx[si], ctx_mask[si], self.ctx_drop)
                temp[:, si, :] = h_tilde
                alpha.append(alpha_si)  # for plot

            h_tilde = self.att_ctx_merge(temp, 1)
        else:
            h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)

        if self.action_space == 6:
            logit = self.decoder2action(h_tilde)  # (100, 6)
        else:
            logit = self.decoder2action(h_tilde, u_features)
            #logit = self.decoder2action(action_input, u_features)

        pred_f = None
        if self.ctrl_feature:
            if self.ctrl_f_net == 'imglinear' or self.ctrl_f_net == 'imgnl': #
                #aux_input = torch.cat((h_1_drop, feature), 1) # with img feature h_1_drop
                aux_input = torch.cat((h_tilde, feature), 1)  # with img feature h_tilde
                pred_f = self.decoder2feature(aux_input)  # , teacher_u_feature)
            else:
                pred_f = self.decoder2feature(h_tilde) #, teacher_u_feature)

        if self.dec_h_type == 'vc':
            return h_tilde, c_1, alpha, logit, pred_f  # h_tilde
        else:
            return h_1, c_1, alpha, logit, pred_f # old verion
