import math

import torch
from torch import nn
import torch.nn.functional as F
import operator
from queue import PriorityQueue
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


def transpose_qkv(X, num_heads):
    """Transposition for parallel computation of multiple attention heads.

    Defined in :numref:`sec_multihead-attention`"""
    # Shape of input `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`).
    # Shape of output `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_heads`,
    # `num_hiddens` / `num_heads`)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # Shape of output `X`:
    # (`batch_size`, `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    X = X.permute(0, 2, 1, 3)

    # Shape of `output`:
    # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """Reverse the operation of `transpose_qkv`.

    Defined in :numref:`sec_multihead-attention`"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis.

    Defined in :numref:`sec_attention-scoring-functions`"""
    # `X`: 3D tensor, `valid_lens`: 1D or 2D tensor
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences.

    Defined in :numref:`sec_seq2seq_decoder`"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFFN, self).__init__()
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.W_2(self.dropout(F.relu(self.W_1(x))))


class AddNorm(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        return self.norm(x + self.dropout(y))


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, d_model))
        x = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / \
            torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
        self.P[:, :, 0::2] = torch.sin(x)
        self.P[:, :, 1::2] = torch.cos(x)

    def forward(self, x):
        x = x + self.P[:, :x.shape[1], :].to(x.device)
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=100):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x, pos):
        # x -> batch * seq * dim
        # pos -> batch * seq
        x = x + self.pos_embedding(pos)
        return self.dropout(x)


class KeywordsEncoding(nn.Module):
    def __init__(self, d_model, dropout, keywords_type=6):
        super(KeywordsEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.type_embedding = nn.Embedding(keywords_type, d_model)

    def forward(self, x, keywords_type):
        x = x + self.type_embedding(keywords_type)
        return self.dropout(x)


class MultiHeadAttentionWithRPR(nn.Module):
    """Multi-head attention.

    Defined in :numref:`sec_multihead-attention`"""

    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, clipping_distance,
                 dropout, bias=False, **kwargs):
        super(MultiHeadAttentionWithRPR, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention_rpr = DotProductAttentionWithRPR(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.relative_pos_v = nn.Embedding(2 * clipping_distance + 1, num_hiddens // num_heads)
        self.relative_pos_k = nn.Embedding(2 * clipping_distance + 1, num_hiddens // num_heads)
        self.clipping_distance = clipping_distance

    def forward(self, queries, keys, values, valid_lens):
        # Shape of `queries`, `keys`, or `values`:
        # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`)
        # Shape of `valid_lens`:
        # (`batch_size`,) or (`batch_size`, no. of queries)
        # After transposing, shape of output `queries`, `keys`, or `values`:
        # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
        # `num_hiddens` / `num_heads`)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        # relative position matrix
        range_queries = torch.arange(queries.size(1), device=queries.device)
        range_keys = torch.arange(keys.size(1), device=keys.device)
        distance_mat = range_keys[None, :] - range_queries[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.clipping_distance, self.clipping_distance) + \
                               self.clipping_distance
        # pos_k, pos_v -> seq_q * seq_k * dim
        pos_k = self.relative_pos_k(distance_mat_clipped)
        pos_v = self.relative_pos_v(distance_mat_clipped)

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for
            # `num_heads` times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # Shape of `output`: (`batch_size` * `num_heads`, no. of queries,
        # `num_hiddens` / `num_heads`)
        output = self.attention_rpr(queries, keys, values, pos_k, pos_v, valid_lens)

        # Shape of `output_concat`:
        # (`batch_size`, no. of queries, `num_hiddens`)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class DotProductAttentionWithRPR(nn.Module):
    """Scaled dot product attention.

    Defined in :numref:`subsec_additive-attention`"""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttentionWithRPR, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value
    # dimension)
    # Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries)
    def forward(self, queries, keys, values, pos_k, pos_v, valid_lens=None):
        d = queries.shape[-1]
        # Set `transpose_b=True` to swap the last two dimensions of `keys`
        scores = torch.bmm(queries, keys.transpose(1, 2))
        scores_pos = torch.bmm(queries.transpose(0, 1), pos_k.transpose(1, 2)).transpose(0, 1)
        scores = (scores + scores_pos) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        output = torch.bmm(self.dropout(self.attention_weights), values)
        output_pos = torch.bmm(self.dropout(self.attention_weights.transpose(0, 1)), pos_v).transpose(0, 1)
        return output + output_pos


class MultiHeadAttention(nn.Module):
    """Multi-head attention.

    Defined in :numref:`sec_multihead-attention`"""

    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # Shape of `queries`, `keys`, or `values`:
        # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`)
        # Shape of `valid_lens`:
        # (`batch_size`,) or (`batch_size`, no. of queries)
        # After transposing, shape of output `queries`, `keys`, or `values`:
        # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
        # `num_hiddens` / `num_heads`)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for
            # `num_heads` times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # Shape of `output`: (`batch_size` * `num_heads`, no. of queries,
        # `num_hiddens` / `num_heads`)
        output = self.attention(queries, keys, values, valid_lens)

        # Shape of `output_concat`:
        # (`batch_size`, no. of queries, `num_hiddens`)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class DotProductAttention(nn.Module):
    """Scaled dot product attention.

    Defined in :numref:`subsec_additive-attention`"""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value
    # dimension)
    # Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Set `transpose_b=True` to swap the last two dimensions of `keys`
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, head_num, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, d_model, d_model, d_model, head_num, dropout)
        self.feedForward = PositionWiseFFN(d_model, d_ff, dropout)
        self.add_norm1 = AddNorm(d_model)
        self.add_norm2 = AddNorm(d_model)

    def forward(self, x, valid_lens):
        y = self.self_attention(x, x, x, valid_lens)
        y = self.add_norm1(x, y)
        z = self.feedForward(y)
        z = self.add_norm2(y, z)
        return z


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, head_num, N=6, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        # self.embedding = nn.Embedding(vocab_size, d_model)
        # self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([EncoderBlock(d_model, d_ff, head_num, dropout) for _ in range(N)])

    def forward(self, x, valid_lens):
        # encoder编码时,输入x是CNN输出的embedding
        # x = self.pos_encoding(self.embedding(x) * math.sqrt(self.d_model))
        for layer in self.layers:
            x = layer(x, valid_lens)

        return x


class EncoderBlockWithRPR(nn.Module):
    def __init__(self, d_model, d_ff, head_num, clipping_distance, dropout=0.1):
        super(EncoderBlockWithRPR, self).__init__()
        self.self_attention = MultiHeadAttentionWithRPR(d_model, d_model, d_model, d_model, head_num,
                                                        clipping_distance, dropout)
        self.feedForward = PositionWiseFFN(d_model, d_ff, dropout)
        self.add_norm1 = AddNorm(d_model)
        self.add_norm2 = AddNorm(d_model)

    def forward(self, x, valid_lens):
        y = self.self_attention(x, x, x, valid_lens)
        y = self.add_norm1(x, y)
        z = self.feedForward(y)
        z = self.add_norm2(y, z)
        return z


class EncoderWithRPR(nn.Module):
    def __init__(self, d_model, d_ff, head_num, clipping_distance, N=6, dropout=0.1):
        super(EncoderWithRPR, self).__init__()
        self.d_model = d_model
        # self.embedding = nn.Embedding(vocab_size, d_model)
        # self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList(
            [EncoderBlockWithRPR(d_model, d_ff, head_num, clipping_distance, dropout) for _ in range(N)])

    def forward(self, x, valid_lens):
        # encoder编码时,输入x是CNN输出的embedding
        # x = self.pos_encoding(self.embedding(x) * math.sqrt(self.d_model))
        for layer in self.layers:
            x = layer(x, valid_lens)

        return x


class DecoderBlockWithKeywords(nn.Module):
    def __init__(self, i, d_model, d_ff, head_num, dropout=0.1):
        super(DecoderBlockWithKeywords, self).__init__()
        self.i = i
        self.masked_self_attention = MultiHeadAttention(d_model, d_model, d_model, d_model, head_num, dropout)
        self.cross_attention_code = MultiHeadAttention(d_model, d_model, d_model, d_model, head_num, dropout)
        self.cross_attention_template = MultiHeadAttention(d_model, d_model, d_model, d_model, head_num, dropout)
        self.cross_attention_keywords = MultiHeadAttention(d_model, d_model, d_model, d_model, head_num, dropout)
        self.gate = nn.Linear(d_model + d_model, 1)
        self.feedForward = PositionWiseFFN(d_model, d_ff, dropout)
        self.add_norm1 = AddNorm(d_model)
        self.add_norm2 = AddNorm(d_model)
        self.add_norm3 = AddNorm(d_model)
        self.add_norm4 = AddNorm(d_model)

    def forward(self, x, state):
        source_code_enc, source_code_len = state[0], state[1]
        template_enc, template_len = state[2], state[3]
        keywords_enc, keywords_len = state[4], state[5]
        if state[6][self.i] is None:
            key_values = x
        else:
            key_values = torch.cat((state[6][self.i], x), axis=1)
        state[6][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = x.shape
            dec_valid_lens = torch.arange(1, num_steps + 1, device=x.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        # 1. self attention
        x2 = self.masked_self_attention(x, key_values, key_values, dec_valid_lens)
        y = self.add_norm1(x, x2)
        # 2. cross attention
        y2_code = self.cross_attention_code(y, source_code_enc, source_code_enc, source_code_len)
        y2_keyword = self.cross_attention_keywords(y, keywords_enc, keywords_enc, keywords_len)
        gate_weight = torch.sigmoid(self.gate(torch.cat([y2_code, y2_keyword], dim=-1)))
        y2 = gate_weight * y2_code + (1. - gate_weight) * y2_keyword
        z = self.add_norm2(y, y2)
        # 3. cross attention
        z2 = self.cross_attention_template(z, template_enc, template_enc, template_len)
        # z2 = z2_keywords + z2_template
        z_end = self.add_norm3(z, z2)
        return self.add_norm4(z_end, self.feedForward(z_end)), state


class DecoderWithKeywords(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, head_num, N=6, dropout=0.1):
        super(DecoderWithKeywords, self).__init__()
        self.num_layers = N
        self.d_model = d_model
        self.layers = nn.ModuleList(
            [DecoderBlockWithKeywords(i, d_model, d_ff, head_num, dropout) for i in range(self.num_layers)])
        self.dense = nn.Linear(d_model, vocab_size)

    def init_state(self, source_code_enc, source_code_len, template_enc, template_len,
                   keywords_enc, keywords_len):
        return [source_code_enc, source_code_len, template_enc, template_len,
                keywords_enc, keywords_len, [None] * self.num_layers]

    def forward(self, x, state):
        for layer in self.layers:
            x, state = layer(x, state)

        return self.dense(x), state


class CodeEncoder(nn.Module):
    def __init__(self, code_embedding, d_model, d_ff, head_num, encoder_layer_num, clipping_distance, dropout=0.1):
        super(CodeEncoder, self).__init__()
        self.code_embedding = code_embedding
        self.code_encoder = EncoderWithRPR(d_model, d_ff, head_num, clipping_distance, encoder_layer_num, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, source_code, source_code_len):
        source_code_embed = self.dropout(self.code_embedding(source_code))
        source_code_enc = self.code_encoder(source_code_embed, source_code_len)

        return source_code_enc, source_code_len


class KeywordsEncoder(nn.Module):
    def __init__(self, comment_embedding, d_model, d_ff, head_num, encoder_layer_num, dropout=0.1):
        super(KeywordsEncoder, self).__init__()
        self.comment_embedding = comment_embedding
        self.keywords_encoder = Encoder(d_model, d_ff, head_num, encoder_layer_num, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, keywords, keywords_len):
        keywords_embed = self.dropout(self.comment_embedding(keywords))
        keywords_enc = self.keywords_encoder(keywords_embed, keywords_len)

        return keywords_enc, keywords_len


class TemplateEncoder(nn.Module):
    def __init__(self, comment_embedding, pos_encoding, d_model, d_ff, head_num, encoder_layer_num, dropout=0.1):
        super(TemplateEncoder, self).__init__()
        self.comment_embedding = comment_embedding
        self.pos_encoding = pos_encoding
        self.template_encoder = Encoder(d_model, d_ff, head_num, encoder_layer_num, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, template, template_len):
        b_, seq_template_num = template.size()
        template_pos = torch.arange(seq_template_num, device=template.device).repeat(b_, 1)
        template_embed = self.pos_encoding(self.comment_embedding(template), template_pos)
        template_enc = self.template_encoder(template_embed, template_len)

        return template_enc, template_len


class Evaluator(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(Evaluator, self).__init__()
        self.linear_proj1 = nn.Linear(d_model, d_ff)
        self.linear_proj2 = nn.Linear(d_ff, d_model)
        self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, source_code_enc, source_code_len, comment_enc, comment_len, template_enc, template_len,
                best_result=None, cur_result=None):
        b_ = source_code_enc.size(0)
        # global mean pooling
        source_code_vec = torch.cumsum(source_code_enc, dim=1)[torch.arange(b_), source_code_len - 1]
        source_code_vec = torch.div(source_code_vec.T, source_code_len).T
        source_code_vec = self.linear_proj2(self.dropout(F.relu(self.linear_proj1(source_code_vec))))

        comment_vec = torch.cumsum(comment_enc, dim=1)[torch.arange(b_), comment_len - 1]
        comment_vec = torch.div(comment_vec.T, comment_len).T
        comment_vec = self.linear_proj2(self.dropout(F.relu(self.linear_proj1(comment_vec))))

        template_vec = torch.cumsum(template_enc, dim=1)[torch.arange(b_), template_len - 1]
        template_vec = torch.div(template_vec.T, template_len).T
        template_vec = self.linear_proj2(self.dropout(F.relu(self.linear_proj1(template_vec))))

        if self.training:
            return source_code_vec, comment_vec, template_vec
        else:
            best_enc = [x for x in template_enc]
            best_len = template_len

            assert best_result is not None
            assert cur_result is not None
            pos_sim = self.cos_sim(source_code_vec, comment_vec)
            neg_sim = self.cos_sim(source_code_vec, template_vec)
            better_index = (pos_sim > neg_sim).nonzero(as_tuple=True)[0]
            for ix in better_index:
                best_result[ix] = cur_result[ix]
                best_enc[ix] = comment_enc[ix]
                best_len[ix] = comment_len[ix]

            return best_result, pad_sequence(best_enc, batch_first=True), best_len


class KeywordsGuidedDecoder(nn.Module):
    def __init__(self, comment_embedding, pos_encoding, d_model, d_ff, head_num, decoder_layer_num, comment_vocab_size,
                 bos_token, eos_token, max_comment_len, dropout=0.1, beam_width=4):
        super(KeywordsGuidedDecoder, self).__init__()
        self.comment_embedding = comment_embedding
        self.pos_encoding = pos_encoding
        self.comment_decoder = DecoderWithKeywords(comment_vocab_size, d_model, d_ff, head_num, decoder_layer_num, dropout)

        self.max_comment_len = max_comment_len
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.d_model = d_model
        self.beam_width = beam_width
        self.num_layers = decoder_layer_num

    def forward(self, source_code_enc, comment, template_enc, keywords_enc,
                source_code_len, template_len, keywords_len):
        b_, seq_comment_num = comment.size()
        dec_state = self.comment_decoder.init_state(source_code_enc, source_code_len,
                                                    template_enc, template_len,
                                                    keywords_enc, keywords_len)

        if self.training:
            comment_pos = torch.arange(seq_comment_num, device=comment.device).repeat(b_, 1)
            comment_embed = self.pos_encoding(self.comment_embedding(comment), comment_pos)

            comment_pred = self.comment_decoder(comment_embed, dec_state)[0]
            return comment_pred
        else:
            if self.beam_width:
                return self.beam_search(b_, comment, dec_state, self.beam_width)
            else:
                return self.greed_search(b_, comment, dec_state)

    def greed_search(self, batch_size, comment, dec_state):
        comment_pred = [[self.bos_token] for _ in range(batch_size)]
        for pos_idx in range(self.max_comment_len):
            comment_pos = torch.arange(pos_idx, pos_idx + 1, device=comment.device).repeat(batch_size, 1)
            comment_embed = self.pos_encoding(self.comment_embedding(comment), comment_pos)
            comment, dec_state = self.comment_decoder(comment_embed, dec_state)
            comment = torch.argmax(comment, -1).detach()
            for i in range(batch_size):
                if comment_pred[i][-1] != self.eos_token:
                    comment_pred[i].append(int(comment[i]))

        comment_pred = [x[1:-1] if x[-1] == self.eos_token and len(x) > 2 else x[1:]
                        for x in comment_pred]
        return comment_pred

    def beam_search(self, batch_size, comment, dec_state, beam_width):
        # comment -> batch * 1
        # first node
        node_list = []
        batchNode_dict = {i: None for i in range(beam_width)}  # 每个时间步都只保留beam_width个node
        # initialization
        for batch_idx in range(batch_size):
            node_comment = comment[batch_idx].unsqueeze(0)
            node_dec_state = [dec_state[0][batch_idx].unsqueeze(0), dec_state[1][batch_idx].unsqueeze(0),
                              dec_state[2][batch_idx].unsqueeze(0), dec_state[3][batch_idx].unsqueeze(0),
                              dec_state[4][batch_idx].unsqueeze(0), dec_state[5][batch_idx].unsqueeze(0),
                              [None] * self.num_layers]
            node_list.append(BeamSearchNode(node_dec_state, None, node_comment, 0, 0))
        batchNode_dict[0] = BatchNodeWithKeywords(node_list)

        # start beam search
        pos_idx = 0
        while pos_idx < self.max_comment_len:
            beamNode_dict = {i: PriorityQueue() for i in range(batch_size)}
            count = 0
            for idx in range(beam_width):
                if batchNode_dict[idx] is None:
                    continue

                batchNode = batchNode_dict[idx]
                # comment -> batch * 1
                comment = batchNode.get_comment()
                dec_state = batchNode.get_dec_state()

                # decode for one step using decoder
                pos = torch.arange(pos_idx, pos_idx + 1, device=comment.device).repeat(batch_size, 1)
                # comment -> batch * d_model
                comment = self.pos_encoding(self.comment_embedding(comment), pos)
                tensor, dec_state = self.comment_decoder(comment, dec_state)
                tensor = F.log_softmax(tensor.squeeze(1), -1).detach()
                # PUT HERE REAL BEAM SEARCH OF TOP
                # log_prob, comment_candidates -> batch * beam_width
                log_prob, comment_candidates = torch.topk(tensor, beam_width, dim=-1)

                for batch_idx in range(batch_size):
                    pre_node = batchNode.list_node[batch_idx]
                    node_dec_state = [dec_state[0][batch_idx].unsqueeze(0), dec_state[1][batch_idx].unsqueeze(0),
                                      dec_state[2][batch_idx].unsqueeze(0), dec_state[3][batch_idx].unsqueeze(0),
                                      dec_state[4][batch_idx].unsqueeze(0), dec_state[5][batch_idx].unsqueeze(0),
                                      [l[batch_idx].unsqueeze(0) for l in dec_state[6]]]
                    if pre_node.history_word[-1] == self.eos_token:
                        new_node = BeamSearchNode(node_dec_state, pre_node.prevNode, pre_node.commentID,
                                                  pre_node.logp, pre_node.leng)
                        # check
                        assert new_node.score == pre_node.score
                        assert new_node.history_word == pre_node.history_word
                        beamNode_dict[batch_idx].put((-new_node.score, count, new_node))
                        count += 1
                        continue

                    for beam_idx in range(beam_width):
                        node_comment = comment_candidates[batch_idx][beam_idx].view(1, -1)
                        node_log_prob = float(log_prob[batch_idx][beam_idx])
                        new_node = BeamSearchNode(node_dec_state, pre_node, node_comment, pre_node.logp + node_log_prob,
                                                  pre_node.leng + 1)
                        beamNode_dict[batch_idx].put((-new_node.score, count, new_node))
                        count += 1

            for beam_idx in range(beam_width):
                node_list = [beamNode_dict[batch_idx].get()[-1] for batch_idx in range(batch_size)]
                batchNode_dict[beam_idx] = BatchNodeWithKeywords(node_list)

            pos_idx += 1
        # the first batchNode in batchNode_dict is the best node
        best_node = batchNode_dict[0]
        comment_pred = []
        for batch_idx in range(batch_size):
            history_word = best_node.list_node[batch_idx].history_word
            if history_word[-1] == self.eos_token and len(history_word) > 2:
                comment_pred.append(history_word[1:-1])
            else:
                comment_pred.append(history_word[1:])

        return comment_pred


class DECOM(nn.Module):
    def __init__(self, d_model, d_ff, head_num, encoder_layer_num, decoder_layer_num, code_vocab_size,
                 comment_vocab_size, bos_token, eos_token, max_comment_len, clipping_distance, max_iter_num,
                 dropout=0.1, beam_width=4):
        super(DECOM, self).__init__()

        self.code_embedding = nn.Embedding(code_vocab_size, d_model)
        self.comment_embedding = nn.Embedding(comment_vocab_size, d_model)
        self.pos_encoding = LearnablePositionalEncoding(d_model, dropout, max_comment_len + 2)
        self.code_encoder = CodeEncoder(self.code_embedding, d_model, d_ff, head_num,
                                        encoder_layer_num, clipping_distance, dropout)
        self.keyword_encoder = KeywordsEncoder(self.comment_embedding, d_model, d_ff, head_num,
                                               encoder_layer_num, dropout)
        self.template_encoder = TemplateEncoder(self.comment_embedding, self.pos_encoding, d_model, d_ff, head_num,
                                                encoder_layer_num, dropout)
        self.deliberation_dec = nn.ModuleList(
            [KeywordsGuidedDecoder(self.comment_embedding, self.pos_encoding, d_model, d_ff, head_num,
                                   decoder_layer_num, comment_vocab_size, bos_token, eos_token, max_comment_len,
                                   dropout, beam_width) if _ == 0 else
             KeywordsGuidedDecoder(self.comment_embedding,
                                   self.pos_encoding, d_model,
                                   d_ff, head_num,
                                   decoder_layer_num,
                                   comment_vocab_size,
                                   bos_token, eos_token,
                                   max_comment_len,
                                   dropout, None)
             for _ in range(max_iter_num)])
        self.evaluator = Evaluator(d_model, d_ff, dropout)
        self.max_iter_num = max_iter_num

    def forward(self, source_code, comment, template, keywords,
                source_code_len, comment_len, template_len, keywords_len):
        source_code_enc, source_code_len = self.code_encoder(source_code, source_code_len)
        keywords_enc, keywords_len = self.keyword_encoder(keywords, keywords_len)
        template_enc, template_len = self.template_encoder(template, template_len)
        if self.training:
            comment_enc, comment_len = self.template_encoder(comment[:, 1:], comment_len)
            anchor, positive, negative = self.evaluator(source_code_enc, source_code_len,
                                                        comment_enc, comment_len, template_enc, template_len)
            memory = []
            for iter_idx in range(self.max_iter_num):
                comment_pred = self.deliberation_dec[iter_idx](source_code_enc, comment, template_enc,
                                                               keywords_enc, source_code_len, template_len, keywords_len)
                memory.append(comment_pred)
                if iter_idx == self.max_iter_num - 1:
                    return memory, anchor, positive, negative
                template = torch.argmax(comment_pred.detach(), -1)
                template_len = comment_len
                template_enc, template_len = self.template_encoder(template, template_len)

        else:
            memory = []
            best_result = [x.tolist()[:leng] for x, leng in zip(template, template_len)]
            best_enc = template_enc
            best_len = template_len
            for iter_idx in range(self.max_iter_num):
                comment_pred = self.deliberation_dec[iter_idx](source_code_enc, comment, template_enc, keywords_enc,
                                                               source_code_len, template_len, keywords_len)
                memory.append(comment_pred)
                template = pad_sequence([torch.tensor(x, device=comment.device) for x in comment_pred],
                                        batch_first=True)
                template_len = torch.tensor([len(x) for x in comment_pred], device=comment.device)
                template_enc, template_len = self.template_encoder(template, template_len)
                best_result, best_enc, best_len = self.evaluator(source_code_enc, source_code_len, template_enc,
                                                                 template_len, best_enc, best_len, best_result, comment_pred)
            memory.append(best_result)
            assert len(memory) == self.max_iter_num + 1
            return memory


class BatchNode(object):
    def __init__(self, list_node):
        self.list_node = list_node

    def get_comment(self):
        comment_list = [node.commentID for node in self.list_node]
        return torch.cat(comment_list, dim=0)

    def get_dec_state(self):
        dec_state_list = [node.dec_state for node in self.list_node]
        batch_dec_state = [torch.cat([batch_state[0] for batch_state in dec_state_list], dim=0),
                           torch.cat([batch_state[1] for batch_state in dec_state_list], dim=0)]
        if dec_state_list[0][2][0] is None:
            batch_dec_state.append(dec_state_list[0][2])
        else:
            state_3 = []
            for i in range(len(dec_state_list[0][2])):
                state_3.append(torch.cat([batch_state[2][i] for batch_state in dec_state_list], dim=0))
            assert len(state_3) == len(dec_state_list[0][2])
            batch_dec_state.append(state_3)
        return batch_dec_state

    def if_allEOS(self, eos_token):
        for node in self.list_node:
            if node.history_word[-1] != eos_token:
                return False
        return True


class BatchNodeWithKeywords(object):
    def __init__(self, list_node):
        self.list_node = list_node

    def get_comment(self):
        comment_list = [node.commentID for node in self.list_node]
        return torch.cat(comment_list, dim=0)

    def get_dec_state(self):
        dec_state_list = [node.dec_state for node in self.list_node]
        batch_dec_state = [torch.cat([batch_state[0] for batch_state in dec_state_list], dim=0),
                           torch.cat([batch_state[1] for batch_state in dec_state_list], dim=0),
                           torch.cat([batch_state[2] for batch_state in dec_state_list], dim=0),
                           torch.cat([batch_state[3] for batch_state in dec_state_list], dim=0),
                           torch.cat([batch_state[4] for batch_state in dec_state_list], dim=0),
                           torch.cat([batch_state[5] for batch_state in dec_state_list], dim=0)]
        # batch_dec_state = [torch.cat([batch_state[i] for batch_state in dec_state_list], dim=0) for i in range(6)]
        if dec_state_list[0][6][0] is None:
            batch_dec_state.append(dec_state_list[0][6])
        else:
            state_3 = []
            for i in range(len(dec_state_list[0][6])):
                state_3.append(torch.cat([batch_state[6][i] for batch_state in dec_state_list], dim=0))
            assert len(state_3) == len(dec_state_list[0][6])
            batch_dec_state.append(state_3)
        return batch_dec_state


class BeamSearchNode(object):
    def __init__(self, dec_state, previousNode, commentID, logProb, length, length_penalty=1):
        '''
        :param dec_state:
        :param previousNode:
        :param commentID:
        :param logProb:
        :param length:
        '''
        self.dec_state = dec_state
        self.prevNode = previousNode
        self.commentID = commentID
        self.logp = logProb
        self.leng = length
        self.length_penalty = length_penalty
        if self.prevNode is None:
            self.history_word = [int(commentID)]
            self.score = -100
        else:
            self.history_word = previousNode.history_word + [int(commentID)]
            self.score = self.eval()

    def eval(self):
        return self.logp / self.leng ** self.length_penalty
