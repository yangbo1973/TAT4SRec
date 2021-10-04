import numpy as np
import torch
from torch import nn as nn
import math
import torch.nn.functional as F

class DiscreteEmbedding(nn.Module):
    def __init__(self,output_dims,num_points,device='cuda'):
        super(DiscreteEmbedding, self).__init__()
        self.output_dims = output_dims
        self.num_points = num_points

        self.time_embedding = nn.Parameter(torch.Tensor(self.num_points,self.output_dims))
        self.points = torch.arange(0,self.num_points).to(device)
        #self.time_embedding = nn.Embedding(self.num_points,self.output_dims)
        #self.one_hot_embedding = torch.Tensor(torch.zeros([x.shape[0],x.shape[1],self.num_points]))

    def _rect_window(self, x, window_size=8):
        w_2 = window_size / 2
        return (torch.sign(x + w_2) - torch.sign(x - w_2)) / 2


    def forward(self,x): # [N,L]

        x *= self.num_points
        x = x.unsqueeze(-1)
        w = x - self.points
        w = self._rect_window(w, window_size=1)

        output = torch.matmul(w, self.time_embedding)
        return output






class ContinuousEmbedding(nn.Module):
    def __init__(self,output_dims,num_points,minval=-1.0,maxval=1.0,window_size = 8,window_type='hann',normalized = True,device='cpu'):
        super(ContinuousEmbedding,self).__init__()
        self.output_dims = output_dims
        self.minval = minval
        self.maxval = maxval
        self.num_points = num_points
        self.window_size = window_size
        assert window_type in {'triangular', 'rectangular', 'hann'}
        self.window_type = window_type
        self.normalized = normalized

        self.embedding = nn.Parameter(torch.Tensor(self.num_points,self.output_dims))
        self.embedding_dim = self.output_dims

        if self.window_type == 'hann':
            self.window_func = self._hann_window
        elif self.window_type == 'triangular':
            self.window_func = self._triangle_window
        else:
            self.window_func = self._rect_window

        self.points = torch.arange(0,self.num_points).to(device)

    def _rect_window(self,x,window_size = 8):
        w_2 = window_size / 2
        return (torch.sign(x+w_2)-torch.sign(x-w_2))/2

    def _triangle_window(self, x, window_size=16):
        w_2 = window_size / 2
        return (torch.abs(x + w_2) + torch.abs(x - w_2) - 2 * torch.abs(x)) / window_size

    def _hann_window(self, x, window_size=16):
        y = torch.cos(math.pi * x / window_size)
        y = y * y * self._rect_window(x, window_size=window_size)
        return y

    def forward(self,x): # x:[N,L]
        x -= self.minval
        x *= self.num_points/(self.maxval-self.minval)
        x = x.unsqueeze(-1)
        #print('x.type',type(x))
        #print('point.type',type(self.points))
        w = x-self.points
        w = self.window_func(w,window_size=self.window_size)
        if self.normalized:

            w = w/w.sum(-1,keepdim=True)

        output = torch.matmul(w,self.embedding)
        return output


class Encoder_layer(nn.Module):
    def __init__(self,block_nums,hidden_units,head_num,dropout_rate,if_point_wise=False,if_gelu=False):
        super(Encoder_layer, self).__init__()

        self.block_nums = block_nums
        self.hidden_units = hidden_units
        self.head_num = head_num
        self.dropout_rate = dropout_rate
        self.if_point_wise = if_point_wise
        self.if_gelu = if_gelu

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)

        for _ in range(self.block_nums):
            new_attn_layernorm = torch.nn.LayerNorm(self.hidden_units,eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            #print(self.hidden_units,self.head_num)
            new_attn_layer = torch.nn.MultiheadAttention(self.hidden_units,
                                                         self.head_num,
                                                         self.dropout_rate)

            self.attention_layers.append(new_attn_layer)
            new_fwd_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            new_fwd_layer = PointWiseFeedForward(self.hidden_units, self.dropout_rate,self.if_point_wise,self.if_gelu)
            self.forward_layers.append(new_fwd_layer)

    def forward(self,seqs,attn_mask=None,key_padding_mask=None):
        for i in range(self.block_nums):
            seqs = torch.transpose(seqs,0,1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs,_ = self.attention_layers[i](Q,seqs,seqs,
                                                     attn_mask = attn_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0 ,1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~key_padding_mask.unsqueeze(-1)

        output = self.last_layernorm(seqs)
        return output

class Decoder_layer(nn.Module):
    def __init__(self, hidden_units, head_num, block_num,dropout_rate,if_point_wise=False,if_gelu=False):
        super(Decoder_layer, self).__init__()
        #self.block_nums = block_nums
        self.hidden_units = hidden_units
        self.head_num = head_num
        self.dropout_rate = dropout_rate
        self.block_num = block_num
        self.if_point_wise = if_point_wise
        self.if_gelu = if_gelu

        #self.attention_layernorms1 = nn.LayerNorm(self.hidden_units, eps=1e-8)
        #self.attention_layernorms2 = nn.LayerNorm(self.hidden_units, eps=1e-8)
        #self.self_attention_layers = nn.MultiheadAttention(self.hidden_units,
        #                                                 self.head_num,
        #                                                 self.dropout_rate)
        #self.mutil_attention_layers = nn.MultiheadAttention(self.hidden_units,
        #                                                 self.head_num,
        #                                                 self.dropout_rate)
        #self.forward_layers = PointWiseFeedForward(self.hidden_units, self.dropout_rate)
        #self.dropout1 = nn.Dropout(self.dropout_rate)
        #self.dropout2 = nn.Dropout(self.dropout_rate)
        #self.last_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)

        self.attention_layernorms1 = nn.ModuleList()
        self.attention_layernorms2 = nn.ModuleList()
        self.self_attention_layers = nn.ModuleList()
        self.mutil_attention_layers = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.dropout1 = nn.ModuleList()
        self.dropout2 = nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)

        for _ in range(self.block_num):
            new_attn_layernorm1 = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)
            self.attention_layernorms1.append(new_attn_layernorm1)
            new_attn_layernorm2 = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)
            self.attention_layernorms2.append(new_attn_layernorm2)

            self_attn_layer = torch.nn.MultiheadAttention(self.hidden_units,
                                                         self.head_num,
                                                         self.dropout_rate)
            self.self_attention_layers.append(self_attn_layer)

            self.dropout1.append(nn.Dropout(self.dropout_rate))
            self.dropout2.append(nn.Dropout(self.dropout_rate))
            mutil_attention_layers = torch.nn.MultiheadAttention(self.hidden_units,
                                                         self.head_num,
                                                         self.dropout_rate)
            self.mutil_attention_layers.append(mutil_attention_layers)

            new_fwd_layer = PointWiseFeedForward(self.hidden_units, self.dropout_rate,self.if_point_wise,self.if_gelu)
            self.forward_layers.append(new_fwd_layer)

    def forward(self,tgt,memory,attn_mask=None,key_padding_mask=None,memory_mask=None):
        memory = torch.transpose(memory, 0, 1)
        for i in range(self.block_num):
            tgt = torch.transpose(tgt, 0, 1)
            #Q = self.attention_layernorms(tgt)
            mha_outputs, _ = self.self_attention_layers[i](tgt, tgt, tgt,
                                                  attn_mask=attn_mask)
            tgt = tgt + self.dropout1[i](mha_outputs)

            tgt =  self.attention_layernorms1[i](tgt)
            if key_padding_mask is not None:
                tgt = torch.transpose(tgt, 0, 1)
                tgt *= ~key_padding_mask.unsqueeze(-1)
                tgt = torch.transpose(tgt,0, 1)

            tgt2,_ = self.mutil_attention_layers[i](tgt,memory,memory,attn_mask=memory_mask)
            #tgt2, _ = self.mutil_attention_layers[i](tgt, memory, memory,attn_mask=None)
            tgt = tgt + self.dropout2[i](tgt2)
            tgt = torch.transpose(tgt, 0, 1)
            tgt = self.attention_layernorms2[i](tgt)
            tgt = self.forward_layers[i](tgt)
            tgt *= ~key_padding_mask.unsqueeze(-1)
        tgt = self.last_layernorm(tgt)
        return tgt

class TAT4Rec(nn.Module):
    def __init__(self,user_num,item_num,args):
        super(TAT4Rec,self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.lag_from_now_emb = ContinuousEmbedding(output_dims=args.hidden_units,
                                                    num_points=args.lag_time_bins,
                                                    minval=0.0, maxval=1.0,
                                                    window_size=args.lagtime_window_size,
                                                    window_type='hann',
                                                    normalized=True,
                                                    device=args.device
                                                    )
        #self.discrete_timeembedding = DiscreteEmbedding(output_dims=args.hidden_units,num_points=args.lag_time_bins,device=args.device)


        self.pos_emb = nn.Embedding(args.maxlen, args.hidden_units_item)
        self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units_item, padding_idx=0)
        self.encoder = Encoder_layer(args.encoder_blocks_num,args.hidden_units,args.encoder_heads_num,args.dropout_rate_encoder,args.if_point_wise_feedforward)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate_decoder)

        self.decoder = Decoder_layer(args.hidden_units, args.decoder_heads_num, args.decoder_blocks_num,args.dropout_rate_decoder,args.if_point_wise_feedforward)
        self.max_len = args.maxlen


    def log2feats(self, user_ids,log_seqs,seq_ln,max_time_lag):

        item = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        #print('item_shape1:', item.shape)
        item *= self.item_emb.embedding_dim ** 0.5

        positions = np.tile(np.array(range(item.shape[1])), [item.shape[0], 1])
        item += self.pos_emb(torch.LongTensor(positions).to(self.dev))  # 加上postional embedding [N,L,E]
        item = self.emb_dropout(item)

        seq_ln = torch.FloatTensor(seq_ln).to(self.dev)
        #seq_ln = torch.FloatTensor(seq_ln)
        seq_ln = torch.clamp(seq_ln,min=0,max=max_time_lag)
        seq_ln = seq_ln/max_time_lag
        #seq_ln = torch.LongTensor(seq_ln).to(self.dev)
        seq_ln = self.lag_from_now_emb(seq_ln)  # [N,L,E]
        #seq_ln = self.discrete_timeembedding(seq_ln)
        #print('seqln:',seq_ln.shape)
        #if self.if_scale:
        #    seq_ln *= self.lag_from_now_emb.embedding_dim ** 0.5

        padding_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)  # [N,L]
        tl = item.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))  # [L,L]
        encoder_output = self.encoder(seq_ln,attn_mask = attention_mask,key_padding_mask =padding_mask )
        decoder_output = self.decoder(item,encoder_output,attn_mask= attention_mask,key_padding_mask=padding_mask,memory_mask=attention_mask)
        return decoder_output

    def forward(self, user_ids, log_seqs, seq_ts,seq_ln,pos_seqs, neg_seqs,x_mask,max_time_lag):
        o = self.log2feats(user_ids,log_seqs,seq_ln,max_time_lag)

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))  # 目标 (N,L,E)
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))  # 负采样 (N,L,E)

        pos_logits = (o * pos_embs).sum(dim=-1)
        neg_logits = (o * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, seq_ln,item_indices,max_time_lag): # for inference
        log_feats = self.log2feats(user_ids,log_seqs,seq_ln,max_time_lag) # user_ids hasn't been used yet [1,L,E]
        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste [1,E]
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # [101,E]
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1) # [1,101]

        return logits # preds # (U, I)




class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate,if_point_wise=False,if_gelu=False):

        super(PointWiseFeedForward, self).__init__()
        self.if_point_wise = if_point_wise
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.if_gelu = if_gelu
        if if_gelu:
            self.act = torch.nn.GELU()
        else:
            self.act = torch.nn.ReLU()
        #self.relu = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)
        if  self.if_point_wise:
            self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
            self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        else:
            self.feedforward1 = torch.nn.Linear(hidden_units,hidden_units)
            self.feedforward2 = torch.nn.Linear(hidden_units, hidden_units)

    def forward(self, inputs):
        if self.if_point_wise:
            outputs = self.dropout2(self.conv2(self.act(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
            outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
            outputs += inputs # 注意此处有残差机制
        else:
            outputs = self.dropout2(self.feedforward2(self.act(self.dropout1(self.feedforward1(inputs)))))
            outputs += inputs  # 注意此处有残差机制
        return outputs
