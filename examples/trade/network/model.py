import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

import torch.nn.functional as F
import math

from ts_model.ns_models.ns_Transformer import NSTransformer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[: x.size(0), :]


class Transformer(nn.Module):
    def __init__(self, d_feat=6, d_model=8, nhead=4, num_layers=2, dropout=0.5, device=None):
        super(Transformer, self).__init__()
        # self.feature_layer = nn.Linear(d_feat, d_model)
        # self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.device = device
        self.d_feat = d_feat

    def forward(self, src, mask = None):
        # src [N, F*T] --> [N, T, F]
        # src = self.feature_layer(src)

        # src [N, T, F] --> [T, N, F], [60, 512, 8]
        src = src.transpose(1, 0)  # not batch first

        # src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)  # [60, 512, 8]

        return output.transpose(1, 0)

class JueNet(nn.Module):
    def __init__(self, d_feat_day=7, d_feat_min=2, d_model=128, nhead=4, num_layers=2, dropout=0.2, device='cuda:0', perfect_info=True, max_min_seq_len=240, **kwargs):
        super(JueNet, self).__init__()
        # self.feature_layer = nn.Linear(d_feat, d_model)
        # self.pos_encoder = PositionalEncoding(d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        # self.encoder_layer_day = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        # self.transformer_encoder_day = nn.TransformerEncoder(self.encoder_layer_day, num_layers=num_layers)

        # self.encoder_layer_min = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        # self.transformer_encoder_min = nn.TransformerEncoder(self.encoder_layer_day, num_layers=num_layers)

        self.device = device
        self.d_feat_day = d_feat_day
        self.d_feat_min = d_feat_min
        self.perfect_info = perfect_info
        print(f"是否是完美信息:{self.perfect_info}")
        self.nhead = nhead
        self.max_min_seq_len = max_min_seq_len

        hidden_size = d_model
        self.position_ids_day = torch.arange(60).expand((1, -1)).cuda()
        self.position_ids_min = torch.arange(max_min_seq_len).expand((1, -1)).cuda()
        self.position_embeddings_day = nn.Embedding(60, hidden_size)
        self.position_embeddings_min = nn.Embedding(max_min_seq_len, hidden_size)


        self.fc_day = nn.Linear(7, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_day.weight)
        self.norm_day = nn.LayerNorm(hidden_size, eps=1e-5, elementwise_affine=True)
        self.fc_min = nn.Linear(2, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_min.weight)
        self.norm_min = nn.LayerNorm(hidden_size, eps=1e-5, elementwise_affine=True)
        self.fc_dec = nn.Linear(4, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_min.weight)
        self.norm_dec = nn.LayerNorm(hidden_size, eps=1e-5, elementwise_affine=True)

        self.fc_bin = nn.Linear(hidden_size, 1)
        torch.nn.init.xavier_uniform_(self.fc_bin.weight)
        self.fc_rank = nn.Linear(hidden_size, 1)
        torch.nn.init.xavier_uniform_(self.fc_rank.weight)

    def inp2tensor(self, inp):
        # public_state: {"day_feat":day_feat, "min_df1":min_df1, "min_df2":min_df2}
        # {"pub_state":self.public_state, "pri_state":list_private_state, "seqlen":seqlen}
        day_feat = inp['pub_state']["day_feat"]
        min_df1 = inp['pub_state']["min_df1"]
        min_df2 = inp['pub_state']["min_df2"]
        pri_state = inp['pri_state']
        # with torch.no_grad():
        x_mask = None
        if not self.perfect_info:
            # df['close'] = (df['close'] / last_close - 1.) * 10.
            # df['amount'] = np.log1p(df['amount']+1e-4) / 10. - 0.45
            x_mask = torch.from_numpy(inp['tgt_mask']).to(self.device)
            with torch.no_grad():
                x_mask = x_mask[:, None, :, :].repeat((1, self.nhead, 1, 1))
                x_mask = x_mask.reshape(-1, x_mask.shape[-2], x_mask.shape[-1])
                x_mask = torch.bmm(x_mask, x_mask.transpose(-1, -2))
            min_df1[inp['tgt_mask']==1] = 0. # 全部等于0算了, 本来是有意思的

            # min_df1[:, seq_len+1:, 1] = - 0.85 # 
        day_feat = torch.from_numpy(day_feat).to(self.device).float()
        min_df1 = torch.from_numpy(min_df1).to(self.device).float()
        min_df2 = torch.from_numpy(min_df2).to(self.device).float()
        pri_state = torch.from_numpy(pri_state).to(self.device).float()
        return day_feat, min_df1, min_df2, pri_state, x_mask

    def forward(self, inp):

        day_feat, min_df1, min_df2, pri_state, mask = self.inp2tensor(inp)
        # src [N, F*T] --> [N, T, F]
        x_day = day_feat.reshape(len(day_feat), self.d_feat_day, -1) # [N, F, T]      
        x_day = x_day.permute(0, 2, 1) # [N, T, F]

        pos_day = self.position_embeddings_day(self.position_ids_day)
        pos_min = self.position_embeddings_min(self.position_ids_min)
        x_day = self.fc_day(x_day) + pos_day
        x_day = self.norm_day(x_day)
        min_df2 = self.fc_min(min_df2)  + pos_min
        min_df2 = self.norm_min(min_df2)

        # 把encoder的coding合并在一起
        x_enc = torch.cat([x_day, min_df2], axis = 1)
        # src [N, T, F] --> [T, N, F], [60, 512, 8]
        x_enc = x_enc.transpose(1, 0)  # not batch first

        x_enc = self.transformer_encoder(x_enc)

        x_dec = torch.cat([min_df1, pri_state], axis=-1)
        x_dec = x_dec.transpose(1, 0)  # not batch first

        x_dec = self.fc_dec(x_dec)
        x = self.transformer_decoder(x_dec, x_enc, tgt_mask=mask)


        return x[-1, :, :]

class HIST_sep(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size


        self.rnn_day = Transformer(
            d_feat = self.hidden_size,
            d_model=self.hidden_size,
            nhead=4,
            dropout=dropout,
            num_layers = num_layers
        )

        self.rnn_min = Transformer(
            d_feat = self.hidden_size,
            d_model=self.hidden_size,
            nhead=4,
            dropout=dropout,
            num_layers = num_layers
        )

        self.rnn_cat = Transformer(
            d_feat = self.hidden_size,
            d_model=self.hidden_size,
            nhead=4,
            dropout=dropout,
            num_layers = num_layers
        )

        self.position_ids_day = torch.arange(60).expand((1, -1)).cuda()
        self.position_ids_min = torch.arange(210).expand((1, -1)).cuda()
        self.position_embeddings_day = nn.Embedding(60, hidden_size)
        self.position_embeddings_min = nn.Embedding(210, hidden_size)


        self.fc_day = nn.Linear(7, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_day.weight)
        self.norm_day = nn.LayerNorm(hidden_size, eps=1e-5, elementwise_affine=True)
        self.fc_min = nn.Linear(2, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_min.weight)
        self.norm_min = nn.LayerNorm(hidden_size, eps=1e-5, elementwise_affine=True)


        self.fc_bin = nn.Linear(hidden_size, 1)
        torch.nn.init.xavier_uniform_(self.fc_bin.weight)
        self.fc_rank = nn.Linear(hidden_size, 1)
        torch.nn.init.xavier_uniform_(self.fc_rank.weight)

        
    def cal_cos_similarity(self, x, y): # the 2nd dimension of x and y are the same
        xy = x.mm(torch.t(y))
        x_norm = torch.sqrt(torch.sum(x*x, dim =1)).reshape(-1, 1)
        y_norm = torch.sqrt(torch.sum(y*y, dim =1)).reshape(-1, 1)
        cos_similarity = xy/x_norm.mm(torch.t(y_norm))
        cos_similarity[cos_similarity != cos_similarity] = 0
        return cos_similarity

    def forward(self, x, min_data):
        # device = torch.device(torch.get_device(x))
        x_day = x.reshape(len(x), self.d_feat, -1) # [N, F, T]      
        x_day = x_day.permute(0, 2, 1) # [N, T, F]

        pos_day = self.position_embeddings_day(self.position_ids_day)
        pos_min = self.position_embeddings_min(self.position_ids_min)
        x_day = self.fc_day(x_day) + pos_day
        x_day = self.norm_day(x_day)
        x_min = self.fc_min(min_data)  + pos_min
        x_min = self.norm_min(x_min)

        x_day = self.rnn_day(x_day)
        x_min = self.rnn_min(x_min)

        pos = torch.cat([pos_day, pos_min], axis=1)
        x_hidden = torch.cat([x_day, x_min], axis=1) + pos
        x_hidden = self.rnn_cat(x_hidden)
        x_hidden = x_hidden[:, -1, :]

        pred_rank = self.fc_rank(x_hidden).squeeze()
        pred_binary = self.fc_bin(x_hidden).squeeze()
        return pred_rank, pred_binary

