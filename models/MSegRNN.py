'''
A complete implementation version containing all code (including ablation components)
'''
import numpy as np
import torch
import torch.nn as nn
from layers.RevIN import RevIN
from models.model import *
from einops import rearrange, repeat, einsum
from mamba_ssm import Mamba as mamba

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # get parameters
        self.batch_size = configs.batch_size
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout

        self.rnn_type = configs.rnn_type
        self.dec_way = configs.dec_way
        self.seg_len = configs.seg_len
        self.channel_id = configs.channel_id
        self.revin = configs.revin
        #we can get the feature_num directly
        #however, this features stands for the task label, it's not the same idea as we need the number of the variate

        assert self.rnn_type in ['rnn', 'gru', 'lstm']
        assert self.dec_way in ['rmf', 'pmf']

        self.seg_num_x = self.seq_len//self.seg_len

        # build model
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )


        if self.rnn_type == "rnn":
            self.rnn = nn.RNN(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)

        if self.dec_way == "rmf":
            self.seg_num_y = self.pred_len // self.seg_len
            self.predict = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model, self.seg_len)
            )
        elif self.dec_way == "pmf":
            self.seg_num_y = self.pred_len // self.seg_len

            if self.channel_id:
                self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))
                self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))
            else:
                self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model))

            self.predict = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model, self.seg_len)
            )
        if self.revin:
            self.revinLayer = RevIN(self.enc_in, affine=False, subtract_last=False)
        #this is for each great block to dimenstion transformation
        self.linear1 = nn.Sequential(
            nn.Linear(1, self.seg_num_x),
        )

        self.linear2 = nn.Sequential(
            nn.Linear(self.seq_len * self.seg_num_x, self.d_model),
        )

        self.linear3 = nn.Sequential(
            nn.Linear(self.seq_len, self.d_model),
        )

        #model statements: from (b, c, l) to (b, c, l)
        # use or not use conv
        if configs.conv:
            self.model = mamba(d_model=self.enc_in, # Model dimension d_model
                                    d_state=configs.d_state,  # SSM state expansion factor
                                    d_conv=2,    # Local convolution width
                                    expand=2,).to("cuda")
        else:
            args = ModelArgs(d_model = self.enc_in, n_layer = 1, vocab_size = configs.enc_in, d_state = configs.d_state) #
            self.model = Mamba(args)

    def forward(self, x):

        # normalization and permute
        if self.revin:
            x = self.revinLayer(x, 'norm')
        else:
            seq_last = x[:, -1:, :].detach()
            x = (x - seq_last)

        # b:batch_size c:channel_size s:seq_len s:seq_len
        # d:d_model w:seg_len n:seg_num_x m:seg_num_y

        #x with shape like batch_size, feature_num, seq_len
        batch_size = x.size(0)
        # x shape: b,l,c

        # get through the mamba layer
        x = self.model(x) # b,l,c
        

        x = x.permute(0, 2, 1) # x_shape: b,c,l
        x = x.unsqueeze(-1) # x_shape: b,c,l,1
        #get through the layer with a linear layer
        seg_tensor = self.linear1(x) # b,c,l,n


        # record the vector, one part go through residual block and the other go through rnn encoding block
        storage = seg_tensor # b,c,l,n

        # the residual block, which storage_hidden is used in RNN output to get final hidden state.
        storage_middel = seg_tensor.reshape(batch_size, self.enc_in, self.seq_len * self.seg_num_x) # b,c, l * n
        storage_hidden = self.linear2(storage_middel)# b,c,d
        storage_hidden = storage_hidden.reshape(1, batch_size * self.enc_in, self.d_model)# 1, b * c, d
        # end of residual block

        # the implicit segment block
        storage = storage.permute(0, 1, 3, 2) # b, c, n, l
        storage = self.linear3(storage) #b, c, n, d
        storage = storage.reshape(batch_size * self.enc_in, self.seg_num_x, self.d_model) # b * c, n, d
        # end of implicit segment block


        # encoding block
        if self.rnn_type == "lstm":
            _, (hn, cn) = self.rnn(storage)
        else:
            _, hn = self.rnn(storage) # bc,n,d  1,bc,d
        #hn : 1, b * c, d
        # end of encoding block

        # add the vector from RNN output and Residual block
        hn = torch.add(hn, storage_hidden)# 1, b * c, d

        # decoding block
        if self.dec_way == "rmf":
            y = []
            for i in range(self.seg_num_y):
                yy = self.predict(hn)    # 1,bc,l
                yy = yy.permute(1,0,2)   # bc,1,l
                y.append(yy)
                yy = self.valueEmbedding(yy)
                if self.rnn_type == "lstm":
                    _, (hn, cn) = self.rnn(yy, (hn, cn))
                else:
                    _, hn = self.rnn(yy, hn)
            y = torch.stack(y, dim=1).squeeze(2).reshape(-1, self.enc_in, self.pred_len) # b,c,s

        #hn : 1, b * c, d
        elif self.dec_way == "pmf":
            if self.channel_id:
                # m,d//2 -> 1,m,d//2 -> c,m,d//2
                # c,d//2 -> c,1,d//2 -> c,m,d//2
                # c,m,d -> cm,1,d -> bcm, 1, d
                pos_emb = torch.cat([
                    self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
                    self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
                ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size,1,1)
            else:
                # m,d -> bcm,d -> bcm, 1, d
                pos_emb = self.pos_emb.repeat(batch_size * self.enc_in, 1).unsqueeze(1)

            # pos_emb: m,d -> bcm,d ->  bcm,1,d
            # hn, cn: 1,bc,d -> 1,bc,md -> 1,bcm,d
            if self.rnn_type == "lstm":
                _, (hy, cy) = self.rnn(pos_emb,
                                       (hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model),
                                        cn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)))
            else:
                _, hy = self.rnn(pos_emb, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model))
            # hy: num_layers * num_directions,batch_size, hidden_size, : 1, 256 * 7 * 15, 512
            # 1,bcm,d -> 1,bcm,w -> b,c,s
            y = self.predict(hy).view(-1, self.enc_in, self.pred_len)

        #end of decoding block

        # permute and denorm
        if self.revin:
            y = self.revinLayer(y.permute(0, 2, 1), 'denorm')
        else:
            y = y.permute(0, 2, 1) + seq_last
        return y

