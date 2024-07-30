'''
A complete implementation version containing all code (including ablation components)
'''
import numpy as np
import torch
import torch.nn as nn
from layers.RevIN import RevIN
from models.model import *
from einops import rearrange, repeat, einsum

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
            nn.Linear(self.seq_len, self.d_model),
        )

        self.linear2 = nn.Sequential(
            nn.Linear(self.seq_len * self.seg_num_x, self.d_model),
        )

        self.softmax_operation = nn.Sequential(
            nn.Softmax(dim = -1),
        )
        self.linear3 = nn.Sequential(
            nn.Linear(1, self.seg_num_x),
        )
        self.linear4 = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.enc_in, self.d_model),
        )
        self.local_length1 = 4
        self.linear5 = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.local_length1, self.local_length1),
        )
        #model statements: from (b, c, l) to (b, c, l)
        args = ModelArgs(d_model = self.enc_in, n_layer = 1, vocab_size = configs.enc_in) #
        self.model = Mamba(args)

    def forward(self, x):

        # normalization and permute     b,s,c : 256, 720, 7
        if self.revin:
            x = self.revinLayer(x, 'norm')
        else:
            seq_last = x[:, -1:, :].detach()
            x = (x - seq_last)

        # b:batch_size c:channel_size s:seq_len s:seq_len
        # d:d_model w:seg_len n:seg_num_x m:seg_num_y

        #x with shape like batch_size, feature_num, seq_len
        batch_size = x.size(0)
        # x shape: 256, 720, 7

        #try to implement the auto seg part
        #create the seg tensor with length seg_num_x

        #we need to create a matrix here for matrix multipuly so we can keep the grad operation

        #get through the softmax part to get the probablity
        

        x = x.permute(0, 2, 1) # x_shape: 256, 7, 720
        x = x.unsqueeze(-1) # x_shape: 256, 7, 720, 1
        #get through the layer with a linear layer
        seg_tensor = self.linear3(x) # 256, 7, 720, 15

        '''
        # # We need to calculate the grad here
        # #create a tensor to store the tensor
        # shape = (batch_size, self.enc_in, self.d_model) # 256, 7, 512
        # storage = torch.zeros(shape) # 256, 7, 512
        #
        # #then we get the segment operation
        # for i in range(self.seg_num_x):# we get through the seg_x_length tensor
        #     # floor the seg_x_length
        #     # seg_x_length[: self.seg_num_x - 1].floor_()
        #     # sum_int_part = torch.sum(seg_x_length[: self.seg_num_x - 1])
        #     # seg_x_length[self.seg_num_x - 1] = self.seq_len - sum_int_part
        #     if i == 0:
        #         x_clone = x[:, :, : int(seg_x_length[i].item())].clone()
        #         zero_tensor = torch.zeros_like(x)
        #         zero_tensor[:, :, : int(seg_x_length[i].item())].add_(x_clone) # 256, 7, 720
        #         #get through the linear function
        #         result_tensor = self.linear1(zero_tensor)# 256, 7 ,512
        #         storage = storage.to(result_tensor.device) + result_tensor # 256, 7, 512
        #         storage = storage.unsqueeze(1) # 256, 1, 7, 512
        #     else:
        #         x_clone = x[:, :, int(torch.sum(seg_x_length[ : i - 1]).item()): int(torch.sum(seg_x_length[ : i - 1]).item() + seg_x_length[i].item())].clone()
        #         zero_tensor = torch.zeros_like(x)
        #         zero_tensor[:, :, int(torch.sum(seg_x_length[ : i - 1]).item()): int(torch.sum(seg_x_length[ : i - 1]).item() + seg_x_length[i].item())].add_(x_clone)  # 256, 7, 720
        #         # get through the linear function
        #         result_tensor = self.linear1(zero_tensor)  # 256, 7 ,512
        #         result_tensor = result_tensor.unsqueeze(1) # 256, 1, 7, 512
        #
        #         storage = torch.cat((storage, result_tensor), dim=1)
        '''
        #define the value
        storage = seg_tensor # 256, 7, 720, 15
        storage_middel = seg_tensor.reshape(batch_size, self.enc_in, self.seq_len * self.seg_num_x) # 256,7, 720 * 15
        storage_hidden = self.linear2(storage_middel)# 256, 7, 512
        
        storage = storage.permute(0, 1, 3, 2) # 256, 7, 15, 720

        #add with the origin data
        storage_hidden = storage_hidden.reshape(1, batch_size * self.enc_in, self.d_model)# 1, 256 * 7, 512

        #use the linear operation
        storage = self.linear1(storage) #256, 7, 15, 512





        # we also need to change the postion of the normalization and permute layer with the layer above(done)


        # segment and embedding    b,c,s -> bc,n,w -> bc,n,d
        # this place, we don't use the the c(feature_num) explicitly
        #reshape : 256 * 7, 15, 48
        #value embedding : 256 * 7, 15, 512
        # x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len))
        storage = storage.reshape(batch_size * self.enc_in, self.seg_num_x, self.d_model) # 256 * 7, 15, 512

        # encoding
        if self.rnn_type == "lstm":
            _, (hn, cn) = self.rnn(storage)
        else:
            _, hn = self.rnn(storage) # bc,n,d  1,bc,d

        #hn : num_layers * num_directions, batch_size(here we get batch_size * channel_size), hidden_size(d_model)
        #hn : 1, 256 * 7, 512

        #combine the hidden state

        hn = torch.add(hn, storage_hidden)# 1, 256 * 7, 512

        # decoding
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
            # we find it, the channel size is recorded as self.enc_in
            y = torch.stack(y, dim=1).squeeze(2).reshape(-1, self.enc_in, self.pred_len) # b,c,s

        # hn : 1, 256 * 7, 512
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
                # 15, 512 --> 256 * 7 * 15, 1, 512
                pos_emb = self.pos_emb.repeat(batch_size * self.enc_in, 1).unsqueeze(1)

            # pos_emb: m,d -> bcm,d ->  bcm,1,d  256 * 7 * 15, 1, 512
            # hn, cn: 1,bc,d -> 1,bc,md -> 1,bcm,d # hn : 1, 256 * 7 , 512 --> 1, 256 * 7, 512 * 15 --> 1, 256 * 7 * 15, 512
            if self.rnn_type == "lstm":
                _, (hy, cy) = self.rnn(pos_emb,
                                       (hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model),
                                        cn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)))
            else:
                _, hy = self.rnn(pos_emb, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model))
            # hy: num_layers * num_directions,batch_size, hidden_size, : 1, 256 * 7 * 15, 512
            # 1,bcm,d -> 1,bcm,w -> b,c,s
            # predict : 1, 256 * 7 * 15, 512 -->1, 256 * 7 * 15 , 48
            #reshape: 1, 256 * 7 * 15 , 48 --> 256, 7, 720
            y = self.predict(hy).view(-1, self.enc_in, self.pred_len)

        # permute and denorm
        if self.revin:
            y = self.revinLayer(y.permute(0, 2, 1), 'denorm')
        else:
            y = y.permute(0, 2, 1) + seq_last

        return y

'''
class Configs:
    def __init__(self):
        self.seq_len = 720  # Sequence length
        self.pred_len = 720  # Prediction length (for the output sequence)
        self.enc_in = 7      # Number of features (channels) of the input
        self.d_model = 512   # Number of features in the hidden state
        self.dropout = 0.1   # Dropout rate
        self.rnn_type = 'lstm'  # Type of RNN ('rnn', 'gru', or 'lstm')
        self.dec_way = 'pmf'    # Decoding way ('rmf' or 'pmf')
        self.seg_len = 48       # Segment length
        self.channel_id = True  # Whether to use channel ID in decoding
        self.revin = True       # Whether to use reversible instance normalization
        self.batch_size = 256

# Create an instance of the configurations.
configs = Configs()

# Instantiate the model with the specified configurations.
model = Model(configs)
# Generate random input data (batch_size x seq_len x num_features).
x = torch.randn(256, configs.seq_len, configs.enc_in)  # For example, a batch size of 256

# Generate random target data with the correct dimensions.
y_true = torch.randn(256, configs.pred_len, configs.enc_in)

# Choose a loss function, e.g., mean squared error for a regression task.
criterion = nn.MSELoss()

# Choose an optimizer, e.g., Adam.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Set the model to training mode.
model.train()

# Training loop for a certain number of epochs.
num_epochs = 5  # This is just an example. Typically, you'd use many more epochs.

for epoch in range(num_epochs):
    # Forward pass: Compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute the loss.
    loss = criterion(y_pred, y_true)

    # Print the loss every epoch (or every few iterations).
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
'''