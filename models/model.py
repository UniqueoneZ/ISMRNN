# minimal implementation of Mamba
# the parameter meaning
'''
b : batch_size
l : sequence length
d or d_model : the hidden dim
n or d_state : latent state dim
expand : expansion factor
d_in or d_inner : d * expand
A, B, C, D : state space parameters
Δ or delta : input-dependent step size
dt-rank : rank of Δ
'''

# import the base codebase
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from typing import Union


# define data class, the d_rank is set to be 1 and
@dataclass
class ModelArgs:
    d_model: int # 720
    n_layer: int # 2
    vocab_size: int # 512
    d_state: int
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    pad_vocab_size_multiple: int = 2
    bias: bool = False

    # define preprocess operations
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)

        if self.dt_rank == 'auto':
            self.dt_rank = 1

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple - self.vocab_size % self.pad_vocab_size_multiple)


# define the mamba structure
class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        """Full Mamba model."""
        super().__init__()
        self.args = args
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
    def forward(self, input_ids):

        x = input_ids
        for layer in self.layers:
            x = layer(x)  # (b, l, d_model)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """
        A block that wraps mamba block with normalization and residual connection
        """
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)

    def forward(self, x):
        """
        Args:
            x: shape(b, l, d)

        Returns:
            output: shape(b, l, d)
        """

        # the process way: norm, mamba, add

        # we drop the conv layer and the normalization part, leaving the local information and normalization part be handle by MSegRNN
        output = self.mixer(x) + x

        return output


class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """A single Mamba block"""
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        # x_proj takes in 'x' and outputs the input-specific Δ, B, C
        # dt_rank stands for the Δ, d_state stands for B and C, which is of the same size
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)

        # dt_proj projects Δ from dt_rank to d_in
        # dt_rank stands for Δ and d_inner stands for A, d_inner = d_model * expand
        # which d_model stands for hidden dim and d_state stands for latent state dim
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        # first we generate a new list with length n(latent state dim)
        # then we repeat it d_inner（d_model * expand） times to get d_inner * d_state(or n)
        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner) # (1024, 32)

        # then we discretize A and make it trainable
        self.A_log = nn.Parameter(torch.log(A))  # d_inner * d_state (1024, 32)
        self.D = nn.Parameter(torch.ones(args.d_inner))  # d_inner 1024
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        """
        Mamba block forward

        Args:
            x: shape(b, l, d)

        Returns:
            output: shape(b, l, d)
        """

        # get the shape parameter first
        (b, l, d) = x.shape

        # from h(t-1) to A, one part is taken as the x and the res
        x_and_res = self.in_proj(x)  # shape(b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)


        # get through the activation layer， silu activation is useful in deeper network
        x = F.silu(x)  # (b, l, d_in)

        # get throught the ssm layer, this layer makes no change on the dimension
        y = self.ssm(x)  # (b, l, d_in)

        # multiple with the res structure
        y = y * F.silu(res)  # multiple the corresponding place (b, l, d_in)

        # make the final projection to get h(t)
        output = self.out_proj(y)  # (b, l, d_model)

        return output

    # define the SSM structure
    def ssm(self, x):
        """
        Implement of SSM structure

        Args:
            x: shape(b, l, d_in)

        Returns:
            output: shape(b, l ,d_in)
        """
        (d_in, n) = self.A_log.shape

        # compute Δ, A, B, C, D the state spacce parameters
        # A, D are input independent
        # Δ, B, C are input-dependent

        A = -torch.exp(self.A_log.float())  # shape(d_in, n)
        D = self.D.float()  # d_in

        x_dbl = self.x_proj(x)  # (b, l ,dt_rank + 2 * n)

        # assign x_dbl to Δ, B, C
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
        # delta (b, l, dt_rank), B and C:(b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        # this function makes no difference to the dimension

        y = self.selective_scan(x, delta, A, B, C, D)

        return y

    # we now trying to discover the selective scan operation, which is the kernel operation of the model
    def selective_scan(self, u, delta, A, B, C, D):
        """
        Args:
            u: shape(b, l, d_in)
            delta: shape(b, l, d_in)
            A: shape(d_in, n)
            B: shape(b, l, n)
            C: shape(b, l, n)
            D: shape(d_in,)

        Returns:
            outputs: shape (b, l, d_in)
        """
        (b, l, d_in) = u.shape
        n = A.shape[1] #32

        # Discretize continuous parmeters (A, B)
        # multiple the matrix
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))  # (b, l, d_in, n)

        # multiple the matrix'element on by one of delta and u, then we multiple the matrix with B to get b l d_in
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')  # (b, l, d_in, n)

        # Perform the selective scan
        x = torch.zeros((b, d_in, n), device=deltaA.device)  # (b, d_in, n)
        ys = []
        for i in range(l):
            # Delta is trainingable, makes it selective, however, the dimenstion is worng
            # it should select the length dimenstion instead of the seglength
            x = deltaA[:, i] * x + deltaB_u[:, i]  # (b, d_in, n) * (b, d_in, n) + (b, d_in, n)
            # multiple the corresponding place and sum for n dimension
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')  # (b, d_in)
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)  #

        y = y + u * D

        return y #




