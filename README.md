# MSegRNN

This is the official implement of MSegRNN, a Enhanced SegRNN model in "MSegRNN:Enhanced SegRNN Model with Mamba for Long-Term Time Series Forecasting". The main code framework is from the official SegRNN code available at:https://github.com/lss-1138/SegRNN. And the official minimal implementation of Mamba can be found at:https://github.com/johnma2006/mamba-minimal. We would like to express our sincere gratitude.

We made several changes in SegRNN models to enhance it's behavior when the look-back windows is short:

- We add the mamba structure to preprocessing the time series data.
- We replace the fixed segmentation of SegRNN into implicit Segmentation, which allows a dense information processing in the Segmentation Stage.
- We add a residual block from time series to encoder output, reduce the information loss in the RNN structure.
