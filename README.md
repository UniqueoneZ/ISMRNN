# MSegRNN

This is the official implement of MSegRNN, a Enhanced SegRNN model in "MSegRNN:Enhanced SegRNN Model with Mamba for Long-Term Time Series Forecasting". The main code framework is from the official SegRNN code available at:https://github.com/lss-1138/SegRNN. And the official minimal implementation of Mamba can be found at:https://github.com/johnma2006/mamba-minimal. We would like to express our sincere gratitude.

We made several changes in SegRNN models to enhance it's behavior when the look-back windows is short:

- We add the mamba structure to preprocessing the time series data.
- We replace the fixed segmentation of SegRNN into implicit Segmentation, which allows a dense information processing in the Segmentation Stage.
- We add a residual block from time series to encoder output, reduce the information loss in the RNN structure.

The overall structure is shown as follow:
![](image/overall_structure.png)

The implicit segmentation we used is shown as follow:
![](image/Implicit_Segmentation.png)

By enhancing these three changes, we conduct the experiments, the result is shown as follow:
![](image/result.jpg)

To use the MSegRNN, first:
- git clone https://github.com/UniqueoneZ/MSegRNN.git.
- cd the MsegRNN file locaion.
- pip install -r requirements.txt
- sh run_main.sh

Note that there is a certain variance in the experimental data for the model, and the hyperparameters have not yet been finely tuned. We will continue to improve this work in the future.

If you find our working useful, you may cite our paper if need:
@article{zhao2024msegrnn,
  title={MSegRNN: Enhanced SegRNN Model with Mamba for Long-Term Time Series Forecasting},
  author={Zhao, GaoXiang and Wang, XiaoQiang},
  journal={arXiv preprint arXiv:2407.10768},
  year={2024}
}
