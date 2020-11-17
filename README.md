# Stable-Weight-Decay-Fixing-Weight-Decay-in-Deep-Learning-Libraries

The PyTorch Implementation of Stable Weight Decay.

The algorithms are proposed in the paper: 

"Stable Weight Decay Regularization: Fixing Weight Decay in Deep Learning Libraries".


# Why Stable Weight Decay?

We proposed the Stable Weight Decay (SWD) method to fix weight decay in modern deep learning libraries.

SWD make significantly improvements over $L_{2}$ regularization and decoupled weight decay.

Simply fixing weight decay in Adam by SWD, with no extra hyperparameter, can usually outperform complex Adam variants, which have more hyperparameters.


# The environment is as bellow:

Python 3.7.3 

PyTorch >= 1.4.0


# Usage

#You may use it as a standard PyTorch optimizer.

```python
import swd_optim

optimizer = adai_optim.AdamS(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4, amsgrad=True)
```
