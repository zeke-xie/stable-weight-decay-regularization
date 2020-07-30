# Stable-Weight-Decay-Fixing-Weight-Decay-in-Deep-Learning-Libraries

The PyTorch Implementation of Stable Weight Decay.

SGDS, AdaiS, and AdamS are available.

The algorithms are proposed in the paper: 

"Stable Weight Decay: Fixing Weight Decay in Deep Learning Libraries".


# The environment is as bellow:

Python 3.7.3 

PyTorch >= 1.4.0


# Usage

#You may use it as a standard PyTorch optimizer.

```python
import swd_optim

optimizer = adai_optim.AdamS(net.parameters(), lr=1e-3, betas=(0.1, 0.999), eps=1e-03, weight_decay=1e-4, amsgrad=True)
```
