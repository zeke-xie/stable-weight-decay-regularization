# Scheduled(Stable)-Weight-Decay-Regularization

The PyTorch Implementation of Scheduled (Stable) Weight Decay.

The algorithms were first proposed in our arxiv paper.

A formal version with major revision and theoretical mechanism ["On the Overlooked Pitfalls of Weight Decay and How to Mitigate Them: A Gradient-Norm Perspective"](https://openreview.net/pdf?id=vnGcubtzR1) is accepted at NeurIPS 2023.


# Why Scheduled (Stable) Weight Decay?

We proposed the Scheduled (Stable) Weight Decay (SWD) method to mitigate overlooked large-gradient-norm pitfalls of weight decay in modern deep learning libraries.

- SWD can **penalize the large gradient norms** at the final phase of training.

- SWD usually makes significant improvements over both L2 regularization and decoupled weight decay.

- Simply fixing weight decay in Adam by SWD, with no extra hyperparameter, can usually outperform complex Adam variants, which have more hyperparameters.



# The environment is as bellow:

Python 3.7.3 

PyTorch >= 1.4.0


# Usage

You may use it as a standard PyTorch optimizer.

```python
import swd_optim

optimizer = swd_optim.AdamS(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4, amsgrad=False)
```


# Test performance



| Dataset   | Model       | AdamS                     | SGD M                | Adam                 | AMSGrad              | AdamW                | AdaBound             | Padam                | Yogi                 | RAdam                |
|:----------|:------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| CIFAR-10  | ResNet18    | **4.91**<sub>0.04</sub>  | 5.01<sub>0.03</sub>  | 6.53<sub>0.03</sub>  | 6.16<sub>0.18</sub>  | 5.08<sub>0.07</sub>  | 5.65<sub>0.08</sub>  | 5.12<sub>0.04</sub>  | 5.87<sub>0.12</sub>  | 6.01<sub>0.10</sub>  |
|           | VGG16       | **6.09**<sub>0.11</sub>  | 6.42<sub>0.02</sub>  | 7.31<sub>0.25</sub>  | 7.14<sub>0.14</sub>  | 6.48<sub>0.13</sub>  | 6.76<sub>0.12</sub>  | 6.15<sub>0.06</sub>  | 6.90<sub>0.22</sub>  | 6.56<sub>0.04</sub>  |
| CIFAR-100 | DenseNet121 | 20.52<sub>0.26</sub> | **19.81**<sub>0.33</sub> | 25.11<sub>0.15</sub> | 24.43<sub>0.09</sub> | 21.55<sub>0.14</sub> | 22.69<sub>0.15</sub> | 21.10<sub>0.23</sub> | 22.15<sub>0.36</sub> | 22.27<sub>0.22</sub> |
|           | GoogLeNet   | **21.05**<sub>0.18</sub> | 21.21<sub>0.29</sub> | 26.12<sub>0.33</sub> | 25.53<sub>0.17</sub> | 21.29<sub>0.17</sub> | 23.18<sub>0.31</sub> | 21.82<sub>0.17</sub> | 24.24<sub>0.16</sub> | 22.23<sub>0.15</sub> |

# Citing

If you use Scheduled (Stable) Weight Decay in your work, please cite ["On the Overlooked Pitfalls of Weight Decay and How to Mitigate Them: A Gradient-Norm Perspective"](https://openreview.net/pdf?id=vnGcubtzR1).

```
@inproceedings{xie2023onwd,
    title={On the Overlooked Pitfalls of Weight Decay and How to Mitigate Them: A Gradient-Norm Perspective},
    author={Xie, Zeke and Xu, Zhiqiang and Zhang, Jingzhao and Sato, Issei and Sugiyama, Masashi},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023}
}
```
