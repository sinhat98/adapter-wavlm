# Abstract
Fine-tuning of self-supervised models is a powerful transfer learning method in a variety of fields, including speech pro- cessing, since it can utilize generic feature representations obtained from large amounts of unlabeled data. Fine-tuning however, requires a new parameter set for each downstream task, which is parameter inefficient. Adapter architecture is proposed to partially solve this issue by inserting lightweight learnable modules into a frozen pre-trained model. However, existing adapter architectures fail to adaptively leverage low- to high-level features stored in different layers, which is necessary for solving various kinds of speech processing tasks. Thus, we propose a new adapter architecture to acquire fea- ture representations more flexibly for various speech tasks. In experiments, we applied this adapter to WavLM on four speech tasks. It performed on par or better than na ̈ıve fine-tuning with only 11% of learnable parameters. It also outperformed an existing adapter architecture.


# Adapter Architecture
<img width="337" alt="layeradapter" src="https://user-images.githubusercontent.com/48460458/189790212-b1863b1a-c985-4e1f-86a4-363cd9f31ffc.png">
<!-- ![layeradapter5](https://user-images.githubusercontent.com/48460458/189691649-a3fd4264-3de6-4e61-abe5-3c60a67b07ac.png) -->

The proposed adapter architecture incorporates two types of adapters, namely Layer adapters (L-adapters) and Encoder adapter (E-adapters), into a frozen backbone. The L-adapters bridge each intermediate layer and the top layer as shown in Figure 1a. They help the model to quickly adapt speech representations to various downstream tasks, and also to reduce dependency on the initialization of adapter parameters. The E-adapters are inserted to each encoder layer in a similar way as previous work (https://arxiv.org/pdf/2202.03218.pdf) as shown in Figure 1b. In contrast to the previous work, our architecture does not have adapters after the multi-head self-attention (MHSA) modules, and alternatively has L-adapters.
We use wavlm-base-plus as the model backbone.

# Installation and Running experiments
You need to install packages necessary for running experiments. Please run the following command.
```python
pip install -r requirement.txt
```

The following command provides an example of training in the proposed method. Please select task_name from ASV, ER, ASR, or IC.
```sh
# ./run.sh task_name
./run.sh ASR
```

# Experiment and Results
<!-- ![result](https://user-images.githubusercontent.com/48460458/189800739-e711e953-9095-45d6-bdec-f509581965bb.png) -->

We demonstrate the effectiveness of the proposed method on
four downstream tasks: automatic speaker verification (ASV), emotion recognition (ER), automatic speech recognition (ASR) and intent classification (IC). 
We conduct experiments to compare the performance of different five training methods in four tasks.

We run experiments on five training methods as follows.
    <li> Fine-tuning the top $l$ layers for $l = 1, 2, \dots ,12$. 
    <li> Conventional method: Adapters are inserted after MHSA and feedforward modules in the top $l$ layers of for $l = 1, 2, \dots , 12$.
    <li> Proposed method: L-adapters are attached to the top $k$ layers for $k = 1, 2, \dots , 12$ and E-adapters are inserted in the $l$ layers from the second layer from the top for $l = 1, 2, \dots, 11$.
    <li> L-adapters-only: L-adapters are attached to all layers without E-adapters.
    <li> E-adapters-only: E-adapters are inserted into all layers without L-adapters.

The performance comparison is shown in the figure and the table below. The table shows the error rate values of the right ends of curves in the figure.

![result](https://user-images.githubusercontent.com/48460458/190456712-4f6252a6-f931-45ee-bea1-68a022d7738f.png)


| Method              | # Params | ASV                                 | ER                                     | ASR                      | IC                        | 
| ------------------- | -------- | --------------------------------------- | ----------------------------------------- | ---------------------------- | ----------------------------- | 
| Fine-tuning         | 85.1 M   | $4.42\pm 0.25$                          | $21.0 \pm 0.62$                          | $\boldsymbol{7.87} \pm 0.08$ | $0.35 \pm 0.08$             | 
| Conventional method | 9.53 M   | $3.95 \pm0.29$                          | $20.8 \pm 0.44$                          | $8.92 \pm 0.13$              | $0.39 \pm 0.04$             | 
| Proposed method     | 9.13 M   | $\boldsymbol{2.63}\pm 0.09$ | $\boldsymbol{20.0} \pm 0.31$ | $7.90 \pm 0.06$              | $\boldsymbol{0.33} \pm 0.04$             | 
| L-adapters-only     | 4.74 M   | $2.74\pm 0.09$                          | $21.1 \pm 0.52$                          | $9.50 \pm 0.08$              | $\boldsymbol{0.33}\pm 0.04$ | 
| E-adapters-only     | 4.79 M   | $4.82\pm 0.02$                          | $23.1 \pm 0.48$                          | $9.00 \pm 0.16$              | $0.34 \pm 0.04$             | 



## Optimal learning rates
We used a scheduler that warms to the maximum
learning rate and then decays for ASV, ASR all but the
conventional method, and IC. A scheduler that decays every
certain step size from the initial learning rate is used for the
others. We chose the best maximum and initial learning rates
from {1e-3, 5e-4, 1e-4, 5e-5, 1e-5} for each architecture all but the down stream head.


| Method              | Module                                                         | ASV                                    | ER                                     | ASR                              | IC                                      | 
| ------------------- | -------------------------------------------------------------- | -------------------------------------- | -------------------------------------- | -------------------------------- | --------------------------------------- | 
| Fime-tuning         | Downstream head <br>Encoder                                    | 5e-4<br>1e-4                       | 5e-4<br>5e-5                      | 1e-2<br>1e-4                   | 5e-4<br>1e-4                        | 
| Conventional method | Downstream head<br>Adapters<br>Layernorm layer                 | 5e-4<br>1e-5<br>1e-5           | 5e-4<br>1e-5<br>1e-5           | 1e-3<br>1e-5<br>1e-5      | 5e-4<br>1e-5<br>1e-5            | 
| Proposed method     | Downstream head<br>L-adapters<br>E-adapters<br>Layernorm layer | 5e-4<br>1e-4<br>1e-5<br>1e-5 |5e-4 <br>1e-4<br>5e-5<br>5e-5 | 2e-3<br>1e-3<br>1e-3<br>1e-3 | 5e-4<br>1e-5 <br>1e-5 <br>1e-5 | 
| L-adapters-only     | Downstream head<br>L-adapters<br>Layernorm layer               | 5e-4<br>5e-4<br>5e-4             | 5e-4<br>5e-4<br>5e-4             | 2e-3<br>1e-3<br>1e-3          | 5e-4<br>1e-4<br>1e-4              | 
| E-adapters-only     | Downstream head<br>E-adapters<br>Layernorm layer               | 5e-4<br>1e-5<br>1e-5           | 5e-4<br>1e-5<br>1e-5           | 2e-3<br>1e-5<br>1e-5      | 5e-4<br>1e-5<br>1e-5            | 
