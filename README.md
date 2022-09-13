<!-- # Introduction
This repository contains a version of WavLM that can be trained using adapters.
Adapters are lightweight modules that can be inserted into some intermediate layers of a frozen pre-trained model. We proposed a novel adapter architecture for multiple speech processing tasks. The proposed adapter architecture produces effective features for downstream tasks to use a weighted sum of outputs of intermediate layers of the WavLM encoder. -->
# Abstract
Fine-tuning of self-supervised models is a powerful transfer learning method in a variety of fields, including speech processing, since it can utilize generic feature representations obtained from large unlabeled data. Fine-tuning, however, requires a new parameter set for each downstream task, which is parameter inefficient. Adapter architectures can improve the parameter efficiency by inserting lightweight learnable modules into a frozen pre-trained model. However, existing adapter architectures fail to adaptively leverage low- to high-level features stored in different layers, which is necessary for tasks other than automatic speech recognition. Thus, we propose a new adapter architecture to acquire feature representations more flexibly for various speech tasks. In experiments, we applied this adapter to WavLM on four speech tasks. It performed on par or better than na ̈ıve fine-tuning, with less than 10% of learnable parameters. It also outperformed an existing adapter architecture.

# Adapter Architecture
<img width="337" alt="layeradapter" src="https://user-images.githubusercontent.com/48460458/189790212-b1863b1a-c985-4e1f-86a4-363cd9f31ffc.png">
<!-- ![layeradapter5](https://user-images.githubusercontent.com/48460458/189691649-a3fd4264-3de6-4e61-abe5-3c60a67b07ac.png) -->
The proposed adapter architecture incorporates two types of adapters, namely Layer adapters (L-adapters) and Encoder adapter (E-adapters), into a frozen backbone. The L-adapters bridge each intermediate layer and the top layer as shown in Figure 1a. They help the model to quickly adapt speech representations to various downstream tasks, and also to reduce dependency on the initialization of adapter parameters. The E-adapters are inserted to each encoder layer in a similar way as previous work ([EFFICIENT ADAPTER TRANSFER OF SELF-SUPERVISED SPEECH MODELS FOR AUTOMATIC SPEECH RECOGNITION](https://arxiv.org/pdf/2202.03218.pdf)) as shown in Figure 1b. In contrast to the previous work, our architecture does not have adapters after the multi-head self-attention (MHSA) modules, and alternatively has L-adapters.
We use wavlm-base-plus as the model backbone.

# Experiment and Results
<!-- ![result](https://user-images.githubusercontent.com/48460458/189800739-e711e953-9095-45d6-bdec-f509581965bb.png) -->
<img width="478" alt="result_cap" src="https://user-images.githubusercontent.com/48460458/189801754-5514febf-d25e-4be9-8375-bf26d2bd1dd5.png">
We demonstrate the effectiveness of the proposed method on
four downstream tasks: automatic speaker verification (ASV), emotion recognition (ER), automatic speech recognition (ASR) and intent classification (IC). 
We conduct experiments to compare the performance of different five training methods in four tasks.
<ol type=i>
    <li>Fine-tuning: Fine-tuning all layers of the WavLM encoder
    <li>Conventional method: Adapters are inserted into after a MHSA module and a feedfoward module in each layer of the WavLM encoder.
    <li>Proposed method: There are L-adapters and E-adapters attached on the WavLM encoder. L-adapters are attached to all layers and E-adapters are inserted into all but the final layer.
    <li> L-Adapters-only: L-adapters are attached to all layers of the WavLM encoder.
    <li> E-adapters-only: E-adapters are inserting into all layers of the WavLM encoder.
</ol>
