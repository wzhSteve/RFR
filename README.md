# A robust feature reinforcement framework for heterogeneous graphs neural networks
This paper has been accepted by Future Generation Computer Systems


cite this paper as 
@article{WANG2023143,
title = {A robust feature reinforcement framework for heterogeneous graphs neural networks},
journal = {Future Generation Computer Systems},
volume = {141},
pages = {143-153},
year = {2023},
issn = {0167-739X},
doi = {https://doi.org/10.1016/j.future.2022.11.009},
url = {https://www.sciencedirect.com/science/article/pii/S0167739X22003703},
author = {Zehao Wang and Huifeng Wu and Jin Fan and Danfeng Sun and Jia Wu},
keywords = {Heterogeneous graph embedding, Node classification, Contrastive learning, Graph neural networks},
abstract = {In the real world, various kinds of data are able to be represented as heterogeneous graph structures. Heterogeneous graphs with multi-typed nodes and edges contain rich messages of heterogeneity and complex semantic information. Recently, diverse heterogeneous graph neural networks (HGNNs) have emerged to solve a range of tasks in this advanced area, such as node classification, knowledge graphs, etc. Heterogeneous graph embedding is a crucial step in HGNNs. It aims to embed rich information from heterogeneous graphs into low-dimensional eigenspaces to improve the performance of downstream tasks. Yet existing methods only project high-dimensional node features into the same low-dimensional space and subsequently aggregate those heterogeneous features directly. This approach ignores the balance between the informative dimensions and the redundant dimensions in the hidden layers. Further, after the dimensionality has been reduced, all kinds of nodes features are projected into the same eigenspace but in a mixed up fashion. One final problem with HGNNs is that their experimental results are always unstable and not reproducible. To solve these issues, we design a general framework named Robust Feature Reinforcement (RFR) for HGNNs to optimize embedding performance. RFR consists of three mechanisms: separate mapping, co-segregating and population-based bandits. The separate mapping mechanism improves the ability to preserve the most informative dimensions when projecting high-dimensional vectors into a low-dimensional eigenspace. The co-segregating mechanism minimizes the contrastive loss to ensure there is a distinction between the features extracted from different types of nodes in the latent feature layers. The population-based bandits mechanism further assures the stability of the experimental results with classification tasks. Supported by rigorous experimentation on three datasets, we assessed the performance of the designed framework and can verify that our models outperform the current state-of-the-arts.}
}
