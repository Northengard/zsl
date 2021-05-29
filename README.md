# zsl
this repo contains my research in zero short learing task

[Some theory.](Zero_Shot_Learning.md)

# Experiments
## Classification
Setup:
1. Dataset: [omniglot](https://pytorch.org/vision/stable/datasets.html). Train: all except val. Val set:
   1. 'Mkhedruli_(Georgian)',
   2. 'N_Ko',
   3. 'Ojibwe_(Canadian_Aboriginal_Syllabics)',
   4. 'Sanskrit', 'Syriac_(Estrangelo)',
   5. 'Tagalog',
   6. 'Tifinagh'
2. Model: [base_model](models/simple_zsl_classification.py)

| Experiment                                                                                                               | accuracy | precision | recall |
|:-------------------------------------------------------------------------------------------------------------------------|:---------|:----------|:-------|
| [def_cfg.yaml](experiments/def_cfg.yaml)                                                                                 |   94.67  |    98     |  91.31 |
| [omni_bl_vs200_bs128.yaml](experiments/omniglot/omni_bl_vs200_bs128.yaml)                                                         |   92.48  |    99     |  86.63 |
| [omni_bl_vs200_bs128_ep200_lr3e-2.yaml](experiments/omniglot/omni_bl_vs200_bs128_ep200_lr3e-2.yaml)                               |   94.3   |    98.5   |  90    |
| [omni_bl_vs200_bs64_ep200_lr3e-2_plateau.yaml](experiments/omniglot/omni_bl_vs200_bs64_ep200_lr3e-2_plateau.yaml)                 |   95     |    98.5   |  92    |
| [omni_bl_vs200_bs64_ep200_lr4e-2_plateau.yaml](experiments/omniglot/omni_bl_vs200_bs64_ep200_lr4e-2_plateau.yaml)                 |   95.69  |    97.96  |  93.3  |
| [omni_bl_vs200_bs64_ep200_lr4e-2_plateau_margin2.yaml](experiments/omniglot/omni_bl_vs200_bs64_ep200_lr4e-2_plateau_margin2.yaml) |   70     |    63     |  96    |
| [omni_bl_vs300_bs64_ep200_lr4e-2_plateau_margin2.yaml](experiments/omniglot/omni_bl_vs300_bs64_ep200_lr4e-2_plateau_margin2.yaml) |   79     |    75     |  91.3  |
