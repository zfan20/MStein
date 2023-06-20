# MStein
This is the implementation for the paper:
TheWebConf'23. You may find it on [Arxiv](https://arxiv.org/pdf/2301.12197.pdf)

### Update on the theoretical connection between KL divergence and mutual information
The mutual information is not asymmetric but symmetric. I have updated the correct version to arXiv (might take a few days to process) by removing the asymmetric deficiency. However, the paper's conclusions and observations are still valid even without the asymmetric deficiency. Thanks to a reader for pointing out this mistake.

### Example code
`python main.py --model_name=CoDistSAModel --data_name=Beauty --lr=0.001 --hidden_size=64 --max_seq_length=50 --hidden_dropout_prob=0.1 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=2 --attention_probs_dropout_prob=0.0 --substitute_rate=0.5 --insert_rate=0.3 --augment_threshold=16 --augmentation_warm_up_epoches=160 --cf_weight=0.005 --batch_size=1024 --pvn_weight=0.01 --temperature=1e-06`

### Please cite our papers if you use the code:
```bibtex
@inproceedings{fan2023mstein,
  title={Mutual Wasserstein Discrepancy Minimization for Sequential Recommendation},
  author={Fan, Ziwei and Liu, Zhiwei and Peng, Hao and Yu, Philip S},
  journal={WWW},
  year={2023}
}

@inproceedings{fan2022sequential,
  title={Sequential Recommendation via Stochastic Self-Attention},
  author={Fan, Ziwei and Liu, Zhiwei and Wang, Alice and Nazari, Zahra and Zheng, Lei and Peng, Hao and Yu, Philip S},
  journal={WWW},
  year={2022}
}

@inproceedings{fan2021modeling,
  title={Modeling Sequences as Distributions with Uncertainty for Sequential Recommendation},
  author={Fan, Ziwei and Liu, Zhiwei and Wang, Shen and Zheng, Lei and Yu, Philip S},
  booktitle={Proceedings of the 30th ACM International Conference on Information \& Knowledge Management},
  pages={3019--3023},
  year={2021}
}
```

## Code introduction
The code is implemented based on [S3-Rec](https://github.com/RUCAIBox/CIKM2020-S3Rec) and [STOSA](https://github.com/zfan20/STOSA).

## Datasets
We use the Amazon Review datasets Beauty and some more. The data split is done in the
leave-one-out setting. Make sure you download the datasets from the [link](https://jmcauley.ucsd.edu/data/amazon/).
