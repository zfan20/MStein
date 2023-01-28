# MStein
WWW 2023 code for MStein

Example code:
`python main.py --model_name=CoDistSAModel --data_name=Beauty --lr=0.001 --hidden_size=64 --max_seq_length=50 --hidden_dropout_prob=0.1 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=2 --attention_probs_dropout_prob=0.0 --substitute_rate=0.5 --insert_rate=0.3 --augment_threshold=16 --augmentation_warm_up_epoches=160 --cf_weight=0.005 --batch_size=1024 --pvn_weight=0.01 --temperature=1e-06`

Please cite our paper if you use the code:
```bibtex
@inproceedings{fan2023mstein,
  title={Mutual Wasserstein Discrepancy Minimization for Sequential Recommendation},
  author={Fan, Ziwei and Liu, Zhiwei and Peng, Hao and Yu, Philip S},
  journal={WWW},
  year={2023}
}
```
