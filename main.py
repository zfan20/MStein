# -*- coding: utf-8 -*-
import os
import numpy as np
import random
import torch
import argparse

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import RecWithContrastiveLearningDataset, SASRecDataset, RecWithModeAugContrastiveLearningDataset, RecWithCL4RecContrastiveLearningDataset

from trainers import CoSeRecTrainer, FinetuneTrainer, DistSAModelTrainer, CoDistSAModelTrainer
from models import SASRecModel, OfflineItemSimilarity, OnlineItemSimilarity, DistSAModel, WassersteinOnlineItemSimilarity
from utils import EarlyStopping, get_user_seqs, get_item2attribute_json, check_path, set_seed
from trainers import SRMATrainer
from models import SASRecModelAugment

def show_args_info(args):
    print(f"--------------------Configure Info:------------")
    for arg in vars(args):
        print(f"{arg:<30} : {getattr(args, arg):>35}")

def main():
    parser = argparse.ArgumentParser()
    #system args
    parser.add_argument('--data_dir', default='../data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--data_name', default='Sports_and_Outdoors', type=str)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--model_idx', default=0, type=int, help="model idenfier 10, 20, 30...")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    #data augmentation args
    parser.add_argument('--noise_ratio', default=0.0, type=float, \
                        help="percentage of negative interactions in a sequence - robustness analysis")
    parser.add_argument('--training_data_ratio', default=1.0, type=float, \
                        help="percentage of training samples used for training - robustness analysis")
    parser.add_argument('--augment_threshold', default=4, type=int, \
                        help="control augmentations on short and long sequences.\
                        default:-1, means all augmentations types are allowed for all sequences.\
                        For sequence length < augment_threshold: Insert, and Substitute methods are allowed \
                        For sequence length > augment_threshold: Crop, Reorder, Substitute, and Mask \
                        are allowed.")
    parser.add_argument('--similarity_model_name', default='ItemCF_IUF', type=str, \
                        help="Method to generate item similarity score. choices: \
                        Random, ItemCF, ItemCF_IUF(Inverse user frequency), Item2Vec, LightGCN")
    parser.add_argument("--augmentation_warm_up_epoches", type=float, default=160, \
                        help="number of epochs to switch from \
                        memory-based similarity model to \
                        hybrid similarity model.")
    parser.add_argument('--base_augment_type', default='random', type=str, \
                        help="default data augmentation types. Chosen from: \
                        mask, crop, reorder, substitute, insert, random, \
                        combinatorial_enumerate (for multi-view).")
    parser.add_argument('--augment_type_for_short', default='SIM', type=str, \
                        help="data augmentation types for short sequences. Chosen from: \
                        SI, SIM, SIR, SIC, SIMR, SIMC, SIRC, SIMRC.")
    parser.add_argument("--tao", type=float, default=0.2, help="crop ratio for crop operator")
    parser.add_argument("--gamma", type=float, default=0.7, help="mask ratio for mask operator")
    parser.add_argument("--beta", type=float, default=0.2, help="reorder ratio for reorder operator") 
    parser.add_argument("--substitute_rate", type=float, default=0.1, \
                        help="substitute ratio for substitute operator")
    parser.add_argument("--insert_rate", type=float, default=0.4, \
                        help="insert ratio for insert operator")
    parser.add_argument("--max_insert_num_per_pos", type=int, default=1, \
                        help="maximum insert items per position for insert operator - not studied")

    ## STOSA specific parameters:
    parser.add_argument('--distance_metric', default='wasserstein', type=str)
    parser.add_argument('--pvn_weight', default=0.1, type=float)
    parser.add_argument('--kernel_param', default=1.0, type=float)

    ## contrastive learning task args
    parser.add_argument('--temperature', default= 1.0, type=float,
                        help='softmax temperature (default:  1.0) - not studied.')
    parser.add_argument('--n_views', default=2, type=int, metavar='N',
                        help='Number of augmented data for each sequence - not studied.')

    # model args
    parser.add_argument("--model_name", default='CoSeRec', type=str)
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str) # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=50, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--cf_weight", type=float, default=0.1, \
                        help="weight of contrastive learning task")
    parser.add_argument("--rec_weight", type=float, default=1.0, \
                        help="weight of contrastive learning task")

    #learning related
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")

    #model aug related
    parser.add_argument("--use_pretrain", action='store_true',help='whether using the pretrain model')
    parser.add_argument("--pretrain_dir", default="./pretrain/", type=str, help='the directory for pretrained models.')
    parser.add_argument("--pretrain_label", default='best', type=str, help='the name for loading the pre-trained encoder in CL.')

    parser.add_argument("--aug_layer_type", default='single', type=str, \
                        help='the layer type for the augmented layers, [single,inter,sasrec]')
    parser.add_argument("--layer_drop_num", type=int, default=1, help="number of layers to drop")
    parser.add_argument("--num_augmented_layers", type=int, default=3, help="number of augmented layers")
    parser.add_argument("--layer_drop_thres", type=float, default=-1, help="probabilty for layer drop")
    parser.add_argument("--rec_model_aug", action='store_true')
    parser.add_argument("--model_aug_in_batch", default='same', type=str, \
                        help='the model augmentation methods adopted in batch, [same, distinct, sasrec-static]')
    parser.add_argument("--attention_dropout_aug", action='store_true', help="using the attention dropout as the model augmentation. \
                        The probability ranges from [0, attention_probs_dropout_prob]")
    parser.add_argument("--hidden_dropout_aug", action='store_true', help="using the hidden dropout layer as the model augmentation. \
                        The probability ranges from [0, hidden_dropout_prob]")
    parser.add_argument("--en_weight", type=float, default=0.1, \
                        help="weight of encoder complementing ")


    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    print("Using Cuda:", torch.cuda.is_available())
    args.data_file = args.data_dir + args.data_name + '.txt'

    user_seq, max_item, valid_rating_matrix, test_rating_matrix, num_users = \
        get_user_seqs(args.data_file)

    args.item_size = max_item + 2
    args.num_users = num_users
    args.mask_id = max_item + 1

    # save model args
    #args_str = f'{args.model_name}-{args.data_name}-{args.model_idx}'
    if args.model_name == 'CoSeRec':
        args_str = f'{args.model_name}-{args.data_name}-{args.hidden_size}-{args.num_hidden_layers}-{args.num_attention_heads}-{args.hidden_act}-{args.attention_probs_dropout_prob}-{args.hidden_dropout_prob}-{args.max_seq_length}-{args.lr}-{args.weight_decay}-{args.model_idx}-{args.substitute_rate}-{args.insert_rate}-{args.augment_threshold}-{args.augmentation_warm_up_epoches}-{args.batch_size}-{args.cf_weight}'
    elif args.model_name == 'SASRec':
        args_str = f'{args.model_name}-{args.data_name}-{args.hidden_size}-{args.num_hidden_layers}-{args.num_attention_heads}-{args.hidden_act}-{args.attention_probs_dropout_prob}-{args.hidden_dropout_prob}-{args.max_seq_length}-{args.lr}-{args.weight_decay}-{args.model_idx}'
    elif args.model_name == 'SRMA':
        args_str = f'{args.model_name}-{args.data_name}-{args.hidden_size}-{args.num_hidden_layers}-{args.num_attention_heads}-{args.hidden_act}-{args.attention_probs_dropout_prob}-{args.hidden_dropout_prob}-{args.max_seq_length}-{args.lr}-{args.weight_decay}-{args.model_idx}-{args.batch_size}-{args.cf_weight}-{args.tao}-{args.gamma}-{args.aug_layer_type}-{args.layer_drop_num}-{args.num_augmented_layers}-{args.layer_drop_thres}-{args.model_aug_in_batch}-{args.attention_dropout_aug}-{args.hidden_dropout_aug}-{args.en_weight}'
    else:
        args_str = f'{args.model_name}-{args.data_name}-{args.hidden_size}-{args.num_hidden_layers}-{args.num_attention_heads}-{args.hidden_act}-{args.attention_probs_dropout_prob}-{args.hidden_dropout_prob}-{args.max_seq_length}-{args.lr}-{args.weight_decay}-{args.model_idx}-{args.pvn_weight}-{args.distance_metric}-{args.kernel_param}-{args.temperature}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    
    show_args_info(args)

    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    # -----------   pre-computation for item similarity   ------------ #
    args.similarity_model_path = os.path.join(args.data_dir,\
                            args.data_name+'_'+args.similarity_model_name+'_similarity.pkl')

    offline_similarity_model = OfflineItemSimilarity(data_file=args.data_file,
                            similarity_path=args.similarity_model_path,
                            model_name=args.similarity_model_name,
                            dataset_name=args.data_name)
    args.offline_similarity_model = offline_similarity_model

    # -----------   online based on shared item embedding for item similarity --------- #
    if args.model_name == 'CoDistSAModel':
        online_similarity_model = WassersteinOnlineItemSimilarity(item_size=args.item_size)
    else:
        online_similarity_model = OnlineItemSimilarity(item_size=args.item_size)
    args.online_similarity_model = online_similarity_model
    
    
    # training data for node classification
    if args.model_name == 'CoSeRec':
        model = SASRecModel(args=args)
        train_dataset = RecWithContrastiveLearningDataset(args, 
                                        user_seq[:int(len(user_seq)*args.training_data_ratio)], \
                                        data_type='train')
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

        eval_dataset = RecWithContrastiveLearningDataset(args, user_seq, data_type='valid')
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

        test_dataset = RecWithContrastiveLearningDataset(args, user_seq, data_type='test')
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

        trainer = CoSeRecTrainer(model, train_dataloader, eval_dataloader,
                              test_dataloader, args)
    if args.model_name == 'CL4Rec':
        model = SASRecModel(args=args)
        train_dataset = RecWithCL4RecContrastiveLearningDataset(args,
                                        user_seq[:int(len(user_seq)*args.training_data_ratio)], \
                                        data_type='train')
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

        eval_dataset = RecWithCL4RecContrastiveLearningDataset(args, user_seq, data_type='valid')
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

        test_dataset = RecWithCL4RecContrastiveLearningDataset(args, user_seq, data_type='test')
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

        trainer = CoSeRecTrainer(model, train_dataloader, eval_dataloader,
                              test_dataloader, args)

    elif args.model_name == 'SRMA':
        model = SASRecModelAugment(args=args)
        train_dataset = RecWithModeAugContrastiveLearningDataset(args,
                                    user_seq[:int(len(user_seq)*args.training_data_ratio)], \
                                    data_type='train')
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

        eval_dataset = RecWithModeAugContrastiveLearningDataset(args, user_seq, data_type='valid')
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

        test_dataset = RecWithModeAugContrastiveLearningDataset(args, user_seq, data_type='test')
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)
        # the cl encoder model complementing.
        pretrain_str = f'{args.model_name}-{args.data_name}-{args.pretrain_label}'
        pretrain_model = pretrain_str+ '.pt'
        path_pretrain_model = os.path.join(args.pretrain_dir, pretrain_model)
        # instantiate the pretrain_encoder type. Can change to any other types
        pretrain_encoder = SASRecModelAugment(args=args)

        if args.use_pretrain:
            print('-----loading pretrained model---------')
            pretrain_encoder.load_state_dict(torch.load(path_pretrain_model))
            print('loaded the pretrained model')
        trainer = SRMATrainer(model, pretrain_encoder, train_dataloader, eval_dataloader, test_dataloader, args)

    elif args.model_name == 'SASRec':#base sasrec
        model = SASRecModel(args=args)
        train_dataset = SASRecDataset(args, user_seq, data_type='train')
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

        eval_dataset = SASRecDataset(args, user_seq, data_type='valid')
        eval_sampler = SequentialSampler(eval_dataset)

        test_dataset = SASRecDataset(args, user_seq, data_type='test')
        test_sampler = SequentialSampler(test_dataset)

        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

        trainer = FinetuneTrainer(model, train_dataloader, eval_dataloader,
                                test_dataloader, args)
    elif args.model_name == 'DistSAModel': #DistSAModel
        model = DistSAModel(args=args)
        train_dataset = SASRecDataset(args, user_seq, data_type='train')
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

        eval_dataset = SASRecDataset(args, user_seq, data_type='valid')
        eval_sampler = SequentialSampler(eval_dataset)

        test_dataset = SASRecDataset(args, user_seq, data_type='test')
        test_sampler = SequentialSampler(test_dataset)

        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)
        trainer = DistSAModelTrainer(model, train_dataloader, eval_dataloader,
                                    test_dataloader, args)
    elif args.model_name == 'CoDistSAModel': #wasserstein CL DistSAModel
        model = DistSAModel(args=args)
        train_dataset = RecWithContrastiveLearningDataset(args,
                                        user_seq[:int(len(user_seq)*args.training_data_ratio)], \
                                        data_type='train')
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

        eval_dataset = RecWithContrastiveLearningDataset(args, user_seq, data_type='valid')
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

        test_dataset = RecWithContrastiveLearningDataset(args, user_seq, data_type='test')
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)
        trainer = CoDistSAModelTrainer(model, train_dataloader, eval_dataloader,
                              test_dataloader, args)


    if args.do_eval:
        trainer.args.train_matrix = test_rating_matrix
        trainer.load(args.checkpoint_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        #scores, result_info = trainer.test(0, full_sort=True)
        scores, result_info, _ = trainer.complicated_eval(user_seq, args)

    else:
        print(f'Train ' + args.model_name)
        if args.model_name == 'DistSAModel' or args.model_name == 'CoDistSAModel':
            early_stopping = EarlyStopping(args.checkpoint_path, patience=100, verbose=True)
        else:
            early_stopping = EarlyStopping(args.checkpoint_path, patience=40, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            # evaluate on MRR
            scores, _, _ = trainer.valid(epoch, full_sort=True)
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        print('---------------Change to test_rating_matrix!-------------------')
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        valid_scores, _, _ = trainer.valid('best', full_sort=True)
        trainer.args.train_matrix = test_rating_matrix
        scores, result_info, _ = trainer.test('best', full_sort=True)

    print(args_str)
    #print(result_info)
    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')
main()
