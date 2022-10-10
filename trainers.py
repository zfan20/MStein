# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
import random

import torch
import torch.nn as nn
from torch.optim import Adam
from collections import defaultdict

from torch.utils.data import DataLoader, RandomSampler
from datasets import RecWithContrastiveLearningDataset
from modules import NCELoss, NTXent, WassersteinNCELoss
from utils import recall_at_k, ndcg_k, get_metric, get_user_seqs, nCr, cal_mrr, get_user_performance_perpopularity, get_item_performance_perpopularity
from modules import wasserstein_distance, kl_distance, wasserstein_distance_matmul, d2s_gaussiannormal, d2s_1overx, kl_distance_matmul

class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, 
                 args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        self.online_similarity_model = args.online_similarity_model

        self.total_augmentaion_pairs = nCr(self.args.n_views, 2)
        #projection head for contrastive learn task
        self.projection = nn.Sequential(nn.Linear(self.args.max_seq_length*self.args.hidden_size, \
                                        512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), 
                                        nn.Linear(512, self.args.hidden_size, bias=True))
        if self.cuda_condition:
            self.model.cuda()
            self.projection.cuda()
        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        self.cf_criterion = NCELoss(self.args.temperature, self.device)
        # self.cf_criterion = NTXent()
        #print("self.cf_criterion:", self.cf_criterion.__class__.__name__)
        
    def __refresh_training_dataset(self, item_embeddings):
        """
        use for updating item embedding
        """
        user_seq, _, _, _, _ = get_user_seqs(self.args.data_file)
        self.args.online_similarity_model.update_embedding_matrix(item_embeddings)
        # training data for node classification
        train_dataset = RecWithContrastiveLearningDataset(self.args, user_seq, 
                                        data_type='train', similarity_model_type='online')
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.args.batch_size)
        return train_dataloader

    def __Wasserstein_refresh_training_dataset(self, item_mean_embeddings, item_cov_embeddings):
        user_seq, _, _, _, _ = get_user_seqs(self.args.data_file)
        self.args.online_similarity_model.update_embedding_matrix(item_mean_embeddings, item_cov_embeddings)
        # training data for node classification
        train_dataset = RecWithContrastiveLearningDataset(self.args, user_seq,
                                        data_type='train', similarity_model_type='online')
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.args.batch_size)
        return train_dataloader
        
    def train(self, epoch):
        # start to use online item similarity
        if epoch > self.args.augmentation_warm_up_epoches:
            print("refresh dataset with updated item embedding")
            if self.args.model_name == 'CoDistSAModel':
                self.train_dataloader = self.__Wasserstein_refresh_training_dataset(self.model.item_mean_embeddings, self.model.item_cov_embeddings)
            else:
                self.train_dataloader = self.__refresh_training_dataset(self.model.item_embeddings)
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort=full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort=full_sort, train=False)

    def complicated_eval(self, user_seq, args):
        return self.eval_analysis(self.test_dataloader, user_seq, args)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def eval_analysis(self, dataloader, seqs):
        raise NotImplementedError

    def qualitative_analysis(self, dataloader, seqs, args):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)
    
    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg, mrr = [], [], 0
        recall_dict_list = []
        ndcg_dict_list = []
        for k in [1, 5, 10, 15, 20, 40]:
            recall_result, recall_dict_k = recall_at_k(answers, pred_list, k)
            recall.append(recall_result)
            recall_dict_list.append(recall_dict_k)
            ndcg_result, ndcg_dict_k = ndcg_k(answers, pred_list, k)
            ndcg.append(ndcg_result)
            ndcg_dict_list.append(ndcg_dict_k)
        mrr, mrr_dict = cal_mrr(answers, pred_list)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.8f}'.format(recall[0]), "NDCG@1": '{:.8f}'.format(ndcg[0]),
            "HIT@5": '{:.8f}'.format(recall[1]), "NDCG@5": '{:.8f}'.format(ndcg[1]),
            "HIT@10": '{:.8f}'.format(recall[2]), "NDCG@10": '{:.8f}'.format(ndcg[2]),
            "HIT@15": '{:.8f}'.format(recall[3]), "NDCG@15": '{:.8f}'.format(ndcg[3]),
            "HIT@20": '{:.8f}'.format(recall[4]), "NDCG@20": '{:.8f}'.format(ndcg[4]),
            "HIT@40": '{:.8f}'.format(recall[5]), "NDCG@40": '{:.8f}'.format(ndcg[5]),
            "MRR": '{:.8f}'.format(mrr)
        }
        print(post_fix, flush=True)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[2], ndcg[2], recall[3], ndcg[3], recall[4], ndcg[4], recall[5], ndcg[5], mrr], str(post_fix), [recall_dict_list, ndcg_dict_list, mrr_dict]

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float() # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        auc = torch.sum(
            ((torch.sign(pos_logits - neg_logits) + 1) / 2) * istarget
        ) / torch.sum(istarget)

        return loss, auc

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class CoSeRecTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, 
                 args):
        super(CoSeRecTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, 
            args
        )
        print("self.cf_criterion:", self.cf_criterion.__class__.__name__)

    
    def eval_analysis(self, dataloader, user_seq, args):
        rec_data_iter = tqdm(enumerate(dataloader),
                                  desc=f"Recommendation Test Analysis",
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        #rec_data_iter = dataloader

        Ks = [1, 5, 10, 15, 20, 40]
        item_freq = defaultdict(int)
        train = {}
        for user_id, seq in enumerate(user_seq):
            train_seq = seq[:-3]
            train[user_id] = train_seq
            for itemid in train_seq:
                item_freq[itemid] += 1
        freq_quantiles = np.array([3, 7, 20, 50])
        items_in_freqintervals = [[] for _ in range(len(freq_quantiles)+1)]
        for item, freq_i in item_freq.items():
            interval_ind = -1
            for quant_ind, quant_freq in enumerate(freq_quantiles):
                if freq_i <= quant_freq:
                    interval_ind = quant_ind
                    break
            if interval_ind == -1:
                interval_ind = len(items_in_freqintervals) - 1
            items_in_freqintervals[interval_ind].append(item)


        self.model.eval()
        all_pos_items_ranks = defaultdict(list)
        pred_list = None
        with torch.no_grad():
            for i, batch in rec_data_iter:
            #i = 0
            #for batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, target_pos, target_neg, answers = batch
                recommend_output = self.model.transformer_encoder(input_ids)

                recommend_output = recommend_output[:, -1, :]

                rating_pred = self.predict_full(recommend_output)

                rating_pred = rating_pred.cpu().data.numpy().copy()
                batch_user_index = user_ids.cpu().numpy()
                rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0

                batch_pred_list = np.argsort(-rating_pred, axis=1)
                pos_items = answers.cpu().data.numpy()

                pos_ranks = np.where(batch_pred_list==pos_items)[1]+1
                for each_pos_item, each_rank in zip(pos_items, pos_ranks):
                    all_pos_items_ranks[each_pos_item[0]].append(each_rank)

                partial_batch_pred_list = batch_pred_list[:, :40]

                if i == 0:
                    pred_list = partial_batch_pred_list
                    answer_list = answers.cpu().data.numpy()
                else:
                    pred_list = np.append(pred_list, partial_batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                #i += 1
            scores, result_info, [recall_dict_list, ndcg_dict_list, mrr_dict] = self.get_full_sort_score('best', answer_list, pred_list)

            get_user_performance_perpopularity(train, [recall_dict_list, ndcg_dict_list, mrr_dict], Ks)
            get_item_performance_perpopularity(items_in_freqintervals, all_pos_items_ranks, Ks, freq_quantiles, args.item_size)
            return scores, result_info, None


    def _one_pair_contrastive_learning(self, inputs):
        '''
        contrastive learning given one pair sequences (batch)
        inputs: [batch1_augmented_data, batch2_augmentated_data]
        '''
        cl_batch = torch.cat(inputs, dim=0)
        cl_batch = cl_batch.to(self.device)
        cl_sequence_output = self.model.transformer_encoder(cl_batch)
        # cf_sequence_output = cf_sequence_output[:, -1, :]
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        # cf_output = self.projection(cf_sequence_flatten)
        batch_size = cl_batch.shape[0]//2
        cl_output_slice = torch.split(cl_sequence_flatten, batch_size)
        cl_loss = self.cf_criterion(cl_output_slice[0], 
                                cl_output_slice[1])
        return cl_loss

    def iteration(self, epoch, dataloader, full_sort=True, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar

        if train:
            self.model.train()
            rec_avg_loss = 0.0
            cl_individual_avg_losses = [0.0 for i in range(self.total_augmentaion_pairs)]
            cl_sum_avg_loss = 0.0
            joint_avg_loss = 0.0

            print(f"rec dataset length: {len(dataloader)}")
            #rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))
            rec_cf_data_iter = dataloader

            for i, (rec_batch, cl_batches) in enumerate(rec_cf_data_iter):
                '''
                rec_batch shape: key_name x batch_size x feature_dim
                cl_batches shape: 
                    list of n_views x batch_size x feature_dim tensors
                '''
                # 0. batch_data will be sent into the device(GPU or CPU)
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_ids, target_pos, target_neg, _ = rec_batch

                # ---------- recommendation task ---------------#
                sequence_output = self.model.transformer_encoder(input_ids)
                rec_loss, auc = self.cross_entropy(sequence_output, target_pos, target_neg)

                # ---------- contrastive learning task -------------#
                cl_losses = []
                for cl_batch in cl_batches:
                    cl_loss = self._one_pair_contrastive_learning(cl_batch)
                    cl_losses.append(cl_loss)

                joint_loss = self.args.rec_weight * rec_loss
                for cl_loss in cl_losses:
                    joint_loss += self.args.cf_weight * cl_loss
                self.optim.zero_grad()
                joint_loss.backward()
                self.optim.step()

                rec_avg_loss += rec_loss.item()

                for i, cl_loss in enumerate(cl_losses):
                    cl_individual_avg_losses[i] += cl_loss.item()
                    cl_sum_avg_loss += cl_loss.item()
                joint_avg_loss += joint_loss.item()


            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_cf_data_iter)),
                "joint_avg_loss": '{:.4f}'.format(joint_avg_loss / len(rec_cf_data_iter)),
                "cl_avg_loss": '{:.4f}'.format(cl_sum_avg_loss / (len(rec_cf_data_iter)*self.total_augmentaion_pairs)),
            }
            for i, cl_individual_avg_loss in enumerate(cl_individual_avg_losses):
                post_fix['cl_pair_'+str(i)+'_loss'] = '{:.4f}'.format(cl_individual_avg_loss / len(rec_cf_data_iter))

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            #rec_data_iter = tqdm(enumerate(dataloader),
            #                      desc="Recommendation EP_%s:%d" % (str_code, epoch),
            #                      total=len(dataloader),
            #                      bar_format="{l_bar}{r_bar}")
            rec_data_iter = dataloader
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in enumerate(rec_data_iter):
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output = self.model.transformer_encoder(input_ids)

                    recommend_output = recommend_output[:, -1, :]
                    # recommendation results

                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition T: O(n)  argsort O(nlogn)
                    ind = np.argpartition(rating_pred, -40)[:, -40:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                for i, batch in enumerate(rec_data_iter):
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)

                return self.get_sample_scores(epoch, pred_list)


class FinetuneTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(FinetuneTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )

    def qualitative_analysis(self, dataloader, user_seq, args):
        rec_data_iter = tqdm(enumerate(dataloader),
                                  desc=f"Recommendation Test Analysis",
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        item_freq = defaultdict(int)
        user_freq = defaultdict(int)
        for user_id, seq in enumerate(user_seq):
            train_seq = seq[:-3]
            train[user_id] = len(train_seq)
            for itemid in train_seq:
                item_freq[itemid] += 1
        self.model.eval()
        pred_list = None
        with torch.no_grad():
            for i, batch in rec_data_iter:
            #i = 0
            #for batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, target_pos, target_neg, answers = batch
                recommend_output, att_scores = self.model.finetune(input_ids)

                recommend_output = recommend_output[:, -1, :]

                att_scores = att_scores.cpu().data.numpy().copy()

                rating_pred = self.predict_full(recommend_output)

                rating_pred = rating_pred.cpu().data.numpy().copy()
                batch_user_index = user_ids.cpu().numpy()
                rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0

                batch_pred_list = np.argsort(-rating_pred, axis=1)
                pos_items = answers.cpu().data.numpy()

                pos_ranks = np.where(batch_pred_list==pos_items)[1]+1
                for each_pos_item, each_rank in zip(pos_items, pos_ranks):
                    all_pos_items_ranks[each_pos_item[0]].append(each_rank)

                partial_batch_pred_list = batch_pred_list[:, :40]

                if i == 0:
                    pred_list = partial_batch_pred_list
                    answer_list = answers.cpu().data.numpy()
                else:
                    pred_list = np.append(pred_list, partial_batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
            scores, result_info, [recall_dict_list, ndcg_dict_list, mrr_dict] = self.get_full_sort_score('best', answer_list, pred_list)
        return scores, result_info, None

    
    def eval_analysis(self, dataloader, user_seq, args):
        rec_data_iter = tqdm(enumerate(dataloader),
                                  desc=f"Recommendation Test Analysis",
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        #rec_data_iter = dataloader

        Ks = [1, 5, 10, 15, 20, 40]
        item_freq = defaultdict(int)
        train = {}
        for user_id, seq in enumerate(user_seq):
            train_seq = seq[:-3]
            train[user_id] = train_seq
            for itemid in train_seq:
                item_freq[itemid] += 1
        freq_quantiles = np.array([3, 7, 20, 50])
        items_in_freqintervals = [[] for _ in range(len(freq_quantiles)+1)]
        for item, freq_i in item_freq.items():
            interval_ind = -1
            for quant_ind, quant_freq in enumerate(freq_quantiles):
                if freq_i <= quant_freq:
                    interval_ind = quant_ind
                    break
            if interval_ind == -1:
                interval_ind = len(items_in_freqintervals) - 1
            items_in_freqintervals[interval_ind].append(item)


        self.model.eval()
        all_pos_items_ranks = defaultdict(list)
        pred_list = None
        with torch.no_grad():
            for i, batch in rec_data_iter:
            #i = 0
            #for batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, target_pos, target_neg, answers = batch
                recommend_output, _ = self.model.finetune(input_ids)

                recommend_output = recommend_output[:, -1, :]

                rating_pred = self.predict_full(recommend_output)

                rating_pred = rating_pred.cpu().data.numpy().copy()
                batch_user_index = user_ids.cpu().numpy()
                rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0

                batch_pred_list = np.argsort(-rating_pred, axis=1)
                pos_items = answers.cpu().data.numpy()

                pos_ranks = np.where(batch_pred_list==pos_items)[1]+1
                for each_pos_item, each_rank in zip(pos_items, pos_ranks):
                    all_pos_items_ranks[each_pos_item[0]].append(each_rank)

                partial_batch_pred_list = batch_pred_list[:, :40]

                if i == 0:
                    pred_list = partial_batch_pred_list
                    answer_list = answers.cpu().data.numpy()
                else:
                    pred_list = np.append(pred_list, partial_batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                #i += 1
            scores, result_info, [recall_dict_list, ndcg_dict_list, mrr_dict] = self.get_full_sort_score('best', answer_list, pred_list)

            get_user_performance_perpopularity(train, [recall_dict_list, ndcg_dict_list, mrr_dict], Ks)
            get_item_performance_perpopularity(items_in_freqintervals, all_pos_items_ranks, Ks, freq_quantiles, args.item_size)
            return scores, result_info, None


    def iteration(self, epoch, dataloader, full_sort=False, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar

        #rec_data_iter = tqdm.tqdm(enumerate(dataloader),
        #                          desc="Recommendation EP_%s:%d" % (str_code, epoch),
        #                          total=len(dataloader),
        #                          bar_format="{l_bar}{r_bar}")
        rec_data_iter = dataloader
        if train:
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0
            rec_avg_auc = 0.0

            #for i, batch in rec_data_iter:
            for batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                _, input_ids, target_pos, target_neg, _ = batch
                # Binary cross_entropy
                sequence_output = self.model.transformer_encoder(input_ids)
                loss, batch_auc = self.cross_entropy(sequence_output, target_pos, target_neg)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()
                rec_avg_auc += batch_auc.item()
            
            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
                "rec_avg_auc": '{:.4f}'.format(rec_avg_auc / len(rec_data_iter)),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix), flush=True)

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')
        else:
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                #for i, batch in rec_data_iter:
                i = 0
                for batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output = self.model.transformer_encoder(input_ids)

                    recommend_output = recommend_output[:, -1, :]
                    # 推荐的结果

                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition 时间复杂度O(n)  argsort O(nlogn) 只会做
                    # 加负号"-"表示取大的值
                    ind = np.argpartition(rating_pred, -40)[:, -40:]
                    # 根据返回的下标 从对应维度分别取对应的值 得到每行topk的子表
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    # 对子表进行排序 得到从大到小的顺序
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    # 再取一次 从ind中取回 原来的下标
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                    i += 1
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                #for i, batch in rec_data_iter:
                i = 0
                for batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)
                    i += 1

                return self.get_sample_scores(epoch, pred_list)


class DistSAModelTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(DistSAModelTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )

    def bpr_optimization(self, seq_mean_out, seq_cov_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        activation = nn.ELU()
        pos_mean_emb = self.model.item_mean_embeddings(pos_ids)
        pos_cov_emb = activation(self.model.item_cov_embeddings(pos_ids)) + 1
        neg_mean_emb = self.model.item_mean_embeddings(neg_ids)
        neg_cov_emb = activation(self.model.item_cov_embeddings(neg_ids)) + 1

        # [batch*seq_len hidden_size]
        pos_mean = pos_mean_emb.view(-1, pos_mean_emb.size(2))
        pos_cov = pos_cov_emb.view(-1, pos_cov_emb.size(2))
        neg_mean = neg_mean_emb.view(-1, neg_mean_emb.size(2))
        neg_cov = neg_cov_emb.view(-1, neg_cov_emb.size(2))
        seq_mean_emb = seq_mean_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]
        seq_cov_emb = seq_cov_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]

        if self.args.distance_metric == 'wasserstein':
            pos_logits = wasserstein_distance(seq_mean_emb, seq_cov_emb, pos_mean, pos_cov)
            neg_logits = wasserstein_distance(seq_mean_emb, seq_cov_emb, neg_mean, neg_cov)
            pos_vs_neg = wasserstein_distance(pos_mean, pos_cov, neg_mean, neg_cov)
        else:
            pos_logits = kl_distance(seq_mean_emb, seq_cov_emb, pos_mean, pos_cov)
            neg_logits = kl_distance(seq_mean_emb, seq_cov_emb, neg_mean, neg_cov)
            pos_vs_neg = kl_distance(pos_mean, pos_cov, neg_mean, neg_cov)

        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float() # [batch*seq_len]
        loss = torch.sum(-torch.log(torch.sigmoid(neg_logits - pos_logits + 1e-24)) * istarget) / torch.sum(istarget)

        pvn_loss = self.args.pvn_weight * torch.sum(torch.clamp(pos_logits - pos_vs_neg, 0) * istarget) / torch.sum(istarget)
        auc = torch.sum(
            ((torch.sign(neg_logits - pos_logits) + 1) / 2) * istarget
        ) / torch.sum(istarget)

        return loss, auc, pvn_loss

    def ce_optimization(self, seq_mean_out, seq_cov_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        activation = nn.ELU()
        pos_mean_emb = self.model.item_mean_embeddings(pos_ids)
        pos_cov_emb = activation(self.model.item_cov_embeddings(pos_ids)) + 1
        neg_mean_emb = self.model.item_mean_embeddings(neg_ids)
        neg_cov_emb = activation(self.model.item_cov_embeddings(neg_ids)) + 1

        # [batch*seq_len hidden_size]
        pos_mean = pos_mean_emb.view(-1, pos_mean_emb.size(2))
        pos_cov = pos_cov_emb.view(-1, pos_cov_emb.size(2))
        neg_mean = neg_mean_emb.view(-1, neg_mean_emb.size(2))
        neg_cov = neg_cov_emb.view(-1, neg_cov_emb.size(2))
        seq_mean_emb = seq_mean_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]
        seq_cov_emb = seq_cov_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]


        #pos_logits = d2s_gaussiannormal(wasserstein_distance(seq_mean_emb, seq_cov_emb, pos_mean, pos_cov), self.args.kernel_param)
        pos_logits = -wasserstein_distance(seq_mean_emb, seq_cov_emb, pos_mean, pos_cov)
        #neg_logits = d2s_gaussiannormal(wasserstein_distance(seq_mean_emb, seq_cov_emb, neg_mean, neg_cov), self.args.kernel_param)
        neg_logits = -wasserstein_distance(seq_mean_emb, seq_cov_emb, neg_mean, neg_cov)

        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float() # [batch*seq_len]

        loss = torch.sum(
            - torch.log(torch.sigmoid(neg_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(pos_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        auc = torch.sum(
            ((torch.sign(neg_logits - pos_logits) + 1) / 2) * istarget
        ) / torch.sum(istarget)


        return loss, auc

    def dist_predict_full(self, seq_mean_out, seq_cov_out):
        elu_activation = torch.nn.ELU()
        test_item_mean_emb = self.model.item_mean_embeddings.weight
        test_item_cov_emb = elu_activation(self.model.item_cov_embeddings.weight) + 1
        #num_items, emb_size = test_item_cov_emb.shape

        #seq_mean_out = seq_mean_out.unsqueeze(1).expand(-1, num_items, -1).reshape(-1, emb_size)
        #seq_cov_out = seq_cov_out.unsqueeze(1).expand(-1, num_items, -1).reshape(-1, emb_size)

        #if args.distance_metric == 'wasserstein':
        #    return wasserstein_distance(seq_mean_out, seq_cov_out, test_item_mean_emb, test_item_cov_emb)
        #else:
        #    return kl_distance(seq_mean_out, seq_cov_out, test_item_mean_emb, test_item_cov_emb)
        #return d2s_1overx(wasserstein_distance_matmul(seq_mean_out, seq_cov_out, test_item_mean_emb, test_item_cov_emb))
        return wasserstein_distance_matmul(seq_mean_out, seq_cov_out, test_item_mean_emb, test_item_cov_emb)


    def kl_predict_full(self, seq_mean_out, seq_cov_out):
        elu_activation = torch.nn.ELU()
        test_item_mean_emb = self.model.item_mean_embeddings.weight
        test_item_cov_emb = elu_activation(self.model.item_cov_embeddings.weight) + 1

        num_items = test_item_mean_emb.shape[0]
        eval_batch_size = seq_mean_out.shape[0]
        moded_num_items = eval_batch_size - num_items % eval_batch_size
        fake_mean_emb = torch.zeros(moded_num_items, test_item_mean_emb.shape[1], dtype=torch.float32).to(self.device)
        fake_cov_emb = torch.ones(moded_num_items, test_item_mean_emb.shape[1], dtype=torch.float32).to(self.device)

        concated_mean_emb = torch.cat((test_item_mean_emb, fake_mean_emb), 0)
        concated_cov_emb = torch.cat((test_item_cov_emb, fake_cov_emb), 0)

        assert concated_mean_emb.shape[0] == test_item_mean_emb.shape[0] + moded_num_items

        num_batches = int(num_items / eval_batch_size)
        if moded_num_items > 0:
            num_batches += 1

        results = torch.zeros(seq_mean_out.shape[0], concated_mean_emb.shape[0], dtype=torch.float32)
        start_i = 0
        for i_batch in range(num_batches):
            end_i = start_i + eval_batch_size

            results[:, start_i:end_i] = kl_distance_matmul(seq_mean_out, seq_cov_out, concated_mean_emb[start_i:end_i, :], concated_cov_emb[start_i:end_i, :])
            #results[:, start_i:end_i] = d2s_gaussiannormal(kl_distance_matmul(seq_mean_out, seq_cov_out, concated_mean_emb[start_i:end_i, :], concated_cov_emb[start_i:end_i, :]))
            start_i += eval_batch_size

        #print(results[:, :5])
        return results[:, :num_items]


    def iteration(self, epoch, dataloader, full_sort=False, train=True):

        str_code = "train" if train else "test"

        #rec_data_iter = tqdm.tqdm(enumerate(dataloader),
        #                          desc=f"Recommendation EP_{str_code}:{epoch}",
        #                          total=len(dataloader),
        #                          bar_format="{l_bar}{r_bar}")
        rec_data_iter = dataloader

        if train:
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0
            rec_avg_pvn_loss = 0.0
            rec_avg_auc = 0.0

            #for i, batch in rec_data_iter:
            for batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, target_pos, target_neg, _ = batch
                # bpr optimization
                sequence_mean_output, sequence_cov_output, att_scores = self.model.finetune(input_ids)
                #print(att_scores[0, 0, :, :])
                loss, batch_auc, pvn_loss = self.bpr_optimization(sequence_mean_output, sequence_cov_output, target_pos, target_neg)
                #loss, batch_auc = self.ce_optimization(sequence_mean_output, sequence_cov_output, target_pos, target_neg)
                
                
                loss += pvn_loss
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()
                rec_avg_auc += batch_auc.item()
                rec_avg_pvn_loss += pvn_loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
                "rec_avg_auc": '{:.6f}'.format(rec_avg_auc / len(rec_data_iter)),
                "rec_avg_pvn_loss": '{:.6f}'.format(rec_avg_pvn_loss / len(rec_data_iter)),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix), flush=True)

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                with torch.no_grad():
                    #for i, batch in rec_data_iter:
                    i = 0
                    for batch in rec_data_iter:
                        # 0. batch_data will be sent into the device(GPU or cpu)
                        batch = tuple(t.to(self.device) for t in batch)
                        user_ids, input_ids, target_pos, target_neg, answers = batch
                        recommend_mean_output, recommend_cov_output, _ = self.model.finetune(input_ids)

                        recommend_mean_output = recommend_mean_output[:, -1, :]
                        recommend_cov_output = recommend_cov_output[:, -1, :]

                        if self.args.distance_metric == 'kl':
                            rating_pred = self.kl_predict_full(recommend_mean_output, recommend_cov_output)
                        else:
                            rating_pred = self.dist_predict_full(recommend_mean_output, recommend_cov_output)

                        rating_pred = rating_pred.cpu().data.numpy().copy()
                        batch_user_index = user_ids.cpu().numpy()
                        rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 1e+24
                        # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                        ind = np.argpartition(rating_pred, 40)[:, :40]
                        #ind = np.argpartition(rating_pred, -40)[:, -40:]
                        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                        # ascending order
                        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::]
                        #arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                        if i == 0:
                            pred_list = batch_pred_list
                            answer_list = answers.cpu().data.numpy()
                        else:
                            pred_list = np.append(pred_list, batch_pred_list, axis=0)
                            answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                        i += 1
                    return self.get_full_sort_score(epoch, answer_list, pred_list)


class CoDistSAModelTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(CoDistSAModelTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )
        self.cf_criterion = WassersteinNCELoss(self.args.temperature, self.device)
        print("self.cf_criterion:", self.cf_criterion.__class__.__name__)

    def qualitative_analysis(self, dataloader, user_seq, args):
        rec_data_iter = tqdm(enumerate(dataloader),
                                  desc=f"Recommendation Test Analysis",
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        item_freq = defaultdict(int)
        user_freq = defaultdict(int)
        for user_id, seq in enumerate(user_seq):
            train_seq = seq[:-3]
            user_freq[user_id] = len(train_seq)
            for itemid in train_seq:
                item_freq[itemid] += 1
        self.model.eval()
        pred_list = None
        all_att_scores = None
        input_seqs = None

        item_mean_emb = self.model.item_mean_embeddings.weight.cpu().data.numpy().copy()
        item_cov_emb = self.model.item_cov_embeddings.weight.cpu().data.numpy().copy()
        with torch.no_grad():
            for i, batch in rec_data_iter:
            #i = 0
            #for batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, target_pos, target_neg, answers = batch
                recommend_mean_output, recommend_cov_output, att_scores = self.model.finetune(input_ids)

                recommend_mean_output = recommend_mean_output[:, -1, :]
                recommend_cov_output = recommend_cov_output[:, -1, :]

                rating_pred = self.dist_predict_full(recommend_mean_output, recommend_cov_output)

                rating_pred = rating_pred.cpu().data.numpy().copy()
                att_scores = att_scores.cpu().data.numpy().copy()

                batch_user_index = user_ids.cpu().numpy()
                rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 1e+24

                batch_pred_list = np.argsort(rating_pred, axis=1)
                pos_items = answers.cpu().data.numpy()

                partial_batch_pred_list = batch_pred_list[:, :40]
                if i == 0:
                    pred_list = partial_batch_pred_list
                    answer_list = answers.cpu().data.numpy()
                    all_att_scores = att_scores[:, 0, -30:, -30:]
                    input_seqs = input_ids.cpu().data.numpy()
                else:
                    pred_list = np.append(pred_list, partial_batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                    all_att_scores = np.append(all_att_scores, att_scores[:, 0, -30:, -30:], axis=0)
                    input_seqs = np.append(input_seqs, input_ids.cpu().data.numpy(), axis=0)
            scores, result_info, [recall_dict_list, ndcg_dict_list, mrr_dict] = self.get_full_sort_score('best', answer_list, pred_list)

        pred_details = {'ndcg': ndcg_dict_list, 'pred_list': pred_list, 'answer_list': answer_list, 'attentions': all_att_scores, 'user_freq': user_freq, 'item_freq': item_freq, 'train': input_seqs, 'embeddings': [item_mean_emb, item_cov_emb]}
        return scores, result_info, pred_details


    def eval_analysis(self, dataloader, user_seq, args):
        rec_data_iter = tqdm(enumerate(dataloader),
                                  desc=f"Recommendation Analysis",
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        #rec_data_iter = dataloader
        Ks = [1, 5, 10, 15, 20, 40]
        item_freq = defaultdict(int)
        train = {}
        for user_id, seq in enumerate(user_seq):
            train_seq = seq[:-3]
            train[user_id] = train_seq
            for itemid in train_seq:
                item_freq[itemid] += 1
        freq_quantiles = np.array([3, 7, 20, 50])
        items_in_freqintervals = [[] for _ in range(len(freq_quantiles)+1)]
        for item, freq_i in item_freq.items():
            interval_ind = -1
            for quant_ind, quant_freq in enumerate(freq_quantiles):
                if freq_i <= quant_freq:
                    interval_ind = quant_ind
                    break
            if interval_ind == -1:
                interval_ind = len(items_in_freqintervals) - 1
            items_in_freqintervals[interval_ind].append(item)


        self.model.eval()
        all_pos_items_ranks = defaultdict(list)
        pred_list = None
        with torch.no_grad():
            for i, batch in rec_data_iter:
            #i = 0
            #for batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, target_pos, target_neg, answers = batch
                recommend_mean_output, recommend_cov_output, _ = self.model.finetune(input_ids)

                recommend_mean_output = recommend_mean_output[:, -1, :]
                recommend_cov_output = recommend_cov_output[:, -1, :]

                rating_pred = self.dist_predict_full(recommend_mean_output, recommend_cov_output)

                rating_pred = rating_pred.cpu().data.numpy().copy()
                batch_user_index = user_ids.cpu().numpy()
                rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 1e+24

                batch_pred_list = np.argsort(rating_pred, axis=1)
                pos_items = answers.cpu().data.numpy()

                pos_ranks = np.where(batch_pred_list==pos_items)[1]+1
                for each_pos_item, each_rank in zip(pos_items, pos_ranks):
                    all_pos_items_ranks[each_pos_item[0]].append(each_rank)

                partial_batch_pred_list = batch_pred_list[:, :40]

                if i == 0:
                    pred_list = partial_batch_pred_list
                    answer_list = answers.cpu().data.numpy()
                else:
                    pred_list = np.append(pred_list, partial_batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                #i += 1
            scores, result_info, [recall_dict_list, ndcg_dict_list, mrr_dict] = self.get_full_sort_score('best', answer_list, pred_list)

            get_user_performance_perpopularity(train, [recall_dict_list, ndcg_dict_list, mrr_dict], Ks)
            get_item_performance_perpopularity(items_in_freqintervals, all_pos_items_ranks, Ks, freq_quantiles, args.item_size)
            return scores, result_info, None


    def _one_pair_contrastive_learning(self, inputs):
        '''
        contrastive learning given one pair sequences (batch)
        inputs: [batch1_augmented_data, batch2_augmentated_data]
        '''
        cl_batch = torch.cat(inputs, dim=0)
        cl_batch = cl_batch.to(self.device)
        cl_mean_sequence_output, cl_cov_sequence_output, _ = self.model.finetune(cl_batch)
        # cf_sequence_output = cf_sequence_output[:, -1, :]
        cl_mean_sequence_flatten = cl_mean_sequence_output.view(cl_batch.shape[0], -1)
        cl_cov_sequence_flatten = cl_cov_sequence_output.view(cl_batch.shape[0], -1)
        # cf_output = self.projection(cf_sequence_flatten)
        batch_size = cl_batch.shape[0]//2
        cl_mean_output_slice = torch.split(cl_mean_sequence_flatten, batch_size)
        cl_cov_output_slice = torch.split(cl_cov_sequence_flatten, batch_size)
        cl_loss = self.cf_criterion(cl_mean_output_slice[0], cl_cov_output_slice[0],
                                cl_mean_output_slice[1], cl_cov_output_slice[1])
        return cl_loss

    def bpr_optimization(self, seq_mean_out, seq_cov_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        activation = nn.ELU()
        pos_mean_emb = self.model.item_mean_embeddings(pos_ids)
        #pos_mean_emb = nn.functional.normalize(self.model.item_mean_embeddings(pos_ids))
        pos_cov_emb = activation(self.model.item_cov_embeddings(pos_ids)) + 1
        neg_mean_emb = self.model.item_mean_embeddings(neg_ids)
        #neg_mean_emb = nn.functional.normalize(self.model.item_mean_embeddings(neg_ids))
        neg_cov_emb = activation(self.model.item_cov_embeddings(neg_ids)) + 1

        # [batch*seq_len hidden_size]
        pos_mean = pos_mean_emb.view(-1, pos_mean_emb.size(2))
        pos_cov = pos_cov_emb.view(-1, pos_cov_emb.size(2))
        neg_mean = neg_mean_emb.view(-1, neg_mean_emb.size(2))
        neg_cov = neg_cov_emb.view(-1, neg_cov_emb.size(2))
        seq_mean_emb = seq_mean_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]
        seq_cov_emb = seq_cov_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]

        if self.args.distance_metric == 'wasserstein':
            pos_logits = wasserstein_distance(seq_mean_emb, seq_cov_emb, pos_mean, pos_cov)
            neg_logits = wasserstein_distance(seq_mean_emb, seq_cov_emb, neg_mean, neg_cov)
            pos_vs_neg = wasserstein_distance(pos_mean, pos_cov, neg_mean, neg_cov)
        else:
            pos_logits = kl_distance(seq_mean_emb, seq_cov_emb, pos_mean, pos_cov)
            neg_logits = kl_distance(seq_mean_emb, seq_cov_emb, neg_mean, neg_cov)
            pos_vs_neg = kl_distance(pos_mean, pos_cov, neg_mean, neg_cov)

        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float() # [batch*seq_len]
        loss = torch.sum(-torch.log(torch.sigmoid(neg_logits - pos_logits + 1e-24)) * istarget) / torch.sum(istarget)

        pvn_loss = torch.sum(torch.clamp(pos_logits - pos_vs_neg, 0) * istarget) / torch.sum(istarget)
        auc = torch.sum(
            ((torch.sign(neg_logits - pos_logits) + 1) / 2) * istarget
        ) / torch.sum(istarget)

        return loss, auc, pvn_loss

    def dist_predict_full(self, seq_mean_out, seq_cov_out):
        elu_activation = torch.nn.ELU()
        test_item_mean_emb = self.model.item_mean_embeddings.weight
        test_item_cov_emb = elu_activation(self.model.item_cov_embeddings.weight) + 1
        #test_item_mean_emb = nn.functional.normalize(self.model.item_mean_embeddings.weight)
        #test_item_cov_emb = nn.functional.normalize(elu_activation(self.model.item_cov_embeddings.weight) + 1)
        #num_items, emb_size = test_item_cov_emb.shape

        #seq_mean_out = seq_mean_out.unsqueeze(1).expand(-1, num_items, -1).reshape(-1, emb_size)
        #seq_cov_out = seq_cov_out.unsqueeze(1).expand(-1, num_items, -1).reshape(-1, emb_size)

        #if args.distance_metric == 'wasserstein':
        #    return wasserstein_distance(seq_mean_out, seq_cov_out, test_item_mean_emb, test_item_cov_emb)
        #else:
        #    return kl_distance(seq_mean_out, seq_cov_out, test_item_mean_emb, test_item_cov_emb)
        #return d2s_1overx(wasserstein_distance_matmul(seq_mean_out, seq_cov_out, test_item_mean_emb, test_item_cov_emb))
        return wasserstein_distance_matmul(seq_mean_out, seq_cov_out, test_item_mean_emb, test_item_cov_emb)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):

        str_code = "train" if train else "test"

        #rec_data_iter = tqdm.tqdm(enumerate(dataloader),
        #                          desc=f"Recommendation EP_{str_code}:{epoch}",
        #                          total=len(dataloader),
        #                          bar_format="{l_bar}{r_bar}")
        rec_data_iter = dataloader

        if train:
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0
            total_avg_loss = 0.0
            rec_avg_pvn_loss = 0.0
            rec_avg_auc = 0.0
            cl_individual_avg_losses = [0.0 for i in range(self.total_augmentaion_pairs)]
            cl_sum_avg_loss = 0.0
            joint_avg_loss = 0.0

            #for i, batch in rec_data_iter:
            i = 1
            for rec_batch, cl_batches in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                user_ids, input_ids, target_pos, target_neg, _ = rec_batch
                # bpr optimization
                sequence_mean_output, sequence_cov_output, att_scores = self.model.finetune(input_ids)
                #print(att_scores[0, 0, :, :])
                rec_loss, batch_auc, pvn_loss = self.bpr_optimization(sequence_mean_output, sequence_cov_output, target_pos, target_neg)
                #loss, batch_auc = self.ce_optimization(sequence_mean_output, sequence_cov_output, target_pos, target_neg)

                loss = rec_loss + self.args.pvn_weight * pvn_loss

                # ---------- contrastive learning task -------------#
                cl_losses = []
                for cl_batch in cl_batches:
                    cl_loss = self._one_pair_contrastive_learning(cl_batch)
                    cl_losses.append(cl_loss)

                for cl_loss in cl_losses:
                    loss += self.args.cf_weight * cl_loss
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                for i, cl_loss in enumerate(cl_losses):
                    cl_individual_avg_losses[i] += cl_loss.item()
                    cl_sum_avg_loss += self.args.cf_weight * cl_loss.item()

                rec_avg_loss += rec_loss.item()
                rec_cur_loss = rec_loss.item()
                total_avg_loss += loss.item()
                rec_avg_auc += batch_auc.item()
                rec_avg_pvn_loss += self.args.pvn_weight * pvn_loss.item()

            post_fix = {
                "epoch": epoch,
                "total_avg_loss": '{:.4f}'.format(total_avg_loss / len(rec_data_iter)),
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
                "rec_avg_auc": '{:.6f}'.format(rec_avg_auc / len(rec_data_iter)),
                "rec_avg_pvn_loss": '{:.6f}'.format(rec_avg_pvn_loss / len(rec_data_iter)),
                "cl_avg_loss": '{:.4f}'.format(cl_sum_avg_loss / (len(rec_data_iter)*self.total_augmentaion_pairs)),
            }
            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix), flush=True)

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                with torch.no_grad():
                    #for i, batch in rec_data_iter:
                    i = 0
                    for batch in rec_data_iter:
                        # 0. batch_data will be sent into the device(GPU or cpu)
                        batch = tuple(t.to(self.device) for t in batch)
                        user_ids, input_ids, target_pos, target_neg, answers = batch
                        recommend_mean_output, recommend_cov_output, _ = self.model.finetune(input_ids)

                        recommend_mean_output = recommend_mean_output[:, -1, :]
                        recommend_cov_output = recommend_cov_output[:, -1, :]

                        rating_pred = self.dist_predict_full(recommend_mean_output, recommend_cov_output)

                        rating_pred = rating_pred.cpu().data.numpy().copy()
                        batch_user_index = user_ids.cpu().numpy()
                        rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 1e+24
                        # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                        ind = np.argpartition(rating_pred, 40)[:, :40]
                        #ind = np.argpartition(rating_pred, -40)[:, -40:]
                        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                        # ascending order
                        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::]
                        #arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                        if i == 0:
                            pred_list = batch_pred_list
                            answer_list = answers.cpu().data.numpy()
                        else:
                            pred_list = np.append(pred_list, batch_pred_list, axis=0)
                            answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                        i += 1
                    return self.get_full_sort_score(epoch, answer_list, pred_list)


class SRMATrainer(Trainer):

    def __init__(self, model, cs_encoder,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader,
                 args):
        super(SRMATrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            args
        )
        self.cs_encoder = cs_encoder
        if self.cuda_condition:
            self.cs_encoder.cuda()

    def _one_pair_contrastive_learning(self, inputs):
        '''
        contrastive learning given one pair sequences (batch)
        inputs: [batch1_augmented_data, batch2_augmentated_data]
        '''
        cl_batch = torch.cat(inputs, dim=0)
        cl_batch = cl_batch.to(self.device)
        #if self.args.model_augmentation:
        cl_sequence_output = self.model.transformer_encoder(cl_batch,isTrain=True)
        #else:
        #    cl_sequence_output = self.model.transformer_encoder(cl_batch)
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        batch_size = cl_batch.shape[0]//2
        cl_output_slice = torch.split(cl_sequence_flatten, batch_size)
        cl_loss = self.cf_criterion(cl_output_slice[0],
                                cl_output_slice[1])
        return cl_loss

    def _one_pair_cl_model_aug(self, inputs):
        '''the contrastive learning with model augmentation:
        each branch adopts different models
        '''
        cl_sequence_outout = []
        for cl_batch in inputs: # each encoder has distinct model structures
            cl_batch = cl_batch.to(self.device)
            cl_sequence_embedding = self.model.transformer_encoder(cl_batch, isTrain=True)
            cl_sequence_flatten = cl_sequence_embedding.view(cl_batch.shape[0], -1)
            cl_sequence_outout.append(cl_sequence_flatten)
        cl_loss = self.cf_criterion(cl_sequence_outout[0], cl_sequence_outout[1])
        return cl_loss

    def _cl_encoder(self, inputs, original=None, encoder=None, en_weight=0.1):
        '''
        the contrastive learning loss with static encoder
        '''
        cl_batch = torch.cat(inputs, dim=0)
        # print("contrastive learning batches:",  cl_batch.shape)
        cl_batch = cl_batch.to(self.device)
        # cl_sequence_output = self.model.transformer_encoder(cl_batch,isTrain=True)
        #if self.args.model_augmentation:
        cl_sequence_output = self.model.transformer_encoder(cl_batch,isTrain=True)
        #else:
        #    cl_sequence_output = self.model.transformer_encoder(cl_batch)

        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        # cf_output = self.projection(cf_sequence_flatten)
        batch_size = cl_batch.shape[0]//2
        cl_output_slice = torch.split(cl_sequence_flatten, batch_size)
        cl_loss = self.cf_criterion(cl_output_slice[0],
                                cl_output_slice[1])
        # generate the embedding of the original sequence output from the pretrained encoder.
        ori_sequence_output = encoder.transformer_encoder(original)
        ori_sequence_flatten = ori_sequence_output.view(ori_sequence_output.shape[0], -1)
        for slice in cl_output_slice:
            cl_loss += en_weight*self.cf_criterion(slice, ori_sequence_flatten)
        return cl_loss

    def iteration(self, epoch, dataloader, full_sort=True, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar

        if train:
            self.model.train()
            rec_avg_loss = 0.0
            cl_individual_avg_losses = [0.0 for i in range(self.total_augmentaion_pairs)]
            cl_sum_avg_loss = 0.0
            joint_avg_loss = 0.0
            avg_auc = 0.0

            print(f"rec dataset length: {len(dataloader)}")
            #rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            for i, (rec_batch, cl_batches) in enumerate(dataloader):#rec_cf_data_iter:
                '''
                rec_batch shape: key_name x batch_size x feature_dim
                cl_batches shape:
                    list of n_views x batch_size x feature_dim tensors
                '''
                # 0. batch_data will be sent into the device(GPU or CPU)
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_ids, target_pos, target_neg, _ = rec_batch

                # ---------- recommendation task ---------------#
                #if self.args.model_augmentation:
                sequence_output = self.model.transformer_encoder(input_ids, model_aug=self.args.rec_model_aug, isTrain=True)
                #else:
                #    sequence_output = self.model.transformer_encoder(input_ids)


                rec_loss, batch_auc = self.cross_entropy(sequence_output, target_pos, target_neg)

                # ---------- contrastive learning task -------------#
                cl_losses = []
                for cl_batch in cl_batches:
                    # print("contrastive learning batches:",  cl_batch[0].shape)
                    if self.args.model_aug_in_batch == 'same':
                        cl_loss = self._one_pair_contrastive_learning(cl_batch)
                    elif self.args.model_aug_in_batch == 'distinct':
                        cl_loss = self._one_pair_cl_model_aug(cl_batch)
                    elif self.args.model_aug_in_batch == 'sasrec-static':
                        cl_loss = self._cl_encoder(cl_batch, input_ids, encoder=self.cs_encoder, en_weight=self.args.en_weight)
                    else:
                        raise ValueError("no %s model augmentation methods" %self.args.model_aug_in_batch)
                    cl_losses.append(cl_loss)

                joint_loss = self.args.rec_weight * rec_loss
                for cl_loss in cl_losses:
                    joint_loss += self.args.cf_weight * cl_loss
                self.optim.zero_grad()
                joint_loss.backward()
                self.optim.step()

                rec_avg_loss += rec_loss.item()
                avg_auc += batch_auc

                for i, cl_loss in enumerate(cl_losses):
                    cl_individual_avg_losses[i] += cl_loss.item()
                    cl_sum_avg_loss += cl_loss.item()
                joint_avg_loss += joint_loss.item()


            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(dataloader)),
                "auc": '{:.4f}'.format(avg_auc / len(dataloader)),
                "joint_avg_loss": '{:.4f}'.format(joint_avg_loss / len(dataloader)),
                "cl_avg_loss": '{:.4f}'.format(cl_sum_avg_loss / (len(dataloader)*self.total_augmentaion_pairs)),
            }
            for i, cl_individual_avg_loss in enumerate(cl_individual_avg_losses):
                post_fix['cl_pair_'+str(i)+'_loss'] = '{:.4f}'.format(cl_individual_avg_loss / len(dataloader))

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            #rec_data_iter = tqdm(enumerate(dataloader),
            #                      desc="Recommendation EP_%s:%d" % (str_code, epoch),
            #                      total=len(dataloader),
            #                      bar_format="{l_bar}{r_bar}")
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in enumerate(dataloader):#rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output = self.model.transformer_encoder(input_ids)

                    recommend_output = recommend_output[:, -1, :]
                    # recommendation results

                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition T: O(n)  argsort O(nlogn)
                    ind = np.argpartition(rating_pred, -40)[:, -40:]
                    #
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    #
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    #
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)

