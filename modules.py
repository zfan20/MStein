# -*- coding:utf-8 -*-

import numpy as np

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def d2s_gaussiannormal(distance, gamma):

    return torch.exp(-gamma*distance)

def d2s_1overx(distance):

    return 1/(1+distance)

def wasserstein_distance(mean1, cov1, mean2, cov2):
    ret = torch.sum((mean1 - mean2) * (mean1 - mean2), -1)
    cov1_sqrt = torch.sqrt(torch.clamp(cov1, min=1e-24))
    cov2_sqrt = torch.sqrt(torch.clamp(cov2, min=1e-24))
    ret = ret + torch.sum((cov1_sqrt - cov2_sqrt) * (cov1_sqrt - cov2_sqrt), -1)

    return ret

def wasserstein_distance_matmul(mean1, cov1, mean2, cov2):
    mean1_2 = torch.sum(mean1**2, -1, keepdim=True)
    mean2_2 = torch.sum(mean2**2, -1, keepdim=True)
    ret = -2 * torch.matmul(mean1, mean2.transpose(-1, -2)) + mean1_2 + mean2_2.transpose(-1, -2)
    #ret = torch.clamp(-2 * torch.matmul(mean1, mean2.transpose(-1, -2)) + mean1_2 + mean2_2.transpose(-1, -2), min=1e-24)
    #ret = torch.sqrt(ret)

    cov1_2 = torch.sum(cov1, -1, keepdim=True)
    cov2_2 = torch.sum(cov2, -1, keepdim=True)
    #cov_ret = torch.clamp(-2 * torch.matmul(torch.sqrt(torch.clamp(cov1, min=1e-24)), torch.sqrt(torch.clamp(cov2, min=1e-24)).transpose(-1, -2)) + cov1_2 + cov2_2.transpose(-1, -2), min=1e-24)
    #cov_ret = torch.sqrt(cov_ret)
    cov_ret = -2 * torch.matmul(torch.sqrt(torch.clamp(cov1, min=1e-24)), torch.sqrt(torch.clamp(cov2, min=1e-24)).transpose(-1, -2)) + cov1_2 + cov2_2.transpose(-1, -2)

    return ret + cov_ret


def kl_distance(mean1, cov1, mean2, cov2):
    trace_part = torch.sum(cov1 / cov2, -1)
    mean_cov_part = torch.sum((mean2 - mean1) / cov2 * (mean2 - mean1), -1)
    determinant_part = torch.log(torch.prod(cov2, -1) / torch.prod(cov1, -1))

    return (trace_part + mean_cov_part - mean1.shape[1] + determinant_part) / 2

def kl_distance_matmul(mean1, cov1, mean2, cov2):
    cov1_det = 1 / torch.prod(cov1, -1, keepdim=True)
    cov2_det = torch.prod(cov2, -1, keepdim=True)
    log_det = torch.log(torch.matmul(cov1_det, cov2_det.transpose(-1, -2)))

    trace_sum = torch.matmul(1 / cov2, cov1.transpose(-1, -2))

    mean_cov_part = torch.matmul((mean1 - mean2) ** 2, (1/cov2).transpose(-1, -2))

    return (log_det + mean_cov_part + trace_sum - mean1.shape[-1]) / 2


class NCELoss(nn.Module):
    """
    Eq. (12): L_{NCE}
    """
    def __init__(self, temperature, device):
        super(NCELoss, self).__init__()
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.temperature = temperature
        self.cossim = nn.CosineSimilarity(dim=-1).to(self.device)
        
    # #modified based on impl: https://github.com/ae-foster/pytorch-simclr/blob/dc9ac57a35aec5c7d7d5fe6dc070a975f493c1a5/critic.py#L5
    def forward(self, batch_sample_one, batch_sample_two):
        sim11 = torch.matmul(batch_sample_one, batch_sample_one.T) / self.temperature
        sim22 = torch.matmul(batch_sample_two, batch_sample_two.T) / self.temperature
        sim12 = torch.matmul(batch_sample_one, batch_sample_two.T) / self.temperature
        d = sim12.shape[-1]
        sim11[..., range(d), range(d)] = float('-inf')
        sim22[..., range(d), range(d)] = float('-inf')
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * d, dtype=torch.long, device=logits.device)
        nce_loss = self.criterion(logits, labels)
        return nce_loss



class WassersteinNCELoss(nn.Module):
    def __init__(self, temperature, device):
        super(WassersteinNCELoss, self).__init__()
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.temperature = temperature
        self.activation = nn.ELU()

    # #modified based on impl: https://github.com/ae-foster/pytorch-simclr/blob/dc9ac57a35aec5c7d7d5fe6dc070a975f493c1a5/critic.py#L5
    def forward(self, batch_sample_one_mean, batch_sample_one_cov, batch_sample_two_mean, batch_sample_two_cov):
        #batch_sample_one_mean = nn.functional.normalize(batch_sample_one_mean)
        #batch_sample_two_mean = nn.functional.normalize(batch_sample_two_mean)
        #batch_sample_one_cov = nn.functional.normalize(self.activation(batch_sample_one_cov) + 1)
        #batch_sample_two_cov = nn.functional.normalize(self.activation(batch_sample_two_cov) + 1)
        batch_sample_one_cov = self.activation(batch_sample_one_cov) + 1
        batch_sample_two_cov = self.activation(batch_sample_two_cov) + 1
        #batch_sample_one_cov = torch.ones_like(batch_sample_one_cov) 
        #batch_sample_two_cov = torch.ones_like(batch_sample_one_cov) 
        sim11 = d2s_1overx(wasserstein_distance_matmul(batch_sample_one_mean, batch_sample_one_cov, batch_sample_one_mean, batch_sample_one_cov)) / self.temperature
        sim22 = d2s_1overx(wasserstein_distance_matmul(batch_sample_two_mean, batch_sample_two_cov, batch_sample_two_mean, batch_sample_two_cov)) / self.temperature
        sim12 = d2s_1overx(wasserstein_distance_matmul(batch_sample_one_mean, batch_sample_one_cov, batch_sample_two_mean, batch_sample_two_cov)) / self.temperature
        d = sim12.shape[-1]
        sim11[..., range(d), range(d)] = float('-inf')
        sim22[..., range(d), range(d)] = float('-inf')
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * d, dtype=torch.long, device=logits.device)
        nce_loss = self.criterion(logits, labels)
        return nce_loss


class NTXent(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    code: https://github.com/AndrewAtanov/simclr-pytorch/blob/master/models/losses.py
    """
    LARGE_NUMBER = 1e9

    def __init__(self, tau=1., gpu=None, multiplier=2, distributed=False):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed
        self.norm = 1.

    def forward(self, batch_sample_one, batch_sample_two):
        z = torch.cat([batch_sample_one, batch_sample_two], dim=0)
        n = z.shape[0]
        assert n % self.multiplier == 0

        z = F.normalize(z, p=2, dim=1) / np.sqrt(self.tau)
        logits = z @ z.t()
        logits[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER

        logprob = F.log_softmax(logits, dim=1)

        # choose all positive objects for an example, for i it would be (i + k * n/m), where k=0...(m-1)
        m = self.multiplier
        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n//m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)

        # TODO: maybe different terms for each process should only be computed here...
        loss = -logprob[np.repeat(np.arange(n), m-1), labels].sum() / n / (m-1) / self.norm
        return loss

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)



ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Embeddings(nn.Module):
    """Construct the embeddings from item, position.
    """
    def __init__(self, args):
        super(Embeddings, self).__init__()

        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0) # 不要乱用padding_idx
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.args = args

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        items_embeddings = self.item_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = items_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads))
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)

        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class DistSelfAttention(nn.Module):
    def __init__(self, args):
        super(DistSelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads))
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.mean_query = nn.Linear(args.hidden_size, self.all_head_size)
        self.cov_query = nn.Linear(args.hidden_size, self.all_head_size)
        self.mean_key = nn.Linear(args.hidden_size, self.all_head_size)
        self.cov_key = nn.Linear(args.hidden_size, self.all_head_size)
        self.mean_value = nn.Linear(args.hidden_size, self.all_head_size)
        self.cov_value = nn.Linear(args.hidden_size, self.all_head_size)

        self.activation = nn.ELU()

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.mean_dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.cov_dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

        self.distance_metric = args.distance_metric
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.gamma = args.kernel_param


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_mean_tensor, input_cov_tensor, attention_mask):
        mixed_mean_query_layer = self.mean_query(input_mean_tensor)
        mixed_mean_key_layer = self.mean_key(input_mean_tensor)
        mixed_mean_value_layer = self.mean_value(input_mean_tensor)

        mean_query_layer = self.transpose_for_scores(mixed_mean_query_layer)
        mean_key_layer = self.transpose_for_scores(mixed_mean_key_layer)
        mean_value_layer = self.transpose_for_scores(mixed_mean_value_layer)

        mixed_cov_query_layer = self.activation(self.cov_query(input_cov_tensor)) + 1
        mixed_cov_key_layer = self.activation(self.cov_key(input_cov_tensor)) + 1
        mixed_cov_value_layer = self.activation(self.cov_value(input_cov_tensor)) + 1

        cov_query_layer = self.transpose_for_scores(mixed_cov_query_layer)
        cov_key_layer = self.transpose_for_scores(mixed_cov_key_layer)
        cov_value_layer = self.transpose_for_scores(mixed_cov_value_layer)

        if self.distance_metric == 'wasserstein':
            #attention_scores = d2s_gaussiannormal(wasserstein_distance_matmul(mean_query_layer, cov_query_layer, mean_key_layer, cov_key_layer), self.gamma)
            attention_scores = -wasserstein_distance_matmul(mean_query_layer, cov_query_layer, mean_key_layer, cov_key_layer)
        else:
            attention_scores = -kl_distance_matmul(mean_query_layer, cov_query_layer, mean_key_layer, cov_key_layer)

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.attn_dropout(attention_probs)
        mean_context_layer = torch.matmul(attention_probs, mean_value_layer)
        cov_context_layer = torch.matmul(attention_probs ** 2, cov_value_layer)
        mean_context_layer = mean_context_layer.permute(0, 2, 1, 3).contiguous()
        cov_context_layer = cov_context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = mean_context_layer.size()[:-2] + (self.all_head_size,)

        mean_context_layer = mean_context_layer.view(*new_context_layer_shape)
        cov_context_layer = cov_context_layer.view(*new_context_layer_shape)

        mean_hidden_states = self.mean_dense(mean_context_layer)
        mean_hidden_states = self.out_dropout(mean_hidden_states)
        mean_hidden_states = self.LayerNorm(mean_hidden_states + input_mean_tensor)

        cov_hidden_states = self.cov_dense(cov_context_layer)
        cov_hidden_states = self.out_dropout(cov_hidden_states)
        cov_hidden_states = self.LayerNorm(cov_hidden_states + input_cov_tensor)

        return mean_hidden_states, cov_hidden_states, attention_probs



class Intermediate(nn.Module):
    def __init__(self, args):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        if isinstance(args.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[args.hidden_act]
        else:
            self.intermediate_act_fn = args.hidden_act

        self.dense_2 = nn.Linear(args.hidden_size * 4, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)


    def forward(self, input_tensor):

        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class InterLayer(nn.Module):
    def __init__(self, args):
        super(InterLayer, self).__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size)
        if isinstance(args.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[args.hidden_act]
        else:
            self.intermediate_act_fn = args.hidden_act

        self.dense_2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)


    def forward(self, input_tensor):

        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class SingleInterLayer(nn.Module):
    def __init__(self, args):
        super(SingleInterLayer, self).__init__()
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        if isinstance(args.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[args.hidden_act]
        else:
            self.intermediate_act_fn = args.hidden_act
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_tensor):
        hidden_states = self.dense(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class DistIntermediate(nn.Module):
    def __init__(self, args):
        super(DistIntermediate, self).__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        self.intermediate_act_fn = nn.ELU()

        self.dense_2 = nn.Linear(args.hidden_size * 4, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)


    def forward(self, input_tensor):

        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class SingleDistInterLayer(nn.Module):
    def __init__(self, args):
        super(SingleDistInterLayer, self).__init__()
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.intermediate_act_fn = nn.ELU()
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_tensor):
        hidden_states = self.dense(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states



class Layer(nn.Module):
    def __init__(self, args):
        super(Layer, self).__init__()
        self.attention = SelfAttention(args)
        self.intermediate = Intermediate(args)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output


class DistLayer(nn.Module):
    def __init__(self, args):
        super(DistLayer, self).__init__()
        self.attention = DistSelfAttention(args)
        self.mean_intermediate = DistIntermediate(args)
        self.cov_intermediate = DistIntermediate(args)
        self.activation_func = nn.ELU()

    def forward(self, mean_hidden_states, cov_hidden_states, attention_mask):
        mean_attention_output, cov_attention_output, attention_scores = self.attention(mean_hidden_states, cov_hidden_states, attention_mask)
        mean_intermediate_output = self.mean_intermediate(mean_attention_output)
        cov_intermediate_output = self.activation_func(self.cov_intermediate(cov_attention_output)) + 1
        return mean_intermediate_output, cov_intermediate_output, attention_scores


class DistSAEncoder(nn.Module):
    def __init__(self, args):
        super(DistSAEncoder, self).__init__()
        layer = DistLayer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, mean_hidden_states, cov_hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            maen_hidden_states, cov_hidden_states, att_scores = layer_module(mean_hidden_states, cov_hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append([mean_hidden_states, cov_hidden_states, att_scores])
        if not output_all_encoded_layers:
            all_encoder_layers.append([mean_hidden_states, cov_hidden_states, att_scores])
        return all_encoder_layers


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class LayerDrop(nn.Module):
    def __init__(self, args):
        super(LayerDrop, self).__init__()
        self.aug_layer_type = args.aug_layer_type
        if self.aug_layer_type == 'single':
            self.augment_layers = nn.ModuleList(SingleInterLayer(args) for _ in range(args.num_augmented_layers))
        elif self.aug_layer_type == 'inter':
            self.augment_layers = nn.ModuleList(Intermediate(args) for _ in range(args.num_augmented_layers))
        elif self.aug_layer_type == 'sasrec':
            self.augment_layers = nn.ModuleList(Layer(args) for _ in range(args.num_augmented_layers))
        elif self.aug_layer_type == 'single_res':
            self.augment_layers = nn.ModuleList(SingleInterLayer(args) for _ in range(args.num_augmented_layers))
            
        self.layer_drop_num = args.layer_drop_num
        self.num_aug_layers = args.num_augmented_layers
    
    def forward(self, hidden_states, attention_mask, isTrain=False):
        if isTrain: # drop layers during training
            drop_idx = random.sample(range(0, self.num_aug_layers), self.layer_drop_num)
            # print('drop idx:',drop_idx)
            for idx, layer_module in enumerate(self.augment_layers):
                if idx not in drop_idx:
                    if self.aug_layer_type in ['single','inter']:
                        hidden_states = layer_module(hidden_states)
                    elif self.aug_layer_type in ['sasrec']:
                        hidden_states = layer_module(hidden_states,attention_mask)
        else:
            for layer_module in self.augment_layers:
                if self.aug_layer_type in ['single','inter']:
                    hidden_states = layer_module(hidden_states)
                elif self.aug_layer_type in ['sasrec']:
                    hidden_states = layer_module(hidden_states,attention_mask)
        return hidden_states

class LayerDropProb(nn.Module):
    def __init__(self, args):
        super(LayerDropProb, self).__init__()
        self.aug_layer_type = args.aug_layer_type
        if self.aug_layer_type == 'single':
            self.augment_layers = nn.ModuleList(SingleInterLayer(args) for _ in range(args.num_augmented_layers))
        elif self.aug_layer_type == 'inter':
            self.augment_layers = nn.ModuleList(Intermediate(args) for _ in range(args.num_augmented_layers))
        elif self.aug_layer_type == 'sasrec':
            self.augment_layers = nn.ModuleList(Layer(args) for _ in range(args.num_augmented_layers))
        elif self.aug_layer_type == 'single_res':
            self.augment_layers = nn.ModuleList(SingleInterLayer(args) for _ in range(args.num_augmented_layers))
            
        self.layer_drop_num = args.layer_drop_num
        self.num_aug_layers = args.num_augmented_layers
        self.layer_drop_thres = args.layer_drop_thres
    
    def forward(self, hidden_states, attention_mask, isTrain=False):
        if isTrain: # drop layers during training
            for idx, layer_module in enumerate(self.augment_layers):
                if random.random() < self.layer_drop_thres:
                    if self.aug_layer_type in ['single','inter']:
                        hidden_states = layer_module(hidden_states)
                    elif self.aug_layer_type in ['sasrec']:
                        hidden_states = layer_module(hidden_states,attention_mask)
        else:
            for layer_module in self.augment_layers:
                if self.aug_layer_type in ['single','inter']:
                    hidden_states = layer_module(hidden_states)
                elif self.aug_layer_type in ['sasrec']:
                    hidden_states = layer_module(hidden_states,attention_mask)
        return hidden_states


class EncoderAug(nn.Module):
    def __init__(self, args):
        super(EncoderAug, self).__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])
        if args.layer_drop_thres >= 0:
            self.aug_layer = LayerDropProb(args)
        else:
            self.aug_layer = LayerDrop(args)
    
    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, model_aug=True, isTrain=False):
        all_encoder_layers = []
        for idx, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        # add the augmented layers
        if model_aug:
            hidden_states = self.aug_layer(hidden_states,attention_mask,isTrain)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class EncoderDrop(nn.Module):
    def __init__(self, args):
        super(EncoderDrop, self).__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])
        self.aug_layer_type = args.aug_layer_type
        if self.aug_layer_type == 'single':
            self.augment_layers = nn.ModuleList(SingleInterLayer(args) for _ in range(args.num_augmented_layers))
        elif self.aug_layer_type == 'inter':
            self.augment_layers = nn.ModuleList(Intermediate(args) for _ in range(args.num_augmented_layers))
        elif self.aug_layer_type == 'sasrec':
            self.augment_layers = nn.ModuleList(Layer(args) for _ in range(args.num_augmented_layers))
        self.layer_drop_num = args.layer_drop_num
        self.num_aug_layers = args.num_augmented_layers
    
    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, isTrain=False):
        # augmenting only on the last dense layers
        all_encoder_layers = []
        # construct the basic Transformer layers
        for idx, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)    
        # build the model augmentation from intermidate layers
        if isTrain: # drop layers during training
            drop_idx = random.sample(range(0, self.num_aug_layers), self.layer_drop_num)
            for idx, layer_module in enumerate(self.augment_layers):
                if idx not in drop_idx:
                    if self.aug_layer_type in ['single','inter']:
                        hidden_states = layer_module(hidden_states)
                    elif self.aug_layer_type in ['sasrec']:
                        hidden_states = layer_module(hidden_states,attention_mask)
                    all_encoder_layers.append(hidden_states)
        else:
            for layer_module in self.augment_layers:
                if self.aug_layer_type in ['single','inter']:
                    hidden_states = layer_module(hidden_states)
                elif self.aug_layer_type in ['sasrec']:
                    hidden_states = layer_module(hidden_states,attention_mask)
                all_encoder_layers.append(hidden_states)
        return all_encoder_layers
