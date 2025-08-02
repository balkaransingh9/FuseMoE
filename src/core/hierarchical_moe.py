import torch
from torch import nn
import torch.nn.functional as F

import math
from inspect import isfunction
import pdb

# constants

MIN_EXPERT_CAPACITY = 4

# helper functions

def default(val, default_val):
    default_val = default_val() if isfunction(default_val) else default_val
    return val if val is not None else default_val

def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)

# tensor related helper functions

def top1(t):
    values, index = t.topk(k=1, dim=-1)
    values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
    return values, index

def cumsum_exclusive(t, dim=-1):
    '''
    Performs cumsum along the group size dimension
    '''
    num_dims = len(t.shape)
    num_pad_dims = - dim - 1
    pre_padding = (0, 0) * num_pad_dims
    pre_slice   = (slice(None),) * num_pad_dims
    # last two dimensions are padded first
    # TODO: why need an extra dimension?
    padded_t = F.pad(t, (*pre_padding, 1, 0)).cumsum(dim=dim)
    return padded_t[(..., slice(None, -1), *pre_slice)]

# pytorch one hot throws an error if there are out of bound indices.
# tensorflow, in contrast, does not throw an error
def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    return F.one_hot(indexes, max(max_index + 1, max_length))[..., :max_length]

def init_(t):
    dim = t.shape[-1]
    std = 1 / math.sqrt(dim)
    return t.uniform_(-std, std)

# activations

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

# expert class

class Experts(nn.Module):
    def __init__(self,
        input_dim,
        output_dim,
        num_experts = 16,
        hidden_dim = None,
        activation = GELU):
        super().__init__()

        hidden_dim = default(hidden_dim, input_dim * 4)
        num_experts = cast_tuple(num_experts)

        w1 = torch.zeros(*num_experts, input_dim, hidden_dim)
        w2 = torch.zeros(*num_experts, hidden_dim, output_dim)

        w1 = init_(w1)
        w2 = init_(w2)

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.act = activation()

    def forward(self, x):
        hidden = torch.einsum('...nd,...dh->...nh', x, self.w1)
        hidden = self.act(hidden)
        out    = torch.einsum('...nh,...hd->...nd', hidden, self.w2)
        return out

# the below code is almost all transcribed from the official tensorflow version, from which the papers are written
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/moe.py

# TODO: extend this to top-N gating function, at least for level 2, but we stick with 2 levels
# TODO: compare this with sparse MoE on performance and time in CIFAR and MIMIC, think of ways to optimize computation of HME
# TODO: how to adapt sparse_moe in sequential input

class Top2Gating(nn.Module):
    def __init__(
        self,
        dim,
        num_gates,
        eps = 1e-9,
        outer_expert_dims = tuple(),
        second_policy_train = 'random',
        second_policy_eval = 'random',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.):
        super().__init__()

        self.eps = eps
        self.num_gates = num_gates
        self.w_gating = nn.Parameter(torch.randn(*outer_expert_dims, dim, num_gates))

        self.second_policy_train = second_policy_train
        self.second_policy_eval = second_policy_eval
        self.second_threshold_train = second_threshold_train
        self.second_threshold_eval = second_threshold_eval
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval

    def forward(self, x, importance = None):
        # batch size, group size (?), feature dimension
        # *_, b, group_size, dim = x.shape
        if len(x.shape) < 3:
            group_size = x.shape[1]
        else:
            *_, b, group_size, dim = x.shape
        
        num_gates = self.num_gates

        if self.training:
            policy = self.second_policy_train
            threshold = self.second_threshold_train
            capacity_factor = self.capacity_factor_train
        else:
            policy = self.second_policy_eval
            threshold = self.second_threshold_eval
            capacity_factor = self.capacity_factor_eval

        # [4, 1024, 512], [512, 16] -> [4, 1024, 16]
        if len(x.shape) < 3:
            raw_gates = torch.einsum('...nd,...de->...ne', x, self.w_gating)
        else:
            raw_gates = torch.einsum('...bnd,...de->...bne', x, self.w_gating)
        # TODO: here change the softmax gating
        raw_gates = raw_gates.softmax(dim=-1)

        # FIND TOP 2 EXPERTS PER POSITON
        # Find the top expert for each position. shape=[batch, group]

        gate_1, index_1 = top1(raw_gates)
        mask_1 = F.one_hot(index_1, num_gates).float()
        density_1_proxy = raw_gates
        pdb.set_trace()
        if importance is not None:
            equals_one_mask = (importance == 1.).float()
            # added an extra dimension on mask
            mask_1 *= equals_one_mask[..., None]
            gate_1 *= equals_one_mask
            density_1_proxy = density_1_proxy * equals_one_mask[..., None]
            del equals_one_mask

        gates_without_top_1 = raw_gates * (1. - mask_1)

        # the second largest gate and its index
        gate_2, index_2 = top1(gates_without_top_1)
        mask_2 = F.one_hot(index_2, num_gates).float()

        if importance is not None:
            greater_zero_mask = (importance > 0.).float()
            mask_2 *= greater_zero_mask[..., None]
            del greater_zero_mask

        # normalize top2 gate scores
        # gate_1 + gate_2 pretty much equals 1, or less than one
        # so this normalization drag the value closer to 1
        denom = gate_1 + gate_2 + self.eps
        gate_1 /= denom
        gate_2 /= denom

        # BALANCING LOSSES
        # TODO: is this CV loss, how this is computed?
        # shape = [batch, experts]
        # We want to equalize the fraction of the batch assigned to each expert
        density_1 = mask_1.mean(dim=-2)
        # Something continuous that is correlated with what we want to equalize.
        density_1_proxy = density_1_proxy.mean(dim=-2)
        # the more unbalance of the fraction, the greater the loss function is
        loss = (density_1_proxy * density_1).mean() * float(num_gates ** 2)

        # Depending on the policy in the hparams, we may drop out some of the
        # second-place experts.
        if policy == "all":
            pass
        elif policy == "none":
            mask_2 = torch.zeros_like(mask_2)
        elif policy == "threshold":
            mask_2 *= (gate_2 > threshold).float()
        elif policy == "random":
            probs = torch.zeros_like(gate_2).uniform_(0., 1.)
            mask_2 *= (probs < (gate_2 / max(threshold, self.eps))).float().unsqueeze(-1)
        else:
            raise ValueError(f"Unknown policy {policy}")

        # Each sequence sends (at most?) expert_capacity positions to each expert (all experts?).
        # Static expert_capacity dimension is needed for expert batch sizes
        # expert number increase, then capacity will decrease
        # TODO: what is the intuition of the expert_capacity computed?
        expert_capacity = min(group_size, int((group_size * capacity_factor) / num_gates))
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
        expert_capacity_f = float(expert_capacity)

        # COMPUTE ASSIGNMENT TO EXPERTS
        # [batch, group (seq_len), experts]
        # This is the position within the expert's mini-batch for this sequence
        # don't forget it is multiplied by mask_1, so it is not monotonically increasing anymore
        position_in_expert_1 = cumsum_exclusive(mask_1, dim=-2) * mask_1
        # Remove the elements that don't fit. [batch, group, experts]
        # the utility of cumsum is to help truncate the sentence at the position that is beyond capacity
        mask_1 *= (position_in_expert_1 < expert_capacity_f).float()
        # [batch, experts]
        # How many examples in this sequence go to this expert
        mask_1_count = mask_1.sum(dim=-2, keepdim=True)
        # [batch, group] - mostly ones, but zeros where something didn't fit
        # indicates whether an element in a sentence is assigned a top expert or not
        # the reason it does not be assigned is that expert is out of capacity
        # each element has its top-1 expert, which is not the same across different elements
        mask_1_flat = mask_1.sum(dim=-1)
        # [batch, group]
        # indicates how many elements up-to-now in a sentence has been assigned to their top-1 experts
        position_in_expert_1 = position_in_expert_1.sum(dim=-1)
        # Weight assigned to first expert.  [batch, group]
        # i.e. at how much probability that this element at this batch number is assigned to this expert
        gate_1 *= mask_1_flat

        # there are less elements in the sequence (length 1024) that is assigned to the second-largest experts
        # some experts may have already reached capacity due to the first-preference of some elements
        # so need to add the capacity offset for these experts first
        position_in_expert_2 = cumsum_exclusive(mask_2, dim=-2) + mask_1_count
        position_in_expert_2 *= mask_2
        mask_2 *= (position_in_expert_2 < expert_capacity_f).float()
        mask_2_flat = mask_2.sum(dim=-1)

        position_in_expert_2 = position_in_expert_2.sum(dim=-1)
        gate_2 *= mask_2_flat
        
        # [batch, group, experts, expert_capacity]
        # basically extend to 4 dimensions
        combine_tensor = (
            gate_1[..., None, None] # [4, 1024, 1, 1]
            * mask_1_flat[..., None, None] # [4, 1024, 1, 1] duplicate multiplication? but does not matter
            * F.one_hot(index_1, num_gates)[..., None] # [4, 1024, 16, 1] this is just mask_1
            * safe_one_hot(position_in_expert_1.long(), expert_capacity)[..., None, :] + # [4, 1024, 1, 80]
            gate_2[..., None, None]
            * mask_2_flat[..., None, None]
            * F.one_hot(index_2, num_gates)[..., None]
            * safe_one_hot(position_in_expert_2.long(), expert_capacity)[..., None, :]
        ) # [4, 1024, 16, 80]
        dispatch_tensor = combine_tensor.bool().to(combine_tensor)
        return dispatch_tensor, combine_tensor, loss

# plain mixture of experts

class MoE(nn.Module):
    def __init__(self,
        input_dim,
        output_dim = None,
        num_experts = 16,
        hidden_dim = None,
        activation = nn.ReLU,
        second_policy_train = 'random',
        second_policy_eval = 'random',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        loss_coef = 1e-2,
        experts = None):
        super().__init__()

        self.num_experts = num_experts
        if output_dim is None:
            output_dim = input_dim
        self.output_dim = output_dim
        gating_kwargs = {'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval, 'second_threshold_train': second_threshold_train, 'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval}
        self.gate = Top2Gating(input_dim, num_gates = num_experts, **gating_kwargs)
        self.experts = default(experts, lambda: Experts(input_dim, output_dim, num_experts = num_experts, hidden_dim = hidden_dim, activation = activation))
        self.loss_coef = loss_coef

    def forward(self, inputs, **kwargs):
        d, e = inputs.shape[-1], self.num_experts
        dispatch_tensor, combine_tensor, loss = self.gate(inputs)
        # first eliminate the seq_len dimension
        if len(inputs.shape) < 3:
            inputs = inputs[:, None, :]
            dispatch_tensor = dispatch_tensor[:, None, ...]
            combine_tensor = combine_tensor[:, None, ...]
        expert_inputs = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor)

        # Now feed the expert inputs through the experts.
        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(e, -1, d)
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape[:-1], self.output_dim)

        # output should be the same shape with input to propagate to the next MoE layer
        # weighted combine, eliminate the expert dimension
        output = torch.einsum('ebcd,bnec->bnd', expert_outputs, combine_tensor)
        return output, loss * self.loss_coef

# 2-level heirarchical mixture of experts

class HierarchicalMoE(nn.Module):
    def __init__(self,
        input_dim,
        output_dim = None,
        num_experts = (4, 4),
        hidden_dim = None,
        activation = nn.ReLU,
        second_policy_train = 'random',
        second_policy_eval = 'random',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        loss_coef = 1e-2,
        experts = None):
        super().__init__()

        assert len(num_experts) == 2, 'only 2 levels of heirarchy for experts allowed for now'
        if output_dim is None:
            output_dim = input_dim
        num_experts_outer, num_experts_inner = num_experts
        self.num_experts_outer = num_experts_outer
        self.num_experts_inner = num_experts_inner
        self.output_dim = output_dim

        gating_kwargs = {'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval, 'second_threshold_train': second_threshold_train, 'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval}

        self.gate_outer = Top2Gating(input_dim, num_gates = num_experts_outer, **gating_kwargs)
        self.gate_inner = Top2Gating(input_dim, num_gates = num_experts_inner, outer_expert_dims = (num_experts_outer,), **gating_kwargs)

        self.experts = default(experts, lambda: Experts(input_dim, output_dim, num_experts = num_experts, hidden_dim = hidden_dim, activation = activation))
        self.loss_coef = loss_coef

    def forward(self, inputs, **kwargs):
        d, eo, ei = inputs.shape[-1], self.num_experts_outer, self.num_experts_inner
        dispatch_tensor_outer, combine_tensor_outer, loss_outer = self.gate_outer(inputs)
        # (1) transform into 2-dimensions (4, 512), (2) open another two dimensions on experts, capacity
        # dispatch input for higher(?)-level, partition input in a relatively coarse degree
        if len(inputs.shape) < 3:
            inputs = inputs[:, None, :]
            dispatch_tensor_outer = dispatch_tensor_outer[:, None, ...]
            combine_tensor_outer = combine_tensor_outer[:, None, ...]
        expert_inputs_outer = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor_outer)

        # we construct an "importance" Tensor for the inputs to the second-level
        # gating.  The importance of an input is 1.0 if it represents the
        # first-choice expert-group and 0.5 if it represents the second-choice expert
        # group.  This is used by the second-level gating.
        importance = combine_tensor_outer.permute(2, 0, 3, 1).sum(dim=-1) # sum over the seq_len dimension
        # TODO: 0.5 as the boundary?
        importance = 0.5 * ((importance > 0.5).float() + (importance > 0.).float())

        # the extra dimension is because of the one more dimensionality of expert_inputs_outer than inputs
        dispatch_tensor_inner, combine_tensor_inner, loss_inner = self.gate_inner(expert_inputs_outer, importance = importance)
        # further dispatch input for lower-level, partition in a relatively fine-grained degree
        # since taking into account the extra high-level top-2 information, this is an extra dimension
        # the extra dimension e is the number of gates in the first level
        expert_inputs = torch.einsum('ebnd,ebnfc->efbcd', expert_inputs_outer, dispatch_tensor_inner)

        # Now feed the expert inputs through the experts.
        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(eo, ei, -1, d)
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape[:-1], self.output_dim)

        # NOW COMBINE EXPERT OUTPUTS (reversing everything we have done)
        # expert_output has shape [y0, x1, h, d, n]

        # lower-level gating
        expert_outputs_outer = torch.einsum('efbcd,ebnfc->ebnd', expert_outputs, combine_tensor_inner)
        # higher-level gating
        output = torch.einsum('ebcd,bnec->bnd', expert_outputs_outer, combine_tensor_outer)
        return output, (loss_outer + loss_inner) * self.loss_coef