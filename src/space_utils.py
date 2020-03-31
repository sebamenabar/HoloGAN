import math
import torch
from torch.nn.utils.rnn import pad_sequence


def sample_gaussean(mean, loc):
    dist = torch.distributions.normal.Normal(mean, loc)
    return dist.rsample()


def sample_gaussean_sp(mean, log_loc):
    return sample_gaussean(mean, torch.nn.functional.softplus(log_loc))


def sample_bernoulli(p_logits, hard=True, clamp=False, temperature=1):
    dist = torch.distributions.RelaxedBernoulli(temperature=1, logits=p_logits)
    obj_prob = dist.rsample()  # .to(device=p_logits.device)
    if hard:  # Use ST-trick
        obj_prob_hard = (obj_prob >= 0.5).to(dtype=torch.float)
        return (obj_prob_hard - obj_prob).detach() + obj_prob, obj_prob
    else:
        return obj_prob, obj_prob


def manual_sample_obj_pres(p_logits, hard=True, clamp=False, eps=1e-20):
    if clamp:
        # In original SPAIR implementation logits al clamped
        p_logits = torch.clamp(p_logits, -10.0, 10.0)

    # Gumbel-softmax trick
    u = torch.rand(p_logits.size())  # [0,1) uniform
    # Sample Gumbel
    noise = torch.log(u + eps) - torch.log(1.0 - u + eps)
    # Sample bernoulli
    obj_pre_sigmoid = p_logits + noise
    obj_prob = torch.sigmoid(obj_pre_sigmoid)

    # Use
    if hard:
        obj_prob_hard = (obj_prob >= 0.5).to(p_logits.dtype)
        return (obj_prob_hard - obj_prob).detach() + obj_prob, obj_prob
    else:
        return obj_prob, obj_prob


def select_and_pad_on_presence(features, presence):
    indexes = presence >= 0.5
    lengths = indexes.sum((1, 2)).tolist()
    sequences = features[indexes].split(lengths, 0)
    return pad_sequence(sequences, batch_first=True)


def process_decoded_transform(transform):
    transform[..., :3] = torch.fmod(transform[..., :3], math.pi)  # .chunk(3, -1)
    transform[..., 3:6] = torch.clamp(transform[..., 3:6], 0.5, 1.5)  # .chunk(3, -1)
    transform[..., 6:] = torch.clamp(transform[..., 6:], -1, 1)  # .chunk(3, -1)
    return transform
