import torch
import numpy as np
import torch.nn as nn
import itertools
from torch.nn import functional as F
from model.util import Linear, make_subsequent_mask


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = Linear(d_model, vocab, False)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

    def greedyscore(self, x):
        return self.proj(x)


class Beam(object):
    def __init__(self, beam_size):
        self.beam_size = beam_size

        self.candidates = []
        self.scores = []

    def step(self, prob, prev_beam, f_done):
        pre_score = prob.new_tensor(prev_beam.scores)
        score = prob + pre_score.unsqueeze(-1).expand_as(prob)
        nbest_score, nbest_ix = score.view(-1).topk(self.beam_size, largest=False)
        beam_ix = nbest_ix / prob.size(1)
        token_ix = nbest_ix - beam_ix * prob.size(1)

        done_list, remain_list = [], []
        prev_candidates = prev_beam.candidates
        for b_score, b_ix, t_ix in itertools.zip_longest(nbest_score.tolist(), beam_ix.tolist(), token_ix.tolist()):
            candidate = prev_candidates[b_ix] + [t_ix]

            if f_done(candidate):
                done_list.append([candidate, b_score])
            else:
                remain_list.append(b_ix)
                self.candidates.append(candidate)
                self.scores.append(b_score)
        return done_list, remain_list


def greedy(args, model, src, src_mask, initid, eosid):
    encodings = model.encode(src, src_mask)
    # torch.set_printoptions(profile='short')

    B, T, H = encodings.size()

    target_len = int(T * args.length_ratio)

    outs = torch.LongTensor(B, 1).fill_(initid).to(src.device)

    # wordemb + hidden
    # hiddens = [torch.Tensor(B, T, H).zero_() for _ in range(args.n_layers + 1)]

    eos_yet = torch.zeros(B, device=src.device, dtype=torch.uint8)
    for t in range(target_len):
        # hiddens[0][:, t] = model.tgt_embed(outs[:, t])
        # for l in range(args.n_layers):
        #     y = hiddens[l][:, :t+1]
        # B t+1 H
        subseq_mask = make_subsequent_mask(outs.size(1))
        subseq_mask = subseq_mask.to(src.device)

        # print('causual')
        # print(subseq_mask.detach())

        y = model.decode(encodings, src_mask, outs, subseq_mask)
        preds = model.generate(y[:, -1])

        max_value, max_index = preds.max(-1)

        outs = torch.cat([outs, max_index.unsqueeze(1)], -1)

        eos_yet = eos_yet | (max_index == eosid)
        if eos_yet.all():
            break
    return outs[:, 1:]


def beam_search(args, model, src, src_mask, initid, eosid, *objs):
    # src_mask 1 1 T
    obj_mask = objs[-2]
    encoding, encoding_obj = model.encode(src, src_mask, *objs)

    _, T, H = encoding.size()
    # print('eosid', eosid) #3
    W = args.beam_size
    alpha = args.alpha

    max_len = int(T * args.length_ratio)
    min_len = T // 2

    prev_beam = Beam(W)
    prev_beam.candidates = [[initid]]
    prev_beam.scores = [0]
    f_done = (lambda x: x[-1] == eosid)
    valid_size = W
    hyp_list = []

    # all position first
    allposition = model.addposition(encoding.new_zeros(1, max_len, H))
    hiddens = encoding.new_zeros(1, max_len + 1, args.n_layers + 1, H)
    for t in range(max_len):
        candidates = prev_beam.candidates
        # B
        input = src.new_tensor(list(map(lambda cand: cand[-1], candidates)))
        # B H
        hiddens[:, t, 0] = model.tgt_embed[0](input) + allposition[:, t]
        for l in range(args.n_layers):
            # B H
            hiddens[:, t, l + 1] = model.decoder.layers[l].search(hiddens[:, t:t + 1, l], hiddens[:, :t + 1, l],
                                                                      encoding, src_mask).view(-1, H)
        # B V
        log_prob = model.generate(hiddens[:, t, -1])
        if t < min_len:
            log_prob[:, eosid] = -float('inf')
        if t == max_len - 1:
            eos_prob = log_prob[:, eosid].clone()
            log_prob[:, :] = -float('inf')
            log_prob[:, eosid] = eos_prob

        next_beam = Beam(valid_size)
        done_list, remain_list = next_beam.step(-log_prob, prev_beam, f_done)
        hyp_list.extend(done_list)
        valid_size -= len(done_list)

        if valid_size == 0:
            break

        beam_remain_ix = src.new_tensor(remain_list)
        encoding = encoding.index_select(0, beam_remain_ix)  # select batch dim
        src_mask = src_mask.index_select(0, beam_remain_ix)

        encoding_obj = encoding_obj.index_select(0, beam_remain_ix)
        obj_mask = obj_mask.index_select(0, beam_remain_ix)

        hiddens = hiddens.index_select(0, beam_remain_ix)
        prev_beam = next_beam

    score_list = [hyp[1] for hyp in hyp_list]
    hyp_list = [hyp[0][1: hyp[0].index(eosid)] if eosid in hyp[0] else hyp[0][1:] for hyp in hyp_list]

    for k, (hyp, score) in enumerate(zip(hyp_list, score_list)):
        if len(hyp) > 0:
            lp = (5 + len(hyp)) / (5 + 1)
            lp = lp ** alpha
            score_list[k] = score_list[k] / lp

    score = hiddens.new_tensor(score_list)
    sort_score, sort_ix = torch.sort(score)
    output = []
    for ix in sort_ix.tolist():
        output.append((hyp_list[ix], score[ix].item()))
    # add batch dim
    output = src.new_tensor([output[0][0]])
    return output
