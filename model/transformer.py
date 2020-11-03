import re
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F

import time
import subprocess
import pickle
from model.decoder import *
from model.optimizer import NoamOpt, CommonOpt
from model.util import *
from model.generator import Generator, greedy, beam_search


class GATEncoder(nn.Module):
    def __init__(self, d_model, d_hidden, n_heads, dropout=0.1, layer=2):
        super(GATEncoder, self).__init__()
        self.layer = layer
        self.dp = dropout
        self.d_model = d_model
        self.hid = d_hidden
        objcnndim = 2048
        self.trans_obj = nn.Sequential(Linear(objcnndim, d_model), nn.ReLU(), nn.Dropout(dropout),
                                       Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout))

        # text
        self.mhatt_x = clone(MultiHeadedAttention(n_heads, d_model, dropout), layer)
        self.ffn_x = clone(PositionwiseFeedForward(d_model, d_hidden), layer)
        self.res4ffn_x = clone(SublayerConnectionv2(d_model, dropout), layer)
        self.res4mes_x = clone(SublayerConnectionv2(d_model, dropout), layer)
        
		# img
        self.mhatt_o = clone(MultiHeadedAttention(n_heads, d_model, dropout, v=0, output=0), layer)
        self.ffn_o = clone(PositionwiseFeedForward(d_model, d_hidden), layer)
        self.res4mes_o = clone(SublayerConnectionv2(d_model, dropout), layer)
        self.res4ffn_o = clone(SublayerConnectionv2(d_model, dropout), layer)
		
        self.mhatt_x2o = clone(Linear(d_model * 2, d_model), layer)
        self.mhatt_o2x = clone(Linear(d_model * 2, d_model), layer)
        self.xgate = clone(SublayerConnectionv2(d_model, dropout), layer)
        self.ogate = clone(SublayerConnectionv2(d_model, dropout), layer)

    def forward(self, x, mask, *objs):
        #              B 1 O     B T O
        obj_feats, _, obj_mask, matrix = objs
        
        o = self.trans_obj(obj_feats)
        matrix = matrix.unsqueeze(-1)
        # B O T
        matrix4obj = torch.transpose(matrix, 1, 2)

        batch, objn, xn = matrix4obj.size(0), matrix4obj.size(1), matrix4obj.size(2)

        for i in range(self.layer):
			# Textual Self-attention, for text node
			newx = self.res4mes_x[i](x, self.mhatt_x[i](x, x, x, mask))

			# Visual Self-attention, for image node
			newo = self.res4mes_o[i](o, self.mhatt_o[i](o, o, o, obj_mask))

			# Text to Image Gating
			newx_ep = newx.unsqueeze(2).expand(batch, xn, objn, newx.size(-1))
			o_ep = newo.unsqueeze(1).expand(batch, xn, objn, o.size(-1))
			# B T O H
			x2o_gates = torch.sigmoid(self.mhatt_x2o[i](torch.cat((newx_ep, o_ep), -1)))
			x2o = (x2o_gates * matrix * o_ep).sum(2)

			# Image to Text Gating
			x_ep = newx.unsqueeze(1).expand(batch, objn, xn, newx.size(-1))
			newo_ep = newo.unsqueeze(2).expand(batch, objn, xn, o.size(-1))
			# B O T H
			o2x_gates = torch.sigmoid(self.mhatt_o2x[i](torch.cat((x_ep, newo_ep), -1)))
			o2x = (o2x_gates * matrix4obj * x_ep).sum(2)

			newx = self.xgate[i](newx, x2o)
			newo = self.ogate[i](newo, o2x)

			# using ffn to update
			x = self.res4ffn_x[i](newx, self.ffn_x[i](newx))
			o = self.res4ffn_o[i](newo, self.ffn_o[i](newo))

        return x, o

def transformer(args):
    d_model = args.d_model
    d_hidden = args.d_hidden
    n_heads = args.n_heads
    src_vocab = args.src_vocab
    trg_vocab = args.trg_vocab
    n_layers = args.n_layers
    input_dp = args.input_drop_ratio
    model_dp = args.drop_ratio
    src_emb_pos = nn.Sequential(Embeddings(d_model, src_vocab), PositionalEncoding(d_model, input_dp))
    enc_dp = args.enc_dp
    dec_dp = args.dec_dp

    encoder = GATEncoder(d_model, d_hidden, n_heads, enc_dp, args.n_enclayers)
    tgt_emb_pos = nn.Sequential(Embeddings(d_model, trg_vocab), PositionalEncoding(d_model, input_dp))
    decoder = Decoder(clone(DecoderLayer(d_model, n_heads, d_hidden, dec_dp), n_layers))
    generator = Generator(d_model, trg_vocab)
    model = EncoderDecoder(encoder, decoder, src_emb_pos, tgt_emb_pos, generator)

    if args.share_vocab:
        model.src_embed[0].lut.weight = model.tgt_embed[0].lut.weight

    if args.share_embed:
        model.generator.proj.weight = model.tgt_embed[0].lut.weight

    return model


def print_params(model):
    print('total parameters:', sum([np.prod(list(p.size())) for p in model.parameters()]))

def train(args, train_iter, dev, src, tgt, checkpoint):
    model = transformer(args)
    print(model)
    print_params(model)

    best_bleu = 0.0
    best_iter = 0
    offset = 0

    srcpadid = src.vocab.stoi['<pad>']
    tgtpadid = tgt.vocab.stoi['<pad>']


    if checkpoint is not None:
        print('model.load_state_dict(checkpoint[model])')
        model.load_state_dict(checkpoint['model'])
    model.cuda()

    if args.optimizer == 'Noam':
        adamopt = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
        opt = NoamOpt(args.d_model, args.lr, args.warmup, adamopt, args.grad_clip, args.delay)
    else:
        raise NotImplementedError

    if args.resume:
        opt.optimizer.load_state_dict(checkpoint['optim'])
        offset = checkpoint['iters']
        opt.set_steps(offset)
        best_bleu = checkpoint['bleu']
        print('*************************************')
        print('resume from {} iters and best_bleu {}'.format(offset, best_bleu))
        print('*************************************')

    #criterion = nn.NLLLoss(ignore_index=tgtpadid, reduction='sum')
    criterion = nn.KLDivLoss(reduction='sum')

    start = time.time()

    update_steps = args.maximum_steps
    maxsteps = update_steps * args.delay

    patience = args.patience
    
    smoothing = 0.1
    confidence = 1.0 - smoothing

    smoothing_value = smoothing / (args.trg_vocab - 2)
    one_hot = torch.full((args.trg_vocab,), smoothing_value)
    one_hot[tgtpadid] = 0
    one_hot = one_hot.unsqueeze(0).cuda()
    
    allboxfeats = pickle.load(open(args.boxfeat[0], 'rb'))
    valboxfeats = pickle.load(open(args.boxfeat[1], 'rb'))
	
    boxprobs = pickle.load(open(args.boxprobs, 'rb'))
	
    topk = 5
	thre = 0.0
    objdim = args.objdim
	
    for iters, train_batch in enumerate(train_iter):
        iters += offset

        if iters > maxsteps:
            print('reached the maximum updating steps.')
            break

        model.train()

        t1 = time.time()

        sources, source_masks = prepare_sources(train_batch.src, srcpadid, args.share_vocab)
        target_inputs, target_outputs, target_ipmasks, n_tokens = prepare_targets(train_batch.trg, tgtpadid)

        imgs, aligns, regions_num = train_batch.extra_0

        # B Tobj
        obj_feat = sources.new_zeros(sources.size(0), max(regions_num), topk, objdim).float()
        # B 1 Tobj*topk
        obj_mask = source_masks.new_zeros(sources.size(0), max(regions_num)*topk)
        
        # B Tx Tobj*topk
        matrix = sources.new_zeros(sources.size(0), sources.size(1), max(regions_num)*topk).float()
        
        for ib, img in enumerate(imgs):
            # phrase_num, 5, 2048 (numpy)
            boxfeat = torch.tensor(allboxfeats[img]).reshape(-1, 5, objdim)
            
            # phrase_num * 5
            img_boxprobs = torch.tensor(boxprobs[img])
            
            ge_thre = (img_boxprobs >= thre).byte()
            # keep top 1
            ge_thre[list(range(0, ge_thre.size(0), 5))] = 1
            obj_mask[ib, :ge_thre.size(0)] = ge_thre
            obj_feat[ib, :boxfeat.size(0)] = boxfeat[:, :topk]

            for item in aligns[ib]:
                ## item: text_word_id, object_id
                objixs = sources.new_tensor([n+item[1] * topk for n in range(topk)])
                matrix[ib, item[0], objixs] = ge_thre[objixs].float().cuda()

        # batch_size, objectnum, objdim
        obj_feat = obj_feat.view(sources.size(0), -1, objdim)
        obj_mask = obj_mask.unsqueeze(1)

        outputs = model.forward(sources, target_inputs, source_masks, target_ipmasks,
                                obj_feat, None, obj_mask, matrix)
        
        truth_p = one_hot.repeat(target_outputs.size(0), target_outputs.size(1), 1)
        truth_p.scatter_(2, target_outputs.unsqueeze(2), confidence)
        truth_p.masked_fill_((target_outputs == tgtpadid).unsqueeze(2), 0)
        loss = criterion(outputs, truth_p)

        if torch.isnan(loss):
            exit('loss nan!!!!!!!!!')

        norm = n_tokens.float()
        loss = loss / norm
        loss = loss / args.delay

        loss.backward()

        opt.step()

        output_loss = loss.item() * args.delay

        # loss = 0
        t2 = time.time()
        print('iters:{} src:({},{}) tgt:({},{}) loss:{:.2f} t:{:.2f} lr:{:.1e}'.format(iters + 1, *sources.size(),
                                                                                       *target_inputs.size(),
                                                                                       output_loss, t2 - t1, opt._rate))
        # from 0 to check error
        if (iters + 1) % (args.eval_every * args.delay) == 0:
            with torch.no_grad():
                score = valid_model(args, model, dev, src, tgt, valboxfeats, boxprobs)
                print('iters: {} bleu: {} best: {}'.format(iters + 1, score, best_bleu))
                if score > best_bleu:
                    best_bleu = score
                    best_iter = iters

                    print('save best model at iter={}'.format(iters + 1))
                    checkpoint = {'model': model.state_dict(),
                                  'optim': opt.optimizer.state_dict(),
                                  'args': args,
                                  'bleu': best_bleu}

                    torch.save(checkpoint, '{}/{}.best.pt'.format(args.model_path, args.model))


        if (iters + 1) % (args.save_every * args.delay) == 0:
            number = 0
            # for args.resume to continue training
            print('save (back-up) checkpoints at iter={}'.format(iters + 1))
            checkpoint = {'model': model.state_dict(),
                          'optim': opt.optimizer.state_dict(),
                          'args': args,
                          'bleu': best_bleu,
                          'iters': iters + 1}
            torch.save(checkpoint, '{}/{}.{}.backup.pt'.format(args.model_path, args.model, number))

    with torch.no_grad():
        score = valid_model(args, model, dev, src, tgt, valboxfeats, boxprobs)
        print('iters: {} bleu: {} best: {}'.format(iters + 1, score, best_bleu))
        if score > best_bleu:
            best_bleu = score
            best_iter = iters + 1

            print('save best model at last')
            checkpoint = {'model': model.state_dict(),
                          'optim': opt.optimizer.state_dict(),
                          'args': args,
                          'bleu': best_bleu}
            torch.save(checkpoint, '{}/{}.best.pt'.format(args.model_path, args.model))

    print('*******Done********{}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    minutes = (time.time() - start) // 60
    if minutes < 60:
        print('best:{}, iter:{}, time:{} mins, lr:{:.1e}, '.format(best_bleu, best_iter, minutes, opt._rate))
    else:
        hours = minutes / 60
        print('best:{}, iter:{}, time:{:.1f} hours, lr:{:.1e}, '.format(best_bleu, best_iter, hours, opt._rate))


def valid_model(args, model, dev, src, tgt, allboxfeats, boxprobs, dev_metrics=None):
    model.eval()

    initid = tgt.vocab.stoi['<init>']
    eosid = tgt.vocab.stoi['<eos>']
    srcpadid = src.vocab.stoi['<pad>']

    f = open(args.writetrans, 'w', encoding='utf-8')

    dev.init_epoch()
    if args.beam_size != 1:
        print('beam search with beam', args.beam_size)

    decoding_time = []
    
    topk = 5
    objdim = args.objdim
    thre = 0
    for j, dev_batch in enumerate(dev):
        sources, source_masks = prepare_sources(dev_batch.src, srcpadid, args.share_vocab)

        t1 = time.time()

        imgs, aligns, regions_num = dev_batch.extra_0
        
        # B Tobj
        obj_feat = sources.new_zeros(sources.size(0), max(regions_num), topk, objdim).float()
        # B 1 Tobj*topk
        obj_mask = source_masks.new_zeros(sources.size(0), max(regions_num)*topk)
        
        # B Tx Tobj*topk
        matrix = sources.new_zeros(sources.size(0), sources.size(1), max(regions_num)*topk).float()
        for ib, img in enumerate(imgs):
            # phrase_num, 5, 4096 (numpy)
            boxfeat = torch.tensor(allboxfeats[img]).reshape(-1, 5, objdim)
            # phrase_num * 5
            img_boxprobs = torch.tensor(boxprobs[img])
            ge_thre = (img_boxprobs >= thre).byte()
            # keep top 1
            ge_thre[list(range(0, ge_thre.size(0), 5))] = 1
            obj_mask[ib, :ge_thre.size(0)] = ge_thre
            obj_feat[ib, :boxfeat.size(0)] = boxfeat[:, :topk]

            for item in aligns[ib]:
                objixs = sources.new_tensor([n+item[1] * topk for n in range(topk)])
                matrix[ib, item[0], objixs] = ge_thre[objixs].float().cuda()
        
        obj_feat = obj_feat.view(sources.size(0), -1, objdim)
        obj_mask = obj_mask.unsqueeze(1)
        
        
        if args.beam_size == 1:
            translations_id = greedy(args, model, sources, source_masks, initid, eosid)
        else:
            translations_id = beam_search(args, model, sources, source_masks, initid, eosid, obj_feat, None,
                                          obj_mask, matrix)

        t2 = time.time()
        decoding_time.append(t2 - t1)
        translations = tgt.reverse(translations_id.detach(), unbpe=True)
        for trans in translations:
            print(trans, file=f)

    f.close()

    status, bleuinfo = subprocess.getstatusoutput(
        'perl scripts/multi-bleu.perl -lc {} < {}'.format(args.ref, args.writetrans))
    bleu = re.findall(r'BLEU = (.*?),', bleuinfo)

    if len(bleu) == 0:
        print('bleu', bleuinfo)
        return 0
    print('average decoding latency: {} ms'.format(int(np.mean(decoding_time) * 1000)))
    return float(bleu[0])


def decode(args, testset, src, tgt, checkpoint):
    with torch.no_grad():
        model = transformer(args)
        model.cuda()

        print('load parameters')
        model.load_state_dict(checkpoint['model'])

        valboxfeats = pickle.load(open(args.boxfeat[0], 'rb'))
        boxprobs = pickle.load(open(args.boxprobs, 'rb'))

        score = valid_model(args, model, testset, src, tgt, valboxfeats, boxprobs)
        print('bleu', score)



class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.loss_computer = None

    def forward(self, src, tgt, src_mask, tgt_mask, *objs):
        hx, ho = self.encode(src, src_mask, *objs)
        dec_outputs = self.decode(hx, src_mask, tgt, ho, objs[-2], tgt_mask)
        return self.generator(dec_outputs)

    def encode(self, src, src_mask, *objs):
        return self.encoder(self.src_embed(src), src_mask, *objs)

    def decode(self, memory, src_mask, tgt, objmem, objmask, tgt_mask=None):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, objmem, objmask, tgt_mask)

    def generate(self, dec_outputs):
        return self.generator(dec_outputs)

    def addposition(self, x):
        return self.tgt_embed[1](x)

