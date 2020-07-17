import torch
import numpy as np
from torchtext import data
import logging
import random
import argparse

from data.data import *
import time

from model.transformer import train, decode
from pathlib import Path
import json


def dyn_batch_with_padding(new, i, sofar):
    prev_max_len = sofar / (i - 1) if i > 1 else 0
    return max(len(new.src), len(new.trg), prev_max_len) * i


def dyn_batch_without_padding(new, i, sofar):
    return sofar + max(len(new.src), len(new.trg))


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Transformer / FastTransformer.')

    # dataset settings
    parser.add_argument('--corpus_prex', type=str)
    parser.add_argument('--lang', type=str, nargs='+', help="the suffix of the corpus, translation language")
    parser.add_argument('--valid', type=str)

    parser.add_argument('--writetrans', type=str, help='write translations for to a file')
    parser.add_argument('--ref', type=str, help='references, word unit')

    parser.add_argument('--vocab', type=str)
    parser.add_argument('--vocab_size', type=int, default=40000)
    parser.add_argument('--boxfeat', type=str, nargs='+')
    parser.add_argument('--boxprobs', type=str)
	
    parser.add_argument('--n_enclayers', type=int)
    parser.add_argument('--objdim', type=int)
    parser.add_argument('--enc_dp', type=float)
    parser.add_argument('--img_dp', type=float)
    parser.add_argument('--dec_dp', type=float)
    parser.add_argument('--load_vocab', action='store_true', help='load a pre-computed vocabulary')

    parser.add_argument('--max_len', type=int, default=None, help='limit the train set sentences to this many tokens')
    parser.add_argument('--pool', type=int, default=100, help='shuffle batches in the pool')

    # model name
    parser.add_argument('--model', type=str, default='[time]', help='prefix to denote the model, nothing or [time]')

    # network settings
    parser.add_argument('--share_embed', action='store_true', default=False,
                        help='share embeddings and linear out weight')
    parser.add_argument('--share_vocab', action='store_true', default=False,
                        help='share vocabulary between src and target')

    parser.add_argument('--params', type=str, default='user', choices=['user', 'small', 'middle', 'base'],
                        help='Defines the dimension size of the parameter')
    parser.add_argument('--n_layers', type=int, default=5, help='number of layers')
    parser.add_argument('--n_heads', type=int, default=2, help='number of heads')
    parser.add_argument('--d_model', type=int, default=278, help='dimention size for hidden states')
    parser.add_argument('--d_hidden', type=int, default=507, help='dimention size for FFN')
    parser.add_argument('--initnn', default='standard', help='parameter init')

    # running setting
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--seed', type=int, default=1234, help='seed for randomness')

    # training
    parser.add_argument('--eval_every', type=int, default=100, help='validate every * step')
    parser.add_argument('--save_every', type=int, default=-1, help='save model every * step (5000)')

    parser.add_argument('--batch_size', type=int, default=2048, help='# of tokens processed per batch')
    parser.add_argument('--delay', type=int, default=1, help='gradiant accumulation for delayed update for large batch')

    parser.add_argument('--optimizer', type=str, default='Noam')
    parser.add_argument('--lr', type=float, default=1.0, help='learning rate')
    parser.add_argument('--warmup', type=int, default=4000, help='maximum steps to linearly anneal the learning rate')

    # lr decay
    parser.add_argument('--lrdecay', action='store_true', default=False, help='learning rate decay 0.5')
    parser.add_argument('--patience', type=int, default=0, help='learning rate decay 0.5')

    parser.add_argument('--maximum_steps', type=int, default=5000000, help='maximum steps you take to train a model')
    parser.add_argument('--drop_ratio', type=float, default=0.1, help='dropout ratio')
    parser.add_argument('--input_drop_ratio', type=float, default=0.1, help='dropout ratio only for inputs')
    parser.add_argument('--grad_clip', type=float, default=-1.0, help='gradient clipping')
    parser.add_argument('--smoothing', type=float, default=0.0, help='label smoothing')

    # decoding
    parser.add_argument('--length_ratio', type=float, default=2, help='maximum lengths of decoding')
    parser.add_argument('--beam_size', type=int, default=1,
                        help='beam-size used in Beamsearch, default using greedy decoding')
    parser.add_argument('--alpha', type=float, default=0.6, help='length normalization weights')
    # parser.add_argument('--T', type=float, default=1, help='softmax temperature when decoding')
    parser.add_argument('--test', type=str, default=None, help='test src file')

    parser.add_argument('--load_from', default=None)
    parser.add_argument('--resume', action='store_true',
                        help='when resume, need other things besides parameters')
    # save path
    parser.add_argument('--main_path', type=str, default="./")
    parser.add_argument('--model_path', type=str, default="models")
    parser.add_argument('--decoding_path', type=str, default="decoding")

    return parser.parse_args()


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def curtime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def override(args, load_dict, except_name):
    for k in args.__dict__:
        if k not in except_name:
            args.__dict__[k] = load_dict[k]

if __name__ == '__main__':
    args = parse_args()

    if args.mode == 'train':
        if args.load_from is not None:
            load_from = args.load_from
            print('{} load the checkpoint from {} for initilize or resume'.
                  format(curtime(), load_from))
            checkpoint = torch.load(load_from, map_location='cpu')
        else:
            checkpoint = None

        # if not resume(initilize), only need model parameters
        if args.resume:
            print('update args from checkpoint')
            load_dict = checkpoint['args'].__dict__
            except_name = ['mode', 'resume', 'maximum_steps']
            override(args, load_dict, tuple(except_name))

        main_path = Path(args.main_path)
        model_path = main_path / args.model_path
        decoding_path = main_path / args.decoding_path

        for path in [model_path, decoding_path]:
            path.mkdir(parents=True, exist_ok=True)

        args.model_path = str(model_path)
        args.decoding_path = str(decoding_path)

        if args.model == '[time]':
            args.model = time.strftime("%m.%d_%H.%M.", time.gmtime())

        if args.params == 'small':
            hparams = {'d_model': 278, 'd_hidden': 507, 'n_layers': 5, 'n_heads': 2, 'warmup': 746}
            args.__dict__.update(hparams)
        elif args.params == 'base':
            hparams = {'d_model': 512, 'd_hidden': 2048, 'n_layers': 6, 'n_heads': 8}
            args.__dict__.update(hparams)
        elif args.params == 'middle':
            hparams = {'d_model': 256, 'd_hidden': 256, 'n_layers': 5, 'n_heads': 4}
            args.__dict__.update(hparams)

        # setup random seeds
        if args.seed == 1234:
           set_seeds(args.seed)

        DataField = NormalField
        TRG = DataField(init_token='<init>', eos_token='<eos>', batch_first=True)
        SRC = DataField(eos_token='<eos>', batch_first=True) if not args.share_vocab else TRG

        GRA = GraphField(batch_first=True)
        train_data = ParallelDataset(path=args.corpus_prex, exts=args.lang, fields=(SRC, TRG, GRA))

        if args.max_len is not None:
            alldata = train_data.examples
            train_data.examples = list(filter(lambda ex: len(ex.src) <= args.max_len and
                                                         len(ex.trg) <= args.max_len, alldata))

        dev_data = ParallelDataset(path=args.valid, exts=args.lang, fields=(SRC, TRG, GRA))

        # build vocabularies for translation dataset
        vocab_path = Path(args.vocab)
        vocab_path = vocab_path / '{}_{}_{}.pt'.format('{}-{}'.format(args.lang[0], args.lang[1]),
                                                       args.vocab_size,
                                                       'shared' if args.share_vocab else '')
        if args.load_vocab and vocab_path.exists():
            src_vocab, trg_vocab = torch.load(str(vocab_path))
            SRC.vocab = src_vocab
            TRG.vocab = trg_vocab
            print('vocab {} loaded'.format(str(vocab_path)))
        else:
            assert (not train_data is None)
            if not args.share_vocab:
                SRC.build_vocab(train_data, max_size=args.vocab_size)
            TRG.build_vocab(train_data, max_size=args.vocab_size)

			print('save the vocabulary')
			vocab_path.parent.mkdir(parents=True, exist_ok=True)
			torch.save([SRC.vocab, TRG.vocab], str(vocab_path))

        args.__dict__.update({'trg_vocab': len(TRG.vocab), 'src_vocab': len(SRC.vocab)})

        train_flag = True
        train_real = data.BucketIterator(train_data, args.batch_size, device='cuda',
                                         batch_size_fn=dyn_batch_with_padding,
                                         train=train_flag, repeat=train_flag,
                                         shuffle=train_flag, sort_within_batch=True, sort=False)

        devbatch = 20 if args.beam_size == 1 else 1
        dev_real = data.Iterator(dev_data, devbatch, device='cuda', batch_size_fn=None,
                                 train=False, repeat=False, shuffle=False, sort=False)
        args_str = json.dumps(args.__dict__, indent=4, sort_keys=True)
        print(args_str)

        print('{} Start training'.format(curtime()))
        train(args, train_real, dev_real, SRC, TRG, checkpoint)

    else:
        if len(args.load_from) == 1:
            load_from = '{}.best.pt'.format(args.load_from)
            print('{} load the best checkpoint from {}'.format(curtime(), load_from))
            checkpoint = torch.load(load_from, map_location='cpu')
        else:
            raise RuntimeError('must load model')

        # when translate load_dict update args except some
        # print('update args from checkpoint')
        load_dict = checkpoint['args'].__dict__
        except_name = ['boxprobs', 'mode', 'load_from', 'test', 'ref', 'writetrans', 'beam_size', 'batch_size', 'boxfeat']
        override(args, load_dict, tuple(except_name))

        args_str = json.dumps(args.__dict__, indent=4, sort_keys=True)
        print(args_str)

        print('{} Load test set'.format(curtime()))
        DataField = NormalField
        TRG = DataField(init_token='<init>', eos_token='<eos>', batch_first=True)
        SRC = DataField(eos_token='<eos>', batch_first=True) if not args.share_vocab else TRG
        GRA = GraphField(batch_first=True)

        vocab_path = Path(args.vocab)
        vocab_path = vocab_path / '{}_{}_{}.pt'.format('{}-{}'.format(args.lang[0], args.lang[1]), args.vocab_size,
                                                       'shared' if args.share_vocab else '')
        if args.load_vocab and vocab_path.exists():
            src_vocab, trg_vocab = torch.load(str(vocab_path))
            SRC.vocab = src_vocab
            TRG.vocab = trg_vocab
            print('vocab {} loaded'.format(str(vocab_path)))
        else:
            raise RuntimeError('no vocab')
        args.__dict__.update({'trg_vocab': len(TRG.vocab), 'src_vocab': len(SRC.vocab)})

        test_data = ParallelDataset(path=args.test, exts=args.lang, fields=(SRC, TRG, GRA))
        batch_size = 20 if args.beam_size == 1 else 1
        test_real = data.Iterator(test_data, batch_size, device='cuda', batch_size_fn=None,
                                  train=False, repeat=False, shuffle=False, sort=False)

        print('{} Load data done'.format(curtime()))
        start = time.time()
        decode(args, test_real, SRC, TRG, checkpoint)
        print('{} Decode done, time {} mins'.format(curtime(), (time.time() - start) / 60))
