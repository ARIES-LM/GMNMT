#!/bin/bash
datapath=
tgt=de
modelname=${1}
CUDA_VISIBLE_DEVICES=3 python main.py --mode test --load_from models/${modelname} \
--test $datapath/test_2016_flickr.bpe --ref $datapath/test_2016_flickr.lc.norm.tok.$tgt \
--boxfeat $datapath/test_2016_flickr.resxyxy.pkl \
--writetrans decoding/${modelname}.2016.$tgt.b4trans --beam_size 4 >>${modelname}.tranlog 2>&1

