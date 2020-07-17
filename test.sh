#!/bin/bash
datapath=

modelname=${1}
tgt=${2}
gpu=${3}

UDA_VISIBLE_DEVICES=$gpu python -u main.py --mode test --load_from models/${modelname} \
--test $datapath/test_2016_flickr.bpe --ref $datapath/test_2016_flickr.lc.norm.tok.$tgt \
--boxfeat $datapath/test_2016_flickr.resxyxy.pkl --boxprobs $datapath/boxporbs.pkl \
--writetrans decoding/${modelname}.2016.$tgt.b4trans --beam_size 4 >>${modelname}.tranlog 2>&1

CUDA_VISIBLE_DEVICES=$gpu python -u main.py --mode test --load_from models/${modelname} \
--test $datapath/test_2017_flickr.bpe --ref $datapath/test_2017_flickr.lc.norm.tok.$tgt \
--boxfeat $datapath/test_2017_flickr.resxyxy.pkl --boxprobs $datapath/boxporbs.pkl \
--writetrans decoding/${modelname}.2017.$tgt.b4trans --beam_size 4 >>${modelname}.tranlog 2>&1

#cd multeval-0.5.1
#bash run.sh $datapath/test_2016_flickr.lc.norm.tok.$tgt ../decoding/$modelname.2016.$tgt.b4trans $tgt >>../$modelname.tranlog
#bash run.sh $datapath/test_2017_flickr.lc.norm.tok.$tgt ../decoding/$modelname.2017.$tgt.b4trans $tgt >>../$modelname.tranlog

