#!/bin/bash
datapath=

modelname=${1}
gpu=0
tgt=de

CUDA_VISIBLE_DEVICES=$gpu python main.py --mode test --load_from models/${modelname} \
--boxfeat $datapath/test_2017_mscoco.resxyxy.pkl \
--boxprobs $datapath/cocoboxprobs.pkl --test $datapath/test_2017_mscoco.bpe --ref $datapath/test_2017_mscoco.lc.norm.tok.$tgt \
--writetrans decoding/${modelname}.2017coco.b4trans --beam_size 4 >>${modelname}.tranlog 2>&1

cd multeval-0.5.1
bash run.sh $datapath/test_2017_mscoco.lc.norm.tok.$tgt ../decoding/$modelname.2017coco.b4trans $tgt >>../$modelname.tranlog



