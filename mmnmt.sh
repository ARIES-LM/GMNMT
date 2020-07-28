#!/bin/bash
datapath=
tgt=de
modelname=
gpu=$1
CUDA_VISIBLE_DEVICES=$gpu python -u main.py --model ${modelname} \
--corpus_prex $datapath/train.bpe --lang en $tgt graph \
--valid $datapath/val.bpe --img_dp 0.5 --objdim 2048 --seed 1234 \
--boxprobs $datapath/boxporbs.pkl --n_encobj 3 --n_enclayers 3 \
--writetrans decoding/${modelname}.devtrans \
--boxfeat $datapath/train.resxyxy.pkl $datapath/val.resxyxy.pkl \
--ref $datapath/val.lc.norm.tok.$tgt --batch_size 2000 --delay 1 --warmup 4000 \
--vocab $datapath --vocab_size 40000 --load_vocab --smoothing 0.1 --share_embed --share_vocab --beam_size 4 \
--params user --lr 1.0 --init standard --enc_dp 0.5 --dec_dp 0.5 --input_drop_ratio 0.5 \
--n_layers 4 --n_heads 4 --d_model 128 --d_hidden 256 \
--max_len 100 --eval_every 2000 --save_every 5000 --maximum_steps 80000 >${modelname}.train 2>&1

bash test.sh $modelname $tgt $gpu
