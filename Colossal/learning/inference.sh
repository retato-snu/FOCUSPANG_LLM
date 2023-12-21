#! /bin/bash
 CUDA_VISIBLE_DEVICES=1 python inference.py \
    --model polyglotko \
    --pretrain /mnt/hf/polyglot-ko-5.8b \
    --model_path model_output/1220_ppo \
    --input "안녕하세요" \
    --keep_going True \
   #  --early_stopping True 