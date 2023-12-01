#! /bin/bash
 CUDA_VISIBLE_DEVICES=1 python inference.py \
    --model polyglotko \
    --pretrain model_output/polyglot-ko-5.8b-rdata \
    --model_path model_output/ppo_polyglotko-5.8b-125-focus \
    --input "안녕하세요" \
    --keep_going True \
   #  --early_stopping True 