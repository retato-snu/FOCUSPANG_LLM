#! /bin/bash
 CUDA_VISIBLE_DEVICES=1 python inference.py \
    --model polyglotko \
    --pretrain output/polyglot-ko-5.8b-lora-koChat_2 \
    --model_path output/ppo_polyglotko-5.8-1.3 \
    --input "안녕하세요" \
    --keep_going True \
   #  --early_stopping True 