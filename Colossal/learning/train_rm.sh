#!/bin/bash

set_n_least_used_CUDA_VISIBLE_DEVICES() {
    local n=${1:-"9999"}
    echo "GPU Memory Usage:"
    local FIRST_N_GPU_IDS=$(nvidia-smi --query-gpu=memory.used --format=csv |
        tail -n +2 |
        nl -v 0 |
        tee /dev/tty |
        sort -g -k 2 |
        awk '{print $1}' |
        head -n $n)
    export CUDA_VISIBLE_DEVICES=$(echo $FIRST_N_GPU_IDS | sed 's/ /,/g')
    echo "Now CUDA_VISIBLE_DEVICES is set to:"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
}

set_n_least_used_CUDA_VISIBLE_DEVICES 2

torchrun --standalone --nproc_per_node=2 train_reward_model.py \
    --pretrain '/mnt/hf/polyglot-ko-1.3b' \
    --model 'polyglotko' \
    --strategy colossalai_zero2 \
    --loss_fn 'log_sig' \
    --dataset 'json' \
    --data_path /mnt/FOCUSPANG_LLM/FOCUSPANG_Private/Data/Focuspang/rm_dataset/rm_dataset.json \
    --batch_size 4 \
    --max_epochs 12 \
    --save_path model_output/1220_rm2 \
    --data_bool False \
    --use_wandb \
    --without_prompt \
    # --dataset 'Anthropic/hh-rlhf' \

