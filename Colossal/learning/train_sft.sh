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
    echo "Now CUDA_VISIBLE_DEVICE1S is set to:"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
}

set_n_least_used_CUDA_VISIBLE_DEVICES 2

torchrun --standalone --nproc_per_node=2 train_sft.py \
    --pretrain "/mnt/hf/polyglot-ko-5.8b" \
    --model 'polyglotko' \
    --strategy colossalai_zero2_cpu \
    --save_path model_output/polyglot-ko-5.8b-focuspang-peft \
    --dataset /mnt/FOCUSPANG_LLM/FOCUSPANG_Private/Data/Focuspang/dataset_source_231009_created_231009/line_human-touched.json \
    --batch_size 4 \
    --accumulation_steps 8 \
    --lr 2e-5 \
    --max_epochs 5 \
    --lora_rank 8 \
    --language 'ko' \
    --without_prompt \
    --use_wandb
