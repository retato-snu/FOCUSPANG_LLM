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

# torchrun --standalone --nproc_per_node=2 train_prompts.py prompts.csv --strategy colossalai_zero2

torchrun --standalone --nproc_per_node=2 train_prompts.py \
    --model polyglotko \
    --pretrain model_output/1220_sft3 \
    --pretrain_dataset /mnt/FOCUSPANG_LLM/FOCUSPANG_Private/Data/Focuspang/sft_dataset/sft_dataset1.json  \
    --prompt_dataset /mnt/FOCUSPANG_LLM/FOCUSPANG_Private/Data/Focuspang/ppo_dataset/ppo_dataset.json \
    --strategy colossalai_zero2 \
    --rm_model gpt-neox \
    --rm_pretrain "/mnt/hf/polyglot-ko-1.3b" \
    --rm_path model_output/1218_rm2 \
    --num_episodes 100 --num_collect_steps 2 --num_update_steps 1 \
    --train_batch_size 4 \
    --save_path model_output/1220_ppo3 \
    --lora_rank 8 \
    --language 'ko' \
    --instruction_str "instruction"\
    --output_str "output" \
    --ppo_instruction_str "prompt" \
    --use_wandb \