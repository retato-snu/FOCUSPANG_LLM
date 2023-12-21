#!/bin/bash

OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output/jaeyoung/1208/sft
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

deepspeed train_sft.py \
   --data_path rubis_sft \
   --data_split 2,4,4 \
   --model_name_or_path '/mnt/hf/polyglot-ko-5.8b' \
   --per_device_train_batch_size 6 \
   --per_device_eval_batch_size 6 \
   --max_seq_len 512 \
   --learning_rate 9.65e-6 \
   --weight_decay 0. \
   --num_train_epochs 30  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
