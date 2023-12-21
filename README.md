# FOCUSPANG_LLM

A project for 2023-Fall SNU Creative integrated design lecture (Team L)

## FOCUSPANG

포커스팡은 온라인 학습과정에서 학생들의 다양한 행동 패턴을 분석하여 수업에 집중을 잘 하고 있는지 판단을 도와주고, 성적 분포를 분석하여 앞으로의 학습 계획을 도와주는 서비스이다.
이 프로젝트에서는 성적 분포를 분석하여 학생들의 전체적인 학습 수행 능력에 대한 피드백을 주는 LLM 모델을 설계하고 제작하는 것을 목표로 한다.

##

## Training Structure

![image](https://github.com/retato-snu/FOCUSPANG_LLM/assets/50572383/ce8d17fa-2d49-4939-a2bd-94ef1a56c810)

## Colossal

# How to run our codes

## Install dependencies

Before start, Need miniconda3

### 1. Create anaconda environment

Create env

`conda create -n env_name python=3.10.13`

Install torch

`conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia`

Install Transfomer

`git clone https://github.com/hpcaitech/transformers.git`

`cd transformer`

`pip install .`

`cd ..`

Install flash-attention

`pip install flash-attn --no-build-isolation`

Install xformers

`conda install xformers -c xformers`

Install Colossal & Coati

`cd Colossal`

`pip install .`

### 2. Install dependencies for learning

`cd learning`

`pip install -r requirements.txt`

### 3. Clone pretrained model (Additional Option)

`git lfs install`

- polyglot-ko-1.3b `git clone https://huggingface.co/EleutherAI/polyglot-ko-1.3b`
- polyglot-ko-5.8b `git clone https://huggingface.co/EleutherAI/polyglot-ko-5.8b`
- polyglot-ko-12.8b `git clone https://huggingface.co/EleutherAI/polyglot-ko-12.8b`

Without clone pretrained model, Parameter pretained part should be written like: EleutherAI/polyglot-ko-1.3b

This will be saved cache, so clone model and use local file is recommended.

### 4. Additional dependency for learning with 8bit (Additional Option)

If you want to learn with 8bit for limit of GPU memory, use this version: [Quantization learning version](https://github.com/retato-snu/FOCUSPANG_LLM/tree/colossal_load8bit).

You need to additional dependency for this.

`pip install peft`

## Training

### 1. SFT model

In `FOCUSPANG_LLM/Colossal/learning`, you can use `train_sft.sh` script.

### 2. RM model

In `FOCUSPANG_LLM/Colossal/learning`, you can use `train_rm.sh` script.

### 3. PPO model

In `FOCUSPANG_LLM/Colossal/learning`, you can use `train_prompt.sh` script.

## Inference

Example command

```
 CUDA_VISIBLE_DEVICES=1 python inference.py \
    --model polyglotko \
    --pretrain /mnt/hf/polyglot-ko-5.8b \
    --model_path $YOUR_PPO_MODEL_PATH \
    --input $YOUR_INPUT \
```

## Implement Detail

Information about the files/code developed ourselved can be found in the README.md provided below.

- [Colossal](https://github.com/retato-snu/FOCUSPANG_LLM/blob/master/Colossal/README.md)
- [DeepSpeed](https://github.com/retato-snu/FOCUSPANG_LLM/blob/master/Colossal/README.md)
- [Dataset](https://github.com/retato-snu/FOCUSPANG_LLM/blob/master/Colossal/README.md)
