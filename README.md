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
### 1. Create anaconda environment follwing ColssalAI's setting video.  
https://www.youtube.com/watch?v=-qFBZFmOJfg&t=198s  

### 2. Install our dependencies
`pip install -r requirements.txt`  

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


