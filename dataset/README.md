# Datasets

Datasets for project for 2023-Fall SNU Creative integrated design lecture (Team L)

## Structure

강화학습의 3단계(SFT, RM, PPO)에 사용되는 세 종류의 데이터셋으로 구성되어 있다.
각 데이터셋의 위치는 다음과 같지만, 개인정보 보안의 이유로 이 repository에서는 공개되지 않는다.
```
dataset/
├ sft/sft_dataset.json : sft 데이터셋
├ rm/rm_dataset.json : rm 데이터셋
├ ppo/ppo_dataset.json : ppo 데이터셋
├ code/ : 데이터셋 augmentation에 사용된 코드들
```

### 1. SFT(Supervised Fine-Tuning) Dataset

SFT 단계의 데이터셋은 오픈 데이터셋과 실제 사용자 데이터셋 두 가지가 준비되어있다.
데이터셋의 각 데이터의 형식은 다음과 같다.
```
{
    "instruction": (질문),
    "input": (질문과 같이 주는 입력),
    "output": (질문에 대한 답)
}
```

### 2. RM(Reward Model) Dataset

RM 단계에서 사용되는 데이터셋이다. 실제 사용자 데이터의 질문에 대해, GPT로 다양한 quality의 답변을 생성한 후 그 질문에 대한 답변들의 점수를 각각 매겼다.
데이터셋의 각 데이터의 형식은 다음과 같다.
```
{
    "instruction": (질문),
    "completion_0": (대답1),
    "completion_1": (대답2),
    "ranking":[
        (대답1의 점수),
        (대답2의 점수)
    ]
}
```

### 3. PPO(Proximal Policy Optimization) Dataset

PPO 단계에서 사용되는 데이터셋이다. 실제 사용자 데이터의 질문들로 구성되어있다.
데이터셋의 각 데이터의 형식은 다음과 같다.
```
{
    "prompt": (질문)
}
```
