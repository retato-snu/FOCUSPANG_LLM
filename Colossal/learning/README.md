# Learning

## Stage1 - Supervised instructs tuning

Stage1 is supervised instructs fine-tuning, which uses the datasets mentioned earlier to fine-tune the model.

You can run the `train_sft.sh` to start a supervised instructs fine-tuning.

You can also use the following cmd to start a supervised instructs fine-tuning with your own settings.

```bash
torchrun --standalone --nproc_per_node=2 train_sft.py \
    --pretrain "/mnt/hf/polyglot-ko-5.8b" \
    --model 'polyglotko' \
    --strategy colossalai_zero2_cpu \
    --save_path model_output/1215 \
    --dataset /mnt/FOCUSPANG_LLM/FOCUSPANG_Private/Data/OpenData/data_kochatgpt/kochatgpt_1_SFT.jsonl \
    --batch_size 4 \
    --accumulation_steps 8 \
    --lr 2e-5 \
    --max_epochs 12 \
    --lora_rank 8 \
    --language 'ko'
```

**Note**: the supervised dataset follows the following format,

```json
[
    {
        "instruction": "Provide a list of the top 10 most popular mobile games in Asia",
        "input": "",
        "output": "The top 10 most popular mobile games in Asia are:\n1) PUBG Mobile\n2) Pokemon Go\n3) Candy Crush Saga\n4) Free Fire\n5) Clash of Clans\n6) Mario Kart Tour\n7) Arena of Valor\n8) Fantasy Westward Journey\n9) Subway Surfers\n10) ARK Survival Evolved",
        "id": 0
    },
    ...
]
```

You can use data whose name of labels are different with example.
This is reason why parameter input_str, output_str,and instruction_str exist.

### Arg List

- `--strategy`: the strategy using for training, choices=['ddp', 'colossalai_gemini', 'colossalai_zero2'], default='colossalai_zero2'
- `--model`: model type, choices=['gpt2', 'bloom', 'opt', 'llama'], default='bloom'
- `--pretrain`: pretrain model, type=str, default=None
- `--max_datasets_size`: the max size of dataset, type=int, default=None
- `--save_path`: path to save the model, type=str, default='output'
- `--need_optim_ckpt`: whether to save optim ckpt, type=bool, default=False
- `--max_epochs`: max epochs for training, type=int, default=3
- `--batch_size`: batch size while training, type=int, default=4
- `--lora_rank`: low-rank adaptation matrices rank, type=int, default=0
- `--input_str`: label name which will change to input
- `--instruction_str`: label name which will change to instruction
- `--output_str`: label name which will change to output
- `--language`: targt language, choices=['ko', 'en']
- `--without_prompt`: learning without prompt. This is for data that has already got prompt engineering.

## Stage2 - Training reward model

We train a reward model in stage 2, which obtains corresponding scores by manually ranking different outputs for the same prompt and supervises the training of the reward model.

You can run the `train_rm.sh` to start a reward model training.

You can also use the following cmd to start training a reward model.

```bash
torchrun --standalone --nproc_per_node=4 train_reward_model.py \
    --pretrain "/path/to/LLaMa-7B/" \
    --model 'llama' \
    --strategy colossalai_zero2 \
    --loss_fn 'log_exp'\
    --save_path 'rmstatic.pt' \
```

### Features and tricks in RM training

- We support [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)and[rm-static](https://huggingface.co/datasets/Dahoas/rm-static) datasets.
- We support 2 kinds of loss function named `log_sig`(used by OpenAI) and `log_exp`(used by Anthropic).
- We change the loss to `valid_acc` and `pair_dist` to monitor progress during training.
- We add special token to the end of the sequence to get better result.
- We use cosine-reducing lr-scheduler for RM training.
- We set value_head as 1 liner layer and initialize the weight of value_head using N(0，1/(d_model + 1)) distribution.
- We train a Bloom-560m reward model for 1 epoch and find the test acc of the model achieve the performance mentions in [Anthropics paper](https://arxiv.org/abs/2204.05862).

### Arg List

- `--strategy`: the strategy using for training, choices=['ddp', 'colossalai_gemini', 'colossalai_zero2'], default='colossalai_zero2'
- `--model`: model type, choices=['gpt2', 'bloom', 'opt', 'llama'], default='bloom'
- `--pretrain`: pretrain model, type=str, default=None
- `--model_path`: the path of rm model(if continue to train), type=str, default=None
- `--save_path`: path to save the model, type=str, default='output'
- `--need_optim_ckpt`: whether to save optim ckpt, type=bool, default=False
- `--max_epochs`: max epochs for training, type=int, default=3
- `--dataset`: dataset name, type=str, choices=['Anthropic/hh-rlhf', 'Dahoas/rm-static']
- `--subset`: subset of the dataset, type=str, default=None
- `--batch_size`: batch size while training, type=int, default=4
- `--lora_rank`: low-rank adaptation matrices rank, type=int, default=0
- `--loss_func`: which kind of loss function, choices=['log_sig', 'log_exp']
- `--max_len`: max sentence length for generation, type=int, default=512
- `--without_prompt`: learning without prompt. This is for data that has already got prompt engineering.

## Stage3 - Training model using prompts with RL

Stage3 uses reinforcement learning algorithm, which is the most complex part of the training process

```bash
torchrun --standalone --nproc_per_node=4 train_prompts.py \
    --pretrain "/path/to/LLaMa-7B/" \
    --model 'llama' \
    --strategy colossalai_zero2 \
    --prompt_dataset /path/to/your/prompt_dataset \
    --pretrain_dataset /path/to/your/pretrain_dataset \
    --rm_pretrain /your/pretrain/rm/definition \
    --rm_path /your/rm/model/path
```

Prompt dataset: the instruction dataset mentioned in the above figure which includes the instructions, e.g. you can use the [script](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat/examples/generate_prompt_dataset.py) which samples `instinwild_en.json` or `instinwild_ch.json` in [InstructionWild](https://github.com/XueFuzhao/InstructionWild/tree/main/data#instructwild-data) to generate the prompt dataset.
Pretrain dataset: the pretrain dataset including the instruction and corresponding response, e.g. you can use the [InstructWild Data](https://github.com/XueFuzhao/InstructionWild/tree/main/data) in stage 1 supervised instructs tuning.

**Note**: the required datasets follow the following format,

- `pretrain dataset`

  ```json
  [
      {
          "instruction": "Provide a list of the top 10 most popular mobile games in Asia",
          "input": "",
          "output": "The top 10 most popular mobile games in Asia are:\n1) PUBG Mobile\n2) Pokemon Go\n3) Candy Crush Saga\n4) Free Fire\n5) Clash of Clans\n6) Mario Kart Tour\n7) Arena of Valor\n8) Fantasy Westward Journey\n9) Subway Surfers\n10) ARK Survival Evolved",
          "id": 0
      },
      ...
  ]
  ```

- `prompt dataset`

  ```json
  [
      {
          "instruction": "Edit this paragraph to make it more concise: \"Yesterday, I went to the store and bought some things. Then, I came home and put them away. After that, I went for a walk and met some friends.\"",
          "id": 0
      },
      {
          "instruction": "Write a descriptive paragraph about a memorable vacation you went on",
          "id": 1
      },
      ...
  ]
  ```

### Arg List

- `--strategy`: the strategy using for training, choices=['ddp', 'colossalai_gemini', 'colossalai_zero2'], default='colossalai_zero2'
- `--model`: model type of actor, choices=['gpt2', 'bloom', 'opt', 'llama'], default='bloom'
- `--pretrain`: pretrain model, type=str, default=None
- `--rm_model`: reward model type, type=str, choices=['gpt2', 'bloom', 'opt', 'llama'], default=None
- `--rm_pretrain`: pretrain model for reward model, type=str, default=None
- `--rm_path`: the path of rm model, type=str, default=None
- `--save_path`: path to save the model, type=str, default='output'
- `--prompt_dataset`: path of the prompt dataset, type=str, default=None
- `--pretrain_dataset`: path of the ptx dataset, type=str, default=None
- `--need_optim_ckpt`: whether to save optim ckpt, type=bool, default=False
- `--num_episodes`: num of episodes for training, type=int, default=10
- `--num_update_steps`: number of steps to update policy per episode, type=int
- `--num_collect_steps`: number of steps to collect experience per episode, type=int
- `--train_batch_size`: batch size while training, type=int, default=8
- `--ptx_batch_size`: batch size to compute ptx loss, type=int, default=1
- `--experience_batch_size`: batch size to make experience, type=int, default=8
- `--lora_rank`: low-rank adaptation matrices rank, type=int, default=0
- `--kl_coef`: kl_coef using for computing reward, type=float, default=0.1
- `--ptx_coef`: ptx_coef using for computing policy loss, type=float, default=0.9
- `--language`: targt language, choices=['ko', 'en']
- `--(label)_str`: for different name of label

## Attention

The examples are demos for the whole training process.You need to change the hyper-parameters to reach great performance.

#### data

- [x] [rm-static](https://huggingface.co/datasets/Dahoas/rm-static)
- [x] [hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [ ] [openai/summarize_from_feedback](https://huggingface.co/datasets/openai/summarize_from_feedback)
- [ ] [openai/webgpt_comparisons](https://huggingface.co/datasets/openai/webgpt_comparisons)
- [ ] [Dahoas/instruct-synthetic-prompt-responses](https://huggingface.co/datasets/Dahoas/instruct-synthetic-prompt-responses)

## Support Model

### GPT

- [x] GPT2-S (s)
- [x] GPT2-M (m)
- [x] GPT2-L (l)
- [x] GPT2-XL (xl)
- [x] GPT2-4B (4b)
- [ ] GPT2-6B (6b)

### BLOOM

- [x] [BLOOM-560m](https://huggingface.co/bigscience/bloom-560m)
- [x] [BLOOM-1b1](https://huggingface.co/bigscience/bloom-1b1)
- [x] [BLOOM-3b](https://huggingface.co/bigscience/bloom-3b)
- [x] [BLOOM-7b](https://huggingface.co/bigscience/bloom-7b1)
- [ ] [BLOOM-175b](https://huggingface.co/bigscience/bloom)

### OPT

- [x] [OPT-125M](https://huggingface.co/facebook/opt-125m)
- [x] [OPT-350M](https://huggingface.co/facebook/opt-350m)
- [x] [OPT-1.3B](https://huggingface.co/facebook/opt-1.3b)
- [x] [OPT-2.7B](https://huggingface.co/facebook/opt-2.7b)
- [x] [OPT-6.7B](https://huggingface.co/facebook/opt-6.7b)
- [ ] [OPT-13B](https://huggingface.co/facebook/opt-13b)
- [ ] [OPT-30B](https://huggingface.co/facebook/opt-30b)

### [LLaMA](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md)

- [x] LLaMA-7B
- [x] LLaMA-13B
- [ ] LLaMA-33B
- [ ] LLaMA-65B

### For Focuspang

- [x] [GptNeoX](https://huggingface.co/EleutherAI/gpt-neox-20b)
- [x] [Polyglot-Ko](https://huggingface.co/EleutherAI/polyglot-ko-12.8b)
