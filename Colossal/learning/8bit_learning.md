# 8bit Learning

## Need to install additional library

`pip install transformers==4.30.2`

`pip install colossalai=0.3.3`

`pip install peft==0.5.0`

`pip install scipy==1.11.3`


## Changed list

Base Model and actor is modified.
[base](Colossal/coati/models/base/__init__.py)
[actor](Colossal/coati/models/polyglotko/polyglotko_actor.py)
[train](Colossal/learning/train_sft.py)

## Need to modify library

Need to modify library code with my code.

Find location with `pip show colossalai`

Location will be like this: `~/miniconda3/envs/env_name/python3.10/site-packages`

With this location, change files in current directory.

- colossalai/zero/low_level/low_level_optim.py
- peft/tuners/lora.py

## Limitation

Current code is only available for polyglotKo.
If you want to use 8bit learning with different model, you need to implement code for each model.