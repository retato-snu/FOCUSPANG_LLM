from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

from coati.models.polyglotko import PolyglotKoActor

import torch

import bitsandbytes

model = PolyglotKoActor(
    pretrained="/mnt/hf/polyglot-ko-5.8b",
    lora_rank=8,
    checkpoint=False,
)
optim = bitsandbytes.optim.AdamW8bit(model.parameters(), lr=2e-5)
dtype = optim.param_groups[0]["params"][0].dtype
for param_group in optim.param_groups:
    group_params = param_group["params"]
    for param in group_params:
        print(param.dtype)

model = GPTNeoXForCausalLM.from_pretrained(
    "/mnt/hf/polyglot-ko-5.8b", torch_dtype=torch.bfloat16, load_in_8bit=True
)
print(model.config)
