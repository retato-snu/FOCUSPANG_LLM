from typing import Optional
import torch

import torch.nn as nn

from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM


from ..base import Actor

from peft import (
    get_peft_config,
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_int8_training,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)


class PolyglotKoActor(Actor, nn.Module):
    """
     polyglot model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (GPTNeoXConfig): Model config.
        checkpoint (bool): Enable gradient checkpointing.
        lora_rank (int): Rank of the low-rank approximation.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(
        self,
        pretrained: Optional[str] = None,
        config: Optional[GPTNeoXConfig] = None,
        checkpoint: bool = False,
        lora_rank: int = 0,
        lora_train_bias: str = "none",
    ) -> None:
        if pretrained is not None:
            model = GPTNeoXForCausalLM.from_pretrained(
                pretrained, torch_dtype=torch.float16, load_in_8bit=True
            )
        elif config is not None:
            model = GPTNeoXForCausalLM(config)
        else:
            model = GPTNeoXForCausalLM(GPTNeoXConfig(use_cache=False))
        if checkpoint:
            model.gradient_checkpointing_enable()

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=32,
            lora_dropout=0.1,
            bias=lora_train_bias,
        )
        super().__init__()
        self.model = prepare_model_for_int8_training(
            model, use_gradient_checkpointing=checkpoint
        )

        self.model = get_peft_model(self.model, peft_config)
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(model, type(model))
        # super().__init__(model, lora_rank, lora_train_bias)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **model_kwargs,
    ) -> torch.Tensor:
        """Returns model output."""
        # output = self.model(input_ids, attention_mask=attention_mask, **model_kwargs)
        # return output
        return self.model.forward(
            input_ids, attention_mask=attention_mask, **model_kwargs
        )
