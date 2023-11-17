from typing import Optional
import torch


from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM


from ..base import Actor


class GptNeoXActor(Actor):
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
                pretrained, torch_dtype=torch.float16
            )
        elif config is not None:
            model = GPTNeoXForCausalLM(config)
        else:
            model = GPTNeoXForCausalLM(GPTNeoXConfig())
        if checkpoint:
            model.gradient_checkpointing_enable()
        super().__init__(model, lora_rank, lora_train_bias)
