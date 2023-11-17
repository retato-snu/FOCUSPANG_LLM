from typing import Optional

import torch.nn as nn
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXModel

from ..base import Critic


class GptNeoXCritic(Critic):
    """
    OPT Critic model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (OPTConfig): Model config.
        lora_rank (int): Rank of the low-rank approximation.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(
        self,
        pretrained: Optional[str] = None,
        config: Optional[GPTNeoXConfig] = None,
        lora_rank: int = 0,
        lora_train_bias: str = "none",
        **kwargs,
    ) -> None:
        if pretrained is not None:
            model = GPTNeoXModel.from_pretrained(pretrained)
        elif config is not None:
            model = GPTNeoXModel(config)
        else:
            model = GPTNeoXModel(GPTNeoXConfig())

        value_head = nn.Linear(model.config.hidden_size, 1)
        super().__init__(model, value_head, lora_rank, lora_train_bias, **kwargs)
