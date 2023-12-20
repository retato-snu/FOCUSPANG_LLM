import argparse

import torch
import torch.distributed as dist
from coati.dataset import HhRlhfDataset, RmStaticDataset
from coati.models import LogExpLoss, LogSigLoss
from coati.models.bloom import BLOOMRM
from coati.models.gpt import GPTRM
from coati.models.llama import LlamaRM
from coati.models.opt import OPTRM
from coati.models.polyglotko import PolyglotkoRM
from coati.models.gptNeoX import GptNeoXRM
from coati.trainer import RewardModelTrainer
from coati.trainer.strategies import DDPStrategy, GeminiStrategy, LowLevelZeroStrategy
from datasets import load_dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoTokenizer,
    BloomTokenizerFast,
    LlamaTokenizer,
    PreTrainedTokenizerFast,
    GPTNeoXTokenizerFast,
)
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer

from colossalai.nn.optimizer import HybridAdam
from coati.dataset.utils import jload


PROMPT_DICT = {
    "en": {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        ),
    },
    "ko": {
        "prompt_input": (
            "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
            "요청을 적절히 완료하는 응답을 작성하세요.\n\n"
            "### 명령어:\n{instruction}\n\n### 입력:\n{input}\n\n### 응답:"
        ),
        "prompt_no_input": (
            "아래는 작업을 설명하는 명령어입니다.\n\n"
            "명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
            "### 명령어:\n{instruction}\n\n### 응답:"
        ),
    },
}


def train(args):
    # configure strategy
    if args.strategy == "ddp":
        strategy = DDPStrategy()
    elif args.strategy == "colossalai_gemini":
        strategy = GeminiStrategy(placement_policy="auto")
    elif args.strategy == "colossalai_zero2":
        strategy = LowLevelZeroStrategy(stage=2, placement_policy="cpu")
    else:
        raise ValueError(f'Unsupported strategy "{args.strategy}"')

    # configure model
    with strategy.model_init_context():
        if args.model == "bloom":
            model = BLOOMRM(pretrained=args.pretrain, lora_rank=args.lora_rank)
        elif args.model == "opt":
            model = OPTRM(pretrained=args.pretrain, lora_rank=args.lora_rank)
        elif args.model == "gpt2":
            model = GPTRM(pretrained=args.pretrain, lora_rank=args.lora_rank)
        elif args.model == "llama":
            model = LlamaRM(pretrained=args.pretrain, lora_rank=args.lora_rank)
        elif args.model == "polyglotko":
            model = PolyglotkoRM(pretrained=args.pretrain, lora_rank=args.lora_rank)
        elif args.model == "gpt-neox":
            model = GptNeoXRM(pretrained=args.pretrain, lora_rank=args.lora_rank)
        else:
            raise ValueError(f'Unsupported model "{args.model}"')

        model.to(torch.bfloat16).to(torch.cuda.current_device())

        if args.model_path is not None:
            state_dict = torch.load(args.model_path)
            model.load_state_dict(state_dict)

    # configure tokenizer
    if args.model == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained(
            "gpt2" if args.tokenizer is None else args.tokenizer
        )
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == "bloom":
        tokenizer = BloomTokenizerFast.from_pretrained(
            "bigscience/bloom-560m" if args.tokenizer is None else args.tokenizer
        )
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == "opt":
        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/opt-350m" if args.tokenizer is None else args.tokenizer
        )
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == "llama":
        tokenizer = LlamaTokenizer.from_pretrained(
            "hf-internal-testing/llama-tokenizer"
            if args.tokenizer is None
            else args.tokenizer
        )
        tokenizer.eos_token = "<\s>"
        tokenizer.pad_token = tokenizer.unk_token
    elif args.model == "polyglotko":
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            "EleutherAI/polyglot-ko-12.8b"
            if args.tokenizer is None
            else args.tokenizer,
            add_eos_token=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == "gpt-neox":
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrain if args.tokenizer is None else args.tokenizer
        )
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError(f'Unsupported model "{args.model}"')

    # configure optimizer
    if args.strategy.startswith("colossalai"):
        optim = HybridAdam(model.parameters(), lr=args.lr)
    else:
        optim = Adam(model.parameters(), lr=args.lr)

    # configure loss function
    if args.loss_fn == "log_sig":
        loss_fn = LogSigLoss()
    elif args.loss_fn == "log_exp":
        loss_fn = LogExpLoss()
    else:
        raise ValueError(f'Unsupported loss function "{args.loss_fn}"')

    # prepare for data and dataset
    
    if args.dataset == "json":
        if args.data_path is None:
            raise ValueError(f"Need to specify data path for json data")
        data = jload(args.data_path)
        if (integer := len(data[0]["ranking"])) <= 1:
            raise ValueError(
                f'Unsupported data size: need more than 2 data, but "{str(integer)}"'
            )
        entire_data = []
        for example in data:
            for first in range(len(example["ranking"]) - 1):
                for second in range(first + 1, len(example["ranking"])):
                    each_data = {}
                    if args.without_prompt : each_data["prompt"] = example["prompt"]
                    else : 
                        prompt_no_input = PROMPT_DICT['ko']['prompt_no_input']
                        each_data["prompt"] = prompt_no_input.format(example["prompt"])
                    
                    if example["ranking"][first] < example["ranking"][second]:
                        each_data["chosen"] = example["completion_" + str(first)]
                        each_data["rejected"] = example["completion_" + str(second)]
                    else:
                        each_data["chosen"] = example["completion_" + str(second)]
                        each_data["rejected"] = example["completion_" + str(first)]
                    entire_data.append(each_data)

        train_data = entire_data[: int(len(entire_data) * (15 / 16))]
        eval_data = entire_data[int(len(entire_data) * (15 / 16)) :]
    else:
        if args.subset is not None:
            data = load_dataset(args.dataset, data_dir=args.subset)
        else:
            data = load_dataset(args.dataset)

        train_data = data["train"].select(
            range(min(args.max_datasets_size, len(data["train"])))
        )
        eval_data = data["test"].select(
            range(min(args.max_datasets_size, len(data["test"])))
        )

    if args.dataset == "Dahoas/rm-static":
        train_dataset = RmStaticDataset(train_data, tokenizer, args.max_len)
        eval_dataset = RmStaticDataset(eval_data, tokenizer, args.max_len)
    elif args.dataset == "Anthropic/hh-rlhf":
        train_dataset = HhRlhfDataset(train_data, tokenizer, args.max_len)
        eval_dataset = HhRlhfDataset(eval_data, tokenizer, args.max_len)
    else:
        train_dataset = RmStaticDataset(train_data, tokenizer, args.max_len)
        eval_dataset = RmStaticDataset(eval_data, tokenizer, args.max_len)
    #     raise ValueError(f'Unsupported dataset "{args.dataset}"')

    if dist.is_initialized() and dist.get_world_size() > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            shuffle=True,
            seed=42,
            drop_last=True,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
        )
        eval_sampler = DistributedSampler(
            eval_dataset,
            shuffle=True,
            seed=42,
            drop_last=True,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
        )
    else:
        train_sampler = None
        eval_sampler = None

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        batch_size=args.batch_size,
        pin_memory=True,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=(eval_sampler is None),
        sampler=eval_sampler,
        batch_size=args.batch_size,
        pin_memory=True,
    )

    lr_scheduler = CosineAnnealingLR(optim, train_dataloader.__len__() // 100)
    strategy_dict = strategy.prepare(
        dict(model=model, optimizer=optim, lr_scheduler=lr_scheduler)
    )
    model = strategy_dict["model"]
    optim = strategy_dict["optimizer"]
    lr_scheduler = strategy_dict["lr_scheduler"]
    trainer = RewardModelTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        lr_scheduler=lr_scheduler,
        loss_fn=loss_fn,
        max_epochs=args.max_epochs,
    )

    trainer.fit(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        log_dir=args.log_dir,
        use_wandb=args.use_wandb,
    )

    if args.lora_rank > 0 and args.merge_lora_weights:
        from coati.models.lora import LORA_MANAGER

        # NOTE: set model to eval to merge LoRA weights
        LORA_MANAGER.merge_weights = True
        model.eval()
    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, path=args.save_path, only_rank0=True)
    strategy.save_pretrained(
        model, path=args.save_path_folder, only_rank0=True, tokenizer=tokenizer
    )
    # save optimizer checkpoint on all ranks
    if args.need_optim_ckpt:
        strategy.save_optimizer(
            trainer.optimizer,
            "rm_optim_checkpoint_%d.pt" % (torch.cuda.current_device()),
            only_rank0=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        choices=["ddp", "colossalai_gemini", "colossalai_zero2"],
        default="colossalai_zero2",
    )
    parser.add_argument(
        "--model",
        choices=["gpt2", "bloom", "opt", "llama", "polyglotko", "gpt-neox"],
        default="bloom",
    )
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--need_optim_ckpt", type=bool, default=False)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["Anthropic/hh-rlhf", "Dahoas/rm-static", "json"],
        default="Dahoas/rm-static",
    )
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--data_bool", type=bool, default=True)
    parser.add_argument(
        "--subset", type=lambda x: None if x == "None" else x, default=None
    )
    parser.add_argument("--max_datasets_size", type=int, default=1000000)
    parser.add_argument("--save_path", type=str, default="rm_ckpt")
    parser.add_argument("--save_path_folder", type=str, default='model_output/ppo')

    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument(
        "--lora_rank", type=int, default=0, help="low-rank adaptation matrices rank"
    )
    parser.add_argument("--merge_lora_weights", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=9e-6)
    parser.add_argument(
        "--loss_fn", type=str, default="log_sig", choices=["log_sig", "log_exp"]
    )
    parser.add_argument("--log_dir", default="logs", type=str)
    parser.add_argument("--use_wandb", default=False, action="store_true")
    parser.add_argument("--without_prompt", action="store_true", default=False)
    args = parser.parse_args()
    train(args)
