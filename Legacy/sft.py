from datasets import load_dataset
import torch
import transformers
import os

from utils.prompter import Prompter


data_path = (
    "/mnt/FOCUSPANG_LLM/FOCUSPANG_Private/Data/Focuspang/sft_dataset/dataset1.json"
)

data = load_dataset("json", data_files=data_path)
base_model = "/mnt/hf/polyglot-ko-1.3b"

from transformers import GPTNeoXForCausalLM, PreTrainedTokenizerFast
from utils.prompter import Prompter

model = GPTNeoXForCausalLM.from_pretrained(
    base_model,
)
cutoff_len = 4096

prompter = Prompter("polyglot")


tokenizer = PreTrainedTokenizerFast.from_pretrained(base_model, add_eos_token=True)

tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
tokenizer.padding_side = "left"  # Allow batched inference
train_on_inputs: bool = (True,)  # if False, masks out inputs in loss
add_eos_token: bool = (True,)


def tokenize(prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def generate_and_tokenize_prompt(data_point):
    full_prompt = prompter.generate_prompt(
        data_point["instruction"],
        data_point["input"],
        data_point["output"],
    )

    tokenized_full_prompt = tokenize(full_prompt)

    if not train_on_inputs:
        user_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"]
        )
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=add_eos_token)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        if add_eos_token:
            user_prompt_len -= 1

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_prompt


world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1


# keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
model.is_parallelizable = True
model.model_parallel = True

val_set_size = 100

train_val = data["train"].train_test_split(
    test_size=val_set_size, shuffle=True, seed=42
)
train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)

model.gradient_checkpointing_enable()

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        warmup_steps=100,
        num_train_epochs=1,
        learning_rate=3e-4,
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=10,
        output_dir="model_output/slow",
        save_total_limit=50,
        load_best_model_at_end=True if val_set_size > 0 else False,
        ddp_find_unused_parameters=None,
    ),
    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)
model.config.use_cache = False

trainer.train()
