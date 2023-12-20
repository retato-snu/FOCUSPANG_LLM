# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import argparse
import re
import logging
import transformers  # noqa: F401
import os
import json
from transformers import pipeline, set_seed
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast, GPTNeoXForCausalLM, PreTrainedTokenizerFast
# from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",
                        type=str,
                        default='/mnt/FOCUSPANG_LLM/Colossal/learning/model_output/1219/sft_ver2',
                        help="Directory containing trained actor model")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate per response",
    )
    args = parser.parse_args()
    return args


def get_generator(path):
    if os.path.exists(path):
        # Locally tokenizer loading has some issue, so we need to force download
        model_json = os.path.join(path, "config.json")
        print("here1")
        if os.path.exists(model_json):
            model_json_file = json.load(open(model_json))
            model_name = model_json_file["_name_or_path"]
            tokenizer = PreTrainedTokenizerFast.from_pretrained(
            "EleutherAI/polyglot-ko-5.8b",
            add_eos_token=True
            )
            tokenizer.pad_token_id = (0)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, fast_tokenizer=True)
        
    print("here 2")

    tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.from_pretrained(path)
    print("here2.1")
    model_class = GPTNeoXForCausalLM
    print("here2.2")
    model = model_class.from_pretrained(path,
                                        ignore_mismatched_sizes=True,
                                        from_tf=bool(".ckpt" in path),
                                        config=model_config).half()
    
    print("here 2.5")

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    
    # tokenizer = PreTrainedTokenizerFast.from_pretrained('/mnt/hf/polyglot-ko-5.8b')
    # print('2.1')
    # model_config = AutoConfig.from_pretrained(path)
    # # model = AutoModelForCausalLM.from_pretrained(
    # #     path,
    # #     load_in_8bit=True,
    # #     torch_dtype=torch.float16,
    # #     device_map={'':0},
    # # )
    # print('2.2')
    # model = model_class.from_pretrained(
    #         path,
    #         from_tf=bool(".ckpt" in path),
    #         config=model_config)
    # print('2.3')
    # # model = PeftModel.from_pretrained(
    # #     model,
    # #     '/mnt/DeepSpeedExamples/applications/DeepSpeed-Chat/output/jaeyoung/1129/ppo/actor',
    # #     torch_dtype=torch.float16,
    # #     device_map={"":0}
    # # )
    # print('2.4')
    # # unwind broken decapoda-research config
    # model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    # model.config.bos_token_id = 1
    # model.config.eos_token_id = 2

    # if not load_8bit:
    #     model.half()  # seems to fix bugs for some users.

    generator = pipeline("text-generation",
                         model=model,
                         tokenizer=tokenizer,
                         device="cuda:0")
    print("here 3")
    return generator


def get_user_input(user_input):
    tmp = input("Enter input (type 'quit' to exit, 'clear' to clean memory): ")
    new_inputs = f"아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n 요청을 적절히 완료하는 응답을 작성하세요.\n\n### 명령어:\n{tmp}\n\n### 입력:\n\n\n### 응답:"
    user_input += f" {new_inputs}"
    return user_input, tmp == "quit", tmp == "clear"


def get_model_response(generator, user_input, max_new_tokens):
    response = generator(user_input, max_new_tokens=max_new_tokens)
    return response


def process_response(response, num_rounds):
    output = str(response[0]["generated_text"])
    output = output.replace("<|endoftext|></s>", "")
    all_positions = [m.start() for m in re.finditer("Human: ", output)]
    place_of_second_q = -1
    if len(all_positions) > num_rounds:
        place_of_second_q = all_positions[num_rounds]
    if place_of_second_q != -1:
        output = output[0:place_of_second_q]
    return output


def main(args):
    generator = get_generator(args.path)
    set_seed(42)

    user_input = ""
    num_rounds = 0
    while True:
        num_rounds += 1
        user_input, quit, clear = get_user_input(user_input)

        if quit:
            break
        if clear:
            user_input, num_rounds = "", 0
            continue

        response = get_model_response(generator, user_input,
                                      args.max_new_tokens)
        output = process_response(response, num_rounds)

        print("-" * 30 + f" Round {num_rounds} " + "-" * 30)
        print(f"{output}")
        user_input = f"{output}\n\n"


if __name__ == "__main__":
    # Silence warnings about `max_new_tokens` and `max_length` being set
    logging.getLogger("transformers").setLevel(logging.ERROR)

    args = parse_args()
    main(args)

# Example:
"""
 Human: what is internet explorer?
 Assistant:
Internet Explorer is an internet browser developed by Microsoft. It is primarily used for browsing the web, but can also be used to run some applications. Internet Explorer is often considered the best and most popular internet browser currently available, though there are many other options available.

 Human: what is edge?
 Assistant:
 Edge is a newer version of the Microsoft internet browser, developed by Microsoft. It is focused on improving performance and security, and offers a more modern user interface. Edge is currently the most popular internet browser on the market, and is also used heavily by Microsoft employees.
"""
