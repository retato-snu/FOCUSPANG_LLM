"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()

class LS_v1_Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        type: str = None,
        math_status: Union[None, str] = None,
        korean_status: Union[None, str] = None,        
        output_data: Union[None, str] = None
    ) -> str:        
        res = ''

        
        if type == 'learning_status':
            res = self.template['learning_status']['prompt'].format(
                math_status = math_status,
                korean_status = korean_status
            )
                
            if output_data:
                res = f'{res}{output_data}'
        
        elif type == 'how_to_improve':
            res = self.template['how_to_improve']['prompt'].format(
                math_status = math_status,
                korean_status = korean_status
            )
            
            if output_data:
                res = f'{res}{output_data}'
        
        elif type == 'to_parents':
            res = self.template['to_parents']['prompt'].format(
                math_status = math_status,
                korean_status = korean_status
            )
             
            if output_data:
                res = f'{res}{output_data}'
        
        else:
            print('[ERROR] Wrong type for prompt generation!')
            exit()
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()