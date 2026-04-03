# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Callable, Tuple, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .base import BaseModel
from .utils import check_bf16_support

__all__ = ["Qwen3"]


class Qwen3(BaseModel):
    require_system_prompt = True

    def __init__(self, *, config: str | dict = None, **kwargs):
        self.kwargs = kwargs
        self.config = self.load_config(config)

        self.tokenizer = self.load_tokenizer()
        self.model = self.load_model()
        self.generation_config = self.load_generation_config()

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_name"],
            use_fast=False,
            token=self.config.get("auth_token", None),
            trust_remote_code=self.config.get("trust_remote_code", False),
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "left"
        return self.tokenizer

    def load_model(self):
        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16 if check_bf16_support() else torch.float16
            device_map = self.config.get("device_map", "auto")
        else:
            torch_dtype = torch.float32
            device_map = None

        model_kwargs = {
            "trust_remote_code": self.config.get("trust_remote_code", False),
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        if self.config.get("load_8bit", False):
            model_kwargs["load_in_8bit"] = True

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model_name"],
            **model_kwargs,
        )
        self.model.eval()
        return self.model

    def load_generation_config(self):
        max_new_tokens = self.kwargs.get("max_new_tokens", 2048)
        stop_token_ids = []

        if self.tokenizer.eos_token_id is not None:
            stop_token_ids.append(self.tokenizer.eos_token_id)

        im_end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_token_id not in (None, self.tokenizer.unk_token_id):
            stop_token_ids.append(im_end_token_id)

        self.generation_config = GenerationConfig(
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=list(dict.fromkeys(stop_token_ids)) or self.tokenizer.eos_token_id,
        )
        return self.generation_config

    def process_fn(
        self,
        example: Any,
        prompt_construct_fn: Callable[
            [
                Any,
            ],
            Tuple[str],
        ],
    ) -> Any:
        messages = []
        if self.require_system_prompt:
            system_prompt, user_prompt = prompt_construct_fn(example)
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
        else:
            user_prompt = prompt_construct_fn(example)

        messages.append({"role": "user", "content": user_prompt})
        example["message"] = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        example.update(
            self.tokenizer(
                example["message"],
                add_special_tokens=False,
                truncation=True,
                max_length=self.config.get("max_model_len", None),
            )
        )
        return example

    def generate(self, data: Any):
        input_ids = torch.as_tensor(data["input_ids"]).to(self.model.device)
        attention_mask = torch.as_tensor(data["attention_mask"]).to(self.model.device)

        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=self.generation_config,
        )
        output_ids = output_ids[:, input_ids.shape[1] :]

        responses = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return self.post_process(responses)

    def post_process(self, responses: List[str]):
        return [response.strip() for response in responses]
