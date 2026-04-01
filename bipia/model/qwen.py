# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Callable, Tuple

from transformers import AutoTokenizer
from vllm import SamplingParams

from .vllm_worker import vLLMModel

__all__ = ["Qwen3"]


class Qwen3(vLLMModel):
    require_system_prompt = True

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

    def load_generation_config(self):
        max_new_tokens = self.kwargs.get("max_new_tokens", 2048)
        stop_token_ids = []

        if self.tokenizer.eos_token_id is not None:
            stop_token_ids.append(self.tokenizer.eos_token_id)

        im_end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_token_id not in (None, self.tokenizer.unk_token_id):
            stop_token_ids.append(im_end_token_id)

        self.generation_config = SamplingParams(
            temperature=0,
            max_tokens=max_new_tokens,
            stop_token_ids=list(dict.fromkeys(stop_token_ids)) or None,
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
        return example
