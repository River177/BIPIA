# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, List, Any, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor

import logging

from bipia.openai_compat import OpenAICompatClient
from .base import BaseModel

__all__ = ["GPTModel", "GPT35", "GPT4"]

logger = logging.getLogger(__name__)


class GPTModel(BaseModel):
    def __init__(self, *, config: str | dict = None, **kwargs):
        config = self.load_config(config)
        self.config = config
        self.client = OpenAICompatClient(config)

    def chat_completion(
        self,
        messages,
        temperature=None,
        max_tokens=2000,
        frequency_penalty=0,
        presence_penalty=0,
    ):
        return self.client.chat_completion(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

    def completion(
        self,
        messages,
        temperature=None,
        max_tokens=2000,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["<|im_end|>"],
    ):
        return self.client.completion(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
        )

    def generate(self, data: Any, **kwargs):
        temperature = kwargs.pop("temperature", 0)
        max_tokens = kwargs.pop("max_tokens", self.config.get("max_tokens", 2000))
        concurrent_requests = self.config.get("concurrent_requests", 1)
        if self.config["chat"]:
            if concurrent_requests > 1:
                with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
                    responses = executor.map(
                        lambda message: self.chat_completion(
                            message,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        ),
                        data["message"],
                    )
                    rslts = []
                    for rslt in responses:
                        rslts.extend(rslt)
            else:
                rslts = []
                for message in data["message"]:
                    rslt = self.chat_completion(
                        message, temperature=temperature, max_tokens=max_tokens
                    )
                    rslts.extend(rslt)
        else:
            rslts = self.completion(
                data["message"], temperature=temperature, max_tokens=max_tokens
            )
        return rslts


class GPTModelWSystem(GPTModel):
    require_system_prompt = True

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
        system_prompt, user_prompt = prompt_construct_fn(example)

        if self.config["chat"]:
            message = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            example["message"] = message
        else:
            system_message = "<|im_start|>system\n{}\n<|im_end|>".format(system_prompt)
            user_message = "\n<|im_start|>{}\n{}\n<|im_end|>".format(
                "user", user_prompt
            )

            message = system_message + user_message + "\n<|im_start|>assistant\n"
            example["message"] = message
        return example


class GPT35(GPTModelWSystem):
    pass


class GPT4(GPTModelWSystem):
    pass


class GPTModelWOSystem(GPTModel):
    require_system_prompt = False

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
        user_prompt = prompt_construct_fn(example)
        system_prompt = "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible."

        if self.config["chat"]:
            message = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            example["message"] = message
        else:
            system_message = "<|im_start|>system\n{}\n<|im_end|>".format(system_prompt)
            user_message = "\n<|im_start|>{}\n{}\n<|im_end|>".format(
                "user", user_prompt
            )

            message = system_message + user_message + "\n<|im_start|>assistant\n"
            example["message"] = message
        return example


class GPT35WOSystem(GPTModelWOSystem):
    pass


class GPT4WOSystem(GPTModelWOSystem):
    pass
