# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, Dict
import logging
import yaml

from bipia.openai_compat import OpenAICompatClient
from .base import BaseEval

logger = logging.getLogger(__name__)


class ModelEval(BaseEval):
    """Compute evaluate metrics with GPT4"""

    def __init__(self, config: str | dict, judge_fn: Callable, format_fn: Callable):
        super().__init__()

        self.config = self.load_config(config)

        self.judge_fn = judge_fn
        self.format_fn = format_fn
        self.client = OpenAICompatClient(self.config)

    def load_config(self, config: str | dict) -> Dict:
        if isinstance(config, dict):
            return config
        with open(config, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        return config

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

    def _compute_score(self, prediction: str = None, **kwargs):
        messages = self.format_fn(prediction, chat=self.config["chat"])

        if self.config["chat"]:
            response = self.chat_completion(messages, temperature=0, max_tokens=32)
        else:
            response = self.completion(messages, temperature=0, max_tokens=32)

        if len(response) > 0:
            asr = self.judge_fn(response[0])
        else:
            asr = -1
        return asr

    def _batch_compute_score(self, predictions: str = None, **kwargs):
        messages = [
            self.format_fn(prediction, chat=self.config["chat"])
            for prediction in predictions
        ]

        responses = self.completion(messages, temperature=0, max_tokens=32)

        if len(responses) > 0:
            asrs = [self.judge_fn(response) for response in responses]
        else:
            asrs = [-1] * len(messages)
        return asrs

    def add_batch(self, *, predictions=None, **kwargs):
        if self.config["chat"]:
            batch_asrs = []
            for pred in predictions:
                asr = self._compute_score(prediction=pred)
                batch_asrs.append(asr)

            self.asrs.extend(batch_asrs)
        else:
            batch_asrs = self._batch_compute_score(predictions=predictions)
            self.asrs.extend(batch_asrs)

        return batch_asrs
