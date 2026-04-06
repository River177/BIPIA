# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import yaml
from pathlib import Path
from collections import OrderedDict

logger = logging.getLogger(__name__)

LLM_NAME_TO_CLASS = OrderedDict()

try:
    from .gpt import GPT35, GPT4, GPT35WOSystem, GPT4WOSystem
except ModuleNotFoundError as exc:
    logger.warning("Skip GPT model registration because an optional dependency is missing: %s", exc)
else:
    LLM_NAME_TO_CLASS.update(
        [
            ("gpt35", GPT35),
            ("gpt4", GPT4),
            ("gpt35_wosys", GPT35WOSystem),
            ("gpt4_wosys", GPT4WOSystem),
        ]
    )

try:
    from .llama import (
        Alpaca,
        Vicuna,
        Baize,
        StableVicuna,
        Koala,
        GPT4ALL,
        Wizard,
        Guanaco,
        Llama2,
    )
except ModuleNotFoundError as exc:
    logger.warning(
        "Skip llama-family model registration because an optional dependency is missing: %s",
        exc,
    )
else:
    LLM_NAME_TO_CLASS.update(
        [
            ("alpaca", Alpaca),
            ("vicuna", Vicuna),
            ("baize", Baize),
            ("stablevicuna", StableVicuna),
            ("koala", Koala),
            ("gpt4all", GPT4ALL),
            ("wizard", Wizard),
            ("guanaco", Guanaco),
            ("llama2", Llama2),
        ]
    )

try:
    from .vllm_worker import Dolly, StableLM, MPT, Mistral
except ModuleNotFoundError as exc:
    logger.warning(
        "Skip vLLM model registration because an optional dependency is missing: %s",
        exc,
    )
else:
    LLM_NAME_TO_CLASS.update(
        [
            ("stablelm", StableLM),
            ("dolly", Dolly),
            ("mpt", MPT),
            ("mistral", Mistral),
        ]
    )

try:
    from .llm_worker import RwkvModel, OASST, ChatGLM, FastChatT5
except ModuleNotFoundError as exc:
    logger.warning(
        "Skip worker model registration because an optional dependency is missing: %s",
        exc,
    )
else:
    LLM_NAME_TO_CLASS.update(
        [
            ("rwkv", RwkvModel),
            ("oasst", OASST),
            ("chatglm", ChatGLM),
            ("t5", FastChatT5),
        ]
    )

try:
    from .qwen import Qwen3
except ModuleNotFoundError as exc:
    logger.warning(
        "Skip Qwen model registration because an optional dependency is missing: %s",
        exc,
    )
else:
    LLM_NAME_TO_CLASS.update([("qwen3", Qwen3)])


class AutoLLM:
    @classmethod
    def from_name(cls, name: str):
        if name in LLM_NAME_TO_CLASS:
            name = name
        elif Path(name).exists():
            with open(name, "r") as f:
                config = yaml.load(f, Loader=yaml.SafeLoader)
            if "llm_name" not in config:
                raise ValueError("llm_name not in config.")
            name = config["llm_name"]
        else:
            raise ValueError(
                f"Invalid name {name}. AutoLLM.from_name needs llm name or llm config as inputs."
            )

        logger.info(f"Load {name} from name.")

        llm_cls = LLM_NAME_TO_CLASS[name]
        return llm_cls
