import importlib
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "qwen3_4b_instruct_2507.yaml"


def _install_model_dependency_stubs():
    accelerate_module = types.ModuleType("accelerate")
    accelerate_logging_module = types.ModuleType("accelerate.logging")
    accelerate_logging_module.get_logger = lambda name: types.SimpleNamespace(
        info=lambda *args, **kwargs: None
    )
    accelerate_module.logging = accelerate_logging_module

    fastchat_module = types.ModuleType("fastchat")
    fastchat_model_module = types.ModuleType("fastchat.model")
    fastchat_model_module.get_conversation_template = lambda name: types.SimpleNamespace(
        name=name,
        roles=("user", "assistant"),
        stop_str=None,
        stop_token_ids=None,
        set_system_message=lambda message: None,
        append_message=lambda role, message: None,
        get_prompt=lambda: "",
    )
    fastchat_module.model = fastchat_model_module

    transformers_module = types.ModuleType("transformers")

    class DummyAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class DummyGenerationConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class DummyStoppingCriteria:
        pass

    transformers_module.AutoTokenizer = DummyAutoTokenizer
    transformers_module.AutoModel = object
    transformers_module.AutoModelForCausalLM = object
    transformers_module.AutoModelForSeq2SeqLM = object
    transformers_module.GenerationConfig = DummyGenerationConfig
    transformers_module.StoppingCriteria = DummyStoppingCriteria
    transformers_module.StoppingCriteriaList = list

    peft_module = types.ModuleType("peft")
    peft_module.PeftModel = type("DummyPeftModel", (), {})

    openai_module = types.ModuleType("openai")
    openai_error_module = types.ModuleType("openai.error")
    for error_name in [
        "RateLimitError",
        "InvalidRequestError",
        "Timeout",
        "APIConnectionError",
        "ServiceUnavailableError",
        "APIError",
    ]:
        setattr(openai_error_module, error_name, RuntimeError)
    openai_module.error = types.SimpleNamespace(
        RateLimitError=RuntimeError,
        InvalidRequestError=RuntimeError,
        APIError=RuntimeError,
        Timeout=RuntimeError,
        APIConnectionError=RuntimeError,
        ServiceUnavailableError=RuntimeError,
    )

    vllm_module = types.ModuleType("vllm")

    class DummySamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    vllm_module.LLM = type("DummyLLM", (), {})
    vllm_module.SamplingParams = DummySamplingParams

    return {
        "accelerate": accelerate_module,
        "accelerate.logging": accelerate_logging_module,
        "fastchat": fastchat_module,
        "fastchat.model": fastchat_model_module,
        "transformers": transformers_module,
        "peft": peft_module,
        "openai": openai_module,
        "openai.error": openai_error_module,
        "vllm": vllm_module,
    }


@contextmanager
def import_bipia_model_with_stubs():
    stub_modules = _install_model_dependency_stubs()
    saved_modules = {}

    for name, module in stub_modules.items():
        saved_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    for name in list(sys.modules):
        if name == "bipia.model" or name.startswith("bipia.model."):
            saved_modules.setdefault(name, sys.modules.get(name))
            sys.modules.pop(name, None)

    try:
        yield importlib.import_module("bipia.model")
    finally:
        for name in list(sys.modules):
            if name == "bipia.model" or name.startswith("bipia.model."):
                sys.modules.pop(name, None)

        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_qwen3_config_targets_expected_model():
    config = yaml.safe_load(CONFIG_PATH.read_text())

    assert config == {
        "load_8bit": False,
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "llm_name": "qwen3",
    }


def test_auto_llm_loads_qwen3_from_config_path():
    with import_bipia_model_with_stubs() as model_module:
        llm_cls = model_module.AutoLLM.from_name(str(CONFIG_PATH))

    assert llm_cls.__name__ == "Qwen3"


def test_qwen3_process_fn_uses_tokenizer_chat_template():
    with import_bipia_model_with_stubs():
        qwen_module = importlib.import_module("bipia.model.qwen")

    llm = qwen_module.Qwen3.__new__(qwen_module.Qwen3)

    calls = {}

    class DummyTokenizer:
        eos_token_id = 42
        unk_token_id = -1

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            calls["messages"] = messages
            calls["tokenize"] = tokenize
            calls["add_generation_prompt"] = add_generation_prompt
            return "formatted-prompt"

        def convert_tokens_to_ids(self, token):
            return {"<|im_end|>": 151645}.get(token, self.unk_token_id)

    llm.tokenizer = DummyTokenizer()
    llm.kwargs = {"max_new_tokens": 128}
    llm.require_system_prompt = True

    example = {"id": 1}
    processed = llm.process_fn(example, lambda item: ("system prompt", "user prompt"))
    generation_config = llm.load_generation_config()

    assert processed["message"] == "formatted-prompt"
    assert calls == {
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "user prompt"},
        ],
        "tokenize": False,
        "add_generation_prompt": True,
    }
    assert generation_config.max_tokens == 128
    assert generation_config.temperature == 0
    assert generation_config.stop_token_ids == [42, 151645]
