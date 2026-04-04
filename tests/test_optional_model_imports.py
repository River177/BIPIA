import importlib.util
import sys
import types
from pathlib import Path


MODEL_INIT_PATH = Path(__file__).resolve().parents[1] / "bipia" / "model" / "__init__.py"
MODEL_DIR = MODEL_INIT_PATH.parent


def _stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _stub_class(name: str):
    return type(name, (), {})


class _Logger:
    def info(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None


def test_gpt_registration_is_available_without_legacy_openai_error_imports(
    monkeypatch,
):
    package_name = "tmp_bipia_model_optional"
    qwen3_cls = _stub_class("Qwen3")

    monkeypatch.setitem(sys.modules, "accelerate", _stub_module("accelerate"))
    monkeypatch.setitem(
        sys.modules,
        "accelerate.logging",
        _stub_module("accelerate.logging", get_logger=lambda _name: _Logger()),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{package_name}.llama",
        _stub_module(
            f"{package_name}.llama",
            Alpaca=_stub_class("Alpaca"),
            Vicuna=_stub_class("Vicuna"),
            Baize=_stub_class("Baize"),
            StableVicuna=_stub_class("StableVicuna"),
            Koala=_stub_class("Koala"),
            GPT4ALL=_stub_class("GPT4ALL"),
            Wizard=_stub_class("Wizard"),
            Guanaco=_stub_class("Guanaco"),
            Llama2=_stub_class("Llama2"),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{package_name}.qwen",
        _stub_module(f"{package_name}.qwen", Qwen3=qwen3_cls),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{package_name}.vllm_worker",
        _stub_module(
            f"{package_name}.vllm_worker",
            Dolly=_stub_class("Dolly"),
            StableLM=_stub_class("StableLM"),
            MPT=_stub_class("MPT"),
            Mistral=_stub_class("Mistral"),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{package_name}.llm_worker",
        _stub_module(
            f"{package_name}.llm_worker",
            RwkvModel=_stub_class("RwkvModel"),
            OASST=_stub_class("OASST"),
            ChatGLM=_stub_class("ChatGLM"),
            FastChatT5=_stub_class("FastChatT5"),
        ),
    )

    spec = importlib.util.spec_from_file_location(
        package_name,
        MODEL_INIT_PATH,
        submodule_search_locations=[str(MODEL_DIR)],
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, package_name, module)

    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.AutoLLM.from_name("qwen3") is qwen3_cls
    assert "gpt35" in module.LLM_NAME_TO_CLASS
