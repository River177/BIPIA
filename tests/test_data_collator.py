import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path


def _install_data_utils_stubs():
    transformers_module = types.ModuleType("transformers")
    transformers_module.PreTrainedTokenizer = type("PreTrainedTokenizer", (), {})
    transformers_module.BatchEncoding = dict

    punkt_module = types.ModuleType("nltk.tokenize.punkt")
    punkt_module.PunktSentenceTokenizer = type(
        "PunktSentenceTokenizer",
        (),
        {"span_tokenize": lambda self, text: [(0, len(text))]},
    )

    nltk_tokenize_module = types.ModuleType("nltk.tokenize")
    nltk_tokenize_module.punkt = punkt_module

    nltk_module = types.ModuleType("nltk")
    nltk_module.tokenize = nltk_tokenize_module

    return {
        "transformers": transformers_module,
        "nltk": nltk_module,
        "nltk.tokenize": nltk_tokenize_module,
        "nltk.tokenize.punkt": punkt_module,
    }


@contextmanager
def import_data_utils_with_stubs():
    stub_modules = _install_data_utils_stubs()
    saved_modules = {}

    for name, module in stub_modules.items():
        saved_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    module_name = "test_bipia_data_utils"
    module_path = Path(__file__).resolve().parents[1] / "bipia" / "data" / "utils.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)

    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
        yield module
    finally:
        for name, original in saved_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class FakeTokenizer:
    model_input_names = ["input_ids", "attention_mask"]

    def pad(self, features, **kwargs):
        return {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }


def test_data_collator_returns_plain_dict():
    with import_data_utils_with_stubs() as data_utils_module:
        collator = data_utils_module.DataCollatorWithPadding(FakeTokenizer())

        batch = [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "labels": [1, 2, 3],
                "attack_str": "prompt",
            }
        ]

        result = collator(batch)

    assert type(result) is dict
    assert result["attack_str"] == ["prompt"]
    assert result["labels"] == [[1, 2, 3]]
