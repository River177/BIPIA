import importlib
import json
import sys
import tempfile
import types
from contextlib import contextmanager
from types import SimpleNamespace


def _install_finetune_dependency_stubs():
    jsonlines_module = types.ModuleType("jsonlines")
    jsonlines_module.open = lambda *args, **kwargs: None

    wandb_module = types.ModuleType("wandb")
    wandb_module.init = lambda *args, **kwargs: None

    yaml_module = types.ModuleType("yaml")
    yaml_module.SafeLoader = object
    yaml_module.load = lambda *args, **kwargs: {}

    datasets_module = types.ModuleType("datasets")
    datasets_module.concatenate_datasets = lambda items: items
    datasets_module.Dataset = type("Dataset", (), {})

    transformers_module = types.ModuleType("transformers")
    transformers_module.Trainer = type("Trainer", (), {})
    transformers_module.HfArgumentParser = type("HfArgumentParser", (), {})
    transformers_module.TrainingArguments = type("TrainingArguments", (), {})
    transformers_module.PreTrainedTokenizer = type("PreTrainedTokenizer", (), {})
    transformers_module.PreTrainedModel = type("PreTrainedModel", (), {})

    trainer_pt_utils_module = types.ModuleType("transformers.trainer_pt_utils")
    trainer_pt_utils_module.LabelSmoother = types.SimpleNamespace(ignore_index=-100)

    bipia_data_module = types.ModuleType("bipia.data")
    bipia_data_module.AutoPIABuilder = type("AutoPIABuilder", (), {})

    utils_module = types.ModuleType("utils")
    utils_module.DATA_INFO = {
        "qa": ["context", "question", "ideal"],
        "abstract": ["context", "ideal"],
        "email": ["context", "question", "ideal"],
        "table": ["context", "question", "ideal"],
        "code": ["error", "code", "context", "ideal"],
    }
    utils_module.DataCollatorWithPaddingAndLabel = type(
        "DataCollatorWithPaddingAndLabel", (), {}
    )

    return {
        "jsonlines": jsonlines_module,
        "wandb": wandb_module,
        "yaml": yaml_module,
        "datasets": datasets_module,
        "transformers": transformers_module,
        "transformers.trainer_pt_utils": trainer_pt_utils_module,
        "bipia.data": bipia_data_module,
        "utils": utils_module,
    }


@contextmanager
def import_finetune_with_stubs():
    stub_modules = _install_finetune_dependency_stubs()
    saved_modules = {}

    for name, module in stub_modules.items():
        saved_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    module_name = "defense.white_box.finetune"
    saved_modules[module_name] = sys.modules.get(module_name)
    sys.modules.pop(module_name, None)

    try:
        yield importlib.import_module(module_name)
    finally:
        sys.modules.pop(module_name, None)
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_should_keep_example_rejects_empty_ideal_for_original_strategy():
    with import_finetune_with_stubs() as finetune_module:
        assert finetune_module.should_keep_example({"ideal": []}, "original") is False
        assert (
            finetune_module.should_keep_example({"ideal": ["answer"]}, "original")
            is True
        )


def test_should_keep_example_allows_empty_ideal_for_non_original_strategies():
    with import_finetune_with_stubs() as finetune_module:
        assert (
            finetune_module.should_keep_example({"ideal": []}, "self_clean") is True
        )


def test_tokenize_segments_does_not_pass_unsupported_kwargs():
    class RecordingTokenizer:
        def __init__(self):
            self.calls = []

        def tokenize(self, text, **kwargs):
            self.calls.append((text, kwargs))
            return [text]

    with import_finetune_with_stubs() as finetune_module:
        tokenizer = RecordingTokenizer()
        tokens = finetune_module.tokenize_segments(tokenizer, ["a", "b"])

    assert tokens == [["a"], ["b"]]
    assert tokenizer.calls == [("a", {}), ("b", {})]


def test_maybe_adjust_deepspeed_config_disables_cpu_offload_on_cuda_mismatch():
    config = {
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
        }
    }

    with import_finetune_with_stubs() as finetune_module, tempfile.TemporaryDirectory() as tmpdir:
        original_path = f"{tmpdir}/ds_config.json"
        with open(original_path, "w", encoding="utf-8") as handle:
            json.dump(config, handle)

        adjusted_path = finetune_module.maybe_adjust_deepspeed_config(
            original_path,
            tmpdir,
            torch_cuda_version="13.0",
            installed_cuda_version="12.4",
        )

        assert adjusted_path != original_path
        with open(adjusted_path, "r", encoding="utf-8") as handle:
            adjusted = json.load(handle)

    assert adjusted["zero_optimization"]["offload_optimizer"]["device"] == "cpu"
    assert adjusted["zero_force_ds_cpu_optimizer"] is False


def test_apply_adjusted_deepspeed_config_updates_live_training_args():
    adjusted = {
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
        },
        "zero_force_ds_cpu_optimizer": False,
    }

    with import_finetune_with_stubs() as finetune_module, tempfile.TemporaryDirectory() as tmpdir:
        adjusted_path = f"{tmpdir}/ds_config.no_cpu_offload.json"
        with open(adjusted_path, "w", encoding="utf-8") as handle:
            json.dump(adjusted, handle)

        training_args = SimpleNamespace(
            deepspeed="defense/white_box/ds_config.json",
            deepspeed_plugin=SimpleNamespace(
                deepspeed_config={
                    "zero_optimization": {
                        "stage": 3,
                        "offload_optimizer": {"device": "cpu", "pin_memory": True},
                    }
                }
            ),
            hf_deepspeed_config=SimpleNamespace(
                config={
                    "zero_optimization": {
                        "stage": 3,
                        "offload_optimizer": {"device": "cpu", "pin_memory": True},
                    }
                }
            ),
        )

        finetune_module.apply_adjusted_deepspeed_config(
            training_args, adjusted_path
        )

    assert training_args.deepspeed == adjusted_path
    assert (
        training_args.deepspeed_plugin.deepspeed_config["zero_optimization"][
            "offload_optimizer"
        ]["device"]
        == "cpu"
    )
    assert training_args.deepspeed_plugin.deepspeed_config["zero_force_ds_cpu_optimizer"] is False
    assert (
        training_args.hf_deepspeed_config.config["zero_optimization"][
            "offload_optimizer"
        ]["device"]
        == "cpu"
    )


def test_resolve_sequence_boundary_token_ids_falls_back_to_eos():
    tokenizer = SimpleNamespace(bos_token_id=None, eos_token_id=151643)

    with import_finetune_with_stubs() as finetune_module:
        bos_token_id, eos_token_id = finetune_module.resolve_sequence_boundary_token_ids(
            tokenizer
        )

    assert bos_token_id == 151643
    assert eos_token_id == 151643
