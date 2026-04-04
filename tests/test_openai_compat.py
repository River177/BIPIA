from types import SimpleNamespace

from bipia.openai_compat import OpenAICompatClient, create_openai_client


class _FakeResponse:
    def __init__(self, choices):
        self.choices = choices


class _FakeOpenAI:
    class RateLimitError(Exception):
        pass

    class APITimeoutError(Exception):
        pass

    class APIConnectionError(Exception):
        pass

    class APIError(Exception):
        pass

    class InternalServerError(Exception):
        pass

    class APIStatusError(Exception):
        pass

    class BadRequestError(Exception):
        pass

    openai_kwargs = None
    azure_kwargs = None
    chat_calls = []
    completion_calls = []

    class OpenAI:
        def __init__(self, **kwargs):
            _FakeOpenAI.openai_kwargs = kwargs
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=_FakeOpenAI._chat_create)
            )
            self.completions = SimpleNamespace(create=_FakeOpenAI._completion_create)

    class AzureOpenAI:
        def __init__(self, **kwargs):
            _FakeOpenAI.azure_kwargs = kwargs
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=_FakeOpenAI._chat_create)
            )
            self.completions = SimpleNamespace(create=_FakeOpenAI._completion_create)

    @staticmethod
    def reset():
        _FakeOpenAI.openai_kwargs = None
        _FakeOpenAI.azure_kwargs = None
        _FakeOpenAI.chat_calls = []
        _FakeOpenAI.completion_calls = []

    @staticmethod
    def _chat_create(**kwargs):
        _FakeOpenAI.chat_calls.append(kwargs)
        return _FakeResponse(
            [SimpleNamespace(message=SimpleNamespace(content="approved"))]
        )

    @staticmethod
    def _completion_create(**kwargs):
        _FakeOpenAI.completion_calls.append(kwargs)
        return _FakeResponse([SimpleNamespace(text="approved")])


def test_create_openai_client_uses_base_url_for_openai_compatible_servers():
    _FakeOpenAI.reset()

    client = create_openai_client(
        {
            "api_key": "secret",
            "api_type": "openai",
            "api_base": "https://api.deepseek.com/v1",
        },
        openai_module=_FakeOpenAI,
    )

    assert isinstance(client, _FakeOpenAI.OpenAI)
    assert _FakeOpenAI.openai_kwargs == {
        "api_key": "secret",
        "base_url": "https://api.deepseek.com/v1",
    }


def test_create_openai_client_uses_azure_specific_fields():
    _FakeOpenAI.reset()

    client = create_openai_client(
        {
            "api_key": "secret",
            "api_type": "azure",
            "api_base": "https://example-resource.openai.azure.com/",
            "api_version": "2024-02-01",
        },
        openai_module=_FakeOpenAI,
    )

    assert isinstance(client, _FakeOpenAI.AzureOpenAI)
    assert _FakeOpenAI.azure_kwargs == {
        "api_key": "secret",
        "azure_endpoint": "https://example-resource.openai.azure.com/",
        "api_version": "2024-02-01",
    }


def test_chat_completion_uses_model_or_engine_with_new_sdk_response_shape():
    _FakeOpenAI.reset()
    client = OpenAICompatClient(
        {
            "api_key": "secret",
            "api_type": "openai",
            "api_base": "https://open.bigmodel.cn/api/paas/v4/",
            "engine": "glm-4-flash",
        },
        openai_module=_FakeOpenAI,
    )

    result = client.chat_completion(
        [{"role": "user", "content": "hello"}],
        temperature=0,
        max_tokens=32,
    )

    assert result == ["approved"]
    assert _FakeOpenAI.chat_calls == [
        {
            "model": "glm-4-flash",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0,
            "max_tokens": 32,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
    ]


def test_completion_uses_new_sdk_completion_api():
    _FakeOpenAI.reset()
    client = OpenAICompatClient(
        {
            "api_key": "secret",
            "api_type": "openai",
            "model": "deepseek-chat",
        },
        openai_module=_FakeOpenAI,
    )

    result = client.completion("prompt", temperature=0, max_tokens=16)

    assert result == ["approved"]
    assert _FakeOpenAI.completion_calls == [
        {
            "model": "deepseek-chat",
            "prompt": "prompt",
            "temperature": 0,
            "max_tokens": 16,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": ["<|im_end|>"],
        }
    ]
