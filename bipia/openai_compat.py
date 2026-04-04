import re
import time


def get_retry_time(err_info):
    match = re.search(r"after (\d+) seconds", err_info)
    if match:
        return int(match.group(1))
    return 1


def import_openai():
    import openai

    return openai


def _filter_none_values(data):
    return {key: value for key, value in data.items() if value is not None}


def _get_model_name(config):
    return config.get("model") or config.get("engine")


def create_openai_client(config, *, openai_module=None):
    openai_module = openai_module or import_openai()
    api_type = (config.get("api_type") or "openai").lower()

    if api_type == "azure":
        client_kwargs = _filter_none_values(
            {
                "api_key": config.get("api_key"),
                "azure_endpoint": config.get("azure_endpoint")
                or config.get("api_base"),
                "api_version": config.get("api_version"),
            }
        )
        return openai_module.AzureOpenAI(**client_kwargs)

    client_kwargs = _filter_none_values(
        {
            "api_key": config.get("api_key"),
            "base_url": config.get("base_url") or config.get("api_base"),
        }
    )
    return openai_module.OpenAI(**client_kwargs)


def _extract_chat_contents(response):
    contents = []
    for choice in getattr(response, "choices", []):
        message = getattr(choice, "message", None)
        if message is None and isinstance(choice, dict):
            message = choice.get("message")

        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)

        if content is not None:
            contents.append(content)

    return contents


def _extract_completion_texts(response):
    texts = []
    for choice in getattr(response, "choices", []):
        if isinstance(choice, dict):
            text = choice.get("text")
        else:
            text = getattr(choice, "text", None)

        if text is not None:
            texts.append(text)

    return texts


class OpenAICompatClient:
    def __init__(self, config, *, openai_module=None):
        self.config = config
        self.openai_module = openai_module or import_openai()
        self.client = create_openai_client(config, openai_module=self.openai_module)
        self.model_name = _get_model_name(config)

    def _retry_request(self, request_fn):
        openai_module = self.openai_module
        rate_limit_error = getattr(openai_module, "RateLimitError", None)
        timeout_error = getattr(openai_module, "APITimeoutError", None)
        connection_error = getattr(openai_module, "APIConnectionError", None)
        api_error = getattr(openai_module, "APIError", None)
        internal_error = getattr(openai_module, "InternalServerError", None)
        status_error = getattr(openai_module, "APIStatusError", None)
        bad_request_error = getattr(openai_module, "BadRequestError", None)

        while True:
            try:
                return request_fn()
            except tuple(
                err
                for err in (rate_limit_error,)
                if err is not None
            ) as exc:
                time.sleep(get_retry_time(str(exc)))
            except tuple(
                err
                for err in (timeout_error, connection_error, api_error, internal_error)
                if err is not None
            ):
                time.sleep(1)
            except tuple(
                err for err in (bad_request_error, status_error) if err is not None
            ):
                return None
            except Exception:
                return None

    def chat_completion(
        self,
        messages,
        temperature=None,
        max_tokens=2000,
        frequency_penalty=0,
        presence_penalty=0,
    ):
        request_kwargs = _filter_none_values(
            {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
            }
        )
        response = self._retry_request(
            lambda: self.client.chat.completions.create(**request_kwargs)
        )
        if response is None:
            return []
        return _extract_chat_contents(response)

    def completion(
        self,
        messages,
        temperature=None,
        max_tokens=2000,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["<|im_end|>"],
    ):
        request_kwargs = _filter_none_values(
            {
                "model": self.model_name,
                "prompt": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "stop": stop,
            }
        )
        response = self._retry_request(
            lambda: self.client.completions.create(**request_kwargs)
        )
        if response is None:
            return []
        return _extract_completion_texts(response)
