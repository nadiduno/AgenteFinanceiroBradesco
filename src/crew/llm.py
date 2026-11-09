import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from crewai.llms.providers.gemini.completion import GeminiCompletion
from crewai.utilities.types import LLMMessage
from google.genai.errors import APIError
from pydantic import BaseModel, PrivateAttr


LOGGER = logging.getLogger(__name__)
RETRY_DELAYS = (5, 15, 45)
T = TypeVar('T')


class ProviderConfigurationError(RuntimeError):
    pass


def _retry_delay(error: APIError, retries: int) -> int:
    if error.code in (401, 404):
        raise ProviderConfigurationError(
            f'HTTP {error.code}: confira LLM_MODEL e a chave GEMINI_API_KEY '
            'no .env, ou use --dry-run. Não haverá retry.'
        ) from None
    if error.code not in (429, 503):
        raise error
    if retries == len(RETRY_DELAYS):
        LOGGER.error('HTTP %s: esgotados os três retries; execução interrompida.', error.code)
        raise error
    delay = RETRY_DELAYS[retries]
    LOGGER.warning(
        'HTTP %s: retry %s/3 em %s segundos.', error.code, retries + 1, delay
    )
    return delay


def call_with_backoff(operation: Callable[[], T]) -> T:
    retries = 0
    while True:
        try:
            return operation()
        except APIError as error:
            time.sleep(_retry_delay(error, retries))
            retries += 1


async def acall_with_backoff(operation: Callable[[], Awaitable[T]]) -> T:
    retries = 0
    while True:
        try:
            return await operation()
        except APIError as error:
            await asyncio.sleep(_retry_delay(error, retries))
            retries += 1


class ReviewGeminiCompletion(GeminiCompletion):
    _terminal_error: APIError | ProviderConfigurationError | None = PrivateAttr(default=None)
    _call_guard: Callable[[], None] | None = PrivateAttr(default=None)

    def configure_call_guard(self, guard: Callable[[], None]) -> None:
        self._call_guard = guard

    def call(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> Any:
        if self._terminal_error is not None:
            raise self._terminal_error
        try:
            if self._call_guard is not None:
                self._call_guard()
                return super().call(
                    messages, tools, callbacks, available_functions,
                    from_task, from_agent, response_model,
                )
            return call_with_backoff(
                lambda: super(ReviewGeminiCompletion, self).call(
                    messages, tools, callbacks, available_functions,
                    from_task, from_agent, response_model,
                )
            )
        except (APIError, ProviderConfigurationError) as error:
            self._terminal_error = error
            raise

    async def acall(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> Any:
        if self._terminal_error is not None:
            raise self._terminal_error
        try:
            if self._call_guard is not None:
                self._call_guard()
                return await super().acall(
                    messages, tools, callbacks, available_functions,
                    from_task, from_agent, response_model,
                )
            return await acall_with_backoff(
                lambda: super(ReviewGeminiCompletion, self).acall(
                    messages, tools, callbacks, available_functions,
                    from_task, from_agent, response_model,
                )
            )
        except (APIError, ProviderConfigurationError) as error:
            self._terminal_error = error
            raise
