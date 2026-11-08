import os
import unittest
from unittest.mock import AsyncMock, Mock, call, patch

from crewai.llms.providers.gemini.completion import GeminiCompletion
from google.genai.errors import APIError

from src.crew.llm import ReviewGeminiCompletion, acall_with_backoff, call_with_backoff
from src.crew.main import AGENTS_CONFIG, _prepare_model, _read_configuration


def api_error(code):
    return APIError(code, {'error': {'code': code, 'message': 'Erro simulado.'}})


class LLMRetryTests(unittest.TestCase):
    def test_503_and_429_retry_exactly_after_5_15_45_seconds(self):
        for code in (503, 429):
            with self.subTest(code=code):
                operation = Mock(side_effect=[api_error(code)] * 3 + ['relatório'])
                with patch('src.crew.llm.time.sleep') as sleep:
                    self.assertEqual(call_with_backoff(operation), 'relatório')
                self.assertEqual(operation.call_count, 4)
                self.assertEqual(sleep.call_args_list, [call(5), call(15), call(45)])

    def test_503_and_429_stop_after_four_calls_without_extra_sleep(self):
        for code in (503, 429):
            with self.subTest(code=code):
                operation = Mock(side_effect=api_error(code))
                with patch('src.crew.llm.time.sleep') as sleep:
                    with self.assertRaises(APIError):
                        call_with_backoff(operation)
                self.assertEqual(operation.call_count, 4)
                self.assertEqual(sleep.call_args_list, [call(5), call(15), call(45)])

    def test_401_and_404_fail_immediately_with_configuration_guidance(self):
        for code in (401, 404):
            with self.subTest(code=code):
                operation = Mock(side_effect=api_error(code))
                with patch('src.crew.llm.time.sleep') as sleep:
                    with self.assertRaisesRegex(RuntimeError, f'HTTP {code}.*LLM_MODEL.*chave'):
                        call_with_backoff(operation)
                operation.assert_called_once()
                sleep.assert_not_called()

    def test_other_http_errors_and_network_errors_never_retry(self):
        for error in (api_error(400), api_error(403), api_error(500), TimeoutError()):
            with self.subTest(error=type(error).__name__, code=getattr(error, 'code', None)):
                operation = Mock(side_effect=error)
                with patch('src.crew.llm.time.sleep') as sleep:
                    with self.assertRaises(type(error)):
                        call_with_backoff(operation)
                operation.assert_called_once()
                sleep.assert_not_called()

    def test_success_never_sleeps(self):
        operation = Mock(return_value='relatório')
        with patch('src.crew.llm.time.sleep') as sleep:
            self.assertEqual(call_with_backoff(operation), 'relatório')
        operation.assert_called_once()
        sleep.assert_not_called()

    def test_native_sync_call_uses_backoff_without_restarting_crew(self):
        llm = ReviewGeminiCompletion.model_construct(model='modelo-local-simulado')
        with (
            patch.object(GeminiCompletion, 'call', side_effect=[api_error(503), 'relatório']) as native,
            patch('src.crew.llm.time.sleep') as sleep,
        ):
            self.assertEqual(llm.call('mensagem', available_functions={'local': Mock()}), 'relatório')
        self.assertEqual(native.call_count, 2)
        self.assertEqual(native.call_args_list[0], native.call_args_list[1])
        sleep.assert_called_once_with(5)

    def test_terminal_failure_prevents_converter_from_starting_new_retries(self):
        for code in (401, 404, 500, 503, 429):
            with self.subTest(code=code):
                llm = ReviewGeminiCompletion.model_construct(model='modelo-local-simulado')
                expected = RuntimeError if code in (401, 404) else APIError
                with (
                    patch.object(GeminiCompletion, 'call', side_effect=api_error(code)) as native,
                    patch('src.crew.llm.time.sleep') as sleep,
                ):
                    for _ in range(3):
                        with self.assertRaises(expected):
                            llm.call('mensagem')
                self.assertEqual(native.call_count, 4 if code in (503, 429) else 1)
                self.assertEqual(sleep.call_count, 3 if code in (503, 429) else 0)

    def test_sdk_and_agent_retries_are_disabled_and_timeout_is_preserved(self):
        with (
            patch('src.crew.main.load_dotenv'),
            patch.dict(os.environ, {
                'LLM_MODEL': 'gemini/modelo-local-simulado',
                'GEMINI_API_KEY': 'placeholder-de-teste',
            }, clear=True),
            patch.object(GeminiCompletion, '_initialize_client', return_value=Mock()),
        ):
            llm = _prepare_model()
        self.assertIsInstance(llm, ReviewGeminiCompletion)
        options = llm.client_params['http_options']
        self.assertEqual(options.retry_options.attempts, 1)
        self.assertEqual(options.timeout, 60_000)
        self.assertFalse(llm.use_vertexai)
        for config in _read_configuration(AGENTS_CONFIG).values():
            self.assertEqual(config['max_retry_limit'], 0)


class AsyncLLMRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_503_and_429_follow_the_same_backoff(self):
        for code in (503, 429):
            with self.subTest(code=code):
                operation = AsyncMock(side_effect=[api_error(code)] * 3 + ['relatório'])
                with patch('src.crew.llm.asyncio.sleep', new_callable=AsyncMock) as sleep:
                    self.assertEqual(await acall_with_backoff(operation), 'relatório')
                self.assertEqual(operation.await_count, 4)
                self.assertEqual(sleep.await_args_list, [call(5), call(15), call(45)])

    async def test_async_fail_fast_and_exhaustion(self):
        for code in (401, 404, 500, 503, 429):
            with self.subTest(code=code):
                operation = AsyncMock(side_effect=api_error(code))
                expected_exception = RuntimeError if code in (401, 404) else APIError
                with patch('src.crew.llm.asyncio.sleep', new_callable=AsyncMock) as sleep:
                    with self.assertRaises(expected_exception):
                        await acall_with_backoff(operation)
                self.assertEqual(operation.await_count, 4 if code in (503, 429) else 1)
                self.assertEqual(
                    sleep.await_args_list,
                    [call(5), call(15), call(45)] if code in (503, 429) else [],
                )

    async def test_native_async_call_uses_backoff(self):
        llm = ReviewGeminiCompletion.model_construct(model='modelo-local-simulado')
        with (
            patch.object(GeminiCompletion, 'acall', new_callable=AsyncMock) as native,
            patch('src.crew.llm.asyncio.sleep', new_callable=AsyncMock) as sleep,
        ):
            native.side_effect = [api_error(429), 'relatório']
            self.assertEqual(await llm.acall('mensagem'), 'relatório')
        self.assertEqual(native.await_count, 2)
        sleep.assert_awaited_once_with(5)

    async def test_async_terminal_failure_also_blocks_sync_calls(self):
        llm = ReviewGeminiCompletion.model_construct(model='modelo-local-simulado')
        with (
            patch.object(GeminiCompletion, 'acall', new_callable=AsyncMock) as native_async,
            patch.object(GeminiCompletion, 'call') as native_sync,
            patch('src.crew.llm.asyncio.sleep', new_callable=AsyncMock) as sleep,
        ):
            native_async.side_effect = api_error(401)
            with self.assertRaisesRegex(RuntimeError, 'LLM_MODEL.*chave'):
                await llm.acall('mensagem')
            with self.assertRaises(RuntimeError):
                await llm.acall('mensagem')
            with self.assertRaises(RuntimeError):
                llm.call('mensagem')
        native_async.assert_awaited_once()
        native_sync.assert_not_called()
        sleep.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()
