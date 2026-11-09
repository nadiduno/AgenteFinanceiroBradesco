import importlib
import json
import logging
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient

from src.api.app import CallLimitExceeded, CallLimiter, Settings, create_app, deterministic_report
from src.crew import main as crew_main
from src.crew.guardrails import REQUEST_ID
from src.crew.schemas import ExecutiveReport
from src.crew.tools import _get_numeric_top_ten
from src.utils.db_gen import generate_synthetic_data


api_module = importlib.import_module('src.api.app')


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.db_path = Path(cls.temporary.name) / 'transacoes.db'
        generate_synthetic_data(cls.db_path, seed=42)
        cls.settings = Settings(
            api_access_key='chave-api-simulada',
            cors_origins=('https://frontend.example',),
            allow_llm=False,
            db_path=cls.db_path,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        blocker = patch.object(crew_main, '_prepare_model', side_effect=AssertionError('Provedor proibido'))
        self.addCleanup(blocker.stop)
        blocker.start()
        self.client = self.enterContext(TestClient(create_app(self.settings)))
        self.headers = {'X-API-Key': self.settings.api_access_key}

    def test_health_is_public_and_contains_no_configuration(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'status': 'ok'})
        self.assertEqual(len(response.headers['X-Request-ID']), 32)

    def test_missing_wrong_and_gemini_keys_are_unauthorized(self):
        for headers in ({}, {'X-API-Key': 'incorreta'}, {'X-API-Key': 'gemini-simulada'}):
            with self.subTest(headers=headers):
                response = self.client.post('/analisar', json={}, headers=headers)
                self.assertEqual(response.status_code, 401)
                self.assertEqual(response.json()['detail'], 'Não autorizado.')

    def test_missing_server_key_fails_closed_without_breaking_health(self):
        with TestClient(create_app(replace(self.settings, api_access_key=''))) as client:
            self.assertEqual(client.get('/health').status_code, 200)
            self.assertEqual(client.post('/analisar', json={}, headers=self.headers).status_code, 503)

    def test_invalid_input_is_generic_and_never_echoes_input(self):
        for payload in (
            {'top_n': 0}, {'top_n': 21}, {'top_n': '10'}, {'top_n': True},
            {'top_n': 1.5}, {'use_llm': 'true'}, {'use_llm': None},
            {'rotulos_fraude': 'segredo'}, {'top_n': None},
        ):
            with self.subTest(payload=payload):
                response = self.client.post('/analisar', json=payload, headers=self.headers)
                self.assertEqual(response.status_code, 422)
                self.assertEqual(response.json()['detail'], 'Entrada inválida.')
                self.assertNotIn('rotulos_fraude', response.text)
                self.assertNotIn('segredo', response.text)
        response = self.client.post('/analisar', content='{', headers={
            **self.headers, 'Content-Type': 'application/json',
        })
        self.assertEqual(response.status_code, 422)

    def test_disallowed_origin_and_preflight_are_denied(self):
        for method in ('POST', 'OPTIONS'):
            headers = {
                **self.headers, 'Origin': 'https://intruso.example',
                'Access-Control-Request-Method': 'POST',
            }
            response = self.client.request(method, '/analisar', json={}, headers=headers)
            self.assertEqual(response.status_code, 403)
            self.assertNotIn('access-control-allow-origin', response.headers)

    def test_allowed_cors_preflight_and_post(self):
        headers = {**self.headers, 'Origin': self.settings.cors_origins[0]}
        preflight = self.client.options('/analisar', headers={
            **headers, 'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'X-API-Key, Content-Type',
        })
        self.assertEqual(preflight.status_code, 200)
        response = self.client.post('/analisar', json={}, headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['access-control-allow-origin'], self.settings.cors_origins[0])
        self.assertNotEqual(response.headers['access-control-allow-origin'], '*')

    def test_deterministic_default_and_boundaries_preserve_validated_sources(self):
        with patch.object(api_module, '_run_real_report', side_effect=AssertionError('LLM proibido')):
            for top_n in (1, 10, 20):
                with self.subTest(top_n=top_n):
                    payload = {} if top_n == 10 else {'top_n': top_n, 'use_llm': False}
                    response = self.client.post('/analisar', json=payload, headers=self.headers)
                    self.assertEqual(response.status_code, 200, response.text)
                    report = response.json()
                    ExecutiveReport.model_validate(report)
                    self.assertEqual(len(report['casos']), top_n)
                    self.assertEqual(report, deterministic_report(top_n, self.db_path))
                    ranking = _get_numeric_top_ten(self.db_path, top_n)
                    self.assertEqual(
                        [case['id_transacao'] for case in report['casos']],
                        [row['id_transacao'] for row in ranking],
                    )
                    self.assertFalse(report['degradado'])
                    self.assertNotIn('auditoria', report)

    def test_opt_in_is_required_on_both_sides(self):
        with patch.object(api_module, '_run_real_report') as run:
            response = self.client.post('/analisar', json={'use_llm': True}, headers=self.headers)
            self.assertEqual(response.status_code, 403)
            with TestClient(create_app(replace(self.settings, allow_llm=True))) as client:
                self.assertEqual(client.post('/analisar', json={}, headers=self.headers).status_code, 200)
            run.assert_not_called()

    def test_llm_runs_in_worker_with_request_id_and_preserves_degraded_flag(self):
        captured = {}
        report = deterministic_report(1, self.db_path)
        report['degradado'] = True

        def fake_crew(**kwargs):
            captured.update(kwargs)
            captured['thread'] = threading.current_thread().name
            captured['request_id'] = REQUEST_ID.get()
            kwargs['call_guard']()
            return report, {}

        with (
            patch.object(api_module, '_run_real_report', side_effect=fake_crew),
            TestClient(create_app(replace(self.settings, allow_llm=True))) as client,
        ):
            response = client.post('/analisar', json={'use_llm': True, 'top_n': 1}, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['degradado'])
        self.assertIn('worker', captured['thread'].lower())
        self.assertEqual(captured['request_id'], response.headers['X-Request-ID'])
        self.assertEqual(captured['top_n'], 1)
        self.assertEqual(captured['db_path'], self.db_path)

    def test_rate_limit_counts_calls_not_requests_and_is_shared(self):
        report = deterministic_report(1, self.db_path)

        def fake_crew(**kwargs):
            kwargs['call_guard']()
            kwargs['call_guard']()
            return report, {}

        with (
            patch.object(api_module, '_run_real_report', side_effect=fake_crew),
            TestClient(create_app(replace(
                self.settings, allow_llm=True, max_calls_per_minute=2,
            ))) as client,
        ):
            for expected in (200, 429):
                response = client.post('/analisar', json={'use_llm': True, 'top_n': 1}, headers=self.headers)
                self.assertEqual(response.status_code, expected)

    def test_rate_limit_stays_explicit_if_crew_wraps_the_exception(self):
        def fake_crew(**kwargs):
            try:
                kwargs['call_guard']()
                kwargs['call_guard']()
            except CallLimitExceeded:
                raise RuntimeError('Erro encapsulado pela crew.') from None

        with (
            patch.object(api_module, '_run_real_report', side_effect=fake_crew),
            TestClient(create_app(replace(
                self.settings, allow_llm=True, max_calls_per_minute=1,
            ))) as client,
        ):
            response = client.post('/analisar', json={'use_llm': True}, headers=self.headers)
        self.assertEqual(response.status_code, 429)

    def test_timeout_returns_promptly_and_worker_cannot_start_more_calls(self):
        captured = {}
        done = threading.Event()
        started = threading.Event()
        release = threading.Event()
        report = deterministic_report(1, self.db_path)

        def fake_crew(**kwargs):
            try:
                started.set()
                release.wait(10)
                try:
                    kwargs['call_guard']()
                except TimeoutError:
                    captured['blocked'] = True
                return report, {}
            finally:
                done.set()

        with (
            patch.object(api_module, '_run_real_report', side_effect=fake_crew),
            TestClient(create_app(replace(
                self.settings, allow_llm=True, timeout_seconds=0.02,
            ))) as client,
        ):
            try:
                response = client.post('/analisar', json={'use_llm': True}, headers=self.headers)
                self.assertEqual(response.status_code, 504)
                self.assertTrue(started.wait(5))
                self.assertFalse(done.is_set())
            finally:
                release.set()
                self.assertTrue(done.wait(5))
        self.assertTrue(captured['blocked'])

    def test_timeout_stays_explicit_if_crew_wraps_or_suppresses_the_exception(self):
        report = deterministic_report(1, self.db_path)

        async def wait_without_timer(awaitable, **_kwargs):
            return await awaitable

        for suppress in (False, True):
            def fake_crew(**kwargs):
                try:
                    kwargs['call_guard']()
                except TimeoutError:
                    if suppress:
                        return report, {}
                    raise RuntimeError('Erro encapsulado pela crew.') from None

            clock = Mock()
            clock.monotonic.side_effect = [0, 61]
            async_runtime = Mock()
            async_runtime.wait_for = AsyncMock(side_effect=wait_without_timer)
            with (
                self.subTest(suppress=suppress),
                patch.object(api_module, 'time', clock),
                patch.object(api_module, 'asyncio', async_runtime),
                patch.object(api_module, '_run_real_report', side_effect=fake_crew),
                TestClient(create_app(replace(self.settings, allow_llm=True))) as client,
            ):
                response = client.post('/analisar', json={'use_llm': True}, headers=self.headers)
                self.assertEqual(response.status_code, 504)
                self.assertEqual(response.json()['detail'], 'Tempo limite excedido.')
                self.assertNotIn('encapsulado', response.text)

    def test_invalid_llm_output_is_rejected_without_leaking_internal_details(self):
        invalid = {'casos': [], 'segredo': 'rotulos_fraude C:\\privado'}
        with (
            patch.object(api_module, '_run_real_report', return_value=(invalid, {})),
            TestClient(create_app(replace(self.settings, allow_llm=True))) as client,
        ):
            response = client.post('/analisar', json={'use_llm': True}, headers=self.headers)
        self.assertEqual(response.status_code, 500)
        self.assertNotIn('rotulos_fraude', response.text)
        self.assertNotIn('privado', response.text)
        self.assertEqual(response.json()['request_id'], response.headers['X-Request-ID'])

    def test_deterministic_api_works_with_torch_and_transformers_imports_blocked(self):
        script = '''
import importlib.abc
import sys
from pathlib import Path

class BlockHF(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers'}:
            raise AssertionError('Import opcional indevido: ' + fullname)

sys.meta_path.insert(0, BlockHF())
from fastapi.testclient import TestClient
from src.api.app import Settings, create_app
from src.crew.schemas import ExecutiveReport
settings = Settings(api_access_key='chave-simulada', cors_origins=(),
                    allow_llm=False, db_path=Path(sys.argv[1]))
with TestClient(create_app(settings)) as client:
    assert client.get('/health').status_code == 200
    response = client.post('/analisar', json={'top_n': 1},
                           headers={'X-API-Key': 'chave-simulada'})
    assert response.status_code == 200, response.text
    report = ExecutiveReport.model_validate(response.json())
    assert len(report.casos) == 1 and not report.degradado
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
print('API deterministica sem torch/transformers: aprovada.')
'''
        environment = {
            **os.environ,
            'USE_TEXT_FEATURES': 'false',
            'ALLOW_LLM': 'false',
            'CREWAI_TELEMETRY_DISABLED': 'true',
            'OTEL_SDK_DISABLED': 'true',
            'HF_HUB_OFFLINE': '1',
            'TRANSFORMERS_OFFLINE': '1',
        }
        result = subprocess.run(
            [sys.executable, '-c', script, str(self.db_path)],
            cwd=crew_main.PROJECT_ROOT, env=environment,
            capture_output=True, text=True, timeout=180,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('aprovada', result.stdout)

    def test_internal_errors_and_logs_do_not_leak_exception_content(self):
        sensitive = 'rotulos_fraude C:\\privado\\banco.db chave-ultrassecreta'
        with (
            patch.object(api_module, 'deterministic_report', side_effect=RuntimeError(sensitive)),
            self.assertLogs(api_module.LOGGER, level='WARNING') as logs,
        ):
            response = self.client.post('/analisar', json={}, headers=self.headers)
        self.assertEqual(response.status_code, 500)
        request_id = response.headers['X-Request-ID']
        self.assertEqual(response.json()['request_id'], request_id)
        self.assertIn(request_id, '\n'.join(logs.output))
        for word in ('rotulos_fraude', 'privado', 'ultrassecreta', 'Traceback'):
            self.assertNotIn(word, response.text)
            self.assertNotIn(word, '\n'.join(logs.output))

    def test_database_generated_once_on_startup_with_fixed_seed(self):
        path = Path(self.temporary.name) / 'startup.db'
        with patch.object(api_module, 'generate_synthetic_data', wraps=generate_synthetic_data) as generate:
            for _ in range(2):
                with TestClient(create_app(replace(self.settings, db_path=path))) as client:
                    self.assertEqual(client.get('/health').status_code, 200)
            generate.assert_called_once_with(path, seed=42)
        with closing(sqlite3.connect(path)) as generated, closing(sqlite3.connect(self.db_path)) as reference:
            self.assertEqual(
                generated.execute('SELECT * FROM transacoes ORDER BY id').fetchall(),
                reference.execute('SELECT * FROM transacoes ORDER BY id').fetchall(),
            )

    def test_startup_error_is_sanitized(self):
        with patch.object(api_module, 'generate_synthetic_data', side_effect=OSError('C:\\segredo')):
            with self.assertLogs(api_module.LOGGER, level='ERROR'):
                with self.assertRaisesRegex(RuntimeError, '^Não foi possível inicializar a API\\.$'):
                    with TestClient(create_app(replace(
                        self.settings, db_path=Path(self.temporary.name) / 'ausente.db',
                    ))):
                        pass

    def test_dependency_logs_in_worker_are_sanitized_and_correlated(self):
        def fake_report(*_args):
            logging.getLogger('dependencia.simulada').error(
                'rotulos_fraude C:\\privado chave-simulada', exc_info=True,
            )
            return deterministic_report(1, self.db_path)

        with (
            patch.object(api_module, 'deterministic_report', side_effect=fake_report),
            self.assertLogs('dependencia.simulada', level='ERROR') as logs,
        ):
            response = self.client.post('/analisar', json={'top_n': 1}, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.headers['X-Request-ID'], logs.output[0])
        for word in ('rotulos_fraude', 'privado', 'chave-simulada', 'Traceback'):
            self.assertNotIn(word, logs.output[0])

    def test_real_crew_api_wiring_preserves_requested_count_and_database_offline(self):
        import crewai
        from crewai_tools.adapters import mcp_adapter

        ranking = _get_numeric_top_ten(self.db_path, 1)
        report = deterministic_report(1, self.db_path)
        analyst, scientist, writer = Mock(), Mock(), Mock()
        writer.execute_task.return_value = json.dumps(report)
        adapter = Mock()
        adapter.__enter__ = Mock(return_value=[])
        adapter.__exit__ = Mock(return_value=False)
        guard = Mock()
        with (
            patch.object(crew_main, '_prepare_model', return_value=Mock()) as prepare,
            patch.object(crew_main, 'create_numeric_ranking_tool') as tool,
            patch.object(crewai, 'Agent', side_effect=[analyst, scientist, writer]),
            patch.object(crewai, 'Crew') as crew,
            patch.object(crewai, 'Task', side_effect=lambda **kwargs: Mock(**kwargs)) as task,
            patch.object(mcp_adapter, 'MCPServerAdapter', return_value=adapter) as mcp,
        ):
            tool.return_value.run.return_value = json.dumps(ranking)
            actual, audit = crew_main._run_real_report(
                top_n=1, db_path=self.db_path, call_guard=guard, timeout=2,
            )
        self.assertEqual(actual, report)
        self.assertEqual(audit['ids_verificados'], 1)
        prepare.assert_called_once_with(2, guard)
        tool.assert_called_once_with(self.db_path, 1)
        crew.return_value.kickoff.assert_called_once()
        self.assertEqual(writer.execute_task.call_count, 1)
        self.assertIn('Inclua os 1 casos', task.call_args.kwargs['description'])
        self.assertEqual(
            mcp.call_args.args[0].env['DATABASE_URL'],
            f'sqlite:///{self.db_path.resolve()}',
        )


class APIConfigurationTests(unittest.TestCase):
    def settings(self, **values):
        with patch('src.api.app.load_dotenv'), patch.dict(os.environ, values, clear=True):
            return Settings.from_environment()

    def test_environment_defaults_and_explicit_configuration(self):
        settings = self.settings()
        self.assertFalse(settings.allow_llm)
        self.assertEqual(settings.cors_origins, ())
        self.assertEqual(settings.api_access_key, '')
        settings = self.settings(
            API_ACCESS_KEY='api-simulada', GEMINI_API_KEY='gemini-simulada',
            ALLOW_LLM='true', CORS_ORIGINS='["https://frontend.example"]',
            LLM_MAX_CALLS_PER_MINUTE='3', LLM_TIMEOUT_SECONDS='15',
        )
        self.assertTrue(settings.allow_llm)
        self.assertEqual(settings.api_access_key, 'api-simulada')
        self.assertEqual(settings.max_calls_per_minute, 3)
        self.assertEqual(settings.timeout_seconds, 15)

    def test_invalid_configuration_fails_without_echoing_values(self):
        for values in (
            {'CORS_ORIGINS': '*'}, {'CORS_ORIGINS': '["*"]'},
            {'CORS_ORIGINS': '["https://*.example"]'}, {'CORS_ORIGINS': '{}'},
            {'CORS_ORIGINS': '[123]'}, {'CORS_ORIGINS': '["https://example/"]'},
            {'ALLOW_LLM': 'sim'}, {'DATABASE_URL': 'postgres://segredo'},
            {'LLM_MAX_CALLS_PER_MINUTE': '0'}, {'LLM_TIMEOUT_SECONDS': 'nan'},
            {'API_ACCESS_KEY': 'simulada', 'GEMINI_API_KEY': 'simulada'},
        ):
            with self.subTest(values=values):
                with self.assertRaisesRegex(RuntimeError, '^Configuração da API inválida\\.$'):
                    self.settings(**values)

    def test_limiter_window_and_concurrent_access(self):
        limiter = CallLimiter(2)
        with patch('src.api.app.time.monotonic', side_effect=[0, 1, 59, 60, 61]):
            limiter.acquire()
            limiter.acquire()
            with self.assertRaises(CallLimitExceeded):
                limiter.acquire()
            limiter.acquire()
            limiter.acquire()

    def test_api_llm_guard_is_called_for_each_native_call_without_retries(self):
        from crewai.llms.providers.gemini.completion import GeminiCompletion
        from google.genai.errors import APIError
        from src.crew.llm import ReviewGeminiCompletion

        llm = ReviewGeminiCompletion.model_construct(model='simulado')
        guard = Mock()
        llm.configure_call_guard(guard)
        with patch.object(GeminiCompletion, 'call', return_value='ok') as native:
            self.assertEqual(llm.call('simulado'), 'ok')
            self.assertEqual(llm.call('simulado'), 'ok')
            self.assertEqual(guard.call_count, 2)
            self.assertEqual(native.call_count, 2)
        with (
            patch.object(GeminiCompletion, 'call', side_effect=APIError(429, {})) as native,
            patch('src.crew.llm.time.sleep') as sleep,
        ):
            for _ in range(2):
                with self.assertRaises(APIError):
                    llm.call('simulado')
            native.assert_called_once()
            sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main()
