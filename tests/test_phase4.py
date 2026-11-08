import json
import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from mcp_servers import sqlite_server
from src.crew import main as crew_main
from src.crew.tools import (
    _compute_numeric_top_ten,
    _bounded_z_score,
    _historical_value_ratios,
    _get_numeric_top_ten,
    build_deterministic_report,
    numeric_top_ten_json,
    verify_report_sources,
)


class SQLiteAuthorizerTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / 'test.db'
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.executescript(
                '''
                CREATE TABLE transacoes (
                    id INTEGER PRIMARY KEY,
                    valor REAL NOT NULL,
                    data_hora TEXT NOT NULL
                );
                CREATE TABLE rotulos_fraude (
                    transacao_id INTEGER PRIMARY KEY,
                    fraude INTEGER NOT NULL
                );
                INSERT INTO transacoes VALUES (1, 25.50, '2023-01-01 10:00:00');
                INSERT INTO rotulos_fraude VALUES (1, 1);
                '''
            )
        self.database_patch = patch(
            'mcp_servers.sqlite_server._database_path',
            return_value=self.database_path,
        )
        self.database_patch.start()
        self.addCleanup(self.database_patch.stop)
        self.addCleanup(self.temporary_directory.cleanup)

    def test_allowed_table_read_succeeds(self):
        result = sqlite_server.run_select('SELECT id, valor FROM transacoes')
        self.assertEqual(result['rows'], [{'id': 1, 'valor': 25.5}])

    def _assert_denied(self, query):
        with self.assertRaises(sqlite3.DatabaseError):
            sqlite_server.run_select(query)

    def test_rotulos_fraude_select_direto_negado(self):
        self._assert_denied('SELECT fraude FROM rotulos_fraude')

    def test_rotulos_fraude_select_com_aspas_negado(self):
        self._assert_denied('SELECT fraude FROM "rotulos_fraude"')

    def test_rotulos_fraude_alias_negado(self):
        self._assert_denied('SELECT rf.fraude FROM rotulos_fraude AS rf')

    def test_rotulos_fraude_cte_negado(self):
        self._assert_denied(
            'WITH labels AS (SELECT fraude FROM rotulos_fraude) '
            'SELECT fraude FROM labels'
        )

    def test_rotulos_fraude_join_negado(self):
        self._assert_denied(
            'SELECT t.id FROM transacoes AS t '
            'JOIN rotulos_fraude AS rf ON rf.transacao_id = t.id'
        )

    def test_rotulos_fraude_subconsulta_negada(self):
        self._assert_denied('SELECT (SELECT fraude FROM rotulos_fraude)')

    def test_rotulos_fraude_union_negado(self):
        self._assert_denied(
            'SELECT id FROM transacoes UNION SELECT transacao_id FROM rotulos_fraude'
        )

    def test_attach_database_negado_pelo_authorizer(self):
        with closing(sqlite_server._connect_readonly()) as connection:
            connection.set_authorizer(sqlite_server._authorize_readonly)
            with self.assertRaises(sqlite3.DatabaseError):
                connection.execute("ATTACH DATABASE ':memory:' AS extra")

    def test_pragma_rotulos_fraude_negado_pelo_authorizer(self):
        with closing(sqlite_server._connect_readonly()) as connection:
            connection.set_authorizer(sqlite_server._authorize_readonly)
            with self.assertRaises(sqlite3.DatabaseError):
                connection.execute('PRAGMA user_version')
            with self.assertRaises(sqlite3.DatabaseError):
                connection.execute('PRAGMA table_info(rotulos_fraude)')

    def test_select_api_rejects_non_select_sql_before_execution(self):
        for query in (
            "ATTACH DATABASE ':memory:' AS extra",
            'PRAGMA user_version',
        ):
            with self.subTest(query=query):
                with self.assertRaises(ValueError):
                    sqlite_server.run_select(query)

    def test_mcp_stdio_allowed_rows_and_label_read_denied_without_llm(self):
        from crewai_tools.adapters.mcp_adapter import MCPServerAdapter
        from mcp import StdioServerParameters

        parameters = StdioServerParameters(
            command=sys.executable,
            args=[str(crew_main.MCP_SERVER)],
            cwd=str(crew_main.PROJECT_ROOT),
            env={
                'DATABASE_URL': f'sqlite:///{self.database_path}',
                'CREWAI_TELEMETRY_ENABLED': 'false',
                'OTEL_SDK_DISABLED': 'true',
            },
        )
        with MCPServerAdapter(parameters, 'run_select', connect_timeout=60) as tools:
            select = next(tool for tool in tools if tool.name == 'run_select')
            permitted = json.loads(select.run(query='SELECT id, valor FROM transacoes', limit=10))
            self.assertEqual(permitted['rows'], [{'id': 1, 'valor': 25.5}])
            denied = select.run(query='SELECT fraude FROM rotulos_fraude', limit=10)
            self.assertIn('access to rotulos_fraude.fraude is prohibited', denied)
            self.assertNotIn('"rows"', denied)
            permitted_again = json.loads(select.run(query='SELECT id FROM transacoes', limit=10))
            self.assertEqual(permitted_again['rows'], [{'id': 1}])


class PhaseFourToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _compute_numeric_top_ten.cache_clear()

    def test_numeric_tool_returns_exact_top_ten_without_label_data(self):
        with patch.dict(os.environ, {'USE_TEXT_FEATURES': 'false'}):
            ranking = json.loads(numeric_top_ten_json())
        self.assertEqual(len(ranking), 10)
        self.assertEqual([row['posicao'] for row in ranking], list(range(1, 11)))
        self.assertEqual(len({row['id_transacao'] for row in ranking}), 10)
        for row in ranking:
            self.assertTrue({
                'id_transacao', 'posicao', 'tipo', 'canal', 'categoria',
                'cidade', 'data_hora', 'valor', 'features_mais_influentes',
                'descricao_contexto',
            }.issubset(row))
            self.assertEqual(len(row['features_mais_influentes']), 3)
            self.assertEqual(
                row['descricao_contexto']['classificacao'],
                'texto livre, não evidência',
            )
            self.assertNotIn('fraude', row)

    def test_deterministic_report_preserves_source_fields(self):
        ranking = _get_numeric_top_ten()
        report = build_deterministic_report(ranking)
        audit = verify_report_sources(report, ranking)
        self.assertEqual(audit['ids_verificados'], 10)
        self.assertTrue(audit['valores_e_datas_iguais_ao_banco'])
        self.assertTrue(audit['features_iguais_ao_top10'])
        for case, source in zip(report['casos'], ranking):
            self.assertEqual(case['id_transacao'], source['id_transacao'])
            self.assertIn(source['data_hora'], case['resumo'])
            self.assertIn(source['valor_formatado'], case['resumo'])

    def test_z_score_limits_both_signs_and_preserves_boundary_flags(self):
        for raw, clipped, limited in (
            (-100, '-10', True), (100, '10', True),
            (-10, '-10', False), (10, '10', False), (2.5, '2.5', False),
        ):
            with self.subTest(raw=raw):
                self.assertEqual(_bounded_z_score(raw), {'valor': clipped, 'limitado': limited})

    def test_ratio_uses_only_prior_values_of_same_client(self):
        transactions = pd.DataFrame([
            {'id': 4, 'client_id': 'a', 'valor': 60, 'data_hora': '2023-01-04'},
            {'id': 1, 'client_id': 'a', 'valor': 10, 'data_hora': '2023-01-01'},
            {'id': 2, 'client_id': 'b', 'valor': 1000, 'data_hora': '2023-01-02'},
            {'id': 3, 'client_id': 'a', 'valor': 30, 'data_hora': '2023-01-03'},
        ])
        self.assertEqual(_historical_value_ratios(transactions), [3.0, None, None, 3.0])
        transactions.loc[1, 'valor'] = 0
        self.assertIsNone(_historical_value_ratios(transactions)[3])

    def test_raw_z_score_is_only_in_audit_and_report_cites_ratio(self):
        ranking = _get_numeric_top_ten()
        report = build_deterministic_report(ranking)
        audit = verify_report_sources(report, ranking)
        self.assertNotIn('valor_z_cliente_bruto', numeric_top_ten_json())
        self.assertNotIn('valor_z_cliente', json.dumps(report))
        self.assertIn('base inteira', ' '.join(report['limitacoes']))
        self.assertIn('não avaliação de desempenho', ' '.join(report['limitacoes']))
        raw_by_id = {row['id_transacao']: row['valor_z_cliente_bruto'] for row in audit['z_scores_brutos']}
        self.assertTrue(any(abs(value) > 10 for value in raw_by_id.values()))
        for row, case in zip(ranking, report['casos']):
            self.assertEqual(row['valor_z_cliente'], _bounded_z_score(raw_by_id[row['id_transacao']]))
            self.assertIn(f"valor_vs_mediana_cliente={row['valor_vs_mediana_cliente']};", case['resumo'])
            for feature in row['features_mais_influentes']:
                if feature['nome'] == 'valor_z_cliente':
                    self.assertLessEqual(abs(float(feature['valor'])), 10)
                    self.assertEqual(feature['limitado'], row['valor_z_cliente']['limitado'])

    def test_writer_output_preserves_ratio_and_full_database_scope(self):
        ranking = _get_numeric_top_ten()
        candidate = crew_main.ExecutiveReport.model_validate(build_deterministic_report(ranking))
        candidate.limitacoes = ['Saída sintética para revisão humana.']
        validated = crew_main._validate_llm_report(candidate, ranking)
        self.assertIn('base inteira', ' '.join(validated['limitacoes']))
        candidate.casos[0].resumo = 'valor_z_cliente=100'
        with self.assertRaisesRegex(ValueError, 'não z-scores'):
            crew_main._validate_llm_report(candidate, ranking)

    def test_dry_run_is_deterministic_and_does_not_start_llm(self):
        ranking = _get_numeric_top_ten()
        output = []
        with (
            patch.object(crew_main, '_get_numeric_top_ten', return_value=ranking),
            patch.object(crew_main, '_run_real_report', side_effect=AssertionError),
            patch('builtins.print', side_effect=lambda value: output.append(value)),
        ):
            self.assertEqual(crew_main.main(['--dry-run']), 0)
            self.assertEqual(crew_main.main(['--dry-run']), 0)
        self.assertEqual(output[0], output[1])

    def test_real_mode_fails_clearly_when_model_is_unset(self):
        with (
            patch.object(crew_main, 'load_dotenv'),
            patch.dict(os.environ, {'LLM_MODEL': '', 'GEMINI_API_KEY': ''}, clear=True),
        ):
            with self.assertRaisesRegex(RuntimeError, 'LLM_MODEL.*--dry-run'):
                crew_main._prepare_model()


if __name__ == '__main__':
    unittest.main()
