import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from mcp_servers import sqlite_server
from src.crew import main as crew_main
from src.crew.tools import (
    _compute_numeric_top_ten,
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

    def test_label_reads_are_denied_by_authorizer_in_all_query_shapes(self):
        queries = (
            'SELECT fraude FROM rotulos_fraude',
            'SELECT fraude FROM "rotulos_fraude"',
            'SELECT rf.fraude FROM rotulos_fraude AS rf',
            (
                'WITH labels AS (SELECT fraude FROM rotulos_fraude) '
                'SELECT fraude FROM labels'
            ),
            (
                'SELECT t.id FROM transacoes AS t '
                'JOIN rotulos_fraude AS rf ON rf.transacao_id = t.id'
            ),
            'SELECT (SELECT fraude FROM rotulos_fraude)',
            (
                'SELECT id FROM transacoes '
                'UNION SELECT transacao_id FROM rotulos_fraude'
            ),
        )
        for query in queries:
            with self.subTest(query=query):
                with self.assertRaises(sqlite3.DatabaseError):
                    sqlite_server.run_select(query)

    def test_attach_and_pragma_are_denied_by_authorizer(self):
        with closing(sqlite_server._connect_readonly()) as connection:
            connection.set_authorizer(sqlite_server._authorize_readonly)
            with self.assertRaises(sqlite3.DatabaseError):
                connection.execute("ATTACH DATABASE ':memory:' AS extra")
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
