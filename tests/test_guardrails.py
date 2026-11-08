import copy
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock, patch

from pydantic import ValidationError

from src.crew import main as crew_main
from src.crew.guardrails import generate_validated_report, validate_report
from src.crew.schemas import ExecutiveReport
from src.crew.tools import _FEATURE_LABELS, _format_currency_brl, build_deterministic_report


class GuardrailTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.db_path = Path(temporary.name) / 'transacoes.db'
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                'CREATE TABLE transacoes (id INTEGER PRIMARY KEY, valor REAL, '
                'data_hora TEXT, localizacao TEXT, tipo_transacao TEXT, '
                'canal TEXT, categoria_estabelecimento TEXT)'
            )
            connection.executemany(
                'INSERT INTO transacoes VALUES (?, ?, ?, ?, ?, ?, ?)',
                [
                    (index, 125.5 + index, '2023-01-02 10:30:00', 'Recife',
                     'transferencia', 'app', 'transferencias')
                    for index in range(1, 12)
                ],
            )
            connection.commit()
        self.ranking = [
            {
                'id_transacao': index,
                'posicao': index,
                'valor': 125.5 + index,
                'valor_formatado': _format_currency_brl(125.5 + index),
                'data_hora': '2023-01-02 10:30:00',
                'cidade': 'Recife',
                'tipo': 'transferencia',
                'canal': 'app',
                'categoria': 'transferencias',
                'valor_vs_mediana_cliente': '2.5',
                'escopo_ranking': 'Ranking de revisão sobre a base inteira, top-10.',
                'features_mais_influentes': [
                    {'nome': name, 'rotulo': _FEATURE_LABELS[name], 'valor': value}
                    for name, value in (
                        ('cidade_nova_cliente', '1'),
                        ('dispositivo_novo', '1'),
                        ('qtd_transacoes_ultima_hora', '3'),
                    )
                ],
                'descricao_contexto': {'texto': 'Contexto livre com número 987654321.'},
            }
            for index in range(1, 11)
        ]
        self.report = build_deterministic_report(self.ranking)

    def validate(self, candidate=None, ranking=None):
        return validate_report(
            self.report if candidate is None else candidate,
            self.ranking if ranking is None else ranking,
            self.db_path,
        )

    def test_valid_report_and_json_preserve_exact_sources(self):
        for candidate in (self.report, json.dumps(self.report)):
            with self.subTest(type=type(candidate).__name__):
                report, audit = self.validate(candidate)
                self.assertEqual(report, self.report)
                self.assertTrue(audit['cidades_iguais_ao_banco'])
                self.assertTrue(audit['numeros_iguais_a_tool'])
                self.assertEqual(audit['ids_verificados'], 10)
                self.assertFalse(report['degradado'])
                self.assertLessEqual(max(len(case['resumo']) for case in report['casos']), 500)

    def test_nonexistent_id_is_rejected(self):
        self.report['casos'][0]['id_transacao'] = 999999
        with self.assertRaisesRegex(ValueError, 'não existem'):
            self.validate()

    def test_existing_id_outside_top_ten_is_rejected(self):
        self.report['casos'][0]['id_transacao'] = 11
        with self.assertRaisesRegex(ValueError, 'top-10'):
            self.validate()

    def test_altered_value_is_rejected_not_silently_replaced(self):
        source = self.ranking[0]
        self.report['casos'][0]['resumo'] = self.report['casos'][0]['resumo'].replace(
            source['valor_formatado'], 'R$ 999,99',
        )
        with self.assertRaisesRegex(ValueError, 'factual'):
            self.validate()

    def test_altered_date_city_and_additional_citations_are_rejected(self):
        for old, new in (
            ('Recife', 'Paris'),
            ('2023-01-02', '2024-01-02'),
            ('fatores detalhados na lista.', 'fatores detalhados na lista; R$ 126,50 em Paris.'),
        ):
            candidate = copy.deepcopy(self.report)
            candidate['casos'][0]['resumo'] = candidate['casos'][0]['resumo'].replace(old, new)
            with self.subTest(new=new), self.assertRaisesRegex(ValueError, 'factual'):
                self.validate(candidate)

    def test_tool_values_dates_and_cities_are_checked_against_database(self):
        for field, value in (
            ('valor', 999.99), ('data_hora', '2024-01-01'), ('cidade', 'Paris'),
            ('tipo', 'outro'), ('canal', 'outro'), ('categoria', 'outra'),
        ):
            ranking = copy.deepcopy(self.ranking)
            ranking[0][field] = value
            candidate = build_deterministic_report(ranking)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'banco'):
                self.validate(candidate, ranking)

    def test_formatted_tool_currency_is_checked(self):
        ranking = copy.deepcopy(self.ranking)
        ranking[0]['valor_formatado'] = 'R$ 999,99'
        with self.assertRaisesRegex(ValueError, 'formatado'):
            self.validate(ranking=ranking)

    def test_missing_fields_at_all_levels_are_rejected(self):
        for path, key in (
            ((), 'recomendacoes'), (('casos', 0), 'resumo'),
            (('casos', 0, 'fatores', 0), 'valor'),
        ):
            candidate = copy.deepcopy(self.report)
            target = candidate
            for step in path:
                target = target[step]
            del target[key]
            with self.subTest(path=path), self.assertRaises(ValidationError):
                self.validate(candidate)

    def test_extra_fields_at_all_levels_are_rejected(self):
        for path in ((), ('casos', 0), ('casos', 0, 'fatores', 0)):
            candidate = copy.deepcopy(self.report)
            target = candidate
            for step in path:
                target = target[step]
            target['campo_extra'] = 'invenção'
            with self.subTest(path=path), self.assertRaises(ValidationError):
                self.validate(candidate)

    def test_strict_types_and_positive_positions(self):
        for field, value in (
            ('id_transacao', '1'), ('id_transacao', True),
            ('id_transacao', 1.0), ('posicao', '1'), ('posicao', 0),
            ('posicao', -1),
        ):
            candidate = copy.deepcopy(self.report)
            candidate['casos'][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                self.validate(candidate)
        self.report['degradado'] = 'true'
        with self.assertRaises(ValidationError):
            self.validate()

    def test_empty_text_lists_and_long_summary_are_rejected(self):
        for field, value in (
            ('recomendacoes', []), ('limitacoes', []),
            ('recomendacoes', ['   ']), ('limitacoes', ['']),
            ('casos', []),
        ):
            candidate = copy.deepcopy(self.report)
            candidate[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                self.validate(candidate)
        for summary in ('', '   ', 'a' * 501):
            candidate = copy.deepcopy(self.report)
            candidate['casos'][0]['resumo'] = summary
            with self.subTest(summary=len(summary)), self.assertRaises(ValidationError):
                self.validate(candidate)

    def test_factor_names_closed_and_values_exact(self):
        for field, value, error in (
            ('nome', 'fator_inventado', ValidationError),
            ('valor', '99', ValueError), ('rotulo', 'outro', ValueError),
        ):
            candidate = copy.deepcopy(self.report)
            candidate['casos'][0]['fatores'][0][field] = value
            with self.subTest(field=field), self.assertRaises(error):
                self.validate(candidate)
        for factors in (
            self.report['casos'][0]['fatores'][:2],
            [self.report['casos'][0]['fatores'][0]] * 3,
        ):
            candidate = copy.deepcopy(self.report)
            candidate['casos'][0]['fatores'] = factors
            with self.subTest(count=len(factors)), self.assertRaises(ValueError):
                self.validate(candidate)

    def test_prohibited_language_in_recommendations_and_limitations(self):
        for text in (
            '90% de chance de fraude', 'Probabilidade elevada',
            'PROBABILIDADES', 'Chance de fraude', 'Fraude confirmada',
            'Consultar rotulos_fraude', 'Consultar RÓTULOS_FRAUDE',
            'Fraude estimada em 90%.',
        ):
            for field in ('recomendacoes', 'limitacoes'):
                candidate = copy.deepcopy(self.report)
                candidate[field] = [text]
                with self.subTest(text=text, field=field), self.assertRaisesRegex(
                    ValueError, 'Linguagem proibida',
                ):
                    self.validate(candidate)

    def test_unknown_number_and_number_only_in_description_are_rejected(self):
        for text in ('Revisar 999999 casos.', 'Revisar 987654321 casos.'):
            candidate = copy.deepcopy(self.report)
            candidate['recomendacoes'] = [text]
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, 'Todo número'):
                self.validate(candidate)

    def test_known_tool_number_is_allowed_but_factual_citations_are_not_general(self):
        self.report['recomendacoes'] = ['Revisar 10 casos manualmente.']
        self.validate()
        for text in (
            'Valor de R$ 126,50.', 'Cidade Recife.', 'Data 2023-01-02.',
            'Revisar ID 1.', 'Revisar transação 1.',
        ):
            self.report['recomendacoes'] = [text]
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, 'gerais'):
                self.validate()

    def test_invalid_numeric_format_is_a_validation_error(self):
        self.report['recomendacoes'] = ['Revisar 1.2.3 casos.']
        with self.assertRaisesRegex(ValueError, 'Formato numérico'):
            self.validate()

    def test_duplicate_ids_wrong_order_and_positions_are_rejected(self):
        for mutation in ('duplicate', 'order', 'position'):
            candidate = copy.deepcopy(self.report)
            if mutation == 'duplicate':
                candidate['casos'][1]['id_transacao'] = 1
            elif mutation == 'order':
                candidate['casos'].reverse()
            else:
                candidate['casos'][0]['posicao'] = 2
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.validate(candidate)

    def test_success_has_no_retry_and_application_controls_degraded_flag(self):
        candidate = copy.deepcopy(self.report)
        candidate['degradado'] = True
        generate = Mock(return_value=candidate)
        report, _ = generate_validated_report(generate, self.ranking, self.db_path)
        generate.assert_called_once_with(None)
        self.assertFalse(report['degradado'])

    def test_retry_receives_validation_error_and_valid_correction_is_returned(self):
        invalid = copy.deepcopy(self.report)
        del invalid['recomendacoes']
        generate = Mock(side_effect=[invalid, self.report])
        with self.assertLogs('src.crew.guardrails', level='WARNING'):
            report, _ = generate_validated_report(generate, self.ranking, self.db_path)
        self.assertEqual(generate.call_count, 2)
        self.assertIn('recomendacoes', generate.call_args_list[1].args[0])
        self.assertEqual(report, self.report)

    def test_two_failures_return_validated_deterministic_degraded_report(self):
        generate = Mock(side_effect=['not json', {'casos': []}])
        with self.assertLogs('src.crew.guardrails', level='WARNING') as logs:
            report, audit = generate_validated_report(generate, self.ranking, self.db_path)
        self.assertEqual(generate.call_count, 2)
        self.assertIn('degradado', logs.output[-1])
        self.assertTrue(report['degradado'])
        expected = copy.deepcopy(self.report)
        expected['degradado'] = True
        self.assertEqual(report, expected)
        ExecutiveReport.model_validate(report)
        self.assertEqual(audit['ids_verificados'], 10)

    def test_unverifiable_fallback_is_not_returned(self):
        ranking = copy.deepcopy(self.ranking)
        ranking[0]['cidade'] = 'Paris'
        generate = Mock(return_value=self.report)
        with self.assertLogs('src.crew.guardrails', level='WARNING'):
            with self.assertRaisesRegex(ValueError, 'banco'):
                generate_validated_report(generate, ranking, self.db_path)
        self.assertEqual(generate.call_count, 2)

    def test_provider_error_is_not_disguised_as_guardrail_failure(self):
        generate = Mock(side_effect=RuntimeError('provedor indisponível'))
        with self.assertRaisesRegex(RuntimeError, 'provedor indisponível'):
            generate_validated_report(generate, self.ranking, self.db_path)
        generate.assert_called_once_with(None)

    def test_real_path_wiring_uses_raw_writer_and_exactly_one_correction(self):
        import crewai
        from crewai_tools.adapters import mcp_adapter

        analyst, scientist, writer = Mock(), Mock(), Mock()
        writer.execute_task.side_effect = ['not json', json.dumps(self.report)]
        adapter = Mock()
        adapter.__enter__ = Mock(return_value=[])
        adapter.__exit__ = Mock(return_value=False)
        with (
            patch.object(crew_main, '_prepare_model', return_value=Mock()),
            patch.object(crew_main, 'create_numeric_ranking_tool') as tool,
            patch.object(crewai, 'Agent', side_effect=[analyst, scientist, writer]),
            patch.object(crewai, 'Crew') as crew,
            patch.object(crewai, 'Task', side_effect=lambda **kwargs: Mock(**kwargs)) as task,
            patch.object(mcp_adapter, 'MCPServerAdapter', return_value=adapter),
            patch.object(crew_main, 'generate_validated_report', side_effect=lambda generate, ranking:
                         generate_validated_report(generate, ranking, self.db_path)),
            patch.object(crew_main, 'verify_report_sources', side_effect=lambda report, ranking:
                         validate_report(report, ranking, self.db_path)[1]),
            self.assertLogs('src.crew.guardrails', level='WARNING'),
        ):
            tool.return_value.run.return_value = json.dumps(self.ranking)
            report, _ = crew_main._run_real_report()
        self.assertEqual(report, self.report)
        crew.return_value.kickoff.assert_called_once()
        self.assertEqual(writer.execute_task.call_count, 2)
        executive_calls = task.call_args_list[-2:]
        self.assertNotIn('output_pydantic', executive_calls[0].kwargs)
        self.assertIn('json_invalid', executive_calls[1].kwargs['description'])
        context = json.loads(writer.execute_task.call_args.kwargs['context'])
        self.assertEqual(context['dados_tool'], self.ranking)
        self.assertEqual(context['formato_factual_obrigatorio'], self.report)
        self.assertTrue(context['schema']['additionalProperties'] is False)


if __name__ == '__main__':
    unittest.main()
