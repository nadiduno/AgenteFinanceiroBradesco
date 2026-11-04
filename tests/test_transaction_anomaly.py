import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.ml.transaction_anomaly import (
    NORMAL_PROTOTYPES,
    RISK_PROTOTYPES,
    TransactionAnomalyPipeline,
    _client_history_features,
    _risk_scores,
    _split_by_time,
    audit_description_leakage,
    evaluate_scenarios,
    load_transactions,
)
from src.utils.db_gen import (
    NOT_APPLICABLE,
    SUSPICIOUS_DESCRIPTION_MARKERS,
    description_rule_violations,
    format_currency_brl,
    generate_synthetic_data,
)


class SyntheticDataTests(unittest.TestCase):
    def test_currency_uses_brazilian_format(self):
        self.assertEqual(format_currency_brl(1070.44), 'R$ 1.070,44')

    def test_5000_rows_follow_description_rules_without_violations(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / 'transactions.db'
            generate_synthetic_data(db_path, n_rows=5000, anomaly_rate=0.02)
            transactions = load_transactions(db_path)
            self.assertEqual(len(transactions), 5000)
            self.assertEqual(
                description_rule_violations(transactions.to_dict('records')),
                [],
            )
            self.assertFalse(
                transactions['descricao'].str.contains(r'\d|R\$', regex=True).any()
            )
            expected_columns = {
                'categoria_estabelecimento',
                'canal',
                'pais',
                'dispositivo_novo',
                'idade_conta_destino_dias',
                'qtd_transacoes_ultima_hora',
            }
            self.assertTrue(expected_columns.issubset(transactions.columns))
            self.assertTrue(
                transactions.loc[
                    ~transactions['tipo_transacao'].isin(('PIX', 'TED')),
                    'idade_conta_destino_dias',
                ].isna().all()
            )
            self.assertTrue(
                transactions.loc[
                    transactions['tipo_transacao'].isin(('PIX', 'TED')),
                    'idade_conta_destino_dias',
                ].notna().all()
            )
            card_mask = transactions['tipo_transacao'].eq('CARTAO')
            self.assertTrue(transactions.loc[
                ~card_mask, 'categoria_estabelecimento'
            ].eq(NOT_APPLICABLE).all())
            self.assertTrue(transactions.loc[
                card_mask, 'categoria_estabelecimento'
            ].ne(NOT_APPLICABLE).all())
            atm_mask = transactions['canal'].eq('ATM')
            self.assertTrue(transactions.loc[
                atm_mask, 'tipo_transacao'
            ].eq('SAQUE').all())
            self.assertTrue(transactions.loc[
                transactions['tipo_transacao'].eq('SAQUE'), 'canal'
            ].eq('ATM').all())
            self.assertTrue(transactions.loc[
                atm_mask, 'descricao'
            ].str.contains('saque', case=False, regex=False).all())
            online_card_mask = card_mask & transactions['canal'].isin(('app', 'web'))
            self.assertTrue(transactions.loc[
                online_card_mask, 'descricao'
            ].str.startswith('compra online').all())
            pos_fuel_mask = (
                card_mask
                & transactions['canal'].eq('POS')
                & transactions['categoria_estabelecimento'].eq('combustível')
            )
            self.assertTrue(transactions.loc[
                pos_fuel_mask, 'descricao'
            ].str.startswith('abastecimento em posto').all())
            online_only_mask = card_mask & transactions[
                'categoria_estabelecimento'
            ].isin(('jogos online', 'cripto'))
            self.assertFalse(
                transactions.loc[online_only_mask, 'canal'].eq('POS').any()
            )
            web_mask = transactions['canal'].eq('web')
            app_mask = transactions['canal'].eq('app')
            self.assertFalse(
                transactions.loc[web_mask, 'descricao'].str.contains(
                    'pelo aplicativo', case=False, regex=False
                ).any()
            )
            self.assertFalse(
                transactions.loc[app_mask, 'descricao'].str.contains(
                    'pelo internet banking', case=False, regex=False
                ).any()
            )
            self.assertTrue(transactions.loc[
                transactions['tipo_transacao'].eq('BOLETO'), 'descricao'
            ].str.startswith('pagamento').all())

            with closing(sqlite3.connect(db_path)) as connection:
                labels = connection.execute(
                    'SELECT transacao_id, fraude FROM rotulos_fraude ORDER BY transacao_id'
                ).fetchall()
                transaction_count = connection.execute(
                    'SELECT COUNT(*) FROM transacoes'
                ).fetchone()[0]
            self.assertEqual(len(labels), transaction_count)
            label_by_id = dict(labels)
            suspicious = transactions['descricao'].map(
                lambda description: any(
                    marker in description
                    for marker in SUSPICIOUS_DESCRIPTION_MARKERS
                )
            )
            fraud_mask = transactions['id'].map(label_by_id).eq(1)
            fraud_innocent_rate = (~suspicious[fraud_mask]).mean()
            normal_suspicious_rate = suspicious[~fraud_mask].mean()
            self.assertAlmostEqual(fraud_innocent_rate, 0.24, delta=0.10)
            self.assertAlmostEqual(normal_suspicious_rate, 0.06, delta=0.03)

            normal_times = transactions.loc[~fraud_mask, 'data_hora'].str[11:13].astype(int)
            self.assertGreater((normal_times.between(8, 22)).mean(), 0.99)
            self.assertEqual(set(transactions['categoria_estabelecimento']), {
                'supermercado', 'restaurante', 'farmácia', 'combustível',
                'eletrônicos', 'jogos online', 'joalheria', 'cripto',
                NOT_APPLICABLE,
            })

    def test_same_seed_generates_identical_database_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / 'first.db'
            second_path = Path(directory) / 'second.db'
            generate_synthetic_data(first_path, n_rows=5000, seed=8675309)
            generate_synthetic_data(second_path, n_rows=5000, seed=8675309)

            def snapshot(path):
                with closing(sqlite3.connect(path)) as connection:
                    return {
                        table: connection.execute(
                            f'SELECT * FROM {table} ORDER BY rowid'
                        ).fetchall()
                        for table in ('clientes', 'transacoes', 'rotulos_fraude')
                    }

            self.assertEqual(snapshot(first_path), snapshot(second_path))


class TransactionAnomalyTests(unittest.TestCase):
    def test_text_risk_score_is_bounded_and_prototypes_are_nonempty(self):
        pipeline = TransactionAnomalyPipeline()
        vectors = np.eye(len(RISK_PROTOTYPES) + len(NORMAL_PROTOTYPES), dtype=np.float32)
        scores = _risk_scores(
            vectors,
            vectors[:len(RISK_PROTOTYPES)],
            vectors[len(RISK_PROTOTYPES):],
        )
        self.assertTrue(np.all((scores >= 0) & (scores <= 1)))
        self.assertGreater(scores[0], scores[-1])

    def test_hugging_face_encodes_unique_text_and_maps_to_rows(self):
        pipeline = TransactionAnomalyPipeline()
        descriptions = ['texto repetido', 'outro texto', 'texto repetido']
        captured = {}

        def encode_once(unique_texts):
            captured['texts'] = unique_texts
            return np.arange(len(unique_texts) * 3, dtype=np.float32).reshape(-1, 3)

        with patch.object(pipeline, '_encode_unique_hugging_face', side_effect=encode_once):
            embeddings, risk, normal = pipeline._encode_descriptions_and_prototypes(
                descriptions
            )

        self.assertEqual(pipeline.backend, 'huggingface')
        self.assertEqual(pipeline.unique_description_count, 2)
        self.assertEqual(len(captured['texts']), len(set(captured['texts'])))
        self.assertTrue(all(isinstance(text, str) for text in captured['texts']))
        self.assertEqual(embeddings.shape[0], 3)
        np.testing.assert_array_equal(embeddings[0], embeddings[2])
        self.assertEqual(risk.shape[0], len(RISK_PROTOTYPES))
        self.assertEqual(normal.shape[0], len(NORMAL_PROTOTYPES))

    def test_temporal_features_and_split(self):
        transactions = load_transactions()
        features = _client_history_features(transactions)
        self.assertEqual(features.shape, (len(transactions), 6))
        self.assertTrue(np.isfinite(features).all())
        train_mask, test_mask, cutoff = _split_by_time(transactions)
        timestamps = pd.to_datetime(transactions['data_hora'])
        self.assertTrue((timestamps[train_mask] <= cutoff).all())
        self.assertTrue((timestamps[test_mask] > cutoff).all())

    def test_temporal_evaluation_reports_metrics_for_all_scenarios(self):
        with patch.object(
            TransactionAnomalyPipeline,
            '_build_hugging_face_features',
            return_value=np.zeros((5000, 9), dtype=np.float32),
        ):
            result = evaluate_scenarios(seeds=(42,), include_audit_rows=True)
        scenarios = result['cenarios']
        self.assertEqual(set(scenarios), {
            'somente_numericas',
            'numericas_mais_texto_hf',
            'numericas_mais_texto_tfidf_compacto',
        })
        for name in (
            'somente_numericas',
            'numericas_mais_texto_hf',
            'numericas_mais_texto_tfidf_compacto',
        ):
            self.assertIn('pr_auc', scenarios[name])
            self.assertIn('precision_at_100', scenarios[name])
            self.assertIn('ganho_sobre_acaso', scenarios[name])
            self.assertGreaterEqual(scenarios[name]['ganho_sobre_acaso'], 0)
            self.assertGreater(scenarios[name]['fraudes_teste'], 0)
            self.assertEqual(scenarios[name]['n_teste'], 1000)
            self.assertAlmostEqual(
                scenarios[name]['ganho_sobre_acaso'],
                scenarios[name]['precision_at_100'] / scenarios[name]['prevalencia_teste'],
            )
        audit_rows = result['linhas_teste']
        self.assertEqual(set(audit_rows), {
            'fraudes_detectadas',
            'fraudes_perdidas',
            'falsos_positivos_maior_score',
            'normais_score_baixo',
        })
        for rows in audit_rows.values():
            self.assertEqual(len(rows), 3)
            for row in rows:
                self.assertIn('score_numerico', row)
                self.assertIn('score_texto', row)
                self.assertIn('ranking_teste', row)
                self.assertIn('tipo', row)
                self.assertEqual(len(row['features_mais_influentes']), 3)
        self.assertTrue(all(
            row['ranking_teste'] <= 100
            for row in audit_rows['fraudes_detectadas']
        ))
        self.assertTrue(all(
            row['ranking_teste'] > 100
            for row in audit_rows['fraudes_perdidas']
        ))

    def test_travel_phrase_is_present_in_both_groups(self):
        audit = audit_description_leakage()
        self.assertEqual(len(audit['fraudes']), 6)
        self.assertEqual(len(audit['normais']), 6)
        self.assertGreater(audit['durante_viagem_em_fraudes'], 0)
        self.assertGreater(audit['durante_viagem_em_normais'], 0)


if __name__ == '__main__':
    unittest.main()
