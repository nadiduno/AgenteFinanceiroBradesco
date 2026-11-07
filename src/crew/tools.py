import json
import os
import sqlite3
from contextlib import closing
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.ml.transaction_anomaly import (
    DEFAULT_DB_PATH,
    NUMERIC_FEATURE_COLUMNS,
    _client_history_features,
    _fit_detector,
    _local_feature_sensitivities,
    load_transactions,
)


_FEATURE_LABELS = {
    'valor_z_cliente': 'z-score do valor pelo histórico do cliente',
    'cidade_nova_cliente': 'cidade nova para o cliente',
    'hora_fora_padrao_cliente': 'horário fora do padrão do cliente',
    'dispositivo_novo': 'dispositivo novo',
    'idade_conta_destino_dias': 'idade da conta de destino em dias',
    'qtd_transacoes_ultima_hora': 'transações na última hora',
}


def _format_currency_brl(value):
    return f'R$ {value:,.2f}'.replace(',', '\0').replace('.', ',').replace('\0', '.')


def _serialize_feature_value(value):
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f'{number:.6g}'


@lru_cache(maxsize=2)
def _compute_numeric_top_ten(db_path):
    transactions = load_transactions(db_path)
    raw_features = _client_history_features(transactions)
    scaler = StandardScaler()
    model_features = scaler.fit_transform(raw_features)
    detector = _fit_detector(model_features, random_state=42)
    scores = -detector.decision_function(model_features)
    order = np.argsort(-scores, kind='stable')[:10]
    reference_values = np.median(model_features, axis=0)
    influences = _local_feature_sensitivities(
        detector,
        model_features,
        order.tolist(),
        list(NUMERIC_FEATURE_COLUMNS),
        reference_values,
    )

    ranking = []
    for position, (row_index, impacts) in enumerate(zip(order, influences), start=1):
        row = transactions.iloc[int(row_index)]
        raw_row = raw_features[int(row_index)]
        top_features = []
        for feature_name, impact in impacts[:3]:
            feature_index = NUMERIC_FEATURE_COLUMNS.index(feature_name)
            top_features.append({
                'nome': feature_name,
                'rotulo': _FEATURE_LABELS[feature_name],
                'valor': _serialize_feature_value(raw_row[feature_index]),
                'impacto_local_heuristico': round(float(impact), 8),
            })
        ranking.append({
            'id_transacao': int(row['id']),
            'posicao': position,
            'tipo': str(row['tipo_transacao']),
            'canal': str(row['canal']),
            'categoria': str(row['categoria_estabelecimento']),
            'cidade': str(row['localizacao']),
            'data_hora': str(row['data_hora']),
            'valor': float(row['valor']),
            'valor_formatado': _format_currency_brl(float(row['valor'])),
            'features_mais_influentes': top_features,
            'descricao_contexto': {
                'classificacao': 'texto livre, não evidência',
                'texto': str(row['descricao']),
            },
        })
    return ranking


def _get_numeric_top_ten(db_path=DEFAULT_DB_PATH):
    enabled = os.getenv('USE_TEXT_FEATURES', 'false').strip().lower()
    if enabled not in {'true', 'false'}:
        raise ValueError('USE_TEXT_FEATURES deve ser true ou false.')
    if enabled == 'true':
        raise ValueError(
            'A ferramenta da Fase 4 é somente numérica; defina USE_TEXT_FEATURES=false.'
        )
    return _compute_numeric_top_ten(str(Path(db_path).resolve()))


def numeric_top_ten_json():
    return json.dumps(_get_numeric_top_ten(), ensure_ascii=False)


def create_numeric_ranking_tool():
    from crewai.tools import tool

    @tool('rank_numeric_transactions')
    def rank_numeric_transactions() -> str:
        """Calcula o top-10 numérico de revisão com campos e features da tabela transacoes.

        Retorna ID e posição exatos, tipo, canal, categoria, cidade, data/hora,
        valor e as três features locais mais influentes. A descrição aparece apenas
        como contexto identificado por "texto livre, não evidência". Não acessa rótulos.
        """
        return numeric_top_ten_json()

    return rank_numeric_transactions


def build_deterministic_report(ranking):
    cases = []
    for row in ranking:
        factors = [
            {
                'nome': feature['nome'],
                'rotulo': feature['rotulo'],
                'valor': feature['valor'],
            }
            for feature in row['features_mais_influentes']
        ]
        factor_text = ', '.join(
            f"{feature['nome']}={feature['valor']}" for feature in factors
        )
        summary = (
            f"Prioridade de revisão na posição {row['posicao']} do ranking; "
            f"tipo={row['tipo']}; canal={row['canal']}; "
            f"categoria={row['categoria']}; cidade={row['cidade']}; "
            f"data/hora={row['data_hora']}; valor={row['valor_formatado']}; "
            f"features mais influentes: {factor_text}."
        )
        cases.append({
            'id_transacao': row['id_transacao'],
            'posicao': row['posicao'],
            'resumo': summary,
            'fatores': factors,
        })
    return {
        'casos': cases,
        'recomendacoes': [
            'Revisar manualmente os casos pela posição no ranking, sem tratar a ordem como decisão automática.',
        ],
        'limitacoes': [
            'O score é heurístico e representa somente uma posição relativa no ranking.',
            'A descrição é texto livre, não evidência; utilize os campos estruturados e as features.',
            'A ferramenta não lê rótulos e não valida ocorrência de fraude.',
        ],
    }


def verify_report_sources(report, ranking, db_path=DEFAULT_DB_PATH):
    expected_by_id = {row['id_transacao']: row for row in ranking}
    cases = report['casos']
    if len(cases) != len(ranking):
        raise ValueError('O relatório deve conter exatamente os dez casos retornados pela ferramenta.')

    cited_ids = [case['id_transacao'] for case in cases]
    expected_ids = [row['id_transacao'] for row in ranking]
    if cited_ids != expected_ids:
        raise ValueError('IDs do relatório divergem do top-10 retornado pela ferramenta.')

    path = Path(db_path)
    placeholders = ','.join('?' for _ in cited_ids)
    with closing(sqlite3.connect(f'{path.resolve().as_uri()}?mode=ro', uri=True)) as connection:
        database_rows = connection.execute(
            f'SELECT id, valor, data_hora FROM transacoes WHERE id IN ({placeholders})',
            cited_ids,
        ).fetchall()
    database_by_id = {
        int(row[0]): {'valor': float(row[1]), 'data_hora': str(row[2])}
        for row in database_rows
    }
    if set(database_by_id) != set(expected_ids):
        raise ValueError('Um ou mais IDs citados não existem em transacoes.')

    for case in cases:
        source = expected_by_id[case['id_transacao']]
        database = database_by_id[case['id_transacao']]
        if database['valor'] != source['valor'] or database['data_hora'] != source['data_hora']:
            raise ValueError('Valor ou data/hora da fonte mudou durante a execução.')
        if case['posicao'] != source['posicao']:
            raise ValueError('Posição no ranking divergente da ferramenta.')
        if source['data_hora'] not in case['resumo'] or source['valor_formatado'] not in case['resumo']:
            raise ValueError('O resumo alterou ou omitiu o valor ou a data/hora da fonte.')
        expected_factors = {
            (feature['nome'], feature['rotulo'], feature['valor'])
            for feature in source['features_mais_influentes']
        }
        actual_factors = {
            (feature['nome'], feature['rotulo'], feature['valor'])
            for feature in case['fatores']
        }
        if actual_factors != expected_factors:
            raise ValueError('Features ou valores citados divergem da ferramenta.')

    textual_content = json.dumps(report, ensure_ascii=False).casefold()
    if 'probabilidade de fraude' in textual_content:
        raise ValueError('O relatório não pode chamar o score de probabilidade de fraude.')
    for item in report['recomendacoes'] + report['limitacoes']:
        if any(character.isdigit() for character in item):
            raise ValueError('Recomendações e limitações não podem introduzir IDs, valores ou datas.')
    return {
        'ids_verificados': len(cited_ids),
        'valores_e_datas_iguais_ao_banco': True,
        'features_iguais_ao_top10': True,
    }
