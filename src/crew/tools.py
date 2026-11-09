import json
import os
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
RANKING_SCOPE = (
    'O top-10 é calculado sobre a base inteira: ranking de revisão, '
    'não avaliação de desempenho.'
)


def _historical_value_ratios(transactions):
    ordered = transactions.assign(
        timestamp=pd.to_datetime(transactions['data_hora'], errors='raise')
    ).sort_values(['timestamp', 'id'], kind='stable')
    history = {}
    ratios = {}
    for row in ordered.itertuples():
        values = history.setdefault(row.client_id, [])
        median = float(np.median(values)) if values else None
        ratios[row.Index] = (
            float(row.valor) / median if median is not None and median > 0 else None
        )
        values.append(float(row.valor))
    return [ratios[index] for index in transactions.index]


def _report_factors(row):
    factors = []
    for feature in row['features_mais_influentes']:
        if feature['nome'] == 'valor_z_cliente':
            factors.append({
                'nome': 'valor_vs_mediana_cliente',
                'rotulo': 'razão do valor pela mediana histórica do cliente',
                'valor': row['valor_vs_mediana_cliente'],
            })
        else:
            factors.append({
                'nome': feature['nome'],
                'rotulo': feature['rotulo'],
                'valor': feature['valor'],
            })
    return factors


def _format_currency_brl(value):
    return f'R$ {value:,.2f}'.replace(',', '\0').replace('.', ',').replace('\0', '.')


def _serialize_feature_value(value):
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f'{number:.6g}'


def _bounded_z_score(value):
    clipped = float(np.clip(value, -10, 10))
    return {'valor': _serialize_feature_value(clipped), 'limitado': value != clipped}


@lru_cache(maxsize=2)
def _compute_numeric_top_ten(db_path, top_n=10):
    transactions = load_transactions(db_path)
    raw_features = _client_history_features(transactions)
    value_ratios = _historical_value_ratios(transactions)
    scaler = StandardScaler()
    model_features = scaler.fit_transform(raw_features)
    detector = _fit_detector(model_features, random_state=42)
    scores = -detector.decision_function(model_features)
    order = np.argsort(-scores, kind='stable')[:top_n]
    reference_values = np.median(model_features, axis=0)
    influences = _local_feature_sensitivities(
        detector,
        model_features,
        order.tolist(),
        list(NUMERIC_FEATURE_COLUMNS),
        reference_values,
    )

    ranking = []
    audit_rows = []
    for position, (row_index, impacts) in enumerate(zip(order, influences), start=1):
        row = transactions.iloc[int(row_index)]
        raw_row = raw_features[int(row_index)]
        raw_z = float(raw_row[0])
        bounded_z = _bounded_z_score(raw_z)
        ratio = value_ratios[int(row_index)]
        audit_rows.append({
            'id_transacao': int(row['id']),
            'valor_z_cliente_bruto': raw_z,
        })
        top_features = []
        for feature_name, impact in impacts[:3]:
            feature_index = NUMERIC_FEATURE_COLUMNS.index(feature_name)
            feature = {
                'nome': feature_name,
                'rotulo': _FEATURE_LABELS[feature_name],
                'valor': bounded_z['valor'] if feature_name == 'valor_z_cliente'
                else _serialize_feature_value(raw_row[feature_index]),
                'impacto_local_heuristico': round(float(impact), 8),
            }
            if feature_name == 'valor_z_cliente':
                feature['limitado'] = bounded_z['limitado']
            top_features.append(feature)
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
            'valor_z_cliente': bounded_z,
            'valor_vs_mediana_cliente': (
                _serialize_feature_value(ratio) if ratio is not None
                else 'indisponível: histórico ausente ou mediana não positiva'
            ),
            'escopo_ranking': RANKING_SCOPE.replace('top-10', f'top-{top_n}'),
            'features_mais_influentes': top_features,
            'descricao_contexto': {
                'classificacao': 'texto livre, não evidência',
                'texto': str(row['descricao']),
            },
        })
    return {'ranking': ranking, 'auditoria': audit_rows}


def _get_numeric_top_ten(db_path=DEFAULT_DB_PATH, top_n=10):
    if type(top_n) is not int or not 1 <= top_n <= 20:
        raise ValueError('top_n deve ser inteiro entre 1 e 20.')
    enabled = os.getenv('USE_TEXT_FEATURES', 'false').strip().lower()
    if enabled not in {'true', 'false'}:
        raise ValueError('USE_TEXT_FEATURES deve ser true ou false.')
    if enabled == 'true':
        raise ValueError(
            'A ferramenta da Fase 4 é somente numérica; defina USE_TEXT_FEATURES=false.'
        )
    if top_n == 10:
        return _compute_numeric_top_ten(str(Path(db_path).resolve()))['ranking']
    return _compute_numeric_top_ten(str(Path(db_path).resolve()), top_n)['ranking']


def numeric_top_ten_json():
    return json.dumps(_get_numeric_top_ten(), ensure_ascii=False)


def create_numeric_ranking_tool(db_path=DEFAULT_DB_PATH, top_n=10):
    from crewai.tools import tool

    @tool('rank_numeric_transactions')
    def rank_numeric_transactions() -> str:
        """Calcula o top solicitado de revisão com campos e features de transacoes.

        Retorna ID e posição exatos, tipo, canal, categoria, cidade, data/hora,
        valor e as três features locais mais influentes. A descrição aparece apenas
        como contexto identificado por "texto livre, não evidência". Não acessa rótulos.
        Ranking sobre a base inteira, não avaliação de desempenho. Z-score limitado
        a ±10; cite valor_vs_mediana_cliente, nunca o z-score bruto.
        """
        return json.dumps(_get_numeric_top_ten(db_path, top_n), ensure_ascii=False)

    return rank_numeric_transactions


def build_deterministic_report(ranking):
    cases = []
    for row in ranking:
        factors = _report_factors(row)
        summary = (
            f"Prioridade de revisão na posição {row['posicao']} do ranking; "
            f"suspeita a verificar; "
            f"tipo={row['tipo']}; canal={row['canal']}; "
            f"categoria={row['categoria']}; cidade={row['cidade']}; "
            f"data/hora={row['data_hora']}; valor={_format_currency_brl(row['valor'])}; "
            f"valor_vs_mediana_cliente={row['valor_vs_mediana_cliente']}; "
            f"fatores detalhados na lista."
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
            RANKING_SCOPE.replace('top-10', f'top-{len(ranking)}'),
            'O score é heurístico e representa somente uma posição relativa no ranking.',
            'A descrição é texto livre, não evidência; utilize os campos estruturados e as features.',
            'A ferramenta não lê rótulos e não valida ocorrência de fraude.',
        ],
        'degradado': False,
    }


def verify_report_sources(report, ranking, db_path=DEFAULT_DB_PATH, top_n=10):
    from src.crew.guardrails import validate_report

    audit = validate_report(report, ranking, db_path, top_n)[1]
    if top_n == 10:
        computed = _compute_numeric_top_ten(str(Path(db_path).resolve()))
    else:
        computed = _compute_numeric_top_ten(str(Path(db_path).resolve()), top_n)
    audit['z_scores_brutos'] = computed['auditoria']
    return audit
