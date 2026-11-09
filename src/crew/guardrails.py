import logging
import re
import sqlite3
import unicodedata
from collections.abc import Callable
from contextlib import closing
from contextvars import ContextVar
from decimal import Decimal, InvalidOperation
from pathlib import Path

from src.crew.schemas import ExecutiveReport
from src.crew.tools import _format_currency_brl, build_deterministic_report
from src.ml.transaction_anomaly import DEFAULT_DB_PATH


LOGGER = logging.getLogger(__name__)
REQUEST_ID = ContextVar('request_id', default='cli')
NUMBER = re.compile(r'[-+]?\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?')
FORBIDDEN = re.compile(
    r'probabilidad\w*|chance\s+de\s+fraude|fraude\s+confirmad\w*|rotulos_fraude'
)


def _normalize_text(text):
    return ''.join(
        character for character in unicodedata.normalize('NFKD', text.casefold())
        if not unicodedata.combining(character)
    )


def _number_value(token):
    if ',' in token:
        token = token.replace('.', '').replace(',', '.')
    try:
        return Decimal(token)
    except InvalidOperation as error:
        raise ValueError('Formato numérico inválido no texto.') from error


def _numbers(text):
    return {_number_value(match.group()) for match in NUMBER.finditer(text)}


def _tool_numbers(ranking):
    numbers = set()

    def collect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key != 'descricao_contexto':
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
            numbers.update(_numbers(str(value)))

    collect(ranking)
    return numbers


def validate_report(candidate, ranking, db_path=DEFAULT_DB_PATH, top_n=10):
    if isinstance(candidate, ExecutiveReport):
        candidate = candidate.model_dump()
    if isinstance(candidate, str):
        parsed = ExecutiveReport.model_validate_json(candidate)
    elif isinstance(candidate, dict):
        parsed = ExecutiveReport.model_validate(candidate)
    else:
        raise ValueError('O Redator deve retornar um objeto JSON de relatório.')
    report = parsed.model_dump()
    if type(top_n) is not int or not 1 <= top_n <= 20:
        raise ValueError('top_n deve ser inteiro entre 1 e 20.')
    if len(ranking) != top_n or [row['posicao'] for row in ranking] != list(range(1, top_n + 1)):
        raise ValueError('A ferramenta deve retornar exatamente as posições solicitadas.')
    expected_ids = [row['id_transacao'] for row in ranking]
    if len(set(expected_ids)) != top_n:
        raise ValueError('A ferramenta retornou IDs duplicados.')

    cited_ids = [case['id_transacao'] for case in report['casos']]
    all_ids = list(set(expected_ids + cited_ids))
    placeholders = ','.join('?' for _ in all_ids)
    path = Path(db_path).resolve()
    with closing(sqlite3.connect(f'{path.as_uri()}?mode=ro', uri=True)) as connection:
        rows = connection.execute(
            f'SELECT id, valor, data_hora, localizacao, tipo_transacao, canal, '
            f'categoria_estabelecimento FROM transacoes WHERE id IN ({placeholders})',
            all_ids,
        ).fetchall()
    database = {row[0]: row[1:] for row in rows}
    if any(identifier not in database for identifier in all_ids):
        raise ValueError('Um ou mais IDs não existem em transacoes.')
    if cited_ids != expected_ids:
        raise ValueError('IDs ou ordem do relatório divergem do top-10 da ferramenta.')

    for row in ranking:
        facts = (
            row['valor'], row['data_hora'], row['cidade'], row['tipo'],
            row['canal'], row['categoria'],
        )
        if database[row['id_transacao']] != facts:
            raise ValueError('Valor, data, cidade ou campos estruturados divergem do banco.')
        if row['valor_formatado'] != _format_currency_brl(row['valor']):
            raise ValueError('Valor formatado da ferramenta diverge do banco.')

    deterministic = build_deterministic_report(ranking)
    for case, source in zip(report['casos'], deterministic['casos']):
        if case['posicao'] != source['posicao']:
            raise ValueError('Posição no ranking divergente da ferramenta.')
        if case['fatores'] != source['fatores']:
            raise ValueError('Fatores, ordem, rótulos ou valores divergem da ferramenta.')
        # O formato fechado impede citações adicionais sem vínculo com a transação.
        if case['resumo'] != source['resumo']:
            raise ValueError(
                'Resumo deve preservar o formato factual canônico da ferramenta '
                'e citar a razão, não z-scores.'
            )

    report['limitacoes'] = list(dict.fromkeys(
        deterministic['limitacoes'] + report['limitacoes']
    ))

    texts = [
        case['resumo'] for case in report['casos']
    ] + [
        factor[field]
        for case in report['casos'] for factor in case['fatores']
        for field in ('nome', 'rotulo', 'valor')
    ] + report['recomendacoes'] + report['limitacoes']
    normalized = _normalize_text(' '.join(texts))
    if FORBIDDEN.search(normalized) or ('%' in normalized and 'fraude' in normalized):
        raise ValueError('Linguagem proibida; use prioridade de revisão e suspeita.')
    allowed_numbers = _tool_numbers(ranking)
    if any(_numbers(text) - allowed_numbers for text in texts):
        raise ValueError('Todo número no texto deve aparecer nos dados estruturados da tool.')
    for text in report['recomendacoes'] + report['limitacoes']:
        if text not in deterministic['limitacoes'] and re.search(
            r'r\$|\b(?:valor|data|cidade|localizacao|id|transacao)\b|'
            r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}',
            _normalize_text(text),
        ):
            raise ValueError('Recomendações e limitações devem ser gerais, sem citações factuais.')
    return report, {
        'ids_verificados': len(cited_ids),
        'valores_e_datas_iguais_ao_banco': True,
        'cidades_iguais_ao_banco': True,
        'features_iguais_ao_top10': True,
        'numeros_iguais_a_tool': True,
        'linguagem_validada': True,
    }


def generate_validated_report(
    generate: Callable[[str | None], object], ranking, db_path=DEFAULT_DB_PATH,
    top_n=10,
):
    error_message = None
    for attempt in range(2):
        candidate = generate(error_message)
        try:
            report, audit = validate_report(candidate, ranking, db_path, top_n)
            report['degradado'] = False
            return report, audit
        except ValueError as error:
            error_message = str(error)
            LOGGER.warning(
                'request_id=%s relatório rejeitado na tentativa %s.',
                REQUEST_ID.get(), attempt + 1,
            )
    LOGGER.error(
        'request_id=%s correção rejeitada; devolvendo relatório determinístico degradado.',
        REQUEST_ID.get(),
    )
    fallback = build_deterministic_report(ranking)
    fallback['degradado'] = True
    return validate_report(fallback, ranking, db_path, top_n)
