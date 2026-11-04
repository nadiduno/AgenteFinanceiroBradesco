import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / 'data' / 'transacoes.db'
CITIES = ('SAO PAULO', 'RIO DE JANEIRO', 'CURITIBA', 'SALVADOR', 'RECIFE', 'BRASILIA')
COUNTRIES = ('Brasil', 'Argentina', 'Chile', 'Estados Unidos', 'Portugal')
CATEGORIES = (
    'supermercado',
    'restaurante',
    'farmácia',
    'combustível',
    'eletrônicos',
    'jogos online',
    'joalheria',
    'cripto',
)
CHANNELS = ('app', 'web', 'POS', 'ATM')
TRANSACTION_TYPES = ('PIX', 'TED', 'DOC', 'BOLETO', 'CARTAO', 'SAQUE')
NOT_APPLICABLE = 'não aplicável'
DESCRIPTION_RULES = {
    ('SAQUE', 'ATM', NOT_APPLICABLE): (('saque', 'vários saques'), None),
    ('PIX', 'app', NOT_APPLICABLE): (('pix ', 'transferência '), None),
    ('PIX', 'web', NOT_APPLICABLE): (('pix ', 'transferência '), None),
    ('TED', 'app', NOT_APPLICABLE): (('transferência ',), None),
    ('TED', 'web', NOT_APPLICABLE): (('transferência ',), None),
    ('DOC', 'app', NOT_APPLICABLE): (('transferência ',), None),
    ('DOC', 'web', NOT_APPLICABLE): (('transferência ',), None),
    ('BOLETO', 'app', NOT_APPLICABLE): (('pagamento ',), None),
    ('BOLETO', 'web', NOT_APPLICABLE): (('pagamento ',), None),
}
_CARD_CATEGORY_MARKERS = {
    'supermercado': 'supermercado',
    'restaurante': 'restaurante',
    'farmácia': 'farmácia',
    'combustível': 'combustível',
    'eletrônicos': 'eletrônicos',
    'jogos online': 'jogos online',
    'joalheria': 'joalheria',
    'cripto': 'criptoativos',
}
for _channel in ('app', 'web'):
    for _category, _marker in _CARD_CATEGORY_MARKERS.items():
        DESCRIPTION_RULES[('CARTAO', _channel, _category)] = (
            ('compra online',),
            _marker,
        )
for _category, _marker in _CARD_CATEGORY_MARKERS.items():
    if _category in ('jogos online', 'cripto'):
        continue
    DESCRIPTION_RULES[('CARTAO', 'POS', _category)] = (
        ('abastecimento em posto',) if _category == 'combustível'
        else ('compra presencial',),
        _marker,
    )
SUSPICIOUS_DESCRIPTION_MARKERS = (
    'poucos dias',
    'novo favorecido',
    'recém-cadastrada',
    'dispositivo novo',
    'fora do padrão',
    'horário incomum',
    'vários saques seguidos',
    'baixo valor repetida',
    'internacional',
    'local não habitual',
    'cobrança inesperada',
)


def description_rule_violations(transactions):
    """Retorna transações que não respeitam o mapa tipo/canal/categoria."""
    violations = []
    for transaction in transactions:
        key = (
            transaction['tipo_transacao'],
            transaction['canal'],
            transaction['categoria_estabelecimento'],
        )
        rule = DESCRIPTION_RULES.get(key)
        description = transaction['descricao'].casefold()
        if rule is None:
            violations.append((transaction['id'], 'combinação sem regra'))
            continue
        allowed_prefixes, required_marker = rule
        if not description.startswith(allowed_prefixes):
            violations.append((transaction['id'], 'prefixo incompatível'))
        elif required_marker and required_marker not in description:
            violations.append((transaction['id'], 'categoria não descrita'))
        elif (
            transaction['canal'] == 'app'
            and 'pelo internet banking' in description
        ) or (
            transaction['canal'] == 'web'
            and 'pelo aplicativo' in description
        ):
            violations.append((transaction['id'], 'canal descrito incorretamente'))
    return violations


def format_currency_brl(amount):
    formatted = f'{amount:,.2f}'
    formatted = formatted.replace(',', '\0').replace('.', ',').replace('\0', '.')
    return f'R$ {formatted}'


def build_transaction_description(transaction, rng):
    """Gera texto coerente com tipo, canal e categoria sem consultar rótulos."""
    hour = transaction['timestamp'].hour
    amount_ratio = transaction['amount'] / transaction['client_median']
    has_risk_context = any((
        transaction['destination_age'] is not None
        and transaction['destination_age'] <= 7,
        transaction['device_new'],
        transaction['recent_count'] >= 4,
        hour < 6,
        transaction['country'] != 'Brasil',
        transaction['city'] != transaction['home_city'],
        amount_ratio >= 6 or amount_ratio <= 0.12,
    ))
    suspicious = (
        (has_risk_context and rng.random() < 0.82)
        or (not has_risk_context and rng.random() < 0.025)
    )

    kind = transaction['kind']
    channel = transaction['channel']
    category = transaction['category']
    channel_method = (
        'pelo aplicativo' if channel == 'app' else 'pelo internet banking'
    )

    if kind == 'SAQUE':
        if channel != 'ATM':
            raise ValueError('Transações de saque devem usar o canal ATM.')
        descriptions = (
            (
                'vários saques seguidos em caixa eletrônico',
                'saque em caixa eletrônico fora do padrão habitual',
                'saque em caixa eletrônico durante viagem',
                'saque em caixa eletrônico em local não habitual',
            )
            if suspicious
            else (
                'saque em caixa eletrônico próximo',
                'saque eventual no caixa eletrônico',
                'saque em caixa eletrônico durante viagem',
                'saque em terminal bancário',
            )
        )
        return str(rng.choice(descriptions))

    if kind in ('PIX', 'TED', 'DOC'):
        if channel not in ('app', 'web'):
            raise ValueError('Transferências devem usar app ou web.')
        if suspicious:
            descriptions = (
                (
                    'PIX para conta aberta há poucos dias',
                    'transferência para novo favorecido',
                    'transferência para conta recém-cadastrada',
                    'transferência após acesso por dispositivo novo',
                )
                if kind == 'PIX'
                else (
                    'transferência para novo favorecido',
                    'transferência para conta recém-cadastrada',
                    'transferência internacional fora do padrão',
                )
            )
        else:
            if kind == 'PIX':
                descriptions = (
                    'PIX para contato conhecido',
                    'transferência recorrente para favorecido habitual',
                    f'transferência {channel_method} para contato conhecido',
                    f'transferência bancária {channel_method}',
                )
            else:
                descriptions = (
                    'transferência recorrente para favorecido habitual',
                    f'transferência {channel_method} para contato conhecido',
                    f'transferência bancária {channel_method}',
                )
        return str(rng.choice(descriptions))

    if kind == 'BOLETO':
        if channel not in ('app', 'web'):
            raise ValueError('Pagamentos de boleto devem usar app ou web.')
        descriptions = (
            (
                f'pagamento de boleto fora do padrão {channel_method}',
                'pagamento de boleto após acesso por dispositivo novo',
                'pagamento online de cobrança inesperada',
            )
            if suspicious
            else (
                f'pagamento habitual de boleto {channel_method}',
                'pagamento recorrente de serviço',
                f'pagamento de conta {channel_method}',
            )
        )
        return str(rng.choice(descriptions))

    if kind != 'CARTAO':
        raise ValueError(f'Tipo de transação não suportado: {kind}')
    if channel not in ('app', 'web', 'POS'):
        raise ValueError('Compras com cartão devem usar app, web ou POS.')

    category_phrases = {
        'supermercado': 'no supermercado',
        'restaurante': 'em restaurante',
        'farmácia': 'em farmácia',
        'combustível': 'em posto de combustível',
        'eletrônicos': 'de eletrônicos',
        'jogos online': 'de jogos online',
        'joalheria': 'em joalheria',
        'cripto': 'de criptoativos',
    }
    if category not in category_phrases:
        raise ValueError(f'Categoria incompatível com compra: {category}')

    online = channel in ('app', 'web')
    purchase_channel = 'online' if online else 'presencial'
    category_phrase = category_phrases[category]
    if category == 'combustível' and not online:
        normal_base = 'abastecimento em posto de combustível'
    else:
        normal_base = f'compra {purchase_channel} {category_phrase}'

    if suspicious:
        descriptions = [
            f'{normal_base} fora do padrão habitual',
            f'{normal_base} em horário incomum',
        ]
        if online:
            descriptions.extend((
                f'compra online internacional {category_phrase}',
                f'compra online {category_phrase} de baixo valor repetida como teste de cartão',
                f'compra online {category_phrase} após acesso por dispositivo novo',
            ))
        if transaction['country'] != 'Brasil':
            descriptions.append(f'{normal_base} internacional durante viagem')
        return str(rng.choice(descriptions))

    if transaction['country'] != 'Brasil':
        return f'{normal_base} internacional durante viagem'
    return str(rng.choice((
        normal_base,
        f'{normal_base} habitual',
        f'{normal_base} em estabelecimento conhecido',
    )))


def generate_synthetic_data(
    db_path=DEFAULT_DB_PATH,
    n_rows=5000,
    anomaly_rate=0.02,
    seed=42,
):
    """Gera transações sintéticas com descrições independentes dos rótulos."""
    if n_rows < 1:
        raise ValueError('n_rows deve ser maior que zero.')
    if not 0 <= anomaly_rate <= 1:
        raise ValueError('anomaly_rate deve estar entre zero e um.')

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    client_ids = [f'CLI{number:05d}' for number in range(1, 501)]
    client_medians = dict(zip(client_ids, np.exp(rng.uniform(4.3, 6.2, len(client_ids)))))
    home_cities = dict(zip(client_ids, rng.choice(CITIES, len(client_ids))))
    clients = [
        (client_id, f'Cliente {client_id}', int(rng.integers(300, 1001)))
        for client_id in client_ids
    ]

    n_fraud = int(n_rows * anomaly_rate)
    fraud_indices = set(rng.choice(n_rows, n_fraud, replace=False).tolist())
    transaction_rows = []
    labels = []
    previous_client_times = {}

    for index in range(n_rows):
        transaction_id = index + 1
        client_id = str(rng.choice(client_ids))
        client_median = client_medians[client_id]
        is_fraud_case = index in fraud_indices
        timestamp = datetime(2023, 1, 1) + timedelta(
            days=int(rng.integers(0, 365)),
            hours=int(np.clip(rng.normal(15, 3.2), 8, 22)),
            minutes=int(rng.integers(0, 60)),
        )
        amount = float(rng.lognormal(np.log(client_median), 0.55))
        kind = str(rng.choice(TRANSACTION_TYPES, p=(0.43, 0.11, 0.04, 0.13, 0.21, 0.08)))
        if kind == 'SAQUE':
            category = NOT_APPLICABLE
            channel = 'ATM'
        elif kind == 'CARTAO':
            category = str(rng.choice(
                CATEGORIES,
                p=(0.27, 0.20, 0.12, 0.12, 0.12, 0.07, 0.07, 0.03),
            ))
            channel = str(rng.choice(('app', 'web', 'POS'), p=(0.25, 0.20, 0.55)))
        else:
            category = NOT_APPLICABLE
            channel = str(rng.choice(('app', 'web'), p=(0.70, 0.30)))
        home_city = home_cities[client_id]
        city = home_city if rng.random() < 0.98 else str(rng.choice(CITIES))
        country = 'Brasil'
        device_new = int(rng.random() < 0.01)
        destination_age = (
            int(rng.integers(8, 3000)) if kind in ('PIX', 'TED') else None
        )
        recent_count = int(rng.choice((0, 1, 2, 3), p=(0.50, 0.30, 0.16, 0.04)))
        if kind in ('CARTAO', 'SAQUE') and rng.random() < 0.05:
            country = str(rng.choice(COUNTRIES[1:]))
            if kind == 'CARTAO':
                category = str(rng.choice(('eletrônicos', 'restaurante', 'combustível')))
                channel = str(rng.choice(('web', 'POS')))

        if is_fraud_case:
            scenario = str(rng.choice(('amount', 'night', 'city', 'burst', 'combined')))
            if scenario in ('amount', 'combined'):
                amount *= float(rng.choice((rng.uniform(8, 18), rng.uniform(0.025, 0.12))))
            if scenario in ('night', 'combined'):
                timestamp = timestamp.replace(hour=int(rng.integers(0, 6)))
            if scenario in ('city', 'combined'):
                city = str(rng.choice([candidate for candidate in CITIES if candidate != home_city]))
            if scenario in ('burst', 'combined'):
                recent_count = int(rng.integers(4, 13))
            if rng.random() < 0.60:
                device_new = 1
            if kind in ('PIX', 'TED') and rng.random() < 0.65:
                destination_age = int(rng.integers(0, 8))
            if rng.random() < 0.18:
                country = str(rng.choice(COUNTRIES[1:]))
            if rng.random() < 0.30:
                if kind == 'CARTAO':
                    category = str(rng.choice(('jogos online', 'joalheria', 'cripto', 'eletrônicos')))
            if scenario == 'burst':
                if rng.random() < 0.60:
                    kind = 'SAQUE'
                    category = NOT_APPLICABLE
                    channel = 'ATM'
                    destination_age = None

        if kind == 'CARTAO' and category in ('jogos online', 'cripto') and channel == 'POS':
            channel = str(rng.choice(('app', 'web')))

        previous_time = previous_client_times.get(client_id)
        if (
            not is_fraud_case
            and previous_time is not None
            and 8 <= previous_time.hour < 22
            and rng.random() < 0.08
        ):
            timestamp = previous_time + timedelta(minutes=int(rng.integers(1, 60)))
        previous_client_times[client_id] = timestamp

        transaction = {
            'id': transaction_id,
            'client_id': client_id,
            'amount': round(amount, 2),
            'timestamp': timestamp,
            'kind': kind,
            'city': city,
            'home_city': home_city,
            'category': category,
            'channel': channel,
            'country': country,
            'device_new': device_new,
            'destination_age': destination_age,
            'recent_count': recent_count,
            'client_median': client_median,
        }
        description_rng = np.random.default_rng(
            np.random.SeedSequence([seed, transaction_id, 1])
        )
        description = build_transaction_description(transaction, description_rng)
        transaction_rows.append((
            transaction_id,
            client_id,
            transaction['amount'],
            timestamp.isoformat(sep=' '),
            kind,
            city,
            category,
            channel,
            country,
            device_new,
            destination_age,
            recent_count,
            description,
        ))
        labels.append((transaction_id, int(is_fraud_case)))

    connection = sqlite3.connect(db_path)
    try:
        connection.execute('PRAGMA foreign_keys = ON')
        connection.executescript(
            '''
            DROP TABLE IF EXISTS rotulos_fraude;
            DROP TABLE IF EXISTS transacoes;
            DROP TABLE IF EXISTS clientes;

            CREATE TABLE clientes (
                client_id TEXT PRIMARY KEY,
                nome TEXT NOT NULL,
                score_credito INTEGER NOT NULL CHECK (score_credito BETWEEN 300 AND 1000)
            );

            CREATE TABLE transacoes (
                id INTEGER PRIMARY KEY,
                client_id TEXT NOT NULL,
                valor REAL NOT NULL CHECK (valor >= 0),
                data_hora TEXT NOT NULL,
                tipo_transacao TEXT NOT NULL,
                localizacao TEXT NOT NULL,
                categoria_estabelecimento TEXT NOT NULL,
                canal TEXT NOT NULL,
                pais TEXT NOT NULL,
                dispositivo_novo INTEGER NOT NULL CHECK (dispositivo_novo IN (0, 1)),
                idade_conta_destino_dias INTEGER,
                qtd_transacoes_ultima_hora INTEGER NOT NULL,
                descricao TEXT NOT NULL,
                FOREIGN KEY (client_id) REFERENCES clientes(client_id)
            );

            CREATE TABLE rotulos_fraude (
                transacao_id INTEGER PRIMARY KEY,
                fraude INTEGER NOT NULL CHECK (fraude IN (0, 1)),
                FOREIGN KEY (transacao_id) REFERENCES transacoes(id)
            );
            '''
        )
        connection.executemany(
            'INSERT INTO clientes (client_id, nome, score_credito) VALUES (?, ?, ?)',
            clients,
        )
        connection.executemany(
            '''
            INSERT INTO transacoes (
                id, client_id, valor, data_hora, tipo_transacao, localizacao,
                categoria_estabelecimento, canal, pais, dispositivo_novo,
                idade_conta_destino_dias, qtd_transacoes_ultima_hora, descricao
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            transaction_rows,
        )
        connection.executemany(
            'INSERT INTO rotulos_fraude (transacao_id, fraude) VALUES (?, ?)',
            labels,
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    print(f'Base sintética gerada em: {db_path.resolve()}')
    print(f'Transações: {n_rows}; rótulos positivos: {n_fraud}')


if __name__ == '__main__':
    generate_synthetic_data()
