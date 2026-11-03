import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / 'data' / 'transacoes.db'


def generate_synthetic_data(db_path=DEFAULT_DB_PATH, n_rows=5000, anomaly_rate=0.02):
    """Gera clientes, transações sintéticas e rótulos de fraude separados."""
    if n_rows < 1:
        raise ValueError('n_rows deve ser maior que zero.')
    if not 0 <= anomaly_rate <= 1:
        raise ValueError('anomaly_rate deve estar entre zero e um.')

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    client_ids = [f'CLI{number:05d}' for number in range(1, 501)]
    clients = [
        (client_id, f'Cliente {client_id}', int(rng.integers(300, 1001)))
        for client_id in client_ids
    ]
    transactions = [
        [
            transaction_id,
            rng.choice(client_ids),
            round(float(rng.exponential(scale=500)), 2),
            datetime(2023, 1, 1) + timedelta(minutes=int(rng.integers(0, 525600))),
            rng.choice(['PIX', 'TED', 'DOC', 'BOLETO', 'CARTAO']),
            rng.choice(['SAO PAULO', 'RIO DE JANEIRO', 'CURITIBA', 'SALVADOR', 'ONLINE']),
        ]
        for transaction_id in range(1, n_rows + 1)
    ]

    n_anomalies = int(n_rows * anomaly_rate)
    anomaly_indices = rng.choice(n_rows, n_anomalies, replace=False)
    fraud_labels = [0] * n_rows
    for index in anomaly_indices:
        fraud_labels[index] = 1
        if rng.random() > 0.5:
            transactions[index][2] = round(float(rng.uniform(5000, 20000)), 2)
        else:
            transaction_time = transactions[index][3]
            transactions[index][3] = transaction_time.replace(hour=int(rng.integers(2, 5)))
            transactions[index][2] = round(float(rng.uniform(1000, 5000)), 2)

    labels = [
        (transaction_id, fraud_labels[transaction_id - 1])
        for transaction_id in range(1, n_rows + 1)
    ]

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
            INSERT INTO transacoes
                (id, client_id, valor, data_hora, tipo_transacao, localizacao)
            VALUES (?, ?, ?, ?, ?, ?)
            ''',
            [
                (transaction_id, client_id, amount, timestamp.isoformat(sep=' '), kind, location)
                for transaction_id, client_id, amount, timestamp, kind, location in transactions
            ],
        )
        connection.executemany(
            'INSERT INTO rotulos_fraude (transacao_id, fraude) VALUES (?, ?)',
            labels,
        )
        connection.commit()
    finally:
        connection.close()

    print(f'Base sintética gerada em: {db_path.resolve()}')
    print(f'Transações: {n_rows}; rótulos de fraude positivos: {n_anomalies}')

if __name__ == "__main__":
    generate_synthetic_data()
