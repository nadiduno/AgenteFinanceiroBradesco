import sqlite3
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_synthetic_data(db_path='data/transacoes.db', n_rows=5000, anomaly_rate=0.02):
    """
    Gera dados sintéticos de transações bancárias com anomalias.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    np.random.seed(42)
    
    # Criando IDs de clientes
    client_ids = [f'CLI{str(i).zfill(5)}' for i in range(1, 501)]
    
    data = {
        'id': range(1, n_rows + 1),
        'client_id': np.random.choice(client_ids, n_rows),
        'valor': np.random.exponential(scale=500, size=n_rows).round(2),
        'data_hora': [datetime(2023, 1, 1) + timedelta(minutes=np.random.randint(0, 525600)) for _ in range(n_rows)],
        'tipo_transacao': np.random.choice(['PIX', 'TED', 'DOC', 'BOLETO', 'CARTAO'], n_rows),
        'localizacao': np.random.choice(['SAO PAULO', 'RIO DE JANEIRO', 'CURITIBA', 'SALVADOR', 'ONLINE'], n_rows),
        'is_anomaly': 0
    }
    
    df = pd.DataFrame(data)
    
    # Inserindo anomalias (2%)
    n_anomalies = int(n_rows * anomaly_rate)
    anomaly_indices = np.random.choice(df.index, n_anomalies, replace=False)
    
    for idx in anomaly_indices:
        df.at[idx, 'is_anomaly'] = 1
        # Anomalia de valor: valor muito alto
        if np.random.random() > 0.5:
            df.at[idx, 'valor'] = np.random.uniform(5000, 20000).round(2)
        # Anomalia de frequência/horário: transações de madrugada com valor alto
        else:
            df.at[idx, 'data_hora'] = df.at[idx, 'data_hora'].replace(hour=np.random.randint(2, 5))
            df.at[idx, 'valor'] = np.random.uniform(1000, 5000).round(2)

    # Conectando ao SQLite e salvando
    conn = sqlite3.connect(db_path)
    df.to_sql('transacoes', conn, if_exists='replace', index=False)
    
    # Criando uma tabela de metadados para o esquema
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS clientes (client_id TEXT PRIMARY KEY, nome TEXT, score_credito INTEGER)")
    
    unique_clients = df['client_id'].unique()
    clients_data = [(cid, f'Cliente {cid}', np.random.randint(300, 1000)) for cid in unique_clients]
    cursor.executemany("INSERT OR IGNORE INTO clientes VALUES (?, ?, ?)", clients_data)
    
    conn.commit()
    conn.close()
    print(f"Base de dados gerada com sucesso em: {db_path}")
    print(f"Total de registros: {n_rows}")
    print(f"Anomalias inseridas: {n_anomalies} (~{anomaly_rate*100}%)")

if __name__ == "__main__":
    generate_synthetic_data()
