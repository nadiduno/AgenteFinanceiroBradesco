import logging
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / 'data' / 'transacoes.db'
MODEL_ID = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
RISK_PROTOTYPES = (
    'PIX para conta aberta há poucos dias',
    'transferência para novo favorecido',
    'compra internacional em loja online de eletrônicos',
    'vários saques seguidos em caixa eletrônico',
    'compra de valor baixo repetida como teste de cartão',
    'acesso pelo aplicativo em dispositivo não reconhecido',
    'compra em estabelecimento fora do padrão habitual',
    'transferência para carteira de criptoativos',
)
NORMAL_PROTOTYPES = (
    'compra semanal no supermercado do bairro',
    'pagamento habitual em restaurante',
    'compra de medicamento em farmácia',
    'abastecimento em posto de combustível conhecido',
    'compra presencial de eletrônicos',
    'transferência para contato conhecido',
    'pagamento de conta pelo aplicativo',
    'compra internacional durante viagem',
)
NUMERIC_FEATURE_COLUMNS = (
    'valor_z_cliente',
    'cidade_nova_cliente',
    'hora_fora_padrao_cliente',
    'dispositivo_novo',
    'idade_conta_destino_dias',
    'qtd_transacoes_ultima_hora',
)


def load_transactions(db_path=DEFAULT_DB_PATH):
    """Lê apenas transações; rótulos são carregados à parte na avaliação."""
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f'Banco SQLite não encontrado: {path}')

    with closing(sqlite3.connect(f'{path.as_uri()}?mode=ro', uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            '''
            SELECT
                id, client_id, valor, data_hora, tipo_transacao, localizacao,
                categoria_estabelecimento, canal, pais, dispositivo_novo,
                idade_conta_destino_dias, qtd_transacoes_ultima_hora, descricao
            FROM transacoes
            ORDER BY id
            '''
        ).fetchall()

    if not rows:
        raise ValueError('A tabela transacoes não contém registros.')
    transactions = pd.DataFrame([dict(row) for row in rows])
    if transactions['descricao'].isna().any() or transactions['descricao'].str.strip().eq('').any():
        raise ValueError('Todas as transações devem ter uma descricao não vazia.')
    return transactions


def _client_history_features(transactions):
    ordered = transactions.copy()
    ordered['timestamp'] = pd.to_datetime(ordered['data_hora'], errors='raise')
    ordered = ordered.sort_values(['timestamp', 'id'], kind='stable')
    features = np.zeros((len(ordered), len(NUMERIC_FEATURE_COLUMNS)), dtype=np.float64)
    client_state = {}

    for row_index, row in enumerate(ordered.itertuples(index=False)):
        state = client_state.setdefault(
            row.client_id,
            {'values': [], 'hours': [], 'cities': set()},
        )
        values = state['values']
        hours = state['hours']
        if len(values) >= 2:
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1))
            if std > 0:
                features[row_index, 0] = (float(row.valor) - mean) / std
        features[row_index, 1] = float(
            bool(state['cities']) and row.localizacao not in state['cities']
        )
        if len(hours) >= 5:
            hour_mean = float(np.mean(hours))
            hour_std = float(np.std(hours, ddof=1))
            distance = abs((row.timestamp.hour - hour_mean + 12) % 24 - 12)
            features[row_index, 2] = float(distance > max(2.0, 2 * hour_std))
        features[row_index, 3] = float(row.dispositivo_novo)
        features[row_index, 4] = (
            float(row.idade_conta_destino_dias)
            if pd.notna(row.idade_conta_destino_dias)
            else -1.0
        )
        features[row_index, 5] = float(row.qtd_transacoes_ultima_hora)

        values.append(float(row.valor))
        hours.append(float(row.timestamp.hour))
        state['cities'].add(row.localizacao)

    return pd.DataFrame(
        features,
        index=ordered.index,
        columns=NUMERIC_FEATURE_COLUMNS,
    ).loc[transactions.index].to_numpy()


def _risk_scores(embeddings, risk_prototypes, normal_prototypes):
    similarities = np.concatenate((
        cosine_similarity(embeddings, risk_prototypes),
        cosine_similarity(embeddings, normal_prototypes),
    ), axis=1)
    logits = similarities / 0.1
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities[:, :risk_prototypes.shape[0]].sum(axis=1)


class TransactionAnomalyPipeline:
    def __init__(
        self,
        model_id=MODEL_ID,
        batch_size=32,
        n_estimators=200,
    ):
        if batch_size < 1:
            raise ValueError('batch_size deve ser maior que zero.')
        if n_estimators < 1:
            raise ValueError('n_estimators deve ser maior que zero.')
        self.model_id = model_id
        self.batch_size = batch_size
        self.n_estimators = n_estimators
        self.backend = None
        self.unique_description_count = 0
        self._tokenizer = None
        self._model = None

    def _encode_unique_hugging_face(self, unique_texts):
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self._tokenizer is None or self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModel.from_pretrained(self.model_id)
            self._model.to('cpu')
            self._model.eval()

        batches = []
        with torch.no_grad():
            for start in range(0, len(unique_texts), self.batch_size):
                tokens = self._tokenizer(
                    unique_texts[start:start + self.batch_size],
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                )
                output = self._model(**tokens)
                attention_mask = tokens['attention_mask'].unsqueeze(-1)
                token_embeddings = output.last_hidden_state
                pooled = (token_embeddings * attention_mask).sum(dim=1)
                pooled = pooled / attention_mask.sum(dim=1).clamp(min=1)
                batches.append(torch.nn.functional.normalize(pooled, p=2, dim=1))
        if not batches:
            raise ValueError('Nenhum texto foi fornecido para gerar embeddings.')
        return torch.cat(batches).cpu().numpy().astype(np.float32)

    def _encode_descriptions_and_prototypes(self, descriptions):
        description_texts = list(descriptions)
        all_texts = (
            description_texts
            + list(RISK_PROTOTYPES)
            + list(NORMAL_PROTOTYPES)
        )
        unique_values, inverse = np.unique(all_texts, return_inverse=True)
        unique_texts = [str(text) for text in unique_values]
        self.unique_description_count = len(set(description_texts))
        vectors = self._encode_unique_hugging_face(unique_texts)
        mapped = vectors[inverse]
        n_descriptions = len(description_texts)
        risk_start = n_descriptions
        normal_start = risk_start + len(RISK_PROTOTYPES)
        self.backend = 'huggingface'
        return (
            mapped[:n_descriptions],
            mapped[risk_start:normal_start],
            mapped[normal_start:],
        )

    @staticmethod
    def _build_tfidf_compact_features(train_descriptions, test_descriptions):
        risk_and_normal = list(RISK_PROTOTYPES) + list(NORMAL_PROTOTYPES)
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            ngram_range=(1, 2),
            max_features=4096,
            dtype=np.float32,
        )
        train_unique, train_inverse = np.unique(
            list(train_descriptions),
            return_inverse=True,
        )
        test_unique, test_inverse = np.unique(
            list(test_descriptions),
            return_inverse=True,
        )
        fit_texts = list(dict.fromkeys(list(train_unique) + risk_and_normal))
        fit_matrix = vectorizer.fit_transform(fit_texts)
        train_matrix = vectorizer.transform(train_unique)
        test_matrix = vectorizer.transform(test_unique)
        prototype_matrix = vectorizer.transform(risk_and_normal)

        n_components = min(8, fit_matrix.shape[0] - 1, fit_matrix.shape[1] - 1)
        if n_components < 1:
            raise ValueError('Descrições insuficientes para TruncatedSVD.')
        reducer = TruncatedSVD(n_components=n_components, random_state=42)
        reducer.fit(fit_matrix)

        train_scores = _risk_scores(
            train_matrix,
            prototype_matrix[:len(RISK_PROTOTYPES)],
            prototype_matrix[len(RISK_PROTOTYPES):],
        )
        test_scores = _risk_scores(
            test_matrix,
            prototype_matrix[:len(RISK_PROTOTYPES)],
            prototype_matrix[len(RISK_PROTOTYPES):],
        )
        train_compact = reducer.transform(train_matrix)
        test_compact = reducer.transform(test_matrix)
        return (
            np.column_stack((train_scores[train_inverse], train_compact[train_inverse])),
            np.column_stack((test_scores[test_inverse], test_compact[test_inverse])),
        )

    @staticmethod
    def _historical_embedding_means(transactions, embeddings):
        ordered_indices = transactions.assign(
            timestamp=pd.to_datetime(transactions['data_hora'], errors='raise')
        ).sort_values(['timestamp', 'id'], kind='stable').index
        client_sums = {}
        client_counts = {}
        result = np.zeros_like(embeddings)
        for index in ordered_indices:
            client_id = transactions.at[index, 'client_id']
            count = client_counts.get(client_id, 0)
            if count:
                result[index] = client_sums[client_id] / count
            client_sums[client_id] = client_sums.get(
                client_id,
                np.zeros(embeddings.shape[1], dtype=np.float64),
            ) + embeddings[index]
            client_counts[client_id] = count + 1
        return result

    def _build_hugging_face_features(self, transactions, train_mask):
        descriptions = transactions['descricao'].tolist()
        embeddings, risk_prototypes, normal_prototypes = (
            self._encode_descriptions_and_prototypes(descriptions)
        )
        text_risk = _risk_scores(embeddings, risk_prototypes, normal_prototypes)
        historical_means = self._historical_embedding_means(transactions, embeddings)
        train_profiles = historical_means[train_mask]
        train_nonzero = np.linalg.norm(train_profiles, axis=1) > 0
        if train_nonzero.sum() < 2:
            raise ValueError('PCA temporal requer pelo menos dois históricos de cliente.')
        n_components = min(8, int(train_nonzero.sum()) - 1, embeddings.shape[1])
        reducer = PCA(n_components=n_components, random_state=42)
        reducer.fit(train_profiles[train_nonzero])
        compact_profiles = reducer.transform(historical_means)
        return np.column_stack((text_risk, compact_profiles))


def _split_by_time(transactions, train_fraction=0.8):
    timestamps = pd.to_datetime(transactions['data_hora'], errors='raise')
    if not 0.5 <= train_fraction < 1:
        raise ValueError('train_fraction deve estar entre 0.5 e 1.')
    cutoff = timestamps.quantile(train_fraction)
    train_mask = timestamps <= cutoff
    test_mask = timestamps > cutoff
    if not train_mask.any() or not test_mask.any():
        raise ValueError('A divisão temporal não produziu treino e teste.')
    return train_mask.to_numpy(), test_mask.to_numpy(), cutoff


def _build_structured_features(transactions, train_mask):
    history = _client_history_features(transactions)
    scaler = StandardScaler()
    numeric_train = scaler.fit_transform(history[train_mask])
    numeric_test = scaler.transform(history[~train_mask])
    return numeric_train, numeric_test


def _fit_detector(train_features, random_state=42):
    detector = IsolationForest(
        n_estimators=200,
        contamination='auto',
        random_state=random_state,
        n_jobs=1,
    )
    detector.fit(train_features)
    return detector


def _ranking_metrics(labels_test, anomaly_scores):
    ranking = np.argsort(-anomaly_scores, kind='stable')
    top_100 = ranking[:min(100, len(ranking))]
    precision_at_100 = float(np.asarray(labels_test)[top_100].mean())
    prevalence = float(np.mean(labels_test))
    return {
        'pr_auc': float(average_precision_score(labels_test, anomaly_scores)),
        'precision_at_100': precision_at_100,
        'ganho_sobre_acaso': precision_at_100 / prevalence if prevalence else 0.0,
        'prevalencia_teste': prevalence,
        'fraudes_teste': int(np.sum(labels_test)),
    }


def _select_audit_rows(
    test_transactions,
    labels_test,
    numeric_scores,
    text_scores,
    combined_scores,
):
    ranking = np.argsort(-combined_scores, kind='stable')
    rank_positions = np.empty(len(ranking), dtype=int)
    rank_positions[ranking] = np.arange(1, len(ranking) + 1)
    fraud_mask = np.asarray(labels_test, dtype=bool)
    top_mask = rank_positions <= 100
    group_positions = {
        'fraudes_detectadas': sorted(
            np.flatnonzero(fraud_mask & top_mask),
            key=lambda index: combined_scores[index],
            reverse=True,
        )[:3],
        'fraudes_perdidas': sorted(
            np.flatnonzero(fraud_mask & ~top_mask),
            key=lambda index: combined_scores[index],
            reverse=True,
        )[:3],
        'falsos_positivos_maior_score': sorted(
            np.flatnonzero(~fraud_mask & top_mask),
            key=lambda index: combined_scores[index],
            reverse=True,
        )[:3],
        'normais_score_baixo': sorted(
            np.flatnonzero(~fraud_mask),
            key=lambda index: combined_scores[index],
        )[:3],
    }

    table = {}
    for group, positions in group_positions.items():
        table[group] = []
        for position in positions:
            row = test_transactions.iloc[position]
            table[group].append({
                'id': int(row['id']),
                'cliente_id': row['client_id'],
                'data_hora': row['data_hora'],
                'valor': float(row['valor']),
                'tipo': row['tipo_transacao'],
                'cidade': row['localizacao'],
                'canal': row['canal'],
                'categoria': row['categoria_estabelecimento'],
                'descricao': row['descricao'],
                'score_numerico': float(numeric_scores[position]),
                'score_texto': float(text_scores[position]),
                'score_combinado': float(combined_scores[position]),
                'ranking_teste': int(rank_positions[position]),
                'rotulo_auditoria': int(fraud_mask[position]),
            })
    return table


def _local_feature_sensitivities(
    detector,
    test_features,
    row_positions,
    feature_names,
    reference_values,
):
    original_scores = -detector.decision_function(test_features[row_positions])
    explanations = []
    for row_number, position in enumerate(row_positions):
        perturbed = np.repeat(
            test_features[[position]],
            repeats=len(feature_names),
            axis=0,
        )
        perturbed[np.arange(len(feature_names)), np.arange(len(feature_names))] = (
            reference_values
        )
        perturbed_scores = -detector.decision_function(perturbed)
        impacts = [
            (
                feature_name,
                float(original_scores[row_number] - perturbed_score),
            )
            for feature_name, perturbed_score in zip(feature_names, perturbed_scores)
        ]
        explanations.append(sorted(impacts, key=lambda item: item[1], reverse=True))
    return explanations


def evaluate_scenarios(
    db_path=DEFAULT_DB_PATH,
    seeds=(42, 43, 44, 45, 46),
    include_audit_rows=False,
):
    """Avalia três cenários com split temporal; rótulos entram só após o scoring."""
    transactions = load_transactions(db_path)
    train_mask, test_mask, cutoff = _split_by_time(transactions)
    if not seeds:
        raise ValueError('Informe ao menos uma semente.')

    structured_train, structured_test = _build_structured_features(
        transactions,
        train_mask,
    )
    scenario_features = {
        'somente_numericas': (structured_train, structured_test),
    }
    pipeline = TransactionAnomalyPipeline()
    try:
        compact_hf = pipeline._build_hugging_face_features(transactions, train_mask)
        hf_train = np.column_stack((structured_train, compact_hf[train_mask]))
        hf_test = np.column_stack((structured_test, compact_hf[test_mask]))
        scenario_features['numericas_mais_texto_hf'] = (hf_train, hf_test)
        hf_metadata = {
            'backend': 'huggingface',
            'descricoes_unicas_codificadas': pipeline.unique_description_count,
        }
    except Exception as error:
        LOGGER.warning(
            'Não foi possível gerar embeddings Hugging Face; cenário HF indisponível: %s',
            error,
        )
        scenario_features['numericas_mais_texto_hf'] = None
        hf_metadata = {
            'indisponivel': True,
            'erro': str(error),
            'backend': 'indisponivel',
        }

    train_indices = np.flatnonzero(train_mask)
    test_indices = np.flatnonzero(test_mask)
    compact_train, compact_test = TransactionAnomalyPipeline._build_tfidf_compact_features(
        transactions.iloc[train_indices]['descricao'].tolist(),
        transactions.iloc[test_indices]['descricao'].tolist(),
    )
    scenario_features['numericas_mais_texto_tfidf_compacto'] = (
        np.column_stack((structured_train, compact_train)),
        np.column_stack((structured_test, compact_test)),
    )

    scores_by_scenario = {}
    detectors_by_scenario = {}
    for name, features in scenario_features.items():
        if features is None:
            continue
        train_features, test_features = features
        detectors = []
        scores = []
        for seed in seeds:
            detector = _fit_detector(train_features, seed)
            detectors.append(detector)
            scores.append(-detector.decision_function(test_features))
        scores_by_scenario[name] = scores
        detectors_by_scenario[name] = detectors

    scenario_metadata = {
        'somente_numericas': {},
        'numericas_mais_texto_hf': hf_metadata,
        'numericas_mais_texto_tfidf_compacto': {
            'backend': 'tfidf_truncated_svd',
        },
    }
    path = Path(db_path)
    with closing(sqlite3.connect(f'{path.resolve().as_uri()}?mode=ro', uri=True)) as connection:
        label_rows = connection.execute(
            'SELECT transacao_id, fraude FROM rotulos_fraude ORDER BY transacao_id'
        ).fetchall()
    label_by_id = dict(label_rows)
    if len(label_by_id) != len(transactions) or set(label_by_id) != set(transactions['id']):
        raise ValueError('Transações e rótulos não correspondem um a um.')
    labels = transactions['id'].map(label_by_id).to_numpy(dtype=np.int8)
    labels_test = labels[test_mask]
    scenarios = {}
    for name, repetitions in scores_by_scenario.items():
        metrics = [
            _ranking_metrics(labels_test, scores)
            for scores in repetitions
        ]
        summary = {}
        for metric in ('pr_auc', 'precision_at_100', 'ganho_sobre_acaso'):
            values = np.asarray([result[metric] for result in metrics], dtype=float)
            summary[metric] = float(values.mean())
            summary[f'{metric}_std'] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        summary.update({
            'n_treino': int(train_mask.sum()),
            'n_teste': int(test_mask.sum()),
            'fraudes_teste': int(labels_test.sum()),
            'prevalencia_teste': float(labels_test.mean()),
            'sementes': list(seeds),
            **scenario_metadata[name],
        })
        scenarios[name] = summary
    result = {
        'cutoff_temporal': str(cutoff),
        'baseline_acaso': float(labels_test.mean()),
        'sementes': list(seeds),
        'cenarios': scenarios,
    }
    if include_audit_rows:
        if scenario_features.get('numericas_mais_texto_hf') is None:
            raise RuntimeError('Auditoria detalhada requer embeddings Hugging Face.')
        primary_index = list(seeds).index(42) if 42 in seeds else 0
        numeric_scores = scores_by_scenario['somente_numericas'][primary_index]
        hf_scores = scores_by_scenario['numericas_mais_texto_hf'][primary_index]
        hf_test = scenario_features['numericas_mais_texto_hf'][1]
        audit_transactions = transactions.iloc[np.flatnonzero(test_mask)].reset_index(drop=True)
        audit_rows = _select_audit_rows(
            audit_transactions,
            labels_test,
            numeric_scores,
            compact_hf[test_mask, 0],
            hf_scores,
        )

        selected_ids = [
            row['id']
            for group_rows in audit_rows.values()
            for row in group_rows
        ]
        id_to_position = {
            int(transaction_id): position
            for position, transaction_id in enumerate(audit_transactions['id'])
        }
        selected_positions = list(dict.fromkeys(
            id_to_position[transaction_id] for transaction_id in selected_ids
        ))
        feature_names = list(NUMERIC_FEATURE_COLUMNS) + [
            'score_risco_textual',
            *[
                f'embedding_pca_{index + 1}'
                for index in range(compact_hf.shape[1] - 1)
            ],
        ]
        reference_values = np.median(scenario_features['numericas_mais_texto_hf'][0], axis=0)
        explanations = _local_feature_sensitivities(
            detectors_by_scenario['numericas_mais_texto_hf'][primary_index],
            hf_test,
            selected_positions,
            feature_names,
            reference_values,
        )
        explanation_by_id = dict(zip(
            [int(audit_transactions.iloc[position]['id']) for position in selected_positions],
            explanations,
        ))
        for group_rows in audit_rows.values():
            for row in group_rows:
                row['features_mais_influentes'] = explanation_by_id[row['id']][:3]
        result['linhas_teste'] = audit_rows
    return result


def audit_description_leakage(db_path=DEFAULT_DB_PATH, sample_size=6, seed=42):
    """Audita frases por grupo de rótulo sem expor o rótulo ao gerador/modelo."""
    path = Path(db_path)
    with closing(sqlite3.connect(f'{path.resolve().as_uri()}?mode=ro', uri=True)) as connection:
        rows = connection.execute(
            '''
            SELECT t.id, t.descricao, r.fraude
            FROM transacoes AS t
            JOIN rotulos_fraude AS r ON r.transacao_id = t.id
            ORDER BY t.id
            '''
        ).fetchall()
    frame = pd.DataFrame(rows, columns=('id', 'descricao', 'fraude'))
    rng = np.random.default_rng(seed)
    samples = {}
    for label, name in ((1, 'fraudes'), (0, 'normais')):
        group = frame[frame['fraude'] == label]
        if len(group) < sample_size:
            raise ValueError(f'Não há {sample_size} exemplos de {name} para a auditoria.')
        travel_rows = group[
            group['descricao'].str.contains('durante viagem', case=False, regex=False)
        ]
        if travel_rows.empty:
            raise ValueError(f'Não há descrição "durante viagem" no grupo {name}.')
        travel_index = int(rng.choice(travel_rows.index.to_numpy()))
        remaining = group.index[group.index != travel_index].to_numpy()
        selected = np.concatenate((
            np.asarray([travel_index]),
            rng.choice(remaining, size=sample_size - 1, replace=False),
        ))
        samples[name] = group.loc[selected, ['id', 'descricao']].to_dict('records')

    during_travel = frame['descricao'].str.contains('durante viagem', case=False, regex=False)
    return {
        'fraudes': samples['fraudes'],
        'normais': samples['normais'],
        'durante_viagem_em_fraudes': int(
            during_travel[frame['fraude'] == 1].sum()
        ),
        'durante_viagem_em_normais': int(
            during_travel[frame['fraude'] == 0].sum()
        ),
    }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    started = time.perf_counter()
    print('Iniciando avaliação temporal; textos únicos serão codificados apenas uma vez.', flush=True)
    result = evaluate_scenarios()
    print(f'tempo_decorrido_segundos={time.perf_counter() - started:.1f}', flush=True)
    print(result, flush=True)
    print(audit_description_leakage(), flush=True)
