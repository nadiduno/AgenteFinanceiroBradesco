# Especificação — Detecção experimental de anomalias transacionais

Status: Fases 1–3 concluídas. Implementação da Fase 4 e testes concluídos; a tentativa real única foi bloqueada pelo provedor (HTTP 404 para o modelo configurado), então não houve relatório real para auditar. Fase 4 aguarda novo modelo e autorização para outra tentativa; Fase 5 não iniciada.

## 1. Contexto e objetivo

- **Problema / evento a prevenir:** identificar transações sintéticas atípicas para análise experimental.
- **Quem usa e que decisão toma com o resultado:** não definido; a saída é apenas um sinal para inspeção, não uma decisão automática.
- **Horizonte e ação:** não aplicáveis nesta etapa.
- **Custo de erro:** não definido; não usar as saídas para decisões financeiras reais.

## 2. Dados

| Fonte | Local | Período | Sensível? | Observações |
|---|---|---|---|---|
| Transações sintéticas SQLite | `data/transacoes.db`, tabela `transacoes` | Datas sintéticas | Não contém dados reais | Descrições livres sem números/datas; variáveis estruturadas em colunas próprias |

O pipeline lê somente `transacoes`; não acessa `rotulos_fraude`. Os rótulos são consultados separadamente apenas para calcular métricas. O modelo Hugging Face, quando disponível, executa localmente em CPU.

## 3. Critérios de aceitação

- **Métrica principal e meta:** PR-AUC, precision@100 e ganho sobre acaso calculado pela prevalência observada no conjunto de teste, com divisão temporal e repetição em cinco sementes.
- **Baseline:** Isolation Forest com variáveis transacionais estruturadas, comparado ao mesmo detector acrescido do score textual e embeddings por cliente via PCA.
- **Validação:** três cenários (somente numéricas, numéricas + texto Hugging Face, numéricas + TF-IDF compacto); rótulos só entram após gerar scores do teste. Auditoria separada de 6 descrições por classe.
- **Calibração:** scores textuais e de anomalia são heurísticos e não representam probabilidades calibradas.

Resultado na base regenerada e semanticamente consistente (corte temporal 2023-10-21 17:07:36; treino 4.000; teste 1.000; 30 fraudes; prevalência real 3,0%). Métricas são média ± desvio-padrão amostral de cinco sementes do Isolation Forest; o ganho usa a prevalência real de 3,0%.

| Cenário | PR-AUC | Precision@100 | Ganho sobre acaso |
|---|---:|---:|---:|
| Somente numéricas | 0,7753 ± 0,0046 | 25,0% ± 0,0 pp | 8,33× ± 0,00 |
| Numéricas + texto Hugging Face | 0,7329 ± 0,0118 | 25,2% ± 0,84 pp | 8,40× ± 0,28 |
| Numéricas + TF-IDF compacto | 0,5567 ± 0,0190 | 20,6% ± 0,55 pp | 6,87× ± 0,18 |

O score Hugging Face foi calculado no ranking HF combinado; o cenário numérico continua sendo o controle. Cada valor de precision@100 corresponde, em média, a 25,0, 25,2 e 20,6 fraudes entre os cem primeiros, respectivamente. Há apenas um corte temporal e uma base sintética; a variação das cinco sementes mede a aleatoriedade do Isolation Forest, não a incerteza entre novas amostras. Trate as diferenças como dentro do ruído experimental desta avaliação, sem alegação de superioridade. O README futuro deve repetir explicitamente esta ressalva.

### Variação histórica da PR-AUC

A PR-AUC numérica de aproximadamente 0,39 da rodada inicial e a média 0,5760 ± 0,0269 da rodada seguinte não vieram de uma comparação controlada: a primeira era uma execução com uma semente e o conjunto numérico anterior; a segunda agregou cinco sementes após a troca para seis sinais históricos por cliente (`valor_z_cliente`, cidade nova, hora fora do padrão, dispositivo novo, idade da conta destino e quantidade de transações na última hora). A definição da métrica não mudou: ambas usam Average Precision (`average_precision_score`) em divisão temporal. A base e o gerador também evoluíram depois; portanto, não é possível atribuir a diferença a um único fator nem comparar diretamente esses valores com a rodada atual.

O gerador atual deriva o texto de tipo, canal e categoria. Os testes e a auditoria do banco encontraram zero contradições nas combinações verificadas, zero descrições com números/datas, 23% das fraudes com texto inocente e 6,45% das normais com texto suspeito. A frase “durante viagem” aparece nos dois grupos (1 fraude e 123 normais). O pipeline codificou 102 descrições únicas com o modelo Hugging Face em CPU; a avaliação final de cinco sementes e auditoria levou 406,9 segundos.

No treino há 500 clientes; a quantidade de transações por cliente tem mínimo 1, mediana 8, média 8 e máximo 19. Quarenta e sete clientes têm menos de cinco transações. A transação 2003 da base atual está no treino, não no teste: é fraude de R$ 83,26, com score numérico 0,2048 (semente 42; posição 24 entre 4.000). Seus sinais incluem z-score do valor −1,18, cidade nova, dispositivo novo e idade de destino de 7 dias. Isso não é a mesma linha antiga citada como R$ 2.335,54 e score ~0,02: a regeneração substituiu a base e não preservou aquela linha, então seu vetor antigo não pode ser reconstruído. O score do Isolation Forest é um valor bruto de ranking, não uma probabilidade; um valor baixo pode ocorrer quando o valor não é atípico frente ao histórico curto e os demais sinais numéricos não formam um padrão raro. Como 47 clientes têm histórico menor que cinco, recomenda-se avaliar shrinkage entre estatísticas do cliente e globais antes de confiar em z-scores individuais.

### Regras de descrição e qualidade factual

`src/utils/db_gen.py` define `DESCRIPTION_RULES`, uma tabela de chaves `(tipo_transacao, canal, categoria_estabelecimento)` para prefixos permitidos e marcador da categoria; `description_rule_violations` aplica a tabela linha a linha. O teste `test_5000_rows_follow_description_rules_without_violations` verifica 5.000 registros e exige zero violações. `test_same_seed_generates_identical_database_contents` gera dois bancos completos de 5.000 registros com a mesma semente e compara todas as linhas das tabelas `clientes`, `transacoes` e `rotulos_fraude`.

Descrições são texto livre para facilitar leitura, **não são evidência** de que o evento declarado ocorreu. Na base de 5.000 linhas, a auditoria das alegações explicitamente verificáveis encontrou:

| Alegação no texto | Ocorrências | Divergências das colunas/histórico |
|---|---:|---:|
| “horário incomum” vs. `hora_fora_padrao_cliente` | 40 | 36 |
| “dispositivo novo” vs. `dispositivo_novo` | 55 | 35 |
| “local não habitual” vs. `cidade_nova_cliente` | 13 | 10 |
| Linhas com pelo menos uma divergência (deduplicadas) | — | 81 |

O ID 4562, por exemplo, descreve “horário incomum” às 17:43, embora `hora_fora_padrao_cliente` seja 0. Essas divergências decorrem da liberdade textual e do ruído deliberado do gerador; consumidores devem usar as colunas estruturadas para fatos e não inferir fatos a partir da descrição.

### Padrão de modelo e limite da evidência

O modelo numérico é o padrão de execução (`USE_TEXT_FEATURES=false`); texto permanece apenas como comparação experimental. Na última avaliação registrada, adicionar HF reduziu PR-AUC de 0,7753 para 0,7329 e alterou Precision@100 de 25,0% para 25,2% (diferença dentro do ruído observado); TF-IDF compacto teve resultados inferiores. O texto não trouxe ganho consistente porque é derivado dos mesmos sinais transacionais já codificados nas features, adiciona formulações deliberadamente ruidosas e contém afirmações não verificadas. Portanto, ele não representa uma fonte independente de evidência nem justifica torná-lo padrão.

Todos os números desta especificação vêm de dados gerados sinteticamente, de um único split temporal e de cinco sementes do Isolation Forest. As sementes medem variabilidade do algoritmo, não variação entre clientes, períodos, instituições ou dados reais. Resultados não demonstram eficácia em fraude real; são necessários dados representativos, rotulados e avaliação apropriada antes de qualquer uso operacional. Nenhuma métrica foi recalculada nesta auditoria de regras e afirmações.

## 4. Entregáveis

- [x] Pipeline executável com Isolation Forest e embeddings de texto.
- [x] Fallback TF-IDF com aviso de log.
- [x] Testes funcionais.
- [x] Usuário aprovou a amostra de descrições sintéticas.
- [x] Métricas temporais dos três cenários, com média/desvio em cinco sementes, registradas.
- [x] Auditoria de vazamento concluída com exemplos de ambas as classes.

## 5. Requisitos não funcionais

- Inferência de embeddings em CPU.
- Nenhum rótulo de fraude como entrada, feature ou critério de ajuste.
- Nenhuma dependência adicional.
- Nenhum envio das transações a serviços externos; download do modelo é opcional e contém apenas artefatos públicos.

## 6. Fora de escopo

- Aprovar transações, bloquear contas ou tomar decisões financeiras reais.
- Treinar ou calibrar o modelo com `rotulos_fraude`.
- Alegações de desempenho ou qualidade preditiva sem critérios e avaliação confirmados.

## 7. Riscos e hipóteses em aberto

- Descrições textuais são amostradas sem ler os rótulos; atributos estruturados podem influenciar os sinais descritos. Na base atual, 23% das fraudes recebem texto inocente e 6,45% das transações normais recebem texto suspeito.
- A similaridade com protótipos depende da cobertura semântica e não equivale à probabilidade de fraude.
- A redução PCA representa comportamento textual médio por cliente, não um histórico temporal validado.

## 8. Definition of Done

- [x] Código restringe entrada a transações e não lê os rótulos.
- [x] Fallback TF-IDF produz embeddings e emite aviso.
- [x] Testes funcionais da geração verificados.
- [x] Métricas temporais dos três cenários e ganho sobre acaso usando prevalência real registrados.
- [x] Usuário aprovou as descrições.
- [x] Auditoria mostra 6 descrições por classe e verifica frase compartilhada entre grupos.

## 9. Fase 4 — agentes e relatório de prioridade

### Implementação

- `config/agents.yaml` define Analista SQL, Cientista de Dados e Redator Executivo; `config/tasks.yaml` e `Process.sequential` conectam as três tarefas.
- O Analista usa `MCPServerAdapter` por stdio com `sys.executable`. O servidor aplica o authorizer do SQLite e permite leitura somente de `clientes` e `transacoes`; `rotulos_fraude` continua negada pelo próprio SQLite.
- `src/crew/tools.py` calcula o ranking somente com as features numéricas existentes, usando `USE_TEXT_FEATURES=false` como padrão, Isolation Forest com semente fixa e top-10. As três features são ordenadas por sensibilidade local heurística; a descrição é transportada apenas como contexto marcado “texto livre, não evidência”.
- O esquema `output_pydantic` contém casos (`id_transacao`, `posicao`, `resumo`, `fatores`), recomendações e limitações. O relatório final preserva campos/valores da ferramenta e não aceita IDs, posições ou features divergentes. A descrição não fundamenta conclusões; os scores não são probabilidades.
- A execução `--dry-run` não inicializa CrewAI/MCP nem chama LLM; produz relatório determinístico e audita IDs, datas, valores e features diretamente contra o SQLite em modo somente leitura.
- Restrições de custo configuradas: `max_rpm=5`, `max_iter=2`, temperatura `0.1`, saída limitada e top-10.
- `google-genai` foi instalado apenas na `.venv` e incluído em `requirements.txt`; CrewAI instalado: 1.15.34. `pip check` não apontou conflitos.

### Verificações e pendência

- A suíte completa executada com o Python da `.venv`: 16 testes passaram. Os casos de autorização cobrem acesso direto, identificador entre aspas, alias, CTE, JOIN, subconsulta, UNION, ATTACH e PRAGMA.
- O comando `python -m src.crew.main --dry-run` terminou com sucesso; a auditoria determinística confirmou os dez IDs, datas, valores e features contra o banco.
- Foi feita uma única tentativa real. O servidor MCP iniciou, mas o provedor retornou HTTP 404 informando que o modelo configurado não está disponível para novos usuários. Nenhum relatório real foi produzido; por isso, IDs e valores de um relatório real não puderam ser conferidos. Não repetir a chamada sem atualizar `LLM_MODEL` e obter autorização.
- A validação de fatos/IDs é específica da Fase 4; guardrails estritos continuam fora de escopo até a Fase 5.
