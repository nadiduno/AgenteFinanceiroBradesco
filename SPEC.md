# Especificação — Detecção experimental de anomalias transacionais

Status: Fases 1–3 concluídas. Fase 5 implementada e validada offline, integrada à Fase 4. Fase 6 concluída e validada offline. Duas tentativas reais anteriores foram bloqueadas pelo provedor (HTTP 404/503); nenhum relatório real foi produzido para auditar. O `.env` local será preenchido pelo usuário; nenhuma chamada ao provedor nesta fase.

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

- `config/agents.yaml` define Analista SQL, Cientista de Dados e Redator Executivo; `config/tasks.yaml` configura as três tarefas. `Process.sequential` executa Analista e Cientista; em seguida o Redator executa sua tarefa com o JSON original da tool, formato factual e schema. A Fase 5 controla localmente sua única correção, sem reiniciar a crew.
- O Analista usa `MCPServerAdapter` por stdio com `sys.executable`. O servidor aplica o authorizer do SQLite e permite leitura somente de `clientes` e `transacoes`; `rotulos_fraude` continua negada pelo próprio SQLite.
- `src/crew/tools.py` calcula o ranking somente com as features numéricas existentes, usando `USE_TEXT_FEATURES=false` como padrão, Isolation Forest com semente fixa e top-10. As três features são ordenadas por sensibilidade local heurística; a descrição é transportada apenas como contexto marcado “texto livre, não evidência”.
- O top-10 é calculado sobre a **base inteira**: tanto scaler quanto detector são ajustados e aplicados sobre todas as transações. Trata-se de **ranking de revisão, não avaliação de desempenho**; não deve ser comparado diretamente às métricas do split temporal da Fase 3. Essa ressalva consta nas limitações do relatório e em cada registro da tool.
- Na saída da tool, `valor_z_cliente` é limitado a ±10 e traz `limitado=true` quando houve clipping (`false` nos demais casos). O clipping é somente de apresentação: detector, scores, ordem e sensibilidade continuam usando as features numéricas anteriores. O z-score bruto aparece exclusivamente em `auditoria.z_scores_brutos`, nunca no JSON da tool ou no relatório do Redator.
- `valor_vs_mediana_cliente` é a razão do valor atual pela mediana das transações **anteriores do mesmo cliente**, ordenadas por data/hora e ID, excluindo a transação atual. Sem histórico ou com mediana não positiva, a tool informa explicitamente indisponibilidade; não substitui por zero nem inventa baseline. O Redator cita a razão em cada caso e substitui o fator z-score por ela quando ele está entre os três fatores; essa razão é explicativa, não uma nova feature do detector.
- O esquema Pydantic contém casos (`id_transacao`, `posicao`, `resumo`, `fatores`), recomendações e limitações. A integração dos guardrails concorrentes foi confirmada pelo usuário: o JSON do Redator é validado localmente com schema estrito e indicador `degradado`, sem retries ocultos do conversor `output_pydantic`. O relatório final preserva campos/valores da ferramenta e não aceita IDs, posições ou features divergentes. A descrição não fundamenta conclusões; os scores não são probabilidades.
- A execução `--dry-run` não inicializa CrewAI/MCP nem chama LLM; produz relatório determinístico e audita IDs, datas, valores e features diretamente contra o SQLite em modo somente leitura.
- Restrições de custo configuradas: `max_rpm=5`, `max_iter=2`, temperatura `0.1`, saída limitada e top-10.
- Retry por chamada Gemini: uma chamada inicial e até três novas tentativas, aguardando exatamente 5, 15 e 45 segundos, somente para HTTP 503/429. HTTP 401/404 falha imediatamente com orientação para conferir `LLM_MODEL` e a chave; outros códigos e falhas de rede não são repetidos. O SDK usa `attempts=1` e os agentes `max_retry_limit=0`; uma falha HTTP terminal bloqueia novas chamadas da mesma instância, inclusive tentativas do conversor de saída. A crew não é reiniciada. Configuração baseada nas APIs locais de CrewAI 1.15.34 e google-genai 3.0.0, sem novas dependências.
- `google-genai` foi instalado apenas na `.venv` e incluído em `requirements.txt`; CrewAI instalado: 1.15.34. `pip check` não apontou conflitos.

### Verificações e pendência

- A suíte completa executada com o Python da `.venv`: 16 testes passaram. Os casos de autorização cobrem acesso direto, identificador entre aspas, alias, CTE, JOIN, subconsulta, UNION, ATTACH e PRAGMA.
- O comando `python -m src.crew.main --dry-run` terminou com sucesso; a auditoria determinística confirmou os dez IDs, datas, valores e features contra o banco.
- Foram feitas duas execuções reais de uma rodada cada. Na primeira, o MCP iniciou e o provedor retornou HTTP 404 para o modelo configurado. Depois de autorização, houve uma segunda execução com o modelo sugerido fornecido apenas no ambiente daquele processo; o MCP iniciou, mas o provedor retornou HTTP 503 por alta demanda, após retries internos do SDK. O `.env` local continua com o identificador antigo. Nenhum relatório real foi produzido; por isso, IDs e valores de um relatório real não puderam ser conferidos. Não fazer nova tentativa sem atualizar o `.env` para modelo acessível e obter nova autorização.
- Nesta revisão, foram preservados os guardrails estritos recebidos por alterações concorrentes, após confirmação explícita do usuário; não foram feitas chamadas ao provedor.
- 2026-11-08: trabalho exclusivamente offline. Suíte completa: 35 testes passaram (`.venv\Scripts\python.exe -m unittest discover -s tests -v`); compilação e `git diff --check` aprovados. Testes de bloqueio separados por nome para SELECT direto, aspas, alias, CTE, JOIN, subconsulta, UNION, ATTACH e PRAGMA; retries testados com exceções simuladas e sleeps mockados. Dry-run completo conferido: dez IDs, valores/datas iguais ao banco e features iguais ao top-10. Nenhuma chamada real foi feita; autorização anterior não foi consumida nesta etapa.
- Revisão offline subsequente: suíte integrada com **63 testes aprovados**, incluindo clipping positivo/negativo, mediana point-in-time, exclusão do z-score bruto do relatório, preservação da razão e ressalva de base inteira. `test_mcp_stdio_allowed_rows_and_label_read_denied_without_llm` inicia o servidor por `sys.executable` com `MCPServerAdapter`, usa banco temporário e confirma linhas permitidas, erro de autorização para `rotulos_fraude` e consulta permitida novamente após a recusa. Nenhum LLM ou provedor foi chamado.

## 10. Fase 5 — guardrails offline

### Contrato e decisões

- `src/crew/schemas.py`: `extra="forbid"` e `strict=True` em todos os níveis; não aceita coerção de strings, floats ou booleanos para IDs/posições. São exatamente dez casos, com IDs positivos, posição >= 1, resumo não vazio de até 500 caracteres e exatamente três fatores. Recomendações e limitações são listas não vazias de textos não vazios.
- Lista fechada dos fatores apresentados: `valor_vs_mediana_cliente`, `cidade_nova_cliente`, `hora_fora_padrao_cliente`, `dispositivo_novo`, `idade_conta_destino_dias` e `qtd_transacoes_ultima_hora`. A substituição explicativa do z-score pela razão histórica da Fase 4 foi preservada; não muda o detector.
- Resumos curtos são **canônicos**, não paráfrases livres: preservam posição, tipo, canal, categoria, cidade, data/hora, valor e razão histórica. Fatores ficam na lista própria, sem duplicação no resumo. Essa restrição permite verificar todas as citações do caso e impede adicionar cidades, datas ou valores não vinculados à transação. Divergências são rejeitadas, nunca corrigidas silenciosamente.
- `src/crew/guardrails.py` confere todo ID no SQLite somente leitura e exige os mesmos dez IDs, ordem e posições da tool. Compara valor, data/hora, cidade, tipo, canal e categoria da tool com o banco, incluindo a formatação monetária; fatores, rótulos e valores devem corresponder exatamente à saída da tool.
- Cada número nos textos deve ocorrer nos dados estruturados da tool. Comparação por tokens numéricos completos e `Decimal`, com suporte à moeda brasileira e notação científica, não por substring. A descrição livre e os z-scores brutos exclusivos da auditoria não autorizam números no relatório. Recomendações/limitações continuam gerais e não podem incluir citações factuais específicas; a ressalva de base inteira é sempre preservada.
- Linguagem proibida em todos os textos: probabilidade (inclusive plural), chance de fraude, `%` ligado a fraude, fraude confirmada e menção a `rotulos_fraude`, com comparação sem distinção de caixa/acentos. O formato usa **prioridade de revisão** e **suspeita**.
- O Redator devolve JSON bruto, convertido pelo Pydantic **localmente**, em vez de `output_pydantic` do CrewAI: isso evita retries ocultos de conversão. Uma falha de schema, fatos, números ou linguagem gera exatamente **uma** nova execução do Redator com a mensagem do erro; Analista e Cientista não são repetidos.
- Se a correção também falhar, retorna o mesmo relatório determinístico do dry-run com `degradado=true`, validado pelo mesmo caminho. Saídas válidas e dry-run usam `degradado=false`, controlado pela aplicação. Falhas são registradas explicitamente. Se o banco/tool não permitir validar nem o fallback, a execução falha sem apresentar relatório; falhas do provedor não são mascaradas como erros de validação.

### Aceite e limites

- [x] Saídas fabricadas cobrem ID inexistente e fora do top-10, valor/data/cidade alterados, linguagem proibida, campo ausente/extra em todos os níveis, número ausente da tool e formato numérico inválido.
- [x] Testes verificam tipos estritos, textos/listas vazios, limite de resumo, lista fechada de fatores, duplicatas, ordem, divergências entre tool e banco, correção única e fallback validado.
- [x] Caminho real testado com agentes, MCP e geração simulados: uma execução da crew e no máximo duas do Redator, com mensagem do erro e sem conversor automático.
- [x] 23 testes de guardrails aprovados; dry-run completo aprovado, resumo máximo observado de 251 caracteres, compilação e `git diff --check` aprovados.
- [x] Suíte completa: 63 testes aprovados com `.venv\Scripts\python.exe -m unittest discover -s tests -v`, incluindo MCP stdio somente leitura e retries HTTP simulados. Os logs HTTP 429/503 e a negação SQLite são cenários esperados dos testes, não chamadas reais.
- Nenhuma chamada ao provedor, instalação de dependência ou alteração do banco local nesta fase. As mudanças offline concorrentes da Fase 4 foram integradas após confirmação explícita do usuário.
- Os guardrails não calibram scores nem comprovam suspeita; recomendações gerais ainda exigem revisão humana. A integração com resposta real do provedor permanece não exercitada.

## 11. Fase 6 — FastAPI

### Contrato confirmado

- `src/api/app.py`: `GET /health` público e `POST /analisar` autenticado
  por `X-API-Key`, conferido contra `API_ACCESS_KEY`, independente do Gemini.
  Chave ausente/incorreta recebe 401; servidor sem chave configurada, 503.
- Entrada Pydantic estrita, sem campos extras: `top_n` inteiro de 1 a 20
  (padrão 10) e `use_llm` booleano opcional (padrão false).
- Saída direta `ExecutiveReport`, incluindo `degradado`, sem auditoria.
  O schema aceita 1–20 casos; o validador exige exatamente os IDs, posições
  e quantidade do ranking solicitado. Fases 4/5 e CLI preservam top-10.
  Resumos, fatores, números, fatos e linguagem seguem os mesmos guardrails.
- Modo determinístico não inicializa LLM/MCP nem usa provedor.
  LLM somente com opt-in no payload e `ALLOW_LLM=true`; caso contrário,
  a solicitação explícita de LLM recebe 403, não uma resposta silenciosa.
- Crew e cálculo numérico fora do event loop, via threadpool.
  `LLM_MAX_CALLS_PER_MINUTE` (padrão 5) limita chamadas nativas em janela
  móvel de 60 segundos, compartilhada e protegida por lock por processo.
  Limite excedido retorna 429. No caminho API não há retries HTTP;
  a correção única do Redator continua sujeita ao mesmo orçamento.
- `LLM_TIMEOUT_SECONDS` (padrão 60, positivo até 300) limita a espera HTTP
  e o SDK; prazo excedido retorna 504. O guard cancela novas chamadas do
  trabalho após expirar. Threads/HTTP já em curso não são terminados à força.
  A resposta continua 504 se a crew encapsular ou suprimir a exceção do guard;
  uma execução que ultrapassou o prazo não é apresentada como sucesso.
- `CORS_ORIGINS` é array JSON de origens HTTP(S) exatas, sem wildcard,
  credenciais ou caminho. Padrão vazio; origem não permitida recebe 403,
  inclusive no preflight. Clientes sem Origin continuam sujeitos à chave.
- Respostas de erro nunca incluem dados de entrada, exceções, caminhos,
  chaves, traceback ou conteúdo da tabela de rótulos. UUID gerado pelo
  servidor em cada requisição aparece em `X-Request-ID`, erros e logs.
  Logs dos guardrails omitem o conteúdo da validação; logs de dependências
  no contexto da requisição são sanitizados, preservando evento/nível/ID.
- `.env` local não foi editado. Configuração inválida falha com mensagem
  genérica; nenhuma chave é exposta pela rota de saúde ou OpenAPI.
- SQLite inexistente é gerado no lifespan com semente 42, 5.000 transações
  e 500 clientes; banco existente não é sobrescrito. `DATABASE_URL` local
  é compartilhada entre ranking, guardrails e subprocesso MCP.
- `requirements.txt` sem torch/transformers; extras em `requirements-hf.txt`.
  Imports HF já são tardios; modelo padrão segue somente numérico.

### Validação e limites

- TestClient offline cobre saúde, autenticação, 422, CORS permitido/recusado,
  top_n 1/10/20, relatório factual validado, opt-in duplo, threadpool,
  limite entre requisições, timeout sem novas chamadas, erros/logs seguros
  e geração reprodutível apenas se o banco estiver ausente.
- Quota é por processo, não distribuída; múltiplos workers multiplicam o limite.
  Primeira geração simultânea do mesmo banco não é suportada.
- Não foi exercitado provedor real. Não há deploy nem push nesta etapa.
- Suíte completa: **86 testes aprovados** com
  `.venv\Scripts\python.exe -m unittest discover -s tests -v` (328,468 s).
  Após endurecer a tradução de exceções encapsuladas da crew para 429,
  os dois testes específicos passaram com
  `.venv\Scripts\python.exe -m unittest discover -s tests -p test_api.py -k rate_limit -v`
  (65,070 s). São 87 casos distintos na árvore final, 21 deles da API.
  Compilação de `src`/`tests` e `git diff --check` aprovados.
- Conclusão na branch `fase-6`, criada de `7eef22f` com todas as alterações
  pendentes preservadas, sem conflito. Suíte completa final: **90 testes
  aprovados** (91,468 s), incluindo **24 da API**. `pip check`, compilação
  e `git diff --check` aprovados; nenhuma dependência instalada.
- Teste em subprocesso bloqueia os imports de `torch` e `transformers` e
  confirma `/health` e análise determinística com relatório validado.
  Saída LLM inválida é recusada com 500 genérico; timeouts encapsulados
  ou suprimidos são testados com relógio simulado, sem provedor.
